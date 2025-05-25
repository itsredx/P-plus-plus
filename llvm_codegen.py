from llvmlite import ir, binding
from ir_nodes import * 
from typing import Dict, Optional

class LLVMCodeGenerator:
    def __init__(self):
        self.module: Optional[ir.Module] = None
        self.builder: Optional[ir.IRBuilder] = None
        self.current_function: Optional[ir.Function] = None
        
        self.symbol_table: Dict[str, ir.Value] = {}
        self.variable_types: Dict[str, IRType] = {} # Stores IRType for variables
        self.function_cache: Dict[str, ir.Function] = {}

        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        
        # Standard C library function signatures (can be expanded)
        self.STD_LIB_FUNCS = {
            "printf": ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True),
            "puts": ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()]),
            "malloc": ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(64)]), # size_t typically i64
            "free": ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()])
        }


    def _get_std_lib_func(self, name: str) -> ir.Function:
        if name in self.function_cache:
            return self.function_cache[name]
        
        if name not in self.STD_LIB_FUNCS:
            raise NameError(f"Standard library function '{name}' not defined in STD_LIB_FUNCS.")

        func_type = self.STD_LIB_FUNCS[name]
        llvm_func = self.module.globals.get(name)
        if llvm_func is None or not isinstance(llvm_func, ir.Function):
            llvm_func = ir.Function(self.module, func_type, name=name)
        
        self.function_cache[name] = llvm_func
        return llvm_func

    def _ir_type_to_llvm_type(self, ir_type: IRType) -> ir.Type:
        if isinstance(ir_type, IRTypeInt):
            return ir.IntType(ir_type.width)
        elif isinstance(ir_type, IRTypeFloat):
            if ir_type.width == 32: return ir.FloatType()
            elif ir_type.width == 64: return ir.DoubleType()
            else: raise ValueError(f"Unsupported float width: {ir_type.width}")
        elif isinstance(ir_type, IRTypeBool):
            return ir.IntType(1)
        elif isinstance(ir_type, IRTypeString): # Represents char*
            return ir.IntType(8).as_pointer()
        elif isinstance(ir_type, IRTypeVoid):
            return ir.VoidType()
        elif isinstance(ir_type, IRPointerType): # Phase 2
            if isinstance(ir_type.pointee_type, IRTypeVoid):
                # Represent void* as i8* in LLVM
                return ir.IntType(8).as_pointer()
            llvm_pointee_type = self._ir_type_to_llvm_type(ir_type.pointee_type)
            return ir.PointerType(llvm_pointee_type)
        elif isinstance(ir_type, IRTypeCustom):
            # Basic support for named struct types (assuming they are defined elsewhere or opaque)
            existing_type = self.module.context.get_identified_type(ir_type.name)
            if existing_type: return existing_type
            # Create an opaque struct type if it's the first time we see this name
            # return self.module.context.get_identified_type(ir_type.name) # Creates opaque
            raise NotImplementedError(f"Custom type IRTypeCustom('{ir_type.name}') to LLVM: struct definitions not fully supported yet.")
        raise NotImplementedError(f"Unsupported IRType: {type(ir_type)}")

    def _generate_global_string(self, value: str, name_hint: str = ".str") -> ir.GlobalVariable:
        value_bytes = value.encode('utf8') + b'\x00'
        str_type = ir.ArrayType(ir.IntType(8), len(value_bytes))
        for gvar in self.module.globals.values():
            if gvar.name.startswith(".str") and isinstance(gvar.initializer, ir.Constant) and \
               gvar.initializer.type == str_type and gvar.initializer.constant == bytearray(value_bytes):
                return gvar
        global_str = ir.GlobalVariable(self.module, str_type, name=self.module.get_unique_name(name_hint))
        global_str.linkage = 'private'
        global_str.global_constant = True
        global_str.initializer = ir.Constant(str_type, bytearray(value_bytes))
        return global_str

    def _generate_expression(self, ir_expr: IRExpression, is_global_initializer=False) -> ir.Value:
        if isinstance(ir_expr, IRLiteral):
            llvm_type = self._ir_type_to_llvm_type(ir_expr.ir_type)
            if isinstance(ir_expr.ir_type, IRTypeString): # String literal value itself
                if not is_global_initializer:
                    global_str_var = self._generate_global_string(ir_expr.value)
                    return self.builder.gep(global_str_var, 
                                            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], 
                                            name=self.module.get_unique_name("strptr"))
                else: # Global string array initializer
                    value_bytes = ir_expr.value.encode('utf8') + b'\x00'
                    return ir.Constant(ir.ArrayType(ir.IntType(8), len(value_bytes)), bytearray(value_bytes))
            elif isinstance(ir_expr.ir_type, IRPointerType) and ir_expr.value is None: # Null pointer
                return ir.Constant(llvm_type, None) # Returns appropriate null pointer for the type
            return ir.Constant(llvm_type, ir_expr.value)

        if is_global_initializer:
             raise ValueError(f"Global variable initializers must be constant literals. Got {type(ir_expr)}.")

        # --- Non-constant expressions below ---
        if isinstance(ir_expr, IRIdentifier):
            ptr = self.symbol_table.get(ir_expr.name)
            if ptr is None: raise NameError(f"Undefined variable: {ir_expr.name}")
            
            # If the identifier points to a global variable that is an array (e.g. global string defined as array),
            # "using" it as an expression should yield a pointer to its first element.
            if isinstance(ptr, ir.GlobalVariable) and isinstance(ptr.value_type, ir.ArrayType):
                # This is common for global string arrays where we want char* behavior.
                if isinstance(ptr.value_type.element, ir.IntType) and ptr.value_type.element.width == 8:
                    return self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], name=ir_expr.name + ".ptr")
            return self.builder.load(ptr, name=ir_expr.name)

        elif isinstance(ir_expr, IRBinaryOp):
            lhs = self._generate_expression(ir_expr.left)
            rhs = self._generate_expression(ir_expr.right)
            op_type = lhs.type 
            
            if isinstance(op_type, ir.IntType):
                ops = {'+': self.builder.add, '-': self.builder.sub, '*': self.builder.mul, '/': self.builder.sdiv, '%': self.builder.srem,
                       '==': lambda l,r,n: self.builder.icmp_signed('==', l, r, name=n),
                       '!=': lambda l,r,n: self.builder.icmp_signed('!=', l, r, name=n),
                       '<': lambda l,r,n: self.builder.icmp_signed('<', l, r, name=n),
                       '<=': lambda l,r,n: self.builder.icmp_signed('<=', l, r, name=n),
                       '>': lambda l,r,n: self.builder.icmp_signed('>', l, r, name=n),
                       '>=': lambda l,r,n: self.builder.icmp_signed('>=', l, r, name=n)}
                if ir_expr.op in ops: return ops[ir_expr.op](lhs, rhs, name='i_tmp')
            elif isinstance(op_type, (ir.FloatType, ir.DoubleType)):
                ops = {'+': self.builder.fadd, '-': self.builder.fsub, '*': self.builder.fmul, '/': self.builder.fdiv,
                       '==': lambda l,r,n: self.builder.fcmp_ordered('==', l, r, name=n),
                       '!=': lambda l,r,n: self.builder.fcmp_ordered('!=', l, r, name=n),
                       '<': lambda l,r,n: self.builder.fcmp_ordered('<', l, r, name=n),
                       '<=': lambda l,r,n: self.builder.fcmp_ordered('<=', l, r, name=n),
                       '>': lambda l,r,n: self.builder.fcmp_ordered('>', l, r, name=n),
                       '>=': lambda l,r,n: self.builder.fcmp_ordered('>=', l, r, name=n)}
                if ir_expr.op in ops: return ops[ir_expr.op](lhs, rhs, name='f_tmp')
            # Pointer arithmetic (simplified: only + and - with integer)
            elif isinstance(op_type, ir.PointerType) and isinstance(rhs.type, ir.IntType):
                 if ir_expr.op == '+': return self.builder.gep(lhs, [rhs], name='ptr_add')
                 # For ptr - int, GEP with negated int.
                 elif ir_expr.op == '-': 
                     neg_rhs = self.builder.neg(rhs, name='neg_idx')
                     return self.builder.gep(lhs, [neg_rhs], name='ptr_sub')
            # Pointer difference
            elif ir_expr.op == '-' and isinstance(lhs.type, ir.PointerType) and isinstance(rhs.type, ir.PointerType):
                # Ensure pointers are of the same type for a meaningful difference.
                # Result is typically ptrdiff_t (often i64).
                # lhs_int = self.builder.ptrtoint(lhs, ir.IntType(64))
                # rhs_int = self.builder.ptrtoint(rhs, ir.IntType(64))
                # byte_diff = self.builder.sub(lhs_int, rhs_int)
                # pointee_size = lhs.type.pointee.get_abi_size(self.module.data_layout)
                # return self.builder.sdiv(byte_diff, ir.Constant(ir.IntType(64), pointee_size), name='ptr_diff')
                # For now, ptrdiff is complex, let's skip full implementation unless required.
                raise NotImplementedError(f"Pointer difference not fully implemented.")

            raise NotImplementedError(f"Binary operation {ir_expr.op} for type {op_type} or operand combination.")

        elif isinstance(ir_expr, IRFunctionCall):
            # Builtin print handling
            if ir_expr.callee_name == "print" and len(ir_expr.args) == 1:
                arg_val = self._generate_expression(ir_expr.args[0])
                # Determine IRType of the argument for correct print formatting
                arg_ir_type = None
                if isinstance(ir_expr.args[0], IRIdentifier): arg_ir_type = self.variable_types.get(ir_expr.args[0].name)
                elif isinstance(ir_expr.args[0], IRLiteral): arg_ir_type = ir_expr.args[0].ir_type
                elif isinstance(ir_expr.args[0], IRDereference): # Type of *ptr is pointee type of ptr
                    ptr_expr_for_type = ir_expr.args[0].pointer_expr
                    if isinstance(ptr_expr_for_type, IRIdentifier):
                        ptr_ir_type = self.variable_types.get(ptr_expr_for_type.name)
                        if isinstance(ptr_ir_type, IRPointerType): arg_ir_type = ptr_ir_type.pointee_type
                # Add more cases if other expressions can be printed

                if isinstance(arg_ir_type, IRTypeString) or \
                   (isinstance(arg_val.type, ir.PointerType) and isinstance(arg_val.type.pointee, ir.IntType) and arg_val.type.pointee.width == 8):
                    return self.builder.call(self._get_std_lib_func("puts"), [arg_val])
                else:
                    fmt_str, llvm_arg = "", arg_val
                    if isinstance(arg_val.type, ir.IntType) and arg_val.type.width == 1: fmt_str = "%d\\0A\\00" # Bool
                    elif isinstance(arg_val.type, ir.IntType): fmt_str = "%d\\0A\\00" # Int
                    elif isinstance(arg_val.type, (ir.FloatType, ir.DoubleType)): fmt_str = "%f\\0A\\00" # Float/Double
                    elif isinstance(arg_val.type, ir.PointerType): 
                        fmt_str = "%p\\0A\\00" # Pointer address
                        llvm_arg = self.builder.ptrtoint(arg_val, ir.IntType(64)) # Print as integer address
                    else: fmt_str = "Unsupported type for print\\0A\\00"
                    
                    fmt_var = self._generate_global_string(fmt_str, ".fmt")
                    fmt_ptr = self.builder.gep(fmt_var, [ir.Constant(ir.IntType(32),0), ir.Constant(ir.IntType(32),0)])
                    return self.builder.call(self._get_std_lib_func("printf"), [fmt_ptr, llvm_arg])

            # General function call
            callee_func = self.module.globals.get(ir_expr.callee_name)
            if not callee_func or not isinstance(callee_func, ir.Function):
                callee_func = self.function_cache.get(ir_expr.callee_name) # Check cache for stdlib funcs
                if not callee_func: raise NameError(f"Function {ir_expr.callee_name} not found.")
            
            llvm_args = [self._generate_expression(arg) for arg in ir_expr.args]
            return self.builder.call(callee_func, llvm_args)

        # Phase 2 Expression Nodes
        elif isinstance(ir_expr, IRAddressOf):
            ptr = self.symbol_table.get(ir_expr.variable_name)
            if ptr is None: raise NameError(f"Undefined variable for address-of: {ir_expr.variable_name}")
            # ptr from symbol_table is already the address (alloca or global var)
            return ptr 
        elif isinstance(ir_expr, IRDereference):
            llvm_pointer_val = self._generate_expression(ir_expr.pointer_expr)
            if not isinstance(llvm_pointer_val.type, ir.PointerType):
                raise TypeError(f"Cannot dereference non-pointer type: {llvm_pointer_val.type}")
            return self.builder.load(llvm_pointer_val, name="deref_tmp")
        elif isinstance(ir_expr, IRMalloc):
            malloc_func = self._get_std_lib_func("malloc")
            llvm_alloc_type = self._ir_type_to_llvm_type(ir_expr.alloc_ir_type)
            
            # Calculate size: num_elements * sizeof(element_type)
            llvm_num_elements_val = self._generate_expression(ir_expr.size_expr)
            # Ensure num_elements is i64 for multiplication with type_size_llvm
            if llvm_num_elements_val.type.width < 64:
                llvm_num_elements_val = self.builder.sext(llvm_num_elements_val, ir.IntType(64))
            elif llvm_num_elements_val.type.width > 64: # Should not happen if size_expr is int32/int64
                 llvm_num_elements_val = self.builder.trunc(llvm_num_elements_val, ir.IntType(64))


            type_size = llvm_alloc_type.get_abi_size(self.module.data_layout)
            type_size_llvm = ir.Constant(ir.IntType(64), type_size)
            
            total_bytes = self.builder.mul(llvm_num_elements_val, type_size_llvm, name="total_bytes")
            
            raw_ptr = self.builder.call(malloc_func, [total_bytes], name="malloc_raw_ptr")
            typed_ptr = self.builder.bitcast(raw_ptr, ir.PointerType(llvm_alloc_type), name="typed_ptr")
            return typed_ptr

        raise NotImplementedError(f"IRExpression type {type(ir_expr)} not implemented.")

    def _generate_statement(self, ir_stmt: IRStatement):
        if isinstance(ir_stmt, IRLocalVariable):
            var_name = ir_stmt.name
            llvm_type = self._ir_type_to_llvm_type(ir_stmt.ir_type)
            self.variable_types[var_name] = ir_stmt.ir_type 
            
            # Store alloca in the entry block of the current function
            # This is simpler than finding the "current" basic block if builder's position changes.
            # For variables that must be in entry (like parameters), this is fine.
            # For locals within nested blocks (if P-- gets them), this might need refinement.
            current_block = self.builder.block
            with self.builder.goto_entry_block():
                 alloca = self.builder.alloca(llvm_type, name=var_name + ".addr")
            self.builder.position_at_end(current_block) # Restore builder position
            self.symbol_table[var_name] = alloca
            
            if ir_stmt.initializer:
                init_val = self._generate_expression(ir_stmt.initializer)
                # Ensure type compatibility for store, especially for null pointers (i8*) -> typed pointers
                if alloca.type.pointee != init_val.type:
                    if isinstance(alloca.type.pointee, ir.PointerType) and isinstance(init_val.type, ir.PointerType):
                        # This handles cases like i8* (null) being assigned to i32* variable
                        init_val = self.builder.bitcast(init_val, alloca.type.pointee)
                    # Add more sophisticated checks or error if types are fundamentally incompatible
                    # For now, this primarily handles pointer type mismatches for null.
                self.builder.store(init_val, alloca)

        elif isinstance(ir_stmt, IRAssign): # var = value
            value_llvm = self._generate_expression(ir_stmt.value)
            target_ptr = self.symbol_table.get(ir_stmt.target.name)
            if target_ptr is None: raise NameError(f"Assignment to undefined variable: {ir_stmt.target.name}")
            self.builder.store(value_llvm, target_ptr)

        elif isinstance(ir_stmt, IRReturn):
            if ir_stmt.value:
                ret_val = self._generate_expression(ir_stmt.value)
                self.builder.ret(ret_val)
            else:
                self.builder.ret_void()
        
        elif isinstance(ir_stmt, IRExpressionStatement):
            self._generate_expression(ir_stmt.expression) 

        # Phase 2 Statement Nodes
        elif isinstance(ir_stmt, IRPointerAssign): # *ptr_expr = value
            llvm_val_to_store = self._generate_expression(ir_stmt.value_to_assign_expr)
            llvm_ptr_to_store_to = self._generate_expression(ir_stmt.target_pointer_expr)
            if not isinstance(llvm_ptr_to_store_to.type, ir.PointerType):
                 raise TypeError(f"Cannot assign to non-pointer type via pointer assignment: {llvm_ptr_to_store_to.type}")
            self.builder.store(llvm_val_to_store, llvm_ptr_to_store_to)

        elif isinstance(ir_stmt, IRFree):
            free_func = self._get_std_lib_func("free")
            llvm_ptr_to_free = self._generate_expression(ir_stmt.pointer_expr)
            # Free expects i8*, so bitcast if necessary
            if str(llvm_ptr_to_free.type) != "i8*": # Simple string comparison for type
                llvm_ptr_to_free = self.builder.bitcast(llvm_ptr_to_free, ir.IntType(8).as_pointer(), name="i8_ptr_for_free")
            self.builder.call(free_func, [llvm_ptr_to_free])
        else:
            raise NotImplementedError(f"IRStatement type {type(ir_stmt)} not implemented.")

    def _generate_function(self, ir_func: IRFunction):
        # Save and clear function-specific state
        prev_symbol_table_func_scope = self.symbol_table.copy() # Save global + previous func state
        prev_variable_types_func_scope = self.variable_types.copy()
        
        # Only function's own params and locals should be in its direct symbol_table view.
        # Globals are resolved via module.globals or a separate global symbol table.
        # For simplicity here, we are merging scopes. A more robust symbol table would handle nested scopes.
        # We will clear symbol_table for locals/params for this function.
        # Globals are already in self.module.globals and self.symbol_table (if needed for lookup).
        
        # Let's refine: symbol_table for a function should start fresh for its allocas
        # and then be merged with globals for lookups.
        # For now, this simplified approach uses one symbol_table that accumulates.
        # This means variable names must be unique across functions or handled carefully.
        # A better way: `func_sym_table = ScopedSymbolTable(parent=self.global_sym_table)`
        # For now:
        # self.symbol_table.clear() # This would clear globals too, bad.
        # self.variable_types.clear() # Same.

        # Let's use a temporary overlay for function scope
        original_symbol_table = self.symbol_table.copy()
        original_variable_types = self.variable_types.copy()

        llvm_return_type = self._ir_type_to_llvm_type(ir_func.return_type)
        llvm_param_types = [self._ir_type_to_llvm_type(p.ir_type) for p in ir_func.params]
        func_type = ir.FunctionType(llvm_return_type, llvm_param_types)
        
        llvm_f = self.module.globals.get(ir_func.name)
        if llvm_f is None: 
            llvm_f = ir.Function(self.module, func_type, name=ir_func.name)
        elif not isinstance(llvm_f, ir.Function):
             raise TypeError(f"Name {ir_func.name} exists but is not a function.")
        
        self.current_function = llvm_f
        
        entry_block = llvm_f.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)

        # Parameters: create allocas and store initial argument values
        for i, ir_param in enumerate(ir_func.params):
            param_name = ir_param.name
            param_llvm_type = llvm_f.args[i].type # Use type from LLVM function argument
            self.variable_types[param_name] = ir_param.ir_type # Store original IRType

            alloca = self.builder.alloca(param_llvm_type, name=param_name + ".addr")
            self.symbol_table[param_name] = alloca
            self.builder.store(llvm_f.args[i], alloca)
            llvm_f.args[i].name = param_name

        for stmt in ir_func.body:
            self._generate_statement(stmt)

        # Ensure terminator
        if not entry_block.is_terminated:
            if isinstance(llvm_return_type, ir.VoidType):
                self.builder.ret_void()
            else: # Non-void function must have returned explicitly via IRReturn
                  # If we reach here, it's like a missing return in source.
                  # LLVM requires blocks to be terminated. For robust error reporting,
                  # this should ideally be caught earlier (semantic analysis or IR validation).
                  # For now, if main, return 0. Otherwise, it's undefined behavior.
                if ir_func.name == "main" and isinstance(llvm_return_type, ir.IntType):
                    self.builder.ret(ir.Constant(llvm_return_type, 0))
                else:
                    # This will likely lead to LLVM verification error if block is not terminated.
                    # Or insert an 'unreachable' instruction if that's the desired semantic.
                    # self.builder.unreachable()
                    pass

        # Restore symbol table state for outer scope (e.g. globals)
        self.symbol_table = original_symbol_table
        self.variable_types = original_variable_types
        self.current_function = None


    def _generate_global_variable(self, ir_global: IRGlobalVariable):
        var_name = ir_global.name
        self.variable_types[var_name] = ir_global.ir_type 
        
        llvm_var_type: ir.Type
        initializer_val: Optional[ir.Constant] = None

        if ir_global.initializer:
            init_val_const = self._generate_expression(ir_global.initializer, is_global_initializer=True)
            llvm_var_type = self._ir_type_to_llvm_type(ir_global.ir_type) # Get the declared type
            
            # Ensure initializer type matches the global variable's declared type
            # Especially for pointers (e.g. null i8* vs typed pointer like i32*)
            if llvm_var_type != init_val_const.type:
                if isinstance(llvm_var_type, ir.PointerType) and isinstance(init_val_const.type, ir.PointerType):
                    # This allows global ptr = NULL (where NULL might be i8* const) to be cast to specific ptr type
                    # Note: LLVM globals require constant initializers, bitcast of constants is complex.
                    # For simplicity, if it's a null pointer, we use the correctly typed null.
                    if isinstance(init_val_const, ir.Constant) and init_val_const.is_null:
                         initializer_val = ir.Constant(llvm_var_type, None)
                    else:
                        # Direct bitcast of non-null constant pointers might not be allowed or straightforward.
                        # This path should ideally ensure init_val_const is already correctly typed by IRLiteral(None, specific_ptr_type)
                        raise TypeError(f"Global initializer type mismatch for '{var_name}'. Expected {llvm_var_type}, got {init_val_const.type}. Bitcasting constant initializers is tricky.")
                else:
                    raise TypeError(f"Global initializer type mismatch for '{var_name}'. Expected {llvm_var_type}, got {init_val_const.type}.")
            else:
                initializer_val = init_val_const
        else:
            llvm_var_type = self._ir_type_to_llvm_type(ir_global.ir_type)
            initializer_val = ir.Constant(llvm_var_type, None) # Zeroinitializer

        llvm_gvar = ir.GlobalVariable(self.module, llvm_var_type, name=var_name)
        llvm_gvar.linkage = 'internal' 
        llvm_gvar.initializer = initializer_val
        self.symbol_table[var_name] = llvm_gvar


    def generate_llvm_ir(self, ir_program: IRProgram) -> str:
        # Use module name from first function/global or default
        module_name = "main_module"
        if ir_program.declarations:
            # Try to get a name from the first declaration if it's a function or global
            # This is just for cosmetic naming of the LLVM module.
            first_decl_name_attr = getattr(ir_program.declarations[0], 'name', None)
            if isinstance(first_decl_name_attr, str):
                module_name = first_decl_name_attr

        self.module = ir.Module(name=module_name)
        
        target_triple = binding.get_default_triple()
        self.module.triple = target_triple
        target = binding.Target.from_default_triple()
        # TODO: Ensure target_machine is created and used for data_layout
        # This was in original provided snippet but seems to be missing in later edits.
        # It's crucial for get_abi_size.
        self.target_machine = target.create_target_machine() # Store it for access
        self.module.data_layout = self.target_machine.target_data


        self.symbol_table.clear()
        self.variable_types.clear()
        self.function_cache.clear()

        # Process global variables first to populate symbol table and variable_types
        for decl in ir_program.declarations:
            if isinstance(decl, IRGlobalVariable):
                self._generate_global_variable(decl)
        
        # Process functions
        for decl in ir_program.declarations:
            if isinstance(decl, IRFunction):
                self._generate_function(decl)
        
        # IRImport nodes are handled implicitly by _get_std_lib_func when functions like malloc/free/printf are called.

        return str(self.module)

if __name__ == '__main__':
    from ast_to_ir import ASTToIRConverter 
    from tokenizer import tokenize
    from parser import Parser 

    codegen = LLVMCodeGenerator()
    ir_converter = ASTToIRConverter()

    # More complex example for Phase 2
    pointer_code_example = """
    g_ptr AS *int // Global pointer

    DEF manipulate_ptr(p AS *int, val AS int) -> void:
        INDENT
        *p = val + *p // Dereference for read and write
        RETURN
        DEDENT

    DEF main() -> int:
        INDENT
        x AS int = 10
        px AS *int
        
        px = &x
        *px = 20 // x is now 20
        
        manipulate_ptr(px, 5) // x should become 20 + 5 = 25
        
        arr AS *int = malloc(int, 3) // Allocate array of 3 ints
        *arr = 1          // arr[0] = 1
        // arr[1] = 2 // Needs pointer arithmetic: *(arr + 1) = 2
        // For now, direct indexing not supported, but can assign to *arr
        
        free(arr)
        
        RETURN *px // Should be 25
        DEDENT
    """
    print(f"\n--- Testing LLVM Codegen with Pointer Code ---")
    print(f"Source Code:\n{pointer_code_example.strip()}")
    try:
        tokens = tokenize(pointer_code_example)
        # Assume tokenizer produces INDENT/DEDENT or they are filtered/handled before parser
        tokens_for_parser = [t for t in tokens if t.type not in ('SKIP', 'NEWLINE')]
            
        parser = Parser(tokens_for_parser)
        ast_program = parser.parse_program()
        print("\nAST Program (first decl):", ast_program.declarations[0] if ast_program.declarations else "No decls")

        ir_program = ir_converter.translate_program(ast_program)
        print("\nIR Program (first decl):", ir_program.declarations[0] if ir_program.declarations else "No decls")
            
        llvm_ir_str = codegen.generate_llvm_ir(ir_program)
        print("\nGenerated LLVM IR:")
        print(llvm_ir_str)
            
        # Optional: Validate LLVM IR (can be slow, useful for debugging)
        # llvmlite_module_ref = binding.parse_assembly(llvm_ir_str)
        # llvmlite_module_ref.verify() 
        # print("LLVM IR Validated.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------------")
