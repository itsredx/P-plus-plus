from llvmlite import ir, binding
from ir_nodes import * # Assuming this has all the structured IR node definitions
from typing import Dict, Optional

class LLVMCodeGenerator:
    def __init__(self):
        self.module: Optional[ir.Module] = None
        self.builder: Optional[ir.IRBuilder] = None
        self.current_function: Optional[ir.Function] = None
        
        # Symbol table: maps variable names (str) to LLVM value pointers (ir.Value)
        self.symbol_table: Dict[str, ir.Value] = {}
        # Keep track of IRTypes for variables/params for op dispatch
        self.variable_types: Dict[str, IRType] = {}

        # Cache for declared functions (especially external ones like printf)
        self.function_cache: Dict[str, ir.Function] = {}

        # Initialize LLVM (only once)
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

    def _ir_type_to_llvm_type(self, ir_type: IRType) -> ir.Type:
        if isinstance(ir_type, IRTypeInt):
            return ir.IntType(ir_type.width)
        elif isinstance(ir_type, IRTypeFloat):
            if ir_type.width == 32:
                return ir.FloatType()
            elif ir_type.width == 64:
                return ir.DoubleType()
            else:
                raise ValueError(f"Unsupported float width: {ir_type.width}")
        elif isinstance(ir_type, IRTypeBool):
            return ir.IntType(1)
        elif isinstance(ir_type, IRTypeString):
            return ir.IntType(8).as_pointer() # char*
        elif isinstance(ir_type, IRTypeVoid):
            return ir.VoidType()
        elif isinstance(ir_type, IRTypeCustom):
            # Attempt to find an existing named struct type
            existing_type = self.module.context.get_identified_type(ir_type.name)
            if existing_type:
                return existing_type
            # Placeholder: For now, we can't define new structs here easily.
            # This would require a pass to collect all struct definitions first.
            # Fallback to an opaque struct or error.
            # return self.module.context.get_identified_type(ir_type.name) # Creates opaque if not found
            raise NotImplementedError(f"Custom type IRTypeCustom('{ir_type.name}') to LLVM not fully supported yet.")
        raise NotImplementedError(f"Unsupported IRType: {type(ir_type)}")

    def _get_printf(self) -> ir.Function:
        if "printf" in self.function_cache:
            return self.function_cache["printf"]
        
        # Declare printf: int printf(char* format, ...)
        printf_type = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True)
        printf_func = ir.Function(self.module, printf_type, name="printf")
        self.function_cache["printf"] = printf_func
        return printf_func

    def _get_puts(self) -> ir.Function:
        if "puts" in self.function_cache:
            return self.function_cache["puts"]
        # Declare puts: int puts(char* str)
        puts_type = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()])
        puts_func = ir.Function(self.module, puts_type, name="puts")
        self.function_cache["puts"] = puts_func
        return puts_func

    def _generate_global_string(self, value: str, name_hint: str = ".str") -> ir.GlobalVariable:
        # Create a global string constant for use with puts, printf, etc.
        value_bytes = value.encode('utf8') + b'\x00' # Null-terminate
        str_type = ir.ArrayType(ir.IntType(8), len(value_bytes))
        
        # Check if an identical string constant already exists
        # This is a simple check; more robust would be a hash map of string values
        for gvar in self.module.globals.values(): # Iterate over GlobalVariable objects
            if gvar.name.startswith(".str") and isinstance(gvar.initializer, ir.Constant) and \
               gvar.initializer.type == str_type and gvar.initializer.constant == bytearray(value_bytes):
                return gvar

        global_str = ir.GlobalVariable(self.module, str_type, name=self.module.get_unique_name(name_hint))
        global_str.linkage = 'private' # Or 'internal'
        global_str.global_constant = True
        global_str.initializer = ir.Constant(str_type, bytearray(value_bytes))
        return global_str

    def _generate_expression(self, ir_expr: IRExpression, is_global_initializer=False) -> ir.Value:
        if isinstance(ir_expr, IRLiteral):
            llvm_type = self._ir_type_to_llvm_type(ir_expr.ir_type)
            if isinstance(ir_expr.ir_type, IRTypeString):
                if not is_global_initializer: # Runtime string, use puts/printf style
                    global_str_var = self._generate_global_string(ir_expr.value)
                    # Return a pointer to the first element (char*)
                    return self.builder.gep(global_str_var, 
                                            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], 
                                            name=self.module.get_unique_name("strptr"))
                else: # Global string initializer
                    value_bytes = ir_expr.value.encode('utf8') + b'\x00'
                    return ir.Constant(ir.ArrayType(ir.IntType(8), len(value_bytes)), bytearray(value_bytes))
            return ir.Constant(llvm_type, ir_expr.value)

        if is_global_initializer: # Only constants allowed from here on for global initializers
             raise ValueError("Global variable initializers must be constant literals.")

        if isinstance(ir_expr, IRIdentifier):
            ptr = self.symbol_table.get(ir_expr.name)
            if ptr is None:
                raise NameError(f"Undefined variable: {ir_expr.name}")
            return self.builder.load(ptr, name=ir_expr.name)

        elif isinstance(ir_expr, IRBinaryOp):
            lhs = self._generate_expression(ir_expr.left)
            rhs = self._generate_expression(ir_expr.right)
            
            # Type checking/dispatch for operations (simplified)
            # Assume for now types of lhs and rhs are compatible and determine op type
            # A more robust system would use type information from IRType nodes of operands
            op_type = lhs.type # Assume type of lhs dictates op type
            
            if isinstance(op_type, ir.IntType):
                if ir_expr.op == '+': return self.builder.add(lhs, rhs, name='addtmp')
                elif ir_expr.op == '-': return self.builder.sub(lhs, rhs, name='subtmp')
                elif ir_expr.op == '*': return self.builder.mul(lhs, rhs, name='multmp')
                elif ir_expr.op == '/': return self.builder.sdiv(lhs, rhs, name='sdivtmp') # Signed div
                # Comparisons for integers
                elif ir_expr.op == '==': return self.builder.icmp_signed('==', lhs, rhs, name='eqtmp')
                elif ir_expr.op == '!=': return self.builder.icmp_signed('!=', lhs, rhs, name='netmp')
                elif ir_expr.op == '<': return self.builder.icmp_signed('<', lhs, rhs, name='lttmp')
                elif ir_expr.op == '<=': return self.builder.icmp_signed('<=', lhs, rhs, name='letmp')
                elif ir_expr.op == '>': return self.builder.icmp_signed('>', lhs, rhs, name='gttmp')
                elif ir_expr.op == '>=': return self.builder.icmp_signed('>=', lhs, rhs, name='getmp')
            elif isinstance(op_type, (ir.FloatType, ir.DoubleType)):
                if ir_expr.op == '+': return self.builder.fadd(lhs, rhs, name='faddtmp')
                elif ir_expr.op == '-': return self.builder.fsub(lhs, rhs, name='fsubtmp')
                elif ir_expr.op == '*': return self.builder.fmul(lhs, rhs, name='fmultmp')
                elif ir_expr.op == '/': return self.builder.fdiv(lhs, rhs, name='fdivtmp')
                # Comparisons for floats
                elif ir_expr.op == '==': return self.builder.fcmp_ordered('==', lhs, rhs, name='feqtmp')
                elif ir_expr.op == '!=': return self.builder.fcmp_ordered('!=', lhs, rhs, name='fnetmp')
                elif ir_expr.op == '<': return self.builder.fcmp_ordered('<', lhs, rhs, name='flttmp')
                elif ir_expr.op == '<=': return self.builder.fcmp_ordered('<=', lhs, rhs, name='fletmp')
                elif ir_expr.op == '>': return self.builder.fcmp_ordered('>', lhs, rhs, name='fgttmp')
                elif ir_expr.op == '>=': return self.builder.fcmp_ordered('>=', lhs, rhs, name='fgetmp')
            else:
                raise NotImplementedError(f"Binary operation {ir_expr.op} not supported for type {op_type}")
            raise NotImplementedError(f"Binary operator {ir_expr.op} not implemented for the given types.")


        elif isinstance(ir_expr, IRFunctionCall):
            # Special handling for "print"
            if ir_expr.callee_name == "print" and len(ir_expr.args) == 1:
                arg_expr = self._generate_expression(ir_expr.args[0])
                arg_type = self.variable_types.get(ir_expr.args[0].name) if isinstance(ir_expr.args[0], IRIdentifier) else \
                           ir_expr.args[0].ir_type if isinstance(ir_expr.args[0], IRLiteral) else None

                if isinstance(arg_type, IRTypeString) or (isinstance(arg_expr.type, ir.PointerType) and isinstance(arg_expr.type.pointee, ir.IntType) and arg_expr.type.pointee.width == 8):
                    puts_func = self._get_puts()
                    return self.builder.call(puts_func, [arg_expr])
                else: # Assume numeric or bool, use printf with format string
                    printf_func = self._get_printf()
                    fmt_str = ""
                    if isinstance(arg_expr.type, ir.IntType) and arg_expr.type.width == 1: # Bool
                        fmt_str = "%d\\0A\\00" # Print as int 0 or 1, then newline
                    elif isinstance(arg_expr.type, ir.IntType):
                         fmt_str = "%d\\0A\\00"
                    elif isinstance(arg_expr.type, (ir.FloatType, ir.DoubleType)):
                         fmt_str = "%f\\0A\\00"
                    else: # Fallback or error
                        fmt_str = "Unsupported type for print\\0A\\00"

                    fmt_var = self._generate_global_string(fmt_str, ".fmt")
                    fmt_ptr = self.builder.gep(fmt_var, [ir.Constant(ir.IntType(32),0), ir.Constant(ir.IntType(32),0)])
                    return self.builder.call(printf_func, [fmt_ptr, arg_expr])

            # General function call
            callee_func = self.module.globals.get(ir_expr.callee_name)
            if callee_func is None or not isinstance(callee_func, ir.Function):
                 # Try function cache (for declared builtins not yet in module.globals)
                 callee_func = self.function_cache.get(ir_expr.callee_name)
                 if callee_func is None:
                    raise NameError(f"Function {ir_expr.callee_name} not found in module or cache.")

            llvm_args = [self._generate_expression(arg) for arg in ir_expr.args]
            return self.builder.call(callee_func, llvm_args)

        raise NotImplementedError(f"IRExpression type {type(ir_expr)} not implemented.")


    def _generate_statement(self, ir_stmt: IRStatement):
        if isinstance(ir_stmt, IRLocalVariable):
            var_name = ir_stmt.name
            llvm_type = self._ir_type_to_llvm_type(ir_stmt.ir_type)
            # Store type info for later use (e.g. in binary ops)
            self.variable_types[var_name] = ir_stmt.ir_type 
            
            # Allocate variable on the stack in the entry block of the function
            # This is a common approach for mutable local variables.
            # For SSA, this would be different (values, not pointers).
            with self.builder.goto_entry_block():
                 alloca = self.builder.alloca(llvm_type, name=var_name + ".addr")
            self.symbol_table[var_name] = alloca
            
            if ir_stmt.initializer:
                init_val = self._generate_expression(ir_stmt.initializer)
                self.builder.store(init_val, alloca)

        elif isinstance(ir_stmt, IRAssign):
            value_llvm = self._generate_expression(ir_stmt.value)
            target_ptr = self.symbol_table.get(ir_stmt.target.name)
            if target_ptr is None:
                raise NameError(f"Assignment to undefined variable: {ir_stmt.target.name}")
            self.builder.store(value_llvm, target_ptr)

        elif isinstance(ir_stmt, IRReturn):
            if ir_stmt.value:
                ret_val = self._generate_expression(ir_stmt.value)
                self.builder.ret(ret_val)
            else:
                self.builder.ret_void()
        
        elif isinstance(ir_stmt, IRExpressionStatement):
            self._generate_expression(ir_stmt.expression) # Result is discarded

        else:
            raise NotImplementedError(f"IRStatement type {type(ir_stmt)} not implemented.")

    def _generate_function(self, ir_func: IRFunction):
        # Save current state if any (for nested functions in future)
        # prev_builder = self.builder
        # prev_function = self.current_function
        # prev_symbol_table = self.symbol_table.copy()
        # prev_variable_types = self.variable_types.copy()

        llvm_return_type = self._ir_type_to_llvm_type(ir_func.return_type)
        llvm_param_types = [self._ir_type_to_llvm_type(p.ir_type) for p in ir_func.params]
        func_type = ir.FunctionType(llvm_return_type, llvm_param_types)
        
        llvm_f = self.module.globals.get(ir_func.name)
        if llvm_f is None: # Function might have been declared by a call earlier
            llvm_f = ir.Function(self.module, func_type, name=ir_func.name)
        elif not isinstance(llvm_f, ir.Function):
             raise TypeError(f"Name {ir_func.name} already exists in module but is not a function.")
        
        self.current_function = llvm_f
        
        # Create a new scope for parameters and locals
        # Global symbols are still accessible if not shadowed.
        # For simplicity, we extend the main symbol table for now.
        # A proper scoping mechanism would be needed for more complex scenarios.
        # We'll manage this by saving/restoring symbol_table around function body processing if needed,
        # but for now, parameters and locals will be added to the current symbol_table.

        # Create entry block
        entry_block = llvm_f.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)

        # Process parameters: allocate space and store initial values
        for i, ir_param in enumerate(ir_func.params):
            param_name = ir_param.name
            param_llvm_type = self._ir_type_to_llvm_type(ir_param.ir_type)
            self.variable_types[param_name] = ir_param.ir_type # Store type

            alloca = self.builder.alloca(param_llvm_type, name=param_name + ".addr")
            self.symbol_table[param_name] = alloca
            self.builder.store(llvm_f.args[i], alloca)
            llvm_f.args[i].name = param_name # Name the LLVM argument itself

        # Process body statements
        for stmt in ir_func.body:
            self._generate_statement(stmt)

        # Ensure terminator if function is not void and last block doesn't have one
        if not entry_block.is_terminated and not isinstance(llvm_return_type, ir.VoidType):
             # This often indicates an error in the source (missing return)
             # For robustnes, LLVM requires blocks to be terminated.
             # Add a default return if possible (e.g., for `main` returning int)
             if ir_func.name == "main" and isinstance(llvm_return_type, ir.IntType):
                 self.builder.ret(ir.Constant(llvm_return_type, 0))
             else:
                 # Or let LLVM verification fail, or insert unreachable if appropriate
                 # For now, if non-void and no return, it's an issue.
                 pass # Let LLVM verifier catch it or handle based on language semantics.
        elif not entry_block.is_terminated and isinstance(llvm_return_type, ir.VoidType):
            self.builder.ret_void() # Ensure void functions are terminated

        # Restore previous state (if saved)
        # self.builder = prev_builder
        # self.current_function = prev_function
        # self.symbol_table = prev_symbol_table
        # self.variable_types = prev_variable_types


    def _generate_global_variable(self, ir_global: IRGlobalVariable):
        var_name = ir_global.name
        self.variable_types[var_name] = ir_global.ir_type # Store type
        
        llvm_type: ir.Type
        actual_initializer: Optional[ir.Constant] = None

        if ir_global.initializer:
            # Global initializers must be constants.
            init_val_const = self._generate_expression(ir_global.initializer, is_global_initializer=True)
            actual_initializer = init_val_const
            llvm_type = actual_initializer.type # Type of global var must match initializer type
            
            # If the IRType was string, but we got an array from initializer, that's correct.
            # If IRType was string but no initializer, llvm_type becomes i8* (char*)
            # and it will be zeroinitialized (null pointer).
        else:
            llvm_type = self._ir_type_to_llvm_type(ir_global.ir_type)
            actual_initializer = ir.Constant(llvm_type, None) # LLVM zeroinitializer

        llvm_gvar = ir.GlobalVariable(self.module, llvm_type, name=var_name)
        llvm_gvar.linkage = 'internal' # Default, can be changed
        llvm_gvar.initializer = actual_initializer
        
        self.symbol_table[var_name] = llvm_gvar

    # In _generate_expression, for IRIdentifier:
    # We need to adjust how global strings (which are arrays) are handled when loaded.
    # Current code:
    #   ptr = self.symbol_table.get(ir_expr.name)
    #   if ptr is None: raise NameError(...)
    #   return self.builder.load(ptr, name=ir_expr.name)
    # If ptr is a global variable of array type (e.g. global string literal), loading it directly is not what we want.
    # We want a pointer to its first element (char*).

    # This change will be inside _generate_expression for IRIdentifier:
    # (This is conceptual, the actual diff will be against that specific part of _generate_expression)
    # if isinstance(ptr, ir.GlobalVariable) and isinstance(ptr.type.pointee, ir.ArrayType) and \
    #    isinstance(ptr.type.pointee.element, ir.IntType) and ptr.type.pointee.element.width == 8:
    #     # It's a global string array, get pointer to first element
    #     return self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], name=ir_expr.name + ".ptr")
    # else:
    #     return self.builder.load(ptr, name=ir_expr.name)

    # Let's apply this specific change to _generate_expression
    # Find the IRIdentifier handling in _generate_expression
    # ...
    # if isinstance(ir_expr, IRIdentifier):
    #     ptr = self.symbol_table.get(ir_expr.name)
    #     if ptr is None:
    #         raise NameError(f"Undefined variable: {ir_expr.name}")
    #     # MODIFICATION START
    #     if isinstance(ptr, ir.GlobalVariable) and isinstance(ptr.value_type, ir.ArrayType) and \
    #        isinstance(ptr.value_type.element, ir.IntType) and ptr.value_type.element.width == 8:
    #         # It's a global string array (e.g. g_var4 AS string = "hello")
    #         # We want a pointer to its first element (char*) when it's used as a value.
    #         return self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], name=ir_expr.name + ".decayed_ptr")
    #     # MODIFICATION END
    #     return self.builder.load(ptr, name=ir_expr.name)
    # ...


    def generate_llvm_ir(self, ir_program: IRProgram) -> str:
        self.module = ir.Module(name=ir_program.declarations[0].name if ir_program.declarations else "main_module")
        
        target_triple = binding.get_default_triple()
        self.module.triple = target_triple
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        self.module.data_layout = target_machine.target_data

        self.symbol_table.clear()
        self.variable_types.clear()
        self.function_cache.clear()

        # First pass: declare globals and function signatures (if needed for forward calls)
        # For simplicity, we'll process globals first, then functions.
        # IRImport nodes are not directly translated to LLVM instructions here,
        # but they might inform declarations of external functions (like printf).
        # We can pre-declare known external functions.
        
        # For Phase 1, IRImport is mostly for information or for specific known libraries.
        # Example: if "stdio" is imported, ensure printf/puts are declared.
        # This is implicitly handled by _get_printf/_get_puts when called.

        for decl in ir_program.declarations:
            if isinstance(decl, IRGlobalVariable):
                self._generate_global_variable(decl)
        
        for decl in ir_program.declarations:
            if isinstance(decl, IRFunction):
                self._generate_function(decl)
        
        # Verify the module (optional, but good for debugging)
        # binding.llvm.verify_module(self.moduleRef) # Requires moduleRef from binding.parse_assembly("")
        # Or convert to string and parse again for verification via binding.
        # For now, rely on llvmlite's internal checks and downstream tools.

        return str(self.module)

if __name__ == '__main__':
    from ast_to_ir import ASTToIRConverter # Assuming this is the updated one
    from tokenizer import tokenize
    from parser import Parser # Assuming this is the updated one

    codegen = LLVMCodeGenerator()

    test_code_samples = {
        "simple_return": """
DEF main() -> int:
    INDENT
    RETURN 42
    DEDENT
""",
        "global_var_and_func": """
g_var AS int = 100

DEF get_g_var() -> int:
    INDENT
    RETURN g_var
    DEDENT

DEF main() -> int:
    INDENT
    x AS int = get_g_var() + g_var
    RETURN x
    DEDENT
""",
        "local_var_and_print_string": """
DEF main() -> int:
    INDENT
    msg AS string = "Hello LLVM!"
    print(msg)
    y AS int = 10
    print(y)
    z AS bool = TRUE
    print(z)
    RETURN 0
    DEDENT
""",
        "params_and_arith": """
DEF add(a AS int, b AS int) -> int:
    INDENT
    c AS int = a + b
    RETURN c
    DEDENT

DEF main() -> int:
    INDENT
    res AS int = add(5, 7)
    print(res)
    RETURN res
    DEDENT
"""
    }

    for name, code_str in test_code_samples.items():
        print(f"\n--- Testing LLVM Codegen: {name} ---")
        print(f"Source Code:\n{code_str.strip()}")
        try:
            tokens = tokenize(code_str)
            tokens_for_parser = [t for t in tokens if t.type not in ('SKIP', 'NEWLINE')]
            
            parser = Parser(tokens_for_parser)
            ast_program = parser.parse_program()
            # print("\nAST Program:", ast_program)

            ir_converter = ASTToIRConverter()
            ir_program = ir_converter.translate_program(ast_program)
            # print("\nIR Program:", ir_program)
            
            llvm_ir_str = codegen.generate_llvm_ir(ir_program)
            print("\nGenerated LLVM IR:")
            print(llvm_ir_str)
            
            # Optional: Validate LLVM IR using llvmlite.binding
            # llvmlite_module_ref = binding.parse_assembly(llvm_ir_str)
            # llvmlite_module_ref.verify() 
            # print("LLVM IR Validated.")

        except Exception as e:
            print(f"An error occurred during LLVM codegen for '{name}': {e}")
            import traceback
            traceback.print_exc()
        print("----------------------------------")
