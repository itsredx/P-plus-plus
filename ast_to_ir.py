# ast_to_ir.py
from typing import List, Optional, Union, Any, Dict # Added Dict
from ast_nodes import * # Assumes ast_nodes.py has BuiltinTypeNode, PointerTypeNode etc.
from ir_nodes import *  # Assumes ir_nodes.py has IRPointerType, IRAddressOf etc.

class ASTToIRConverter:
    def __init__(self):
        self.global_variable_types: Dict[str, IRType] = {}
        self.current_function_variable_types: Dict[str, IRType] = {}

    def _get_var_ir_type(self, name: str) -> Optional[IRType]:
        # Check current function's scope first, then global scope
        if name in self.current_function_variable_types:
            return self.current_function_variable_types[name]
        if name in self.global_variable_types:
            return self.global_variable_types[name]
        return None

    def _translate_type(self, type_node: Optional[TypeNode]) -> IRType:
        if type_node is None: # Handles optional return types that default to void
            return IRTypeVoid()
        
        if isinstance(type_node, PointerTypeNode):
            pointee_ir_type = self._translate_type(type_node.pointee_type)
            return IRPointerType(pointee_type=pointee_ir_type)
        
        if isinstance(type_node, BuiltinTypeNode): # Explicitly check for BuiltinTypeNode
            name = type_node.name
            if name == 'int': return IRTypeInt(32)
            elif name == 'int8': return IRTypeInt(8)
            elif name == 'int16': return IRTypeInt(16)
            elif name == 'int32': return IRTypeInt(32)
            elif name == 'int64': return IRTypeInt(64)
            elif name == 'uint8': return IRTypeInt(8) # Placeholder, consider IRTypeUInt
            elif name == 'uint16': return IRTypeInt(16)
            elif name == 'uint32': return IRTypeInt(32)
            elif name == 'uint64': return IRTypeInt(64)
            elif name == 'float': return IRTypeFloat(64)
            elif name == 'float32': return IRTypeFloat(32)
            elif name == 'float64': return IRTypeFloat(64)
            elif name == 'bool': return IRTypeBool()
            elif name == 'string': return IRTypeString() # Represents char* or similar
            elif name == 'void': return IRTypeVoid()
            else: # Custom type names
                return IRTypeCustom(name)
        
        # Fallback for old TypeNode if it was directly used with a name (should not happen with updated AST)
        if hasattr(type_node, 'name'):
             return IRTypeCustom(type_node.name) # type: ignore

        raise NotImplementedError(f"Unsupported TypeNode structure: {type(type_node)}")


    def _translate_literal(self, node: LiteralNode) -> IRLiteral:
        value = node.value
        ir_type: IRType
        if value is None: # For null pointer literals
            # Representing null as a void* equivalent or a special NullType.
            # For simplicity, let's use IRPointerType(IRTypeVoid()) as a convention for 'generic pointer' / 'null pointer type'.
            # The actual type might be context-dependent or resolved later.
            ir_type = IRPointerType(IRTypeVoid()) 
        elif isinstance(value, bool):
            ir_type = IRTypeBool()
        elif isinstance(value, int):
            ir_type = IRTypeInt(32)
        elif isinstance(value, float):
            ir_type = IRTypeFloat(64)
        elif isinstance(value, str):
            ir_type = IRTypeString() # This implies it's a pointer to char, typically.
                                     # For string literals, the IR might be ('const_string', "value")
                                     # and its type is effectively char*.
                                     # Let's refine IRTypeString if it's a pointer type.
                                     # For now, IRLiteral("hello", IRTypeString()) is okay.
        else:
            raise NotImplementedError(f"Unknown literal type: {type(value)} for value {value}")
        return IRLiteral(value, ir_type)

    def _translate_expression(self, node: ExpressionNode) -> IRExpression:
        if isinstance(node, LiteralNode):
            return self._translate_literal(node)
        elif isinstance(node, IdentifierNode):
            return IRIdentifier(name=node.name)
        elif isinstance(node, BinaryOpNode):
            left_ir = self._translate_expression(node.left)
            right_ir = self._translate_expression(node.right)
            return IRBinaryOp(left=left_ir, op=node.operator, right=right_ir)
        elif isinstance(node, FunctionCallNode):
            # Handle MallocNode if it's parsed as a FunctionCallNode with callee "malloc"
            if isinstance(node.callee, IdentifierNode) and node.callee.name == "malloc":
                 # This is a deviation from subtask if MallocNode is distinct.
                 # Assuming MallocNode is the AST node from previous step.
                 raise NotImplementedError("Direct 'malloc' FunctionCallNode translation not expected here; use MallocNode.")

            if not isinstance(node.callee, IdentifierNode):
                raise NotImplementedError("Complex callee expressions not yet supported in IR translation")
            callee_name = node.callee.name
            args_ir = [self._translate_expression(arg) for arg in node.arguments]
            return IRFunctionCall(callee_name=callee_name, args=args_ir)
        
        # Phase 2: Pointer and Memory AST Nodes
        elif isinstance(node, AddressOfNode):
            var_name = node.identifier.name
            var_ir_type = self._get_var_ir_type(var_name)
            if var_ir_type is None:
                raise ValueError(f"Type of variable '{var_name}' not found for AddressOf operation.")
            return IRAddressOf(variable_name=var_name, var_ir_type=var_ir_type)
        elif isinstance(node, DereferenceNode):
            ir_pointer_expr = self._translate_expression(node.expression)
            return IRDereference(pointer_expr=ir_pointer_expr)
        elif isinstance(node, MallocNode): # Correct handling for MallocNode
            ir_alloc_type = self._translate_type(node.alloc_type)
            ir_size_expr = self._translate_expression(node.size_expr)
            return IRMalloc(alloc_ir_type=ir_alloc_type, size_expr=ir_size_expr)
        else:
            raise NotImplementedError(f"AST expression node not implemented in IR translation: {type(node)}")

    def _translate_param(self, node: ParamNode) -> IRParam:
        name = node.identifier.name
        ir_type = self._translate_type(node.param_type)
        # Store param type in current function's scope
        self.current_function_variable_types[name] = ir_type 
        return IRParam(name=name, ir_type=ir_type)

    def _translate_statement(self, node: StatementNode) -> Union[IRStatement, List[IRStatement], None]:
        if isinstance(node, GlobalVarDeclNode): 
            # This should ideally only be called by translate_program for top-level
            # If called in a statement context, it's an error or needs specific handling.
            # For now, let it pass to _translate_global_var_decl for type recording.
            # However, _translate_global_var_decl returns IRGlobalVariable, not IRStatement.
            # This indicates a logic error if it's meant to be a statement here.
            # Let's assume this path is not taken for statements within functions.
            raise ValueError("GlobalVarDeclNode should not be processed as a simple statement within a block.")
        elif isinstance(node, LocalVarDeclNode):
            return self._translate_local_var_decl(node)
        elif isinstance(node, VarAssignNode):
            target_ir = IRIdentifier(name=node.identifier.name) 
            value_ir = self._translate_expression(node.expression)
            return IRAssign(target=target_ir, value=value_ir)
        elif isinstance(node, ReturnStmtNode):
            expr_ir: Optional[IRExpression] = None
            if node.expression:
                expr_ir = self._translate_expression(node.expression)
            return IRReturn(value=expr_ir)
        elif isinstance(node, ExpressionStatementNode):
            expr_ir = self._translate_expression(node.expression)
            return IRExpressionStatement(expression=expr_ir)
        elif isinstance(node, BlockNode): 
            return self._translate_block_node_statements(node)
        
        # Phase 2: Pointer and Memory Statement AST Nodes
        elif isinstance(node, PointerAssignmentNode):
            ir_target_pointer_expr = self._translate_expression(node.target_deref.expression)
            ir_value_to_assign_expr = self._translate_expression(node.value)
            return IRPointerAssign(target_pointer_expr=ir_target_pointer_expr, value_to_assign_expr=ir_value_to_assign_expr)
        elif isinstance(node, FreeNode):
            ir_pointer_expr = self._translate_expression(node.pointer_expr)
            return IRFree(pointer_expr=ir_pointer_expr)
        else:
            raise NotImplementedError(f"AST statement node not implemented in IR translation: {type(node)}")

    def _translate_block_node_statements(self, node: BlockNode) -> List[IRStatement]:
        ir_statements: List[IRStatement] = []
        for stmt_node in node.statements:
            translated_stmt = self._translate_statement(stmt_node)
            if isinstance(translated_stmt, list): 
                ir_statements.extend(translated_stmt)
            elif translated_stmt is not None: 
                ir_statements.append(translated_stmt)
        return ir_statements

    def _translate_local_var_decl(self, node: LocalVarDeclNode) -> IRLocalVariable:
        name = node.identifier.name
        ir_type = self._translate_type(node.var_type)
        self.current_function_variable_types[name] = ir_type # Store for current function scope
        
        initializer_ir: Optional[IRExpression] = None
        if node.expression:
            initializer_ir = self._translate_expression(node.expression)
        return IRLocalVariable(name=name, ir_type=ir_type, initializer=initializer_ir)

    def _translate_function_decl(self, node: FunctionDeclNode) -> IRFunction:
        # Prepare new scope for function parameters and locals
        # This simple dict for current_function_variable_types assumes non-nested functions for now.
        # For nested functions, a stack of these dicts or a more complex Environment object would be needed.
        previous_func_var_types = self.current_function_variable_types
        self.current_function_variable_types = {} # New scope for this function

        name = node.identifier.name
        params_ir = [self._translate_param(p) for p in node.params] # Populates current_function_variable_types
        return_ir_type = self._translate_type(node.return_type)
        
        body_ir_stmts = self._translate_block_node_statements(node.body)
        
        # Restore previous scope
        self.current_function_variable_types = previous_func_var_types
        
        return IRFunction(name=name, params=params_ir, return_type=return_ir_type, body=body_ir_stmts)

    def _translate_global_var_decl(self, node: GlobalVarDeclNode) -> IRGlobalVariable:
        name = node.identifier.name
        ir_type = self._translate_type(node.var_type)
        self.global_variable_types[name] = ir_type # Store in global types
        
        initializer_ir: Optional[IRExpression] = None
        if node.expression:
            initializer_ir = self._translate_expression(node.expression)
        return IRGlobalVariable(name=name, ir_type=ir_type, initializer=initializer_ir)

    def _translate_import(self, node: ImportNode) -> IRImport:
        module_name_str = node.module_name.name
        return IRImport(module_name=module_name_str)

    def translate_program(self, program_node: ProgramNode) -> IRProgram:
        self.global_variable_types.clear() # Clear global types for fresh program translation
        ir_declarations: List[Union[IRGlobalVariable, IRFunction, IRImport]] = []
        
        # First pass for global variable types (if needed for forward references by functions, though not strictly necessary here)
        for decl_node in program_node.declarations:
            if isinstance(decl_node, GlobalVarDeclNode):
                # Temporarily translate type to populate global_variable_types
                # This is a bit redundant as _translate_global_var_decl will do it again,
                # but ensures types are available if functions are processed before all globals.
                # A better way might be a dedicated first pass for all type signatures.
                 ir_type = self._translate_type(decl_node.var_type)
                 self.global_variable_types[decl_node.identifier.name] = ir_type
        
        # Main translation pass
        for decl_node in program_node.declarations:
            if isinstance(decl_node, GlobalVarDeclNode):
                ir_declarations.append(self._translate_global_var_decl(decl_node))
            elif isinstance(decl_node, FunctionDeclNode):
                # Clear function-local types before translating a new function
                self.current_function_variable_types.clear() 
                ir_declarations.append(self._translate_function_decl(decl_node))
            elif isinstance(decl_node, ImportNode):
                ir_declarations.append(self._translate_import(decl_node))
            else:
                raise NotImplementedError(f"Top-level AST node not implemented in IR translation: {type(decl_node)}")
        return IRProgram(declarations=ir_declarations)


if __name__ == "__main__":
    from tokenizer import tokenize # Assuming these are up-to-date
    from parser import Parser     # Assuming these are up-to-date

    converter = ASTToIRConverter()
    
    # Example P++ code with pointer operations
    # Note: This requires the Tokenizer and Parser to be updated for Phase 2 syntax
    # (e.g. *, &, malloc, free, pointer types in declarations)
    pointer_code_sample = """
    // Global pointer
    g_ptr AS *int

    DEF main() -> void:
        INDENT
        x AS int = 10
        p AS *int
        
        p = &x          // AddressOfNode -> IRAddressOf
        *p = 20         // PointerAssignmentNode -> IRPointerAssign
                        // DereferenceNode for *p (lhs) is implicit in PointerAssignmentNode's AST structure
                        // but becomes target_pointer_expr in IRPointerAssign

        val AS int = *p // DereferenceNode -> IRDereference
        
        arr AS *int = malloc(int, 5) // MallocNode -> IRMalloc
        // arr[0] = 1; // Array indexing not in this phase's AST/IR nodes yet
        *arr = 100 // PointerAssignment to first element

        free(arr)       // FreeNode -> IRFree
        // free(p) // Error: p points to stack (x), not heap allocated
        RETURN
        DEDENT
    """
    # This __main__ block needs fully functional Tokenizer and Parser for Phase 2.
    # The code below is illustrative and may not run without those prerequisites.
    print(f"--- Example AST to IR Conversion for Pointer Code ---")
    print(f"P++ Code:\n{pointer_code_sample}\n")

    try:
        # 1. Tokenize (requires tokenizer to handle *, &, malloc, free, INDENT/DEDENT)
        # tokens = tokenize(pointer_code_sample) 
        # tokens_for_parser = [t for t in tokens if t.type not in ('SKIP', 'NEWLINE')]
        # print("Tokens (example - would need actual tokenizer output):", tokens_for_parser[:10])

        # 2. Parse (requires parser to handle new AST nodes and syntax)
        # parser = Parser(tokens_for_parser)
        # ast_program = parser.parse_program()
        # print("\nAST Program (example - would need actual parser output):", ast_program.declarations[0]) # Print first decl

        # 3. AST to IR (this is what we are testing here, assuming AST is correctly formed)
        # Manually create a sample AST for demonstration if parser is not ready:
        # This is complex to do manually for the whole program.
        # Let's assume ast_program is available from a correctly parsing Parser.
        
        # Placeholder for manual AST construction or actual parsing result:
        # For now, this if __name__ block won't run the full conversion without a Phase 2 parser.
        # The unit tests would be a better place for focused testing with hand-crafted ASTs or token streams.

        print("Note: Full execution of this __main__ block requires Tokenizer and Parser to be Phase 2 compatible.")
        print("The ASTToIRConverter class itself has been updated for Phase 2 nodes.")
        
        # Example: Translating a manually created PointerTypeNode
        manual_ptr_type_ast = PointerTypeNode(pointee_type=BuiltinTypeNode(name="int"))
        ir_ptr_type = converter._translate_type(manual_ptr_type_ast)
        print(f"\nManually translated AST PointerTypeNode(*int) to IR: {ir_ptr_type}")
        # Expected: IRPointerType(pointee_type=IRTypeInt(width=32))

        # Example: Translating a manually created AddressOfNode
        # To do this, variable_types needs to be populated first.
        converter.global_variable_types["my_var"] = IRTypeInt(32) # Simulate declaration
        manual_addr_of_ast = AddressOfNode(identifier=IdentifierNode(name="my_var"))
        ir_addr_of_expr = converter._translate_expression(manual_addr_of_ast)
        print(f"Manually translated AST AddressOfNode(&my_var) to IR: {ir_addr_of_expr}")
        # Expected: IRAddressOf(variable_name='my_var', var_ir_type=IRTypeInt(width=32))
        converter.global_variable_types.clear() # Clean up for other potential manual tests

    except Exception as e:
        print(f"Error in AST to IR example: {e}")
        import traceback
        traceback.print_exc()

    print("----------------------------------------------------")
