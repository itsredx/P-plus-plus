# ast_to_ir.py
from typing import List, Optional, Union, Any
from ast_nodes import *
from ir_nodes import * # Using the updated ir_nodes.py

class ASTToIRConverter:
    def __init__(self):
        # Symbol table or other state can be added later if needed for more complex translations
        pass

    def _translate_type(self, type_node: Optional[TypeNode]) -> IRType:
        if type_node is None:
            return IRTypeVoid()
        
        name = type_node.name
        # Simple mapping for now. Can be extended with width info from TypeNode if added.
        if name == 'int': return IRTypeInt(32) # Default P-- int is i32
        elif name == 'int8': return IRTypeInt(8)
        elif name == 'int16': return IRTypeInt(16)
        elif name == 'int32': return IRTypeInt(32)
        elif name == 'int64': return IRTypeInt(64)
        # Assuming uint types are also mapped to IRTypeInt for now, width matters.
        # Proper signed/unsigned distinction might need separate IR types or flags.
        elif name == 'uint8': return IRTypeInt(8) # Placeholder, might need IRTypeUInt
        elif name == 'uint16': return IRTypeInt(16)
        elif name == 'uint32': return IRTypeInt(32)
        elif name == 'uint64': return IRTypeInt(64)
        elif name == 'float': return IRTypeFloat(64) # Default P-- float is f64
        elif name == 'float32': return IRTypeFloat(32)
        elif name == 'float64': return IRTypeFloat(64)
        elif name == 'bool': return IRTypeBool()
        elif name == 'string': return IRTypeString()
        elif name == 'void': return IRTypeVoid()
        else:
            # For custom types or types not yet specifically handled
            return IRTypeCustom(name)

    def _translate_literal(self, node: LiteralNode) -> IRLiteral:
        value = node.value
        ir_type: IRType
        if isinstance(value, bool):
            ir_type = IRTypeBool()
        elif isinstance(value, int):
            # TODO: Infer width more accurately if possible, default to i32
            ir_type = IRTypeInt(32)
        elif isinstance(value, float):
            # TODO: Infer width (f32/f64)
            ir_type = IRTypeFloat(64)
        elif isinstance(value, str):
            ir_type = IRTypeString()
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
            # op should be directly usable (e.g., "+", "-", "*", "/")
            return IRBinaryOp(left=left_ir, op=node.operator, right=right_ir)
        elif isinstance(node, FunctionCallNode):
            if not isinstance(node.callee, IdentifierNode):
                raise NotImplementedError("Complex callee expressions not yet supported in IR translation")
            callee_name = node.callee.name
            args_ir = [self._translate_expression(arg) for arg in node.arguments]
            return IRFunctionCall(callee_name=callee_name, args=args_ir)
        else:
            raise NotImplementedError(f"AST expression node not implemented in IR translation: {type(node)}")

    def _translate_param(self, node: ParamNode) -> IRParam:
        name = node.identifier.name
        ir_type = self._translate_type(node.param_type)
        return IRParam(name=name, ir_type=ir_type)

    def _translate_statement(self, node: StatementNode) -> Union[IRStatement, List[IRStatement], None]:
        if isinstance(node, GlobalVarDeclNode): # Should be handled at program level
            # This indicates it's being called from within a block, which is not where globals are defined.
            # However, the structure allows it. For now, let's treat it like a local.
            # Or raise error: raise ValueError("GlobalVarDeclNode cannot appear inside a block")
            # For simplicity, let's assume this function is called for statements *within* blocks.
            # A GlobalVarDeclNode appearing here would be an error in a real compiler or should be a LocalVarDeclNode.
            # This method is mostly for statements *inside* functions.
            # Let's assume if we see it here, it's a LocalVarDeclNode conceptually.
            # This path should ideally not be hit if parser distinguishes global/local properly
            # and translate_program handles GlobalVarDeclNode separately.
             return self._translate_local_var_decl(node) # Treat as local if found in statement context

        elif isinstance(node, LocalVarDeclNode):
            return self._translate_local_var_decl(node)
        elif isinstance(node, VarAssignNode):
            target_ir = IRIdentifier(name=node.identifier.name) # Assign to variable name
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
        elif isinstance(node, BlockNode): # A block itself is not a single statement, but contains them
            return self._translate_block_node_statements(node)
        else:
            raise NotImplementedError(f"AST statement node not implemented in IR translation: {type(node)}")

    def _translate_block_node_statements(self, node: BlockNode) -> List[IRStatement]:
        ir_statements: List[IRStatement] = []
        for stmt_node in node.statements:
            translated_stmt = self._translate_statement(stmt_node)
            if isinstance(translated_stmt, list): # If a block was nested and returned a list
                ir_statements.extend(translated_stmt)
            elif translated_stmt is not None: # Ensure something was returned
                ir_statements.append(translated_stmt)
        return ir_statements

    def _translate_local_var_decl(self, node: Union[LocalVarDeclNode, GlobalVarDeclNode]) -> IRLocalVariable:
        # This handles LocalVarDeclNode. It can also take GlobalVarDeclNode if it's misused in local scope.
        name = node.identifier.name
        ir_type = self._translate_type(node.var_type)
        initializer_ir: Optional[IRExpression] = None
        if node.expression:
            initializer_ir = self._translate_expression(node.expression)
        return IRLocalVariable(name=name, ir_type=ir_type, initializer=initializer_ir)

    def _translate_function_decl(self, node: FunctionDeclNode) -> IRFunction:
        name = node.identifier.name
        params_ir = [self._translate_param(p) for p in node.params]
        return_ir_type = self._translate_type(node.return_type)
        
        # The body of an IRFunction is a list of IRStatement
        # BlockNode's statements are translated by _translate_block_node_statements
        body_ir_stmts = self._translate_block_node_statements(node.body)
        
        return IRFunction(name=name, params=params_ir, return_type=return_ir_type, body=body_ir_stmts)

    def _translate_global_var_decl(self, node: GlobalVarDeclNode) -> IRGlobalVariable:
        name = node.identifier.name
        ir_type = self._translate_type(node.var_type)
        initializer_ir: Optional[IRExpression] = None
        if node.expression:
            initializer_ir = self._translate_expression(node.expression)
        return IRGlobalVariable(name=name, ir_type=ir_type, initializer=initializer_ir)

    def _translate_import(self, node: ImportNode) -> IRImport:
        # Assuming module_name is an IdentifierNode
        module_name_str = node.module_name.name
        return IRImport(module_name=module_name_str)

    def translate_program(self, program_node: ProgramNode) -> IRProgram:
        ir_declarations: List[Union[IRGlobalVariable, IRFunction, IRImport]] = []
        for decl_node in program_node.declarations:
            if isinstance(decl_node, GlobalVarDeclNode):
                ir_declarations.append(self._translate_global_var_decl(decl_node))
            elif isinstance(decl_node, FunctionDeclNode):
                ir_declarations.append(self._translate_function_decl(decl_node))
            elif isinstance(decl_node, ImportNode):
                ir_declarations.append(self._translate_import(decl_node))
            # Note: The ProgramNode in ast_nodes.py is defined to only contain these types.
            # If it could contain other StatementNodes, that logic would be here.
            else:
                raise NotImplementedError(f"Top-level AST node not implemented in IR translation: {type(decl_node)}")
        return IRProgram(declarations=ir_declarations)


if __name__ == "__main__":
    from tokenizer import tokenize
    from parser import Parser # Assumes parser.py uses the same AST nodes

    # Sample P++ code for testing
    # This code needs to be tokenized and parsed into AST that matches the new AST structure.
    # The parser.py would need to be updated to produce INDENT/DEDENT tokens for blocks.
    # For now, let's assume the AST is correctly formed with BlockNode, etc.

    # Test 1: Simple global variable and function
    code1 = """
my_global_var AS int = 42
IMPORT math

DEF my_function(param1 AS int, param2 AS string) -> void:
    INDENT
    local_var AS int = param1 + 10
    param2 = "new string"
    another_local AS bool = TRUE
    RETURN
    DEDENT
"""
    # Test 2: Function call and expression statement
    code2 = """
DEF main():
    INDENT
    print("Hello, world!")
    x AS int = call_func(1, 2.5) + 3
    RETURN x
    DEDENT
"""

    # Test 3: More complex expressions and assignments
    code3 = """
width AS int = 100
height AS int = 200
area AS int

DEF calculate_area():
    INDENT
    area = width * height
    io.print(area)
    DEDENT
"""
    
    test_cases = {
        "simple_global_func_import": code1,
        "func_call_expr_stmt": code2,
        "assignments_and_expressions": code3
    }
    
    converter = ASTToIRConverter()

    for name, code_str in test_cases.items():
        print(f"\n--- Testing: {name} ---")
        print(f"Code:\n{code_str.strip()}")
        try:
            # This assumes tokenizer.py and parser.py are up-to-date and work with INDENT/DEDENT
            # and produce the AST nodes that this converter expects.
            tokens = tokenize(code_str)
            # Filter out SKIP and NEWLINE if parser doesn't handle them after INDENT/DEDENT phase
            tokens_for_parser = [t for t in tokens if t.type not in ('SKIP', 'NEWLINE')]
            # print("Filtered Tokens:", [f"{t.type}({t.value})" for t in tokens_for_parser])
            
            parser = Parser(tokens_for_parser) # Ensure parser.py is compatible
            ast = parser.parse_program()
            print("\nAST:")
            # A simple way to print AST structure if __repr__ is well-defined
            # For complex ASTs, a dedicated AST printer would be better.
            # print(ast) 
            for i, decl in enumerate(ast.declarations):
                print(f"  AST Decl {i}: {decl}")
                if isinstance(decl, FunctionDeclNode):
                    for j, stmt in enumerate(decl.body.statements):
                        print(f"    Func {decl.identifier.name} Stmt {j}: {stmt}")


            ir_program = converter.translate_program(ast)
            print("\nGenerated IRProgram:")
            # print(ir_program) # Relies on good __repr__ in ir_nodes
            for i, ir_decl in enumerate(ir_program.declarations):
                print(f"  IR Decl {i}: {ir_decl}")
                if isinstance(ir_decl, IRFunction):
                    for j, ir_stmt in enumerate(ir_decl.body):
                        print(f"    Func {ir_decl.name} Stmt {j}: {ir_stmt}")
            
        except ImportError as e:
            print(f"ImportError: {e}. Make sure all files (tokenizer, parser, ast_nodes, ir_nodes) are accessible.")
        except AttributeError as e:
            print(f"AttributeError: {e}. Check if AST/IR node structures have changed or if methods are missing.")
            import traceback
            traceback.print_exc()
        except NotImplementedError as e:
            print(f"NotImplementedError: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e: # Catch other errors like ParserError from parser
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
        print("------------------------")
