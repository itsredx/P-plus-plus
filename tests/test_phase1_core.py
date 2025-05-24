import unittest
import io
import sys
from contextlib import redirect_stdout

# Assuming all necessary modules are in the parent directory or PYTHONPATH is set
sys.path.append('..') # Adjust if your project structure is different

from tokenizer import tokenize, Token 
from parser import Parser, ParserError
from ast_nodes import ProgramNode, FunctionCallNode, IdentifierNode, ExpressionStatementNode # Added ExpressionStatementNode
from interpreter import Interpreter, InterpreterError, Environment, UserDefinedFunction
from ast_to_ir import ASTToIRConverter
from ir_nodes import IRProgram
from llvm_codegen import LLVMCodeGenerator

# Helper to manually create token stream with INDENT/DEDENT for testing blocks
def create_token_stream(data):
    tokens = []
    for item in data:
        line = item[2] if len(item) >= 3 else 1
        col = item[3] if len(item) >= 4 else 0
        tokens.append(Token(type=item[0], value=item[1], line=line, column=col))
    return tokens

class TestPhase1Core(unittest.TestCase):

    def setUp(self):
        self.interpreter = Interpreter()
        self.ast_converter = ASTToIRConverter()
        self.llvm_generator = LLVMCodeGenerator()

    # --- Test Case 1: Global Variable Declaration and Access ---
    # P++ Code
    gv_decl_code_tokens = create_token_stream([
        ('ID', 'g_var1', 1, 0), ('AS', 'AS', 1, 7), ('INT', 'int', 1, 10),
        ('ID', 'g_var2', 2, 0), ('AS', 'AS', 2, 7), ('INT', 'int', 2, 10), ('ASSIGN', '=', 2, 14), ('NUMBER', '100', 2, 16),
        ('ID', 'g_var3', 3, 0), ('AS', 'AS', 3, 7), ('BOOL', 'bool', 3, 12), ('ASSIGN', '=', 3, 14), ('TRUE', 'TRUE', 3, 16),
        ('ID', 'g_var4', 4, 0), ('AS', 'AS', 4, 7), ('STRING', 'string', 4, 14), ('ASSIGN', '=', 4, 16), ('STRING', '"hello"', 4, 18),
        
        ('DEF', 'DEF', 6, 0), ('ID', 'get_g_var2_val', 6, 4), ('LPAREN', '(', 6, 11), ('RPAREN', ')', 6, 12), ('ARROW', '->', 6, 14), ('INT', 'int', 6, 17), ('COLON', ':', 6, 20),
        ('INDENT', '<INDENT>', 7, 0),
        ('RETURN', 'RETURN', 8, 4), ('ID', 'g_var2', 8, 11),
        ('DEDENT', '<DEDENT>', 9, 0),
        
        ('DEF', 'DEF', 10, 0), ('ID', 'get_g_var1_val', 10, 4), ('LPAREN', '(', 10, 20), ('RPAREN', ')', 10, 21), ('ARROW', '->', 10, 23), ('INT', 'int', 10, 26), ('COLON', ':', 10, 29),
        ('INDENT', '<INDENT>', 11, 0),
        ('RETURN', 'RETURN', 12, 4), ('ID', 'g_var1', 12, 11),
        ('DEDENT', '<DEDENT>', 13, 0),
        
        # Global to hold result from main_gv
        # Corrected: Removed DEF from global variable declaration, adjusted col numbers for consistency
        ('ID', 'main_gv_test_val_holder', 15,0), ('AS', 'AS', 15,28), ('INT', 'int', 15,31), 
        # Main function to exercise and allow result checking
        ('DEF', 'DEF', 16,0), ('ID', 'main_gv', 16,4), ('LPAREN', '(',16,11),('RPAREN',')',16,12), ('ARROW','->',16,14),('VOID','void',16,17),('COLON',':',16,21),
        ('INDENT', '<INDENT>', 17,0),
        ('ID', 'main_gv_test_val_holder', 18,4), ('ASSIGN', '=', 18,28), ('ID', 'g_var2', 18,30), # Assign g_var2 to the holder
        ('DEDENT', '<DEDENT>', 19,0),
    ])

    def test_interpreter_global_vars(self):
        parser = Parser(self.gv_decl_code_tokens)
        ast = parser.parse_program()
        self.interpreter.interpret(ast) # Defines globals and functions
        
        self.assertIsNone(self.interpreter.global_env.get('g_var1'))
        self.assertEqual(self.interpreter.global_env.get('g_var2'), 100)
        self.assertEqual(self.interpreter.global_env.get('g_var3'), True)
        self.assertEqual(self.interpreter.global_env.get('g_var4'), "hello")

        # Call main_gv to set main_gv_test_val_holder
        main_gv_func = self.interpreter.global_env.get('main_gv')
        self.assertIsInstance(main_gv_func, UserDefinedFunction)
        # Manually creating an AST node for the call for the interpreter
        # This is a bit of a test harness detail.
        call_main_gv_ast = ProgramNode(declarations=[
            ExpressionStatementNode(expression=FunctionCallNode(callee=IdentifierNode('main_gv'), arguments=[]))
        ])
        self.interpreter.interpret(call_main_gv_ast) # Execute the call
        self.assertEqual(self.interpreter.global_env.get('main_gv_test_val_holder'), 100)

    def test_llvm_global_vars(self):
        parser = Parser(self.gv_decl_code_tokens)
        ast = parser.parse_program()
        ir_program = self.ast_converter.translate_program(ast)
        llvm_ir_string = self.llvm_generator.generate_llvm_ir(ir_program)
        
        self.assertTrue(llvm_ir_string)
        self.assertIn("@\"g_var1\"", llvm_ir_string)
        self.assertIn("@\"g_var2\"", llvm_ir_string)
        self.assertIn("@\"g_var4\" = internal global [6 x i8] c\"hello\\00\"", llvm_ir_string) # Check string init
        self.assertIn("define void @\"main_gv\"()", llvm_ir_string)
        # print("\nLLVM - Global Vars:\n", llvm_ir_string)


    # --- Test Case 2: Simple Function Definition and Call ---
    fn_call_code_tokens = create_token_stream([
        ('DEF', 'DEF', 1,0), ('ID', 'greet', 1,4), ('LPAREN', '(', 1,9), ('RPAREN', ')', 1,10), ('ARROW', '->', 1,12), ('VOID', 'void', 1,15), ('COLON', ':', 1,19),
        ('INDENT', '<INDENT>', 2,0),
        ('ID', 'print', 2,4), ('LPAREN', '(', 2,9), ('STRING', '"Hello from greet"', 2,10), ('RPAREN', ')', 2,28),
        ('DEDENT', '<DEDENT>', 3,0),

        ('DEF', 'DEF', 5,0), ('ID', 'add', 5,4), ('LPAREN', '(', 5,7), 
            ('ID', 'a', 5,8), ('AS', 'AS', 5,10), ('INT', 'int', 5,13), ('COMMA', ',', 5,16), 
            ('ID', 'b', 5,18), ('AS', 'AS', 5,20), ('INT', 'int', 5,23), 
        ('RPAREN', ')', 5,26), ('ARROW', '->', 5,28), ('INT', 'int', 5,31), ('COLON', ':', 5,34),
        ('INDENT', '<INDENT>', 6,0),
        ('RETURN', 'RETURN', 7,4), ('ID', 'a', 7,11), ('OP', '+', 7,13), ('ID', 'b', 7,15),
        ('DEDENT', '<DEDENT>', 8,0),
        
        ('ID', 'main_add_result_holder', 9,0), ('AS', 'AS', 9,18), ('INT', 'int', 9,21), # Global to hold result
        ('DEF', 'DEF', 10,0), ('ID', 'main_fn_call', 10,4), ('LPAREN', '(',10,11),('RPAREN',')',10,12), ('ARROW','->',10,14),('VOID','void',10,17),('COLON',':',10,21),
        ('INDENT', '<INDENT>', 11,0),
        ('ID', 'greet', 12,4), ('LPAREN', '(', 12,9), ('RPAREN', ')', 12,10), # Call greet
        ('ID', 'main_add_result_holder', 13,4), ('ASSIGN', '=', 13,20),
            ('ID', 'add', 13,22), ('LPAREN', '(', 13,25), ('NUMBER', '5', 13,26), ('COMMA', ',', 13,27), ('NUMBER', '7', 13,28), ('RPAREN', ')', 13,29),
        ('DEDENT', '<DEDENT>', 14,0),
    ])

    def test_interpreter_func_def_call(self):
        parser = Parser(self.fn_call_code_tokens)
        ast = parser.parse_program()
        
        captured_output = io.StringIO()
        sys.stdout = captured_output # type: ignore
        self.interpreter.interpret(ast) # Defines functions

        # Call main_fn_call
        call_main_ast = ProgramNode(declarations=[
            ExpressionStatementNode(expression=FunctionCallNode(callee=IdentifierNode('main_fn_call'), arguments=[]))
        ])
        self.interpreter.interpret(call_main_ast)
        sys.stdout = sys.__stdout__

        self.assertIn("Hello from greet", captured_output.getvalue())
        self.assertEqual(self.interpreter.global_env.get('main_add_result_holder'), 12)


    def test_llvm_func_def_call(self):
        parser = Parser(self.fn_call_code_tokens)
        ast = parser.parse_program()
        ir_program = self.ast_converter.translate_program(ast)
        llvm_ir_string = self.llvm_generator.generate_llvm_ir(ir_program)

        self.assertTrue(llvm_ir_string)
        self.assertIn("define void @\"greet\"()", llvm_ir_string)
        self.assertIn("define i32 @\"add\"(i32 %\"a\", i32 %\"b\")", llvm_ir_string)
        self.assertIn("@\"main_add_result_holder\"", llvm_ir_string) 
        self.assertIn("call void @\"greet\"()", llvm_ir_string) # In main_fn_call
        self.assertIn("call i32 @\"add\"(i32 5, i32 7)", llvm_ir_string) # In main_fn_call
        # print("\nLLVM - Func Def Call:\n", llvm_ir_string)


    # --- Test Case 3: Function returning a value, global var usage ---
    fn_return_global_code_tokens = create_token_stream([
        ('ID', 'g_val', 1,0), ('AS', 'AS', 1,6), ('INT', 'int', 1,9), ('ASSIGN', '=', 1,11), ('NUMBER', '50', 1,13),
        ('DEF', 'DEF', 3,0), ('ID', 'get_plus_global', 3,4), ('LPAREN', '(', 3,20), 
            ('ID', 'x', 3,21), ('AS', 'AS', 3,23), ('INT', 'int', 3,26), 
        ('RPAREN', ')', 3,29), ('ARROW', '->', 3,31), ('INT', 'int', 3,34), ('COLON', ':', 3,37),
        ('INDENT', '<INDENT>', 4,0),
        ('RETURN', 'RETURN', 5,4), ('ID', 'x', 5,11), ('OP', '+', 5,13), ('ID', 'g_val', 5,15),
        ('DEDENT', '<DEDENT>', 6,0),

        ('ID', 'main_res_holder_fn_ret_glob', 7,0), ('AS', 'AS', 7,24), ('INT', 'int', 7,27), # Global to hold result
        ('DEF', 'DEF', 8,0), ('ID', 'main_fn_ret_glob', 8,4), ('LPAREN', '(',8,11),('RPAREN',')',8,12), ('ARROW','->',8,14),('VOID','void',8,17),('COLON',':',8,21),
        ('INDENT', '<INDENT>', 9,0),
        ('ID', 'main_res_holder_fn_ret_glob', 10,4), ('ASSIGN', '=', 10,20),
            ('ID', 'get_plus_global', 10,22), ('LPAREN', '(', 10,38), ('NUMBER', '10', 10,39), ('RPAREN', ')', 10,41),
        ('DEDENT', '<DEDENT>', 11,0),
    ])

    def test_interpreter_func_return_global(self):
        parser = Parser(self.fn_return_global_code_tokens)
        ast = parser.parse_program()
        self.interpreter.interpret(ast) # Defines

        call_main_ast = ProgramNode(declarations=[
            ExpressionStatementNode(expression=FunctionCallNode(callee=IdentifierNode('main_fn_ret_glob'), arguments=[]))
        ])
        self.interpreter.interpret(call_main_ast) # Executes main

        self.assertEqual(self.interpreter.global_env.get('main_res_holder_fn_ret_glob'), 60)

    def test_llvm_func_return_global(self):
        parser = Parser(self.fn_return_global_code_tokens)
        ast = parser.parse_program()
        ir_program = self.ast_converter.translate_program(ast)
        llvm_ir_string = self.llvm_generator.generate_llvm_ir(ir_program)

        self.assertTrue(llvm_ir_string)
        self.assertIn("@\"g_val\"", llvm_ir_string)
        self.assertIn("define i32 @\"get_plus_global\"(i32 %\"x\")", llvm_ir_string)
        self.assertIn("call i32 @\"get_plus_global\"(i32 10)", llvm_ir_string)
        # print("\nLLVM - Func Return Global:\n", llvm_ir_string)


    # --- Test Case 4: Variable Assignment (Global and Local within a function) ---
    var_assign_code_tokens = create_token_stream([
        ('ID', 'g_assign_outer', 1,0), ('AS', 'AS', 1,13), ('INT', 'int', 1,16), ('ASSIGN', '=', 1,18), ('NUMBER', '10', 1,20),
        ('ID', 'g_final_val_holder', 2,0), ('AS', 'AS', 2,13), ('INT', 'int', 2,16), # Global to hold result
        ('DEF', 'DEF', 4,0), ('ID', 'main_var_assign', 4,4), ('LPAREN', '(',4,11),('RPAREN',')',4,12), ('ARROW','->',4,14),('VOID','void',4,17),('COLON',':',4,21),
        ('INDENT', '<INDENT>', 5,0),
        ('ID', 'g_assign_outer', 6,4), ('ASSIGN', '=', 6,20), ('NUMBER', '25', 6,22), # Assignment to global
        ('ID', 'local_var', 7,4), ('AS', 'AS', 7,14), ('INT', 'int', 7,17), ('ASSIGN', '=', 7,19), ('ID', 'g_assign_outer', 7,21),
        ('ID', 'local_var', 8,4), ('ASSIGN', '=', 8,14), ('ID', 'local_var', 8,16), ('OP', '+', 8,18), ('NUMBER', '5', 8,20), # local_var = 25 + 5 = 30
        ('ID', 'g_final_val_holder', 9,4), ('ASSIGN', '=', 9,20), ('ID', 'local_var', 9,22),
        ('DEDENT', '<DEDENT>', 10,0),
    ])

    def test_interpreter_var_assignment(self):
        parser = Parser(self.var_assign_code_tokens)
        ast = parser.parse_program()
        self.interpreter.interpret(ast) # Defines g_assign_outer, g_final_val_holder, main_var_assign

        self.assertEqual(self.interpreter.global_env.get('g_assign_outer'), 10) # Initial global value

        call_main_ast = ProgramNode(declarations=[
            ExpressionStatementNode(expression=FunctionCallNode(callee=IdentifierNode('main_var_assign'), arguments=[]))
        ])
        self.interpreter.interpret(call_main_ast) # Executes main

        self.assertEqual(self.interpreter.global_env.get('g_assign_outer'), 25) # Modified by main
        self.assertEqual(self.interpreter.global_env.get('g_final_val_holder'), 30)


    def test_llvm_var_assignment(self):
        parser = Parser(self.var_assign_code_tokens)
        ast = parser.parse_program()
        ir_program = self.ast_converter.translate_program(ast)
        llvm_ir_string = self.llvm_generator.generate_llvm_ir(ir_program)

        self.assertTrue(llvm_ir_string)
        self.assertIn("@\"g_assign_outer\"", llvm_ir_string)
        self.assertIn("@\"g_final_val_holder\"", llvm_ir_string)
        self.assertIn("store i32 25, i32* @\"g_assign_outer\"", llvm_ir_string) # Assignment in main, corrected ptr to i32*
        # print("\nLLVM - Var Assignment:\n", llvm_ir_string)

    # --- Test Case 5: Print (inside a function) ---
    print_code_tokens = create_token_stream([
        ('DEF', 'DEF', 1,0), ('ID', 'test_print_func', 1,4), ('LPAREN', '(', 1,14), ('RPAREN', ')', 1,15), ('ARROW', '->', 1,17), ('VOID', 'void', 1,20), ('COLON', ':', 1,24),
        ('INDENT', '<INDENT>', 2,0),
        ('ID', 'print', 3,4), ('LPAREN', '(', 3,9), ('STRING', '"Hello P++ Print!"', 3,10), ('RPAREN', ')', 3,30),
        ('ID', 'val_to_print', 4,4), ('AS', 'AS', 4,17), ('INT', 'int', 4,20), ('ASSIGN', '=', 4,22), ('NUMBER', '12345', 4,24),
        ('ID', 'print', 5,4), ('LPAREN', '(', 5,9), ('ID', 'val_to_print', 5,10), ('RPAREN', ')', 5,22),
        ('DEDENT', '<DEDENT>', 6,0)
    ])
    
    def test_interpreter_print(self):
        parser = Parser(self.print_code_tokens)
        ast = parser.parse_program()
        self.interpreter.interpret(ast) # Defines test_print_func

        captured_output = io.StringIO()
        sys.stdout = captured_output # type: ignore

        # Call test_print_func
        call_test_print_ast = ProgramNode(declarations=[
            ExpressionStatementNode(expression=FunctionCallNode(callee=IdentifierNode('test_print_func'), arguments=[]))
        ])
        self.interpreter.interpret(call_test_print_ast)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("Hello P++ Print!", output)
        self.assertIn("12345", output)

    def test_llvm_print(self):
        parser = Parser(self.print_code_tokens)
        ast = parser.parse_program()
        ir_program = self.ast_converter.translate_program(ast)
        llvm_ir_string = self.llvm_generator.generate_llvm_ir(ir_program)

        # Temporary print for debugging the test_llvm_print failure
        print(f"\n--- LLVM IR for test_llvm_print ---\n{llvm_ir_string}\n---------------------------------")
        
        found_puts = "@puts" in llvm_ir_string
        found_printf = "@printf" in llvm_ir_string
        print(f"DEBUG: Found @puts in llvm_ir_string: {found_puts}")
        print(f"DEBUG: Found @printf in llvm_ir_string: {found_printf}")
        
        # Ensure it's a plain string and print its type for sanity check
        plain_llvm_ir_string = str(llvm_ir_string)
        print(f"DEBUG: Type of llvm_ir_string: {type(plain_llvm_ir_string)}")

        import re # Import re for regex search
        # Normalize whitespace and search
        normalized_ir_string = " ".join(plain_llvm_ir_string.split())
        print(f"DEBUG: Normalized LLVM IR (first 300 chars): {normalized_ir_string[:300]}")


        found_puts_re = bool(re.search(r"@puts", normalized_ir_string))
        found_printf_re = bool(re.search(r"@printf", normalized_ir_string))
        # Also check for quoted versions, just in case, though IR output shows unquoted for these
        found_puts_re_quoted = bool(re.search(r'@\"puts\"', normalized_ir_string))
        found_printf_re_quoted = bool(re.search(r'@\"printf\"', normalized_ir_string))


        print(f"DEBUG: Found @puts via regex (on normalized): {found_puts_re}")
        print(f"DEBUG: Found @printf via regex (on normalized): {found_printf_re}")
        print(f"DEBUG: Found @\"puts\" via regex (on normalized): {found_puts_re_quoted}")
        print(f"DEBUG: Found @\"printf\" via regex (on normalized): {found_printf_re_quoted}")

        self.assertTrue(plain_llvm_ir_string)
        self.assertIn("define void @\"test_print_func\"()", plain_llvm_ir_string) # Check on plain string
        self.assertTrue(found_puts_re or found_printf_re or found_puts_re_quoted or found_printf_re_quoted, 
                        "puts or printf not found for print test (using regex on normalized string)")
        self.assertIn("Hello P++ Print!", plain_llvm_ir_string) # The string constant
        # print("\nLLVM - Print:\n", llvm_ir_string)

if __name__ == '__main__':
    unittest.main()
