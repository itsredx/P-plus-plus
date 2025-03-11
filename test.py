# run_diff_tests.py

from tokenizer import tokenize
from parser import Parser
from ast_to_ir import translate
from llvm_codegen import generate_llvm_ir

# List of test cases: (description, source code)
test_cases = [
    ("Multiplication", 'print(21 * 21)'),
    ("Addition", 'print(21 + 21)'),
    ("Subtraction", 'print(21 - 7)'),
    ("Division", 'print(21 / 7)'),
    ("Comparison Less Than", 'print(21 < 42)'),
    ("Comparison Greater or Equal", 'print(21 >= 21)'),
    ("Combined Expression", 'print((21 + 21) * (42 / 2))'),
]

def run_test(code, description):
    print("======================================")
    print(f"Test: {description}")
    print("Source Code:")
    print(code)
    print("--------------------------------------")
    
    try:
        tokens = tokenize(code)
        parser = Parser(tokens)
        ast = parser.parse_program()
        
        # Translate AST to our custom IR.
        custom_ir = translate(ast)
        print("Custom IR:")
        print(custom_ir)
        
        # Generate LLVM IR from our custom IR.
        llvm_module = generate_llvm_ir(custom_ir)
        print("Generated LLVM IR:")
        print(llvm_module)
    except Exception as e:
        print("Error:", e)
    print("======================================\n")

if __name__ == "__main__":
    for desc, code in test_cases:
        run_test(code, desc)
