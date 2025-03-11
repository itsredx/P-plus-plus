# test_interpreter.py

import io
from contextlib import redirect_stdout
from tokenizer import tokenize
from parser import Parser
from interpreter import Interpreter, InterpreterError

def run_interpreter_test(code):
    """Tokenize, parse, and interpret the given code; capture printed output."""
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse_program()
    interpreter = Interpreter()
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            interpreter.interpret(ast)
        except InterpreterError as e:
            # Print the error message to captured output.
            print(f"Error: {e}")
    return output.getvalue()

def main():
    tests = [
        {
            "description": "Print a string literal",
            "code": 'print("Hello, Test!")',
            "expected": "Hello, Test!\n"
        },
        {
            "description": "Variable assignment and print",
            "code": 'x = 42\nprint(x)',
            "expected": "42.0\n"
        },
        {
            "description": "Arithmetic expression",
            "code": 'y = 5 * 7 + 3\nprint(y)',
            "expected": "38.0\n"  # 5 * 7 = 35, plus 3 = 38
        },
        {
            "description": "Nested arithmetic",
            "code": 'print((2 + 3) * (10 - 4))',
            "expected": "30.0\n"  # (2+3)=5, (10-4)=6, 5*6 = 30
        },
        {
            "description": "Division by zero error",
            "code": 'print(10 / 0)',
            "expected": "Error: Division by zero.\n"
        }
    ]

    all_passed = True
    for test in tests:
        result = run_interpreter_test(test["code"])
        print("======================================")
        print(f"Test: {test['description']}")
        print("Code:")
        print(test["code"])
        print("Expected Output:")
        print(test["expected"])
        print("Interpreter Output:")
        print(result)
        if result != test["expected"]:
            print("Result: FAILED\n")
            all_passed = False
        else:
            print("Result: PASSED\n")
    print("======================================")
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")

if __name__ == "__main__":
    main()
