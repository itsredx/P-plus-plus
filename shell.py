import sys
from interpreter import Interpreter
from compiler import compile_and_run
from tokenizer import tokenize
from parser import Parser

class PPlusShell:
    def __init__(self):
        self.mode = "interpreter"  # Default mode
        self.interpreter = Interpreter()

    def start(self):
        print("P++ Shell | Type ':mode compiler' or ':mode interpreter' to switch modes.")
        print("Type ':exit' to quit.")

        while True:
            try:
                user_input = input("P++> ").strip()
                
                if user_input == ":exit":
                    break
                elif user_input.startswith(":mode"):
                    self.change_mode(user_input)
                else:
                    self.execute(user_input)
            except Exception as e:
                print(f"Error: {e}")

    def change_mode(self, command):
        parts = command.split()
        if len(parts) < 2:
            print("Usage: :mode [interpreter|compiler]")
            return

        new_mode = parts[1].lower()
        if new_mode in ["interpreter", "compiler"]:
            self.mode = new_mode
            print(f"Switched to {new_mode} mode.")
        else:
            print("Invalid mode. Use 'interpreter' or 'compiler'.")

    def execute(self, code):
        tokens = tokenize(code)
        parser = Parser(tokens)
        ast = parser.parse_program()

        if self.mode == "interpreter":
            self.interpreter.interpret(ast)
        else:
            compile_and_run(code, aot=False)  # JIT execution

if __name__ == "__main__":
    PPlusShell().start()
