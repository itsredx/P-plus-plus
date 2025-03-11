# interpreter.py

from ast_nodes import *
import sys
import time

class InterpreterError(Exception):
    pass

class Environment:
    def __init__(self, parent=None):
        self.values = {}
        self.parent = parent

    def define(self, name, value):
        self.values[name] = value

    def assign(self, name, value):
        if name in self.values:
            self.values[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            raise InterpreterError(f"Undefined variable '{name}'.")

    def get(self, name):
        if name in self.values:
            return self.values[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise InterpreterError(f"Undefined variable '{name}'.")

# Built-in functions for the interpreter.
def builtin_print(*args):
    # Print all arguments, converting them to string.
    print(*args)
    return None

class Interpreter:
    def __init__(self):
        # The global environment will store variables and built-in functions.
        self.global_env = Environment()
        # Register built-in functions.
        self.global_env.define("print", builtin_print)

    def interpret(self, ast):
        # Interpret the AST starting from the ProgramNode.
        if not isinstance(ast, ProgramNode):
            raise InterpreterError("AST must be a ProgramNode.")
        for declaration in ast.declarations:
            self.execute(declaration, self.global_env)

    def execute(self, node, env):
        if isinstance(node, ExpressionStatementNode):
            return self.evaluate(node.expression, env)
        elif isinstance(node, VariableDeclarationNode):
            value = self.evaluate(node.expression, env)
            env.define(node.identifier, value)
            return value
        elif isinstance(node, FunctionDeclarationNode):
            # For now, we'll store the function AST in the environment.
            env.define(node.identifier, node)
            return node
        else:
            # For now, assume the node is an expression.
            return self.evaluate(node, env)

    def evaluate(self, node, env):
        if isinstance(node, LiteralNode):
            return node.value
        elif isinstance(node, IdentifierNode):
            return env.get(node.name)
        elif isinstance(node, BinaryOpNode):
            left = self.evaluate(node.left, env)
            right = self.evaluate(node.right, env)
            op = node.operator
            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                if right == 0:
                    raise InterpreterError("Division by zero.")
                return left / right
            else:
                raise InterpreterError(f"Unknown binary operator: {op}")
        elif isinstance(node, FunctionCallNode):
            # Evaluate the callee.
            callee = self.evaluate(node.callee, env)
            # Evaluate each argument.
            args = [self.evaluate(arg, env) for arg in node.arguments]
            # If the callee is a built-in function, call it directly.
            if callable(callee):
                return callee(*args)
            else:
                raise InterpreterError(f"Function '{node.callee}' is not callable.")
        else:
            raise InterpreterError(f"Unsupported AST node: {node}")


if __name__ == "__main__":
    from tokenizer import tokenize
    from parser import Parser

    if len(sys.argv) < 2:
        print("Usage: python interpreter.py <filename.pypp>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, "r") as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse_program()
    print("AST:", ast)

    interpreter = Interpreter()
    try:
        start_time = time.time()
        interpreter.interpret(ast)
        end_time = time.time()

        print(f"Interpreter Execution Time: {end_time - start_time:.6f} seconds")
    except Exception as e:
        print("Interpreter error:", e)

