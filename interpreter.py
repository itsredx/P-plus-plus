# interpreter.py
from typing import Any, List, Optional # Added for type hints

from ast_nodes import * # Assuming this imports all necessary AST node classes
import sys
import time # For the main execution timing

class InterpreterError(Exception):
    """Custom exception for interpreter errors."""
    pass

class ReturnSignal(Exception):
    """Exception used to signal a return from a function call."""
    def __init__(self, value):
        self.value = value

class Environment:
    def __init__(self, parent=None):
        self.values = {}
        self.parent = parent

    def define(self, name: str, value: Any): # type: ignore
        """Defines a variable or function in the current scope."""
        self.values[name] = value

    def assign(self, name: str, value: Any): # type: ignore
        """Assigns a value to an existing variable in the current or enclosing scopes."""
        if name in self.values:
            self.values[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            raise InterpreterError(f"Undefined variable '{name}' for assignment.")

    def get(self, name: str) -> Any: # type: ignore
        """Retrieves a variable or function from the current or enclosing scopes."""
        if name in self.values:
            return self.values[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise InterpreterError(f"Undefined variable '{name}'.")

class UserDefinedFunction:
    """Represents a user-defined function in the interpreter."""
    def __init__(self, name: str, params: List[ParamNode], body: BlockNode, closure_env: Environment, return_type_name: Optional[str]): # type: ignore
        self.name = name
        self.params = params  # List of ParamNode
        self.body = body      # BlockNode
        self.closure_env = closure_env # Environment where the function was defined
        self.return_type_name = return_type_name # Optional, for info/debugging

    def __repr__(self):
        return f"<UserDefinedFunction {self.name}(params={len(self.params)})>"

# Built-in functions
def builtin_print(*args):
    print(*args)
    return None

class Interpreter:
    def __init__(self):
        self.global_env = Environment()
        # Register built-in functions
        self.global_env.define("print", builtin_print)
        # Add more built-ins if needed, e.g., len(), input(), etc.

    def interpret(self, program_node: ProgramNode):
        """Interprets the main program node."""
        if not isinstance(program_node, ProgramNode):
            raise InterpreterError("AST root must be a ProgramNode.")
        try:
            for declaration_node in program_node.declarations:
                self.execute(declaration_node, self.global_env)
        except ReturnSignal:
            raise InterpreterError("Return statement outside of function.")


    def execute(self, node: ASTNode, env: Environment) -> Any: # type: ignore
        """Executes a statement node."""
        if isinstance(node, GlobalVarDeclNode):
            return self._execute_global_var_decl(node, env)
        elif isinstance(node, LocalVarDeclNode): # Handles 'var x as type [= expr]' within functions
            return self._execute_local_var_decl(node, env)
        elif isinstance(node, FunctionDeclNode):
            return self._execute_function_decl(node, env)
        elif isinstance(node, VarAssignNode):
            return self._execute_var_assign(node, env)
        elif isinstance(node, ReturnStmtNode):
            return self._execute_return_stmt(node, env) # This will raise ReturnSignal
        elif isinstance(node, ExpressionStatementNode):
            self.evaluate(node.expression, env) # Result is discarded
            return None
        elif isinstance(node, BlockNode):
            return self._execute_block(node, env) # Pass current env, block might create its own
        elif isinstance(node, ImportNode):
            return self._execute_import(node, env)
        # Note: VariableDeclarationNode (old) is superseded by Global/LocalVarDeclNode
        else:
            raise InterpreterError(f"Unsupported statement node for execution: {type(node)}")


    def evaluate(self, node: ExpressionNode, env: Environment) -> Any: # type: ignore
        """Evaluates an expression node."""
        if isinstance(node, LiteralNode):
            return node.value
        elif isinstance(node, IdentifierNode):
            return env.get(node.name)
        elif isinstance(node, BinaryOpNode):
            left_val = self.evaluate(node.left, env)
            right_val = self.evaluate(node.right, env)
            op = node.operator

            # Arithmetic operators
            if op == '+': return left_val + right_val
            elif op == '-': return left_val - right_val
            elif op == '*': return left_val * right_val
            elif op == '/':
                if right_val == 0: raise InterpreterError("Division by zero.")
                return left_val / right_val
            elif op == '%': 
                if right_val == 0: raise InterpreterError("Modulo by zero.")
                return left_val % right_val
            # Comparison operators
            elif op == '==': return left_val == right_val
            elif op == '!=': return left_val != right_val
            elif op == '<': return left_val < right_val
            elif op == '<=': return left_val <= right_val
            elif op == '>': return left_val > right_val
            elif op == '>=': return left_val >= right_val
            # Logical operators (short-circuiting not implemented here, basic bool eval)
            # For true short-circuiting, 'and' and 'or' would need to be special forms/control structures.
            elif op.lower() == 'and': return bool(left_val and right_val)
            elif op.lower() == 'or': return bool(left_val or right_val)
            else:
                raise InterpreterError(f"Unknown binary operator: {op}")
        elif isinstance(node, FunctionCallNode):
            return self._evaluate_function_call(node, env)
        else:
            raise InterpreterError(f"Unsupported expression node for evaluation: {type(node)}")

    def _execute_global_var_decl(self, node: GlobalVarDeclNode, env: Environment):
        value = None
        if node.expression:
            value = self.evaluate(node.expression, env)
        env.define(node.identifier.name, value)
        # TypeNode (node.var_type) is ignored by interpreter for now

    def _execute_local_var_decl(self, node: LocalVarDeclNode, env: Environment):
        value = None
        if node.expression:
            value = self.evaluate(node.expression, env)
        env.define(node.identifier.name, value) # Define in the current (e.g. function's) environment
        # TypeNode (node.var_type) is ignored by interpreter for now

    def _execute_function_decl(self, node: FunctionDeclNode, env: Environment):
        func_name = node.identifier.name
        return_type_name = node.return_type.name if node.return_type else "void"
        user_func = UserDefinedFunction(
            name=func_name,
            params=node.params,
            body=node.body,
            closure_env=env, # Capture the environment where the function is defined
            return_type_name=return_type_name
        )
        env.define(func_name, user_func)

    def _execute_var_assign(self, node: VarAssignNode, env: Environment):
        value = self.evaluate(node.expression, env)
        env.assign(node.identifier.name, value)

    def _execute_return_stmt(self, node: ReturnStmtNode, env: Environment):
        value = None
        if node.expression:
            value = self.evaluate(node.expression, env)
        raise ReturnSignal(value)

    def _execute_block(self, node: BlockNode, env: Environment) -> Any: # type: ignore
        # For P-- Phase 1, blocks don't create new lexical scopes by default.
        # Variables declared with LocalVarDeclNode within the block will be added to the current 'env'.
        # If strict C-like block scoping is desired later, a new Environment would be created here:
        # block_env = Environment(parent=env)
        # And self.execute(stmt, block_env) would be used.
        # For now, execute statements in the provided environment.
        
        # If a ReturnSignal is raised by any statement, it should propagate up.
        # The function call handler (_evaluate_function_call) is responsible for catching it.
        for stmt_node in node.statements:
            self.execute(stmt_node, env) 
            # If self.execute raises ReturnSignal, it will naturally propagate up from here.
        return None # Block itself doesn't return a value unless a ReturnStmt is hit.

    def _execute_import(self, node: ImportNode, env: Environment):
        # For Phase 1, simply acknowledge the import.
        # Future phases might load modules into the environment.
        print(f"[Interpreter Info] Encountered import: {node.module_name.name}")


    def _evaluate_function_call(self, node: FunctionCallNode, env: Environment) -> Any: # type: ignore
        if not isinstance(node.callee, IdentifierNode):
            # More complex callee expressions (e.g., (get_func())()) are not typical for this phase
            raise InterpreterError("Callee must be an identifier.")
        
        callee_name = node.callee.name
        callee = env.get(callee_name)

        evaluated_args = [self.evaluate(arg, env) for arg in node.arguments]

        if isinstance(callee, UserDefinedFunction):
            if len(evaluated_args) != len(callee.params):
                raise InterpreterError(
                    f"Function '{callee.name}' expected {len(callee.params)} arguments, "
                    f"but got {len(evaluated_args)}."
                )
            
            # Create a new environment for the function call, enclosing the function's definition environment
            func_env = Environment(parent=callee.closure_env)
            
            # Bind arguments to parameter names in the function's new environment
            for i, param_node in enumerate(callee.params):
                func_env.define(param_node.identifier.name, evaluated_args[i])
            
            try:
                # Execute the function body in its new environment
                self.execute(callee.body, func_env)
                # If the function body completes without a ReturnSignal (for a non-void function)
                return None # Implicit return None if no return statement is hit
            except ReturnSignal as rs:
                return rs.value # Return the value from the ReturnStmtNode
        
        elif callable(callee): # For Python-defined built-in functions
            try:
                return callee(*evaluated_args)
            except Exception as e:
                raise InterpreterError(f"Error calling built-in function '{callee_name}': {e}")
        else:
            raise InterpreterError(f"'{callee_name}' is not a function.")


if __name__ == "__main__":
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

    # Assuming tokenizer.py and parser.py are in the same directory or accessible
    # and that parser.py produces ASTs with the new nodes.
    from tokenizer import tokenize
    from parser import Parser 

    print(f"Interpreting file: {filename}\n")
    try:
        tokens = tokenize(code)
        # Filter for parser if needed (e.g. SKIP, some NEWLINEs if INDENT/DEDENT handles structure)
        # tokens_for_parser = [t for t in tokens if t.type not in ('SKIP', 'NEWLINE')] # Example
        tokens_for_parser = [t for t in tokens if t.type not in ('SKIP')] # Basic filtering

        parser = Parser(tokens_for_parser)
        ast_program = parser.parse_program()
        
        # print("--- AST ---")
        # print(ast_program) # Can be very verbose
        # print("------------")

        interpreter = Interpreter()
        start_time = time.time()
        interpreter.interpret(ast_program)
        end_time = time.time()

        print(f"\nInterpreter Execution Time: {end_time - start_time:.6f} seconds")

    except InterpreterError as e:
        print(f"Interpreter Runtime Error: {e}", file=sys.stderr)
    except ParserError as e: # Assuming ParserError is defined in parser.py
         print(f"Parser Error: {e}", file=sys.stderr)
    except SyntaxError as e: # From tokenizer or other syntax issues
         print(f"Syntax Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
