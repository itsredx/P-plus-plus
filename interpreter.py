# interpreter.py
from typing import Any, List, Optional, Dict, Tuple, Union # Added Dict, Tuple, Union

from ast_nodes import * 
import sys
import time 

class InterpreterError(Exception):
    """Custom exception for interpreter errors."""
    pass

class ReturnSignal(Exception):
    """Exception used to signal a return from a function call."""
    def __init__(self, value):
        self.value = value

class Environment:
    def __init__(self, parent: Optional['Environment'] = None):
        self.values: Dict[str, Any] = {}
        self.parent = parent

    def define(self, name: str, value: Any):
        self.values[name] = value

    def assign(self, name: str, value: Any):
        if name in self.values:
            self.values[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            raise InterpreterError(f"Undefined variable '{name}' for assignment.")

    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise InterpreterError(f"Undefined variable '{name}'.")

    def get_env_for_var(self, name: str) -> Optional['Environment']:
        """Finds the environment where a variable is defined."""
        if name in self.values:
            return self
        elif self.parent:
            return self.parent.get_env_for_var(name)
        else:
            return None


class UserDefinedFunction:
    def __init__(self, name: str, params: List[ParamNode], body: BlockNode, closure_env: Environment, return_type_name: Optional[str]):
        self.name = name
        self.params = params
        self.body = body
        self.closure_env = closure_env
        self.return_type_name = return_type_name

    def __repr__(self):
        return f"<UserDefinedFunction {self.name}(params={len(self.params)})>"

# Represents a simulated "address" of a variable stored in an Environment.
SimulatedStackAddress = Tuple[Environment, str] 
# Represents a simulated heap address (index into Interpreter.heap).
SimulatedHeapAddress = int
# Union type for any kind of simulated address the interpreter might handle.
SimulatedAddress = Union[SimulatedStackAddress, SimulatedHeapAddress]


# Built-in functions
def builtin_print(*args):
    # Replace our internal null representation with Python's None for printing
    printable_args = [None if arg is Interpreter.NULL_POINTER_VALUE else arg for arg in args]
    print(*printable_args)
    return None

class Interpreter:
    NULL_POINTER_VALUE: Optional[int] = None # Using Python's None for null pointers internally.
                                          # Could be a special int like -1 if heap addresses are non-negative.

    def __init__(self):
        self.global_env = Environment()
        self.global_env.define("print", builtin_print)
        
        # Simulated Memory Model
        self.heap: List[Any] = [] # Stores actual values for heap allocations
        self.allocations: Dict[SimulatedHeapAddress, Dict[str, Any]] = {} 
        # Example allocation metadata: {'size_elements': int, 'type_name': str, 'freed': bool}
        # self.next_heap_address is implicitly len(self.heap) when appending.

    def _get_type_name_from_ast_type(self, type_node: Optional[TypeNode]) -> str:
        if type_node is None:
            return "unknown_or_void"
        if isinstance(type_node, BuiltinTypeNode):
            return type_node.name
        if isinstance(type_node, PointerTypeNode):
            return "*" + self._get_type_name_from_ast_type(type_node.pointee_type)
        return "complex_type"


    def interpret(self, program_node: ProgramNode):
        if not isinstance(program_node, ProgramNode):
            raise InterpreterError("AST root must be a ProgramNode.")
        try:
            for declaration_node in program_node.declarations:
                self.execute(declaration_node, self.global_env)
        except ReturnSignal:
            raise InterpreterError("Return statement outside of function.")


    def execute(self, node: ASTNode, env: Environment) -> Any:
        if isinstance(node, GlobalVarDeclNode):
            return self._execute_global_var_decl(node, env)
        elif isinstance(node, LocalVarDeclNode): 
            return self._execute_local_var_decl(node, env)
        elif isinstance(node, FunctionDeclNode):
            return self._execute_function_decl(node, env)
        elif isinstance(node, VarAssignNode):
            return self._execute_var_assign(node, env)
        elif isinstance(node, PointerAssignmentNode): # Phase 2
            return self._execute_pointer_assignment(node, env)
        elif isinstance(node, ReturnStmtNode):
            return self._execute_return_stmt(node, env)
        elif isinstance(node, ExpressionStatementNode):
            self.evaluate(node.expression, env) 
            return None
        elif isinstance(node, BlockNode):
            return self._execute_block(node, env)
        elif isinstance(node, ImportNode):
            return self._execute_import(node, env)
        elif isinstance(node, FreeNode): # Phase 2
            return self._execute_free(node, env)
        else:
            raise InterpreterError(f"Unsupported statement node for execution: {type(node)}")


    def evaluate(self, node: ExpressionNode, env: Environment) -> Any:
        if isinstance(node, LiteralNode):
            # LiteralNode(None) will represent a null pointer
            return node.value 
        elif isinstance(node, IdentifierNode):
            return env.get(node.name)
        elif isinstance(node, BinaryOpNode):
            # (Existing binary op logic from Phase 1, assumed to be sufficient for now)
            left_val = self.evaluate(node.left, env)
            right_val = self.evaluate(node.right, env)
            op = node.operator
            if op == '+': return left_val + right_val
            elif op == '-': return left_val - right_val
            elif op == '*': return left_val * right_val
            elif op == '/':
                if right_val == 0: raise InterpreterError("Division by zero.")
                return left_val / right_val
            elif op == '%': 
                if right_val == 0: raise InterpreterError("Modulo by zero.")
                return left_val % right_val
            elif op == '==': return left_val == right_val
            elif op == '!=': return left_val != right_val
            elif op == '<': return left_val < right_val
            elif op == '<=': return left_val <= right_val
            elif op == '>': return left_val > right_val
            elif op == '>=': return left_val >= right_val
            elif op.lower() == 'and': return bool(left_val and right_val)
            elif op.lower() == 'or': return bool(left_val or right_val)
            else: raise InterpreterError(f"Unknown binary operator: {op}")
        elif isinstance(node, FunctionCallNode):
            # Check if it's a "malloc" or "free" identifier being called,
            # which should be parsed as MallocNode/FreeNode, not generic FunctionCallNode.
            if isinstance(node.callee, IdentifierNode):
                if node.callee.name == "malloc":
                    raise InterpreterError("`malloc` should be parsed as MallocNode, not a generic function call.")
                if node.callee.name == "free":
                     raise InterpreterError("`free` should be parsed as FreeNode, not a generic function call.")
            return self._evaluate_function_call(node, env)
        
        # Phase 2 Expression Nodes
        elif isinstance(node, AddressOfNode):
            var_name = node.identifier.name
            # Find the environment where var_name is actually stored
            target_env = env.get_env_for_var(var_name)
            if target_env is None:
                raise InterpreterError(f"Cannot take address of undefined variable '{var_name}'.")
            return (target_env, var_name) # SimulatedStackAddress
        
        elif isinstance(node, DereferenceNode):
            address_repr = self.evaluate(node.expression, env)
            if address_repr is self.NULL_POINTER_VALUE:
                raise InterpreterError("Null pointer dereference.")
            
            if isinstance(address_repr, tuple) and len(address_repr) == 2 and isinstance(address_repr[0], Environment):
                # It's a SimulatedStackAddress
                target_env, var_name = address_repr
                return target_env.get(var_name)
            elif isinstance(address_repr, int): # Assuming heap addresses are integers
                # It's a SimulatedHeapAddress
                if address_repr not in self.allocations or self.allocations[address_repr]['freed']:
                    raise InterpreterError(f"Invalid or freed heap address: {address_repr}.")
                # For simplicity, assume heap stores single elements directly.
                # A real system would use type info from self.allocations[address_repr]['type_name']
                # to interpret bytes if heap stored raw bytes.
                if address_repr >= len(self.heap): # Should not happen if allocations are managed correctly
                    raise InterpreterError(f"Heap address {address_repr} out of bounds for heap size {len(self.heap)}.")
                return self.heap[address_repr] 
            else:
                raise InterpreterError(f"Cannot dereference non-pointer value: {address_repr}")

        elif isinstance(node, MallocNode):
            return self._evaluate_malloc(node, env)
        else:
            raise InterpreterError(f"Unsupported expression node for evaluation: {type(node)}")

    def _execute_global_var_decl(self, node: GlobalVarDeclNode, env: Environment):
        value = self.NULL_POINTER_VALUE # Default for pointers if no initializer
        if isinstance(node.var_type, PointerTypeNode) and node.expression is None:
             pass # value is already NULL_POINTER_VALUE
        elif node.expression:
            value = self.evaluate(node.expression, env)
        env.define(node.identifier.name, value)

    def _execute_local_var_decl(self, node: LocalVarDeclNode, env: Environment):
        value = self.NULL_POINTER_VALUE # Default for pointers if no initializer
        if isinstance(node.var_type, PointerTypeNode) and node.expression is None:
            pass # value is already NULL_POINTER_VALUE
        elif node.expression:
            value = self.evaluate(node.expression, env)
        env.define(node.identifier.name, value)

    def _execute_function_decl(self, node: FunctionDeclNode, env: Environment):
        func_name = node.identifier.name
        return_type_name = self._get_type_name_from_ast_type(node.return_type)
        user_func = UserDefinedFunction(
            name=func_name, params=node.params, body=node.body,
            closure_env=env, return_type_name=return_type_name
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

    def _execute_block(self, node: BlockNode, env: Environment) -> Any:
        # For Phase 2, if strict block scoping is desired for LocalVarDeclNode, 
        # a new Environment should be created: block_env = Environment(parent=env).
        # For now, using current env.
        for stmt_node in node.statements:
            self.execute(stmt_node, env)
        return None 

    def _execute_import(self, node: ImportNode, env: Environment):
        print(f"[Interpreter Info] Encountered import: {node.module_name.name}")

    # --- Phase 2 Specific Execution/Evaluation Methods ---

    def _execute_pointer_assignment(self, node: PointerAssignmentNode, env: Environment):
        # node.target_deref is DereferenceNode. Its .expression is the pointer.
        address_repr = self.evaluate(node.target_deref.expression, env)
        value_to_store = self.evaluate(node.value, env)

        if address_repr is self.NULL_POINTER_VALUE:
            raise InterpreterError("Cannot assign to null pointer.")

        if isinstance(address_repr, tuple) and len(address_repr) == 2 and isinstance(address_repr[0], Environment):
            # SimulatedStackAddress: (target_env, var_name)
            target_env, var_name = address_repr
            target_env.assign(var_name, value_to_store)
        elif isinstance(address_repr, int): # SimulatedHeapAddress
            if address_repr not in self.allocations or self.allocations[address_repr]['freed']:
                raise InterpreterError(f"Cannot assign to invalid or freed heap address: {address_repr}")
            # Assuming single element store for simplicity.
            # Type checking value_to_store against allocation type would be good here.
            self.heap[address_repr] = value_to_store
        else:
            raise InterpreterError(f"Cannot assign to non-pointer address: {address_repr}")
            
    def _evaluate_malloc(self, node: MallocNode, env: Environment) -> SimulatedHeapAddress:
        num_elements_val = self.evaluate(node.size_expr, env)
        if not isinstance(num_elements_val, int) or num_elements_val <= 0:
            raise InterpreterError(f"malloc size must be a positive integer, got {num_elements_val}")

        alloc_type_name = self._get_type_name_from_ast_type(node.alloc_type)
        
        # Simple list-based heap: address is the starting index
        heap_address = len(self.heap) 
        
        self.allocations[heap_address] = {
            'size_elements': num_elements_val, 
            'type_name': alloc_type_name, 
            'freed': False
        }
        # Extend heap with placeholder values for the new allocation
        self.heap.extend([self.NULL_POINTER_VALUE] * num_elements_val) # Initialize with nulls or default
        
        return heap_address

    def _execute_free(self, node: FreeNode, env: Environment):
        address_repr = self.evaluate(node.pointer_expr, env)
        
        if address_repr is self.NULL_POINTER_VALUE:
            # Freeing a null pointer is often a no-op in C.
            # print("[Interpreter Info] free(null) called, no operation.")
            return

        if not isinstance(address_repr, int) or address_repr not in self.allocations:
            raise InterpreterError(f"Invalid address for free: {address_repr}. Not a heap address or not allocated.")
        
        if self.allocations[address_repr]['freed']:
            raise InterpreterError(f"Double free attempt at heap address: {address_repr}.")
            
        self.allocations[address_repr]['freed'] = True
        # For more robust simulation, could also set self.heap elements to a sentinel "freed" value.
        # For simplicity, just marking metadata. Malloc does not currently reuse freed blocks.
        # print(f"[Interpreter Info] Freed memory at heap address: {address_repr}")


    def _evaluate_function_call(self, node: FunctionCallNode, env: Environment) -> Any:
        if not isinstance(node.callee, IdentifierNode):
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
            func_env = Environment(parent=callee.closure_env)
            for i, param_node in enumerate(callee.params):
                func_env.define(param_node.identifier.name, evaluated_args[i])
            try:
                self.execute(callee.body, func_env)
                return None 
            except ReturnSignal as rs:
                return rs.value
        elif callable(callee):
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
        with open(filename, "r") as f: code = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found."); sys.exit(1)

    from tokenizer import tokenize
    from parser import Parser 

    print(f"Interpreting file: {filename}\n")
    try:
        tokens = tokenize(code)
        tokens_for_parser = [t for t in tokens if t.type not in ('SKIP')]
        parser = Parser(tokens_for_parser)
        ast_program = parser.parse_program()
        
        interpreter = Interpreter()
        start_time = time.time()
        interpreter.interpret(ast_program)
        end_time = time.time()
        print(f"\nInterpreter Execution Time: {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Error during interpretation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
