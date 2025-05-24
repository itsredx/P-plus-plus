# ir_nodes.py
from typing import List, Optional, Any, Union

class IRNode:
    """Base class for all IR nodes."""
    def __repr__(self):
        attributes = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in attributes.items())})"

# IR Type System
class IRType(IRNode):
    """Base class for IR types."""
    pass

class IRTypeInt(IRType):
    def __init__(self, width: int = 32): # Default to 32-bit if not specified
        self.width = width

class IRTypeFloat(IRType):
    def __init__(self, width: int = 64): # Default to 64-bit (double precision)
        self.width = width

class IRTypeBool(IRType):
    pass

class IRTypeString(IRType):
    pass

class IRTypeVoid(IRType):
    pass

class IRTypeCustom(IRType): # For user-defined types or unresolved types
    def __init__(self, name: str):
        self.name = name

# IR Expressions
class IRExpression(IRNode):
    """Base class for IR expressions."""
    pass

class IRLiteral(IRExpression):
    def __init__(self, value: Any, ir_type: IRType):
        self.value = value
        self.ir_type = ir_type

class IRIdentifier(IRExpression):
    """Represents an identifier used as a value (e.g., reading a variable)."""
    def __init__(self, name: str):
        self.name = name
        # Type of identifier would be resolved during a later IR pass (type checking/inference)

class IRBinaryOp(IRExpression):
    def __init__(self, left: IRExpression, op: str, right: IRExpression):
        self.left = left
        self.op = op
        self.right = right

class IRFunctionCall(IRExpression):
    def __init__(self, callee_name: str, args: List[IRExpression]):
        self.callee_name = callee_name
        self.args = args

# IR Statements
class IRStatement(IRNode):
    """Base class for IR statements."""
    pass

class IRAssign(IRStatement):
    def __init__(self, target: IRIdentifier, value: IRExpression): # Target is an identifier for now
        self.target = target
        self.value = value

class IRReturn(IRStatement):
    def __init__(self, value: Optional[IRExpression]):
        self.value = value

class IRExpressionStatement(IRStatement):
    def __init__(self, expression: IRExpression):
        self.expression = expression

class IRLocalVariable(IRStatement): # Local variable declaration is a statement
    def __init__(self, name: str, ir_type: IRType, initializer: Optional[IRExpression] = None):
        self.name = name
        self.ir_type = ir_type
        self.initializer = initializer


# Top-Level IR Declarations
class IRGlobalVariable(IRNode): # Not a statement or expression, but a top-level decl
    def __init__(self, name: str, ir_type: IRType, initializer: Optional[IRExpression] = None):
        self.name = name
        self.ir_type = ir_type
        self.initializer = initializer

class IRParam(IRNode):
    def __init__(self, name: str, ir_type: IRType):
        self.name = name
        self.ir_type = ir_type

class IRFunction(IRNode): # Not a statement or expression, but a top-level decl
    def __init__(self, name: str, params: List[IRParam], return_type: IRType, body: List[IRStatement]):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body

class IRImport(IRNode): # Top-level declaration
    def __init__(self, module_name: str):
        self.module_name = module_name

class IRProgram(IRNode): # Represents the whole program as a list of top-level declarations
    def __init__(self, declarations: List[Union[IRGlobalVariable, IRFunction, IRImport]]):
        self.declarations = declarations

# The old IRInstruction, IRBasicBlock might be used in a later lowering pass
# For now, this more structured IR is the target.
# class IRBasicBlock(IRNode):
#     def __init__(self, label):
#         self.label = label
#         self.instructions = []

#     def add_instruction(self, instr):
#         self.instructions.append(instr)

#     def __repr__(self):
#         return f"IRBasicBlock({self.label}, instrs={self.instructions})"

# class IRInstruction(IRNode):
#     def __init__(self, op, operands=None):
#         self.op = op
#         self.operands = operands if operands is not None else []

#     def __repr__(self):
#         return f"IRInstruction({self.op}, {self.operands})"

if __name__ == '__main__':
    # Example Usage
    int_type = IRTypeInt(32)
    float_type = IRTypeFloat(64)
    
    # Global variable: my_global as int = 10
    global_var_init = IRLiteral(10, int_type)
    global_var = IRGlobalVariable(name="my_global", ir_type=int_type, initializer=global_var_init)
    print(global_var)

    # Function parameter: x as int
    param_x = IRParam(name="x", ir_type=int_type)
    print(param_x)

    # Function body statement: y as int = x
    local_var_decl = IRLocalVariable(name="y", ir_type=int_type, initializer=IRIdentifier(name="x"))
    print(local_var_decl)
    
    # Function body statement: return y
    return_stmt = IRReturn(value=IRIdentifier(name="y"))
    print(return_stmt)

    # Function: def my_func(x as int) -> int { y as int = x; return y; }
    func_decl = IRFunction(
        name="my_func",
        params=[param_x],
        return_type=int_type,
        body=[local_var_decl, return_stmt]
    )
    print(func_decl)

    # Import: import my_math
    import_decl = IRImport(module_name="my_math")
    print(import_decl)

    # Program
    program = IRProgram(declarations=[global_var, func_decl, import_decl])
    print(program)

    # Expression: 5 + z
    expr = IRBinaryOp(left=IRLiteral(5, int_type), op='+', right=IRIdentifier(name='z'))
    print(expr)

    # Expression Statement: print("hello")
    expr_stmt = IRExpressionStatement(
        expression=IRFunctionCall(callee_name="print", args=[IRLiteral("hello", IRTypeString())])
    )
    print(expr_stmt)

    # Assignment: temp = 5 + z
    assign_stmt = IRAssign(target=IRIdentifier(name="temp"), value=expr)
    print(assign_stmt)

    # Void type
    void_type = IRTypeVoid()
    print(void_type)

    # Bool type
    bool_type = IRTypeBool()
    true_literal = IRLiteral(True, bool_type)
    print(true_literal)
