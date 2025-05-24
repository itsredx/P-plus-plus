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
    def __init__(self, width: int = 32):
        self.width = width

class IRTypeFloat(IRType):
    def __init__(self, width: int = 64):
        self.width = width

class IRTypeBool(IRType):
    pass

class IRTypeString(IRType): # Represents char* or similar string type
    pass

class IRTypeVoid(IRType):
    pass

class IRTypeCustom(IRType): 
    def __init__(self, name: str):
        self.name = name

class IRPointerType(IRType): # New Pointer Type
    def __init__(self, pointee_type: IRType):
        self.pointee_type = pointee_type


# IR Expressions
class IRExpression(IRNode):
    """Base class for IR expressions."""
    pass

class IRLiteral(IRExpression):
    def __init__(self, value: Any, ir_type: IRType):
        self.value = value
        self.ir_type = ir_type

class IRIdentifier(IRExpression):
    def __init__(self, name: str):
        self.name = name
        # Resolved type of this identifier (an IRType instance) might be added here by a type checking pass
        # or looked up from a symbol table during later IR stages.

class IRBinaryOp(IRExpression):
    def __init__(self, left: IRExpression, op: str, right: IRExpression):
        self.left = left
        self.op = op
        self.right = right

class IRFunctionCall(IRExpression):
    def __init__(self, callee_name: str, args: List[IRExpression]):
        self.callee_name = callee_name
        self.args = args

# New IRExpression Nodes for Phase 2
class IRAddressOf(IRExpression):
    """Represents &variable_name. The type of this expression is IRPointerType(var_ir_type)."""
    def __init__(self, variable_name: str, var_ir_type: IRType):
        self.variable_name = variable_name
        self.var_ir_type = var_ir_type # Type of the variable itself

class IRDereference(IRExpression):
    """Represents *pointer_expression. The type is the pointee_type of the pointer_expression."""
    def __init__(self, pointer_expr: IRExpression):
        self.pointer_expr = pointer_expr

class IRMalloc(IRExpression):
    """Represents malloc(alloc_type, size_expr). Type is IRPointerType(alloc_type)."""
    def __init__(self, alloc_ir_type: IRType, size_expr: IRExpression):
        self.alloc_ir_type = alloc_ir_type # The type of element(s) to allocate
        self.size_expr = size_expr         # Expression for number of elements


# IR Statements
class IRStatement(IRNode):
    """Base class for IR statements."""
    pass

class IRAssign(IRStatement): # Assignment to a simple variable: var = value
    def __init__(self, target: IRIdentifier, value: IRExpression): 
        self.target = target
        self.value = value

class IRReturn(IRStatement):
    def __init__(self, value: Optional[IRExpression]):
        self.value = value

class IRExpressionStatement(IRStatement):
    def __init__(self, expression: IRExpression):
        self.expression = expression

class IRLocalVariable(IRStatement): 
    def __init__(self, name: str, ir_type: IRType, initializer: Optional[IRExpression] = None):
        self.name = name
        self.ir_type = ir_type
        self.initializer = initializer

# New IRStatement Nodes for Phase 2
class IRPointerAssign(IRStatement): # Assignment to a dereferenced pointer: *ptr_expr = value
    def __init__(self, target_pointer_expr: IRExpression, value_to_assign_expr: IRExpression):
        self.target_pointer_expr = target_pointer_expr # e.g. IRIdentifier('p') if *p = ...
        self.value_to_assign_expr = value_to_assign_expr

class IRFree(IRStatement):
    def __init__(self, pointer_expr: IRExpression):
        self.pointer_expr = pointer_expr


# Top-Level IR Declarations
class IRGlobalVariable(IRNode): 
    def __init__(self, name: str, ir_type: IRType, initializer: Optional[IRExpression] = None):
        self.name = name
        self.ir_type = ir_type
        self.initializer = initializer

class IRParam(IRNode):
    def __init__(self, name: str, ir_type: IRType):
        self.name = name
        self.ir_type = ir_type

class IRFunction(IRNode): 
    def __init__(self, name: str, params: List[IRParam], return_type: IRType, body: List[IRStatement]):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body

class IRImport(IRNode): 
    def __init__(self, module_name: str):
        self.module_name = module_name

class IRProgram(IRNode): 
    def __init__(self, declarations: List[Union[IRGlobalVariable, IRFunction, IRImport]]):
        self.declarations = declarations


if __name__ == '__main__':
    # Example Usage of new nodes
    int_type = IRTypeInt(32)
    ptr_to_int = IRPointerType(pointee_type=int_type)
    
    # &my_var (where my_var is int)
    addr_of = IRAddressOf(variable_name="my_var", var_ir_type=int_type)
    print(addr_of) # IRAddressOf(variable_name='my_var', var_ir_type=IRTypeInt(width=32))
                  # This expression's type is IRPointerType(IRTypeInt(32))

    # *p (where p is *int)
    deref = IRDereference(pointer_expr=IRIdentifier(name="p"))
    print(deref) # IRDereference(pointer_expr=IRIdentifier(name='p'))
                 # This expression's type is IRTypeInt(32) if p is *int

    # malloc(int, 10)
    malloc_node = IRMalloc(alloc_ir_type=int_type, size_expr=IRLiteral(10, IRTypeInt(32)))
    print(malloc_node) # IRMalloc(alloc_ir_type=IRTypeInt(width=32), size_expr=IRLiteral(value=10, ir_type=IRTypeInt(width=32)))
                       # This expression's type is IRPointerType(IRTypeInt(32))

    # *p = 5
    ptr_assign = IRPointerAssign(
        target_pointer_expr=IRIdentifier(name="p"), # p itself is the pointer
        value_to_assign_expr=IRLiteral(5, int_type)
    )
    print(ptr_assign) # IRPointerAssign(target_pointer_expr=IRIdentifier(name='p'), value_to_assign_expr=IRLiteral(value=5, ir_type=IRTypeInt(width=32)))

    # free(p)
    free_node = IRFree(pointer_expr=IRIdentifier(name="p"))
    print(free_node) # IRFree(pointer_expr=IRIdentifier(name='p'))

    # Example of a global pointer variable
    g_ptr_var = IRGlobalVariable(name="g_ptr", ir_type=ptr_to_int, initializer=None)
    print(g_ptr_var)

    # Example function using pointer types
    func_with_ptr_param = IRFunction(
        name="process_ptr",
        params=[IRParam(name="ptr_param", ir_type=ptr_to_int)],
        return_type=IRTypeVoid(),
        body=[
            IRPointerAssign(target_pointer_expr=IRIdentifier(name="ptr_param"), value_to_assign_expr=IRLiteral(100, int_type)),
            IRFree(pointer_expr=IRIdentifier(name="ptr_param"))
        ]
    )
    print(func_with_ptr_param)
