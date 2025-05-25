# ast_nodes.py
from typing import List, Optional, Union, Any # Added Any for LiteralNode if it's not already there

class ASTNode:
    """Base class for all AST nodes."""
    def __repr__(self): # Basic repr for easier debugging
        attributes = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in attributes.items())})"

class ExpressionNode(ASTNode):
    """Base class for nodes that represent expressions."""
    pass

class StatementNode(ASTNode):
    """Base class for nodes that represent statements."""
    pass

# --- Type Nodes ---
class TypeNode(ASTNode): # Now a base class for types
    """Represents a type annotation."""
    pass

class BuiltinTypeNode(TypeNode):
    """Represents a built-in type annotation like 'int', 'float', 'void', etc."""
    def __init__(self, name: str):
        self.name = name

class PointerTypeNode(TypeNode):
    """Represents a pointer type, e.g., *int."""
    def __init__(self, pointee_type: TypeNode):
        self.pointee_type = pointee_type # The type the pointer points to


# --- Expression Nodes (Existing and New) ---

class LiteralNode(ExpressionNode):
    def __init__(self, value: Union[int, float, str, bool, None]): # Added None for null literal
        self.value = value

class IdentifierNode(ExpressionNode):
    def __init__(self, name: str):
        self.name = name

class BinaryOpNode(ExpressionNode):
    def __init__(self, left: ExpressionNode, operator: str, right: ExpressionNode):
        self.left = left
        self.operator = operator
        self.right = right

class FunctionCallNode(ExpressionNode):
    def __init__(self, callee: ExpressionNode, arguments: List[ExpressionNode]):
        self.callee = callee
        self.arguments = arguments

class AddressOfNode(ExpressionNode):
    """Represents the address-of operation (&identifier)."""
    def __init__(self, identifier: IdentifierNode):
        self.identifier = identifier

class DereferenceNode(ExpressionNode):
    """Represents the dereference operation (*expression)."""
    def __init__(self, expression: ExpressionNode):
        self.expression = expression # Expression that evaluates to a pointer

class MallocNode(ExpressionNode):
    """Represents memory allocation (malloc(type, size_expr))."""
    def __init__(self, alloc_type: TypeNode, size_expr: ExpressionNode):
        self.alloc_type = alloc_type # Type of data to allocate
        self.size_expr = size_expr   # Expression for number of elements

# --- Statement Nodes (Existing and New) ---

class ImportNode(ASTNode): # Or StatementNode, keeping as ASTNode for now as per original
    def __init__(self, module_name: IdentifierNode):
        self.module_name = module_name

class GlobalVarDeclNode(StatementNode):
    def __init__(self, identifier: IdentifierNode, var_type: TypeNode, expression: Optional[ExpressionNode] = None):
        self.identifier = identifier
        self.var_type = var_type # This will now be TypeNode (e.g. BuiltinTypeNode or PointerTypeNode)
        self.expression = expression

class ParamNode(ASTNode): # Not a statement or expression, but part of FunctionDecl
    def __init__(self, identifier: IdentifierNode, param_type: TypeNode):
        self.identifier = identifier
        self.param_type = param_type # This will now be TypeNode

class BlockNode(StatementNode):
    def __init__(self, statements: List[StatementNode]):
        self.statements = statements

class FunctionDeclNode(StatementNode):
    def __init__(self, identifier: IdentifierNode, params: List[ParamNode], return_type: Optional[TypeNode], body: BlockNode):
        self.identifier = identifier
        self.params = params
        self.return_type = return_type # This will now be TypeNode or None
        self.body = body

class ReturnStmtNode(StatementNode):
    def __init__(self, expression: Optional[ExpressionNode] = None):
        self.expression = expression

class VarAssignNode(StatementNode): # Standard variable assignment: x = value
    def __init__(self, identifier: IdentifierNode, expression: ExpressionNode):
        self.identifier = identifier
        self.expression = expression

class PointerAssignmentNode(StatementNode): # Pointer assignment: *ptr = value
    def __init__(self, target_deref: DereferenceNode, value: ExpressionNode):
        self.target_deref = target_deref # The *ptr part
        self.value = value             # The value to assign

class ExpressionStatementNode(StatementNode):
    def __init__(self, expression: ExpressionNode):
        self.expression = expression

class LocalVarDeclNode(StatementNode):
    def __init__(self, identifier: IdentifierNode, var_type: TypeNode, expression: Optional[ExpressionNode] = None):
        self.identifier = identifier
        self.var_type = var_type # This will now be TypeNode
        self.expression = expression

class FreeNode(StatementNode):
    """Represents freeing allocated memory (free(pointer_expr))."""
    def __init__(self, pointer_expr: ExpressionNode):
        self.pointer_expr = pointer_expr # Expression that evaluates to a pointer

# --- Program Node (Root) ---
class ProgramNode(ASTNode):
    # Assuming top-level declarations can include imports, global vars, and functions
    def __init__(self, declarations: List[Union[GlobalVarDeclNode, FunctionDeclNode, ImportNode]]):
        self.declarations = declarations


if __name__ == '__main__':
    # Example Usage of new and refactored nodes

    # TypeNodes
    int_type = BuiltinTypeNode(name='int')
    ptr_to_int_type = PointerTypeNode(pointee_type=int_type)
    ptr_to_ptr_to_int_type = PointerTypeNode(pointee_type=ptr_to_int_type)
    
    print(f"Int Type: {int_type}")
    print(f"Pointer to Int Type: {ptr_to_int_type}")
    print(f"Pointer to Pointer to Int Type: {ptr_to_ptr_to_int_type}")

    # Variable declaration with pointer type
    # my_ptr AS *int
    ptr_var_decl = LocalVarDeclNode(
        identifier=IdentifierNode(name='my_ptr'),
        var_type=ptr_to_int_type,
        expression=None # Or an initializer like MallocNode or another pointer
    )
    print(f"\nPointer Variable Declaration: {ptr_var_decl}")

    # AddressOfNode: &my_var (assuming my_var is an int variable)
    addr_of_expr = AddressOfNode(identifier=IdentifierNode(name='my_var'))
    print(f"\nAddressOf Expression: {addr_of_expr}")

    # DereferenceNode: *my_ptr
    deref_expr = DereferenceNode(expression=IdentifierNode(name='my_ptr'))
    print(f"\nDereference Expression: {deref_expr}")

    # PointerAssignmentNode: *my_ptr = 10
    ptr_assign_stmt = PointerAssignmentNode(
        target_deref=DereferenceNode(expression=IdentifierNode(name='my_ptr')),
        value=LiteralNode(value=10)
    )
    print(f"\nPointer Assignment Statement: {ptr_assign_stmt}")
    
    # MallocNode: malloc(int, 1)
    malloc_expr = MallocNode(
        alloc_type=int_type,
        size_expr=LiteralNode(value=1) # Allocate space for 1 int
    )
    print(f"\nMalloc Expression: {malloc_expr}")
    
    # Example of using MallocNode in a variable declaration
    # p AS *int = malloc(int, 1)
    ptr_var_decl_with_malloc = LocalVarDeclNode(
        identifier=IdentifierNode(name='p'),
        var_type=ptr_to_int_type,
        expression=malloc_expr
    )
    print(f"Pointer Declaration with Malloc: {ptr_var_decl_with_malloc}")

    # FreeNode: free(p)
    free_stmt = FreeNode(pointer_expr=IdentifierNode(name='p'))
    print(f"\nFree Statement: {free_stmt}")

    # Example function declaration using new TypeNodes
    # DEF my_func(param_ptr AS *int) -> *(*int):
    #    ...
    func_with_pointers = FunctionDeclNode(
        identifier=IdentifierNode(name="my_func"),
        params=[ParamNode(identifier=IdentifierNode(name="param_ptr"), param_type=ptr_to_int_type)],
        return_type=ptr_to_ptr_to_int_type,
        body=BlockNode(statements=[
            ReturnStmtNode(expression=IdentifierNode(name="some_ptr_to_ptr")) # Dummy body
        ])
    )
    print(f"\nFunction with Pointer Types: {func_with_pointers}")

    # Ensure LiteralNode can handle None (for potential null pointers)
    null_literal = LiteralNode(value=None)
    print(f"\nNull Literal: {null_literal}")
    
    # Ensure existing VarAssignNode still works
    simple_assign = VarAssignNode(identifier=IdentifierNode("x"), expression=LiteralNode(10))
    print(f"Simple Assignment: {simple_assign}")

    # Program with a global pointer
    global_ptr_decl = GlobalVarDeclNode(
        identifier=IdentifierNode(name="g_ptr"),
        var_type=PointerTypeNode(pointee_type=BuiltinTypeNode("char")),
        expression=None
    )
    program_with_global_ptr = ProgramNode(declarations=[global_ptr_decl, func_with_pointers])
    print(f"\nProgram with Global Pointer: {program_with_global_ptr}")
