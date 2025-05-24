# ast_nodes.py
from typing import List, Optional, Union

class ASTNode:
    """Base class for all AST nodes."""
    pass

class ExpressionNode(ASTNode):
    """Base class for nodes that represent expressions."""
    pass

class StatementNode(ASTNode):
    """Base class for nodes that represent statements."""
    pass

# Expression Nodes (some existing, some might be new or placeholders)

class LiteralNode(ExpressionNode):
    def __init__(self, value: Union[int, float, str, bool]): # Assuming basic literal types
        self.value = value

    def __repr__(self):
        return f"LiteralNode(value={self.value!r})"

class IdentifierNode(ExpressionNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"IdentifierNode(name='{self.name}')"

class BinaryOpNode(ExpressionNode):
    def __init__(self, left: ExpressionNode, operator: str, right: ExpressionNode):
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, operator='{self.operator}', right={self.right})"

class FunctionCallNode(ExpressionNode):
    def __init__(self, callee: ExpressionNode, arguments: List[ExpressionNode]):
        self.callee = callee
        self.arguments = arguments

    def __repr__(self):
        return f"FunctionCallNode(callee={self.callee}, arguments={self.arguments})"

# New and Updated AST Nodes as per the subtask

class TypeNode(ASTNode):
    """Represents a type annotation."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"TypeNode(name='{self.name}')"

class ImportNode(ASTNode): # Or StatementNode if imports are considered statements
    """Represents an import statement."""
    def __init__(self, module_name: IdentifierNode):
        self.module_name = module_name

    def __repr__(self):
        return f"ImportNode(module_name={self.module_name})"

class GlobalVarDeclNode(StatementNode):
    """Represents a global variable declaration."""
    def __init__(self, identifier: IdentifierNode, var_type: TypeNode, expression: Optional[ExpressionNode] = None):
        self.identifier = identifier
        self.var_type = var_type
        self.expression = expression

    def __repr__(self):
        return f"GlobalVarDeclNode(identifier={self.identifier}, var_type={self.var_type}, expression={self.expression})"

class ParamNode(ASTNode):
    """Represents a function parameter."""
    def __init__(self, identifier: IdentifierNode, param_type: TypeNode):
        self.identifier = identifier
        self.param_type = param_type

    def __repr__(self):
        return f"ParamNode(identifier={self.identifier}, param_type={self.param_type})"

class BlockNode(StatementNode): # BlockNode can also be considered an ASTNode if it can appear in expressions
    """Represents a block of statements."""
    def __init__(self, statements: List[StatementNode]):
        self.statements = statements

    def __repr__(self):
        return f"BlockNode(statements={self.statements})"

class FunctionDeclNode(StatementNode): # Existing, but updated
    """Represents a function declaration."""
    def __init__(self, identifier: IdentifierNode, params: List[ParamNode], return_type: Optional[TypeNode], body: BlockNode):
        self.identifier = identifier
        self.params = params
        self.return_type = return_type  # Could be TypeNode(name='void') or None
        self.body = body

    def __repr__(self):
        return (f"FunctionDeclNode(identifier={self.identifier}, params={self.params}, "
                f"return_type={self.return_type}, body={self.body})")

class ReturnStmtNode(StatementNode):
    """Represents a return statement."""
    def __init__(self, expression: Optional[ExpressionNode] = None):
        self.expression = expression

    def __repr__(self):
        return f"ReturnStmtNode(expression={self.expression})"

class VarAssignNode(StatementNode):
    """Represents a variable assignment."""
    def __init__(self, identifier: IdentifierNode, expression: ExpressionNode):
        self.identifier = identifier
        self.expression = expression

    def __repr__(self):
        return f"VarAssignNode(identifier={self.identifier}, expression={self.expression})"

class ExpressionStatementNode(StatementNode): # Existing, ensured inheritance
    """Represents an expression used as a statement."""
    def __init__(self, expression: ExpressionNode):
        self.expression = expression

    def __repr__(self):
        return f"ExpressionStatementNode(expression={self.expression})"

# Program Node (Root of the AST) - (Existing, but ensure it uses new nodes if applicable)
class ProgramNode(ASTNode):
    def __init__(self, declarations: List[Union[GlobalVarDeclNode, FunctionDeclNode, ImportNode]]): # Example top-level declarations
        self.declarations = declarations

    def __repr__(self):
        return f"ProgramNode(declarations={self.declarations})"


# Example of a local variable declaration, if needed distinct from GlobalVarDeclNode
class LocalVarDeclNode(StatementNode):
    """Represents a local variable declaration within a function body."""
    def __init__(self, identifier: IdentifierNode, var_type: TypeNode, expression: Optional[ExpressionNode] = None):
        self.identifier = identifier
        self.var_type = var_type
        self.expression = expression

    def __repr__(self):
        return f"LocalVarDeclNode(identifier={self.identifier}, var_type={self.var_type}, expression={self.expression})"

# Note: The original VariableDeclarationNode was generic. 
# It's replaced by GlobalVarDeclNode and potentially LocalVarDeclNode for clarity.
# If VariableDeclarationNode is still used elsewhere, it might need to be updated or deprecated.
# For now, assuming it's superseded by the more specific declaration nodes.

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    
    # import my_module
    import_stmt = ImportNode(module_name=IdentifierNode(name='my_module'))
    print(import_stmt)

    # my_var as int = 10
    global_var = GlobalVarDeclNode(
        identifier=IdentifierNode(name='my_var'),
        var_type=TypeNode(name='int'),
        expression=LiteralNode(value=10)
    )
    print(global_var)

    # def my_func(x as int, y as float32) -> float32 { return x + y; }
    func_decl = FunctionDeclNode(
        identifier=IdentifierNode(name='my_func'),
        params=[
            ParamNode(identifier=IdentifierNode(name='x'), param_type=TypeNode(name='int')),
            ParamNode(identifier=IdentifierNode(name='y'), param_type=TypeNode(name='float32'))
        ],
        return_type=TypeNode(name='float32'),
        body=BlockNode(statements=[
            ReturnStmtNode(expression=BinaryOpNode(
                left=IdentifierNode(name='x'),
                operator='+',
                right=IdentifierNode(name='y')
            ))
        ])
    )
    print(func_decl)

    # x = 5
    assign_stmt = VarAssignNode(identifier=IdentifierNode(name='x'), expression=LiteralNode(value=5))
    print(assign_stmt)

    # foo(); (expression as statement)
    expr_stmt = ExpressionStatementNode(expression=FunctionCallNode(callee=IdentifierNode(name='foo'), arguments=[]))
    print(expr_stmt)

    # return;
    return_void = ReturnStmtNode()
    print(return_void)

    # Program example
    program = ProgramNode(declarations=[import_stmt, global_var, func_decl])
    print(program)

    # Local variable declaration example
    local_var = LocalVarDeclNode(
        identifier=IdentifierNode(name='local_val'),
        var_type=TypeNode(name='string'),
        expression=LiteralNode(value="hello")
    )
    # Example of using it in a block
    block_with_local_var = BlockNode(statements=[
        local_var,
        ReturnStmtNode(expression=IdentifierNode(name='local_val'))
    ])
    print(block_with_local_var)

    # Test TypeNode
    type_node_example = TypeNode(name='custom_MyType')
    print(type_node_example)

    # Test ParamNode
    param_node_example = ParamNode(identifier=IdentifierNode(name='arg1'), param_type=TypeNode(name='bool'))
    print(param_node_example)
    
    # Test LiteralNode with various types
    print(LiteralNode(value=100))
    print(LiteralNode(value=3.14))
    print(LiteralNode(value="test string"))
    print(LiteralNode(value=True))

    # Test ProgramNode with only one function
    program_single_func = ProgramNode(declarations=[func_decl])
    print(program_single_func)
    
    # Test FunctionDeclNode with no params and void return
    func_no_param_void_return = FunctionDeclNode(
        identifier=IdentifierNode(name='do_something'),
        params=[],
        return_type=TypeNode(name='void'), # Explicit void
        body=BlockNode(statements=[
            ExpressionStatementNode(expression=FunctionCallNode(callee=IdentifierNode(name='print_status'), arguments=[LiteralNode(value="done")]))
        ])
    )
    print(func_no_param_void_return)

    func_no_param_implicit_void_return = FunctionDeclNode(
        identifier=IdentifierNode(name='do_another_thing'),
        params=[],
        return_type=None, # Implicit void
        body=BlockNode(statements=[])
    )
    print(func_no_param_implicit_void_return)

    # Test BlockNode with multiple statements
    multi_stmt_block = BlockNode(statements=[
        LocalVarDeclNode(IdentifierNode("a"), TypeNode("int"), LiteralNode(1)),
        VarAssignNode(IdentifierNode("a"), BinaryOpNode(IdentifierNode("a"), "+", LiteralNode(2))),
        ReturnStmtNode(IdentifierNode("a"))
    ])
    print(multi_stmt_block)
