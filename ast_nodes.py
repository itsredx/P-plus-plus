# ast_nodes.py

class ASTNode:
    pass

class ProgramNode(ASTNode):
    def __init__(self, declarations):
        self.declarations = declarations

    def __repr__(self):
        return f"Program({self.declarations})"

class VariableDeclarationNode(ASTNode):
    def __init__(self, identifier, type_annotation, expression):
        self.identifier = identifier
        self.type_annotation = type_annotation
        self.expression = expression

    def __repr__(self):
        return f"VarDecl({self.identifier}, {self.type_annotation}, {self.expression})"

class FunctionDeclarationNode(ASTNode):
    def __init__(self, identifier, parameters, return_type, body):
        self.identifier = identifier
        self.parameters = parameters
        self.return_type = return_type
        self.body = body

    def __repr__(self):
        return f"FuncDecl({self.identifier}, params={self.parameters}, return={self.return_type}, body={self.body})"

class ExpressionStatementNode(ASTNode):
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"ExprStmt({self.expression})"

class ExpressionNode(ASTNode):
    pass

class LiteralNode(ExpressionNode):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Literal({self.value})"

class IdentifierNode(ExpressionNode):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Identifier({self.name})"

class BinaryOpNode(ExpressionNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"BinaryOp({self.left} {self.operator} {self.right})"

class FunctionCallNode(ExpressionNode):
    def __init__(self, callee, arguments):
        self.callee = callee
        self.arguments = arguments

    def __repr__(self):
        return f"FunctionCall({self.callee}, {self.arguments})"
