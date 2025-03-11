# parser.py
from ast_nodes import *
from tokenizer import tokenize

class ParserError(Exception):
    pass

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected_type):
        token = self.current_token()
        if token is None:
            raise ParserError("Unexpected end of input")
        if token.type != expected_type:
            raise ParserError(f"Expected token type {expected_type} at line {token.line}, column {token.column}, found {token.type} ('{token.value}')")
        self.pos += 1
        return token

    def match(self, expected_type):
        token = self.current_token()
        if token is not None and token.type == expected_type:
            self.pos += 1
            return token
        return None

    def parse_program(self):
        declarations = []
        while self.current_token() is not None:
            declarations.append(self.parse_declaration())
        return ProgramNode(declarations)

    def parse_declaration(self):
        token = self.current_token()
        if token.type == "DEF":
            return self.parse_function_declaration()
        elif token.type == "ID":
            # Lookahead: if next token is ASSIGN, it's a variable declaration.
            next_token = self.tokens[self.pos+1] if self.pos+1 < len(self.tokens) else None
            if next_token is not None and next_token.type == "ASSIGN":
                return self.parse_variable_declaration()
            else:
                # Otherwise, parse as an expression statement.
                expr = self.parse_expression()
                return ExpressionStatementNode(expr)
        else:
            # Fallback: parse as an expression statement.
            expr = self.parse_expression()
            return ExpressionStatementNode(expr)

    def parse_variable_declaration(self):
        id_token = self.consume("ID")
        self.consume("ASSIGN")
        expr = self.parse_expression()
        return VariableDeclarationNode(id_token.value, None, expr)

    def parse_function_declaration(self):
        self.consume("DEF")
        id_token = self.consume("ID")
        self.consume("LPAREN")
        # For simplicity, parameters are omitted.
        self.consume("RPAREN")
        body = self.parse_block()
        return FunctionDeclarationNode(id_token.value, [], None, body)

    def parse_block(self):
        # For simplicity, treat a block as a single expression.
        return self.parse_expression()

    def parse_expression(self):
        left = self.parse_primary()
        token = self.current_token()
        # Handle binary operators.
        while token is not None and token.type in {"OP", "==", "!=", ">", "<", ">=", "<="}:
            op = self.consume(token.type).value
            right = self.parse_primary()
            left = BinaryOpNode(left, op, right)
            token = self.current_token()
        return left

    def parse_primary(self):
        token = self.current_token()
        if token is None:
            raise ParserError("Unexpected end of input in primary expression")
        if token.type == "NUMBER":
            self.consume("NUMBER")
            return LiteralNode(float(token.value))
        elif token.type == "STRING":
            # Remove the surrounding quotes.
            self.consume("STRING")
            # Assuming the string is well-formed and always has quotes.
            value = token.value[1:-1]
            return LiteralNode(value)
        elif token.type == "ID":
            id_token = self.consume("ID")
            if self.current_token() is not None and self.current_token().type == "LPAREN":
                self.consume("LPAREN")
                args = self.parse_argument_list()
                self.consume("RPAREN")
                return FunctionCallNode(IdentifierNode(id_token.value), args)
            else:
                return IdentifierNode(id_token.value)
        elif token.type == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_expression()
            self.consume("RPAREN")
            return expr
        else:
            raise ParserError(f"Unexpected token {token.type} ('{token.value}') at line {token.line}, column {token.column}")

    def parse_argument_list(self):
        args = []
        if self.current_token() is not None and self.current_token().type == "RPAREN":
            return args
        args.append(self.parse_expression())
        # Extend this later to support comma-separated arguments.
        return args

# For testing the parser with a sample input.
if __name__ == "__main__":
    code = 'print("hello world")'
    try:
        tokens = tokenize(code)
        parser = Parser(tokens)
        ast = parser.parse_program()
        print("AST:", ast)
    except Exception as e:
        print("Parser error:", e)
