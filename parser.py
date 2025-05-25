# parser.py
from ast_nodes import *
from tokenizer import tokenize, Token 
from typing import List, Optional

class ParserError(Exception):
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek_token(self, lookahead: int = 1) -> Optional[Token]:
        peek_pos = self.pos + lookahead -1
        return self.tokens[peek_pos] if peek_pos < len(self.tokens) else None

    def consume(self, expected_type: str) -> Token:
        token = self.current_token()
        if token is None:
            raise ParserError(f"Unexpected end of input. Expected {expected_type}.")
        if token.type != expected_type:
            raise ParserError(
                f"Expected token type {expected_type} at line {token.line}, column {token.column}, "
                f"found {token.type} ('{token.value}')"
            )
        self.pos += 1
        return token

    def match(self, expected_type: str) -> Optional[Token]:
        token = self.current_token()
        if token is not None and token.type == expected_type:
            self.pos += 1
            return token
        return None

    def parse_program(self) -> ProgramNode:
        declarations = []
        while self.current_token() is not None and self.current_token().type != 'DEDENT':
            token_type = self.current_token().type
            if token_type == 'IMPORT':
                declarations.append(self.parse_import())
            elif token_type == 'DEF':
                declarations.append(self.parse_function_declaration())
            elif token_type == 'ID': 
                if self.peek_token(lookahead=2) and self.peek_token(lookahead=2).type == 'AS':
                     declarations.append(self.parse_global_var_declaration())
                else:
                    token = self.current_token()
                    raise ParserError(f"Unexpected token '{token.value}' ({token.type}) at top level at line {token.line}, col {token.column}. Expected IMPORT, DEF, or global variable declaration (ID AS Type).")
            else:
                token = self.current_token()
                raise ParserError(f"Unexpected token '{token.value}' ({token.type}) at top level at line {token.line}, col {token.column}. Expected IMPORT, DEF, or global variable declaration.")
        return ProgramNode(declarations)

    def parse_import(self) -> ImportNode:
        self.consume('IMPORT')
        id_token = self.consume('ID')
        return ImportNode(module_name=IdentifierNode(name=id_token.value))

    def parse_type(self) -> TypeNode:
        if self.match('STAR'):
            pointee_type = self.parse_type() # Recursive call for types like **int
            return PointerTypeNode(pointee_type=pointee_type)
        else:
            token = self.current_token()
            # Assuming basic types are tokenized as their uppercase keyword (e.g., INT, VOID)
            # or as a general ID for custom types.
            TYPE_KEYWORDS = {'INT', 'INT8', 'INT16', 'INT32', 'INT64',
                             'UINT8', 'UINT16', 'UINT32', 'UINT64',
                             'FLOAT', 'FLOAT32', 'FLOAT64',
                             'BOOL', 'STRING', 'VOID', 'ID'} # ID for potential custom types
            if token and token.type in TYPE_KEYWORDS:
                self.pos += 1 # Consume the type token
                return BuiltinTypeNode(name=token.value)
            else:
                err_token = token if token else Token("EOF", "EOF", 0, 0) # Handle case where token is None
                raise ParserError(f"Expected type identifier or STAR at line {err_token.line}, col {err_token.column}, found {err_token.type} ('{err_token.value}')")


    def parse_global_var_declaration(self) -> GlobalVarDeclNode:
        identifier_token = self.consume('ID')
        self.consume('AS')
        var_type = self.parse_type()
        expression = None
        if self.match('ASSIGN'):
            expression = self.parse_expression()
        return GlobalVarDeclNode(identifier=IdentifierNode(name=identifier_token.value), 
                                 var_type=var_type, 
                                 expression=expression)
    
    def parse_local_var_declaration(self) -> LocalVarDeclNode:
        identifier_token = self.consume('ID')
        self.consume('AS')
        var_type = self.parse_type()
        expression = None
        if self.match('ASSIGN'):
            expression = self.parse_expression()
        return LocalVarDeclNode(identifier=IdentifierNode(name=identifier_token.value),
                                var_type=var_type,
                                expression=expression)

    def parse_param_list(self) -> List[ParamNode]:
        params: List[ParamNode] = []
        if self.current_token().type == 'RPAREN':
            return params
        while True:
            id_token = self.consume('ID')
            self.consume('AS')
            param_type = self.parse_type()
            params.append(ParamNode(identifier=IdentifierNode(name=id_token.value), param_type=param_type))
            if not self.match('COMMA'):
                break
        return params

    def parse_function_declaration(self) -> FunctionDeclNode:
        self.consume('DEF')
        id_token = self.consume('ID')
        self.consume('LPAREN')
        params: List[ParamNode] = []
        if self.current_token().type != 'RPAREN':
            params = self.parse_param_list()
        self.consume('RPAREN')
        return_type: Optional[TypeNode] = None
        if self.match('ARROW'):
            return_type = self.parse_type()
        self.consume('COLON')
        body = self.parse_block()
        return FunctionDeclNode(identifier=IdentifierNode(name=id_token.value), 
                                params=params, 
                                return_type=return_type, 
                                body=body)

    def parse_block(self) -> BlockNode:
        self.consume('INDENT')
        statements: List[StatementNode] = []
        while self.current_token() is not None and self.current_token().type != 'DEDENT':
            statements.append(self.parse_statement())
        self.consume('DEDENT')
        return BlockNode(statements=statements)

    def parse_statement(self) -> StatementNode:
        token = self.current_token()
        if token is None:
            raise ParserError("Unexpected end of input, expected a statement.")

        if token.type == 'RETURN':
            return self.parse_return_statement()
        elif token.type == 'ID': # Could be VarAssign, LocalVarDecl, or ExpressionStatement (func call)
            if self.peek_token(lookahead=2) and self.peek_token(lookahead=2).type == 'AS':
                return self.parse_local_var_declaration()
            elif self.peek_token(lookahead=2) and self.peek_token(lookahead=2).type == 'ASSIGN':
                return self.parse_variable_assignment()
            else: # Fallback to expression statement (e.g. function call)
                return ExpressionStatementNode(self.parse_expression())
        elif token.type == 'STAR': # Potential PointerAssignment or an ExpressionStatement starting with Dereference
            # Try to parse an expression starting with STAR. This will be a DereferenceNode or error.
            lhs_expr = self.parse_expression() # This should parse *expr
            if isinstance(lhs_expr, DereferenceNode) and self.match('ASSIGN'):
                rhs_expr = self.parse_expression()
                return PointerAssignmentNode(target_deref=lhs_expr, value=rhs_expr)
            else:
                # If it wasn't an assignment, it's an expression statement (e.g. *ptr; which might be valid)
                # Or if lhs_expr was not DereferenceNode but STAR was consumed, then parse_expression already handled it.
                # This relies on parse_expression correctly parsing *expr as a DereferenceNode.
                return ExpressionStatementNode(expression=lhs_expr)
        elif token.type == 'FREE':
            return self.parse_free_statement()
        else: # Fallback to expression statement for other cases (e.g. literal if allowed, (expr), etc.)
             # This also covers calls to malloc if it's not part of an assignment/declaration
            return ExpressionStatementNode(self.parse_expression())


    def parse_variable_assignment(self) -> VarAssignNode:
        id_token = self.consume('ID')
        self.consume('ASSIGN')
        expression = self.parse_expression()
        return VarAssignNode(identifier=IdentifierNode(name=id_token.value), expression=expression)

    def parse_return_statement(self) -> ReturnStmtNode:
        self.consume('RETURN')
        expression: Optional[ExpressionNode] = None
        # Check if there's an expression to return, carefully avoiding consuming tokens past statement end
        if self.current_token() is not None and self.current_token().type not in ('DEDENT', 'NEWLINE'): # Assuming NEWLINE also ends statements if not in block
            # A more robust check might be needed here if expressions can start with keywords that also end statements.
            # For now, if it's not an obvious statement terminator, try to parse an expression.
            # This list should ideally be specific starter tokens for expressions.
            EXPRESSION_STARTERS = {'ID', 'NUMBER', 'STRING', 'LPAREN', 'TRUE', 'FALSE', 'NULL', 'STAR', 'AMPERSAND', 'MALLOC'}
            if self.current_token().type in EXPRESSION_STARTERS:
                 expression = self.parse_expression()
        return ReturnStmtNode(expression=expression)

    def parse_free_statement(self) -> FreeNode:
        self.consume('FREE')
        self.consume('LPAREN')
        pointer_expr = self.parse_expression()
        self.consume('RPAREN')
        return FreeNode(pointer_expr=pointer_expr)

    # Expression Parsing Hierarchy:
    # parse_expression (handles binary operators, lowest precedence)
    #   parse_unary (handles unary operators like *, &, NOT, -)
    #     parse_primary (handles literals, identifiers, (expr), malloc)

    def parse_expression(self) -> ExpressionNode:
        # This will be expanded to handle binary operator precedence correctly (e.g. Pratt parser or shunting-yard)
        # For now, simple left-associative for OP tokens. STAR for multiplication is assumed to be an OP token.
        # STAR for dereference is handled in parse_unary.
        
        node = self.parse_unary() # Parse unary (which includes primary)

        while self.current_token() is not None and self.current_token().type == 'OP':
            op_token = self.consume('OP') # Consumes tokens like '+', '-', (binary)*, '/'
            # Note: If '*' for multiplication is an 'OP' token, it's handled here.
            # If '*' for dereference is a 'STAR' token, it's handled by parse_unary.
            right_node = self.parse_unary() # Right operand of a binary op is also unary (or primary)
            node = BinaryOpNode(left=node, operator=op_token.value, right=right_node)
        return node

    def parse_unary(self) -> ExpressionNode:
        token = self.current_token()
        if token is None:
            raise ParserError("Unexpected end of input, expected unary expression or primary.")

        if token.type == 'STAR': # Dereference: *expr
            self.consume('STAR')
            expr = self.parse_unary() # Recursively call parse_unary for chained dereferences like **ptr
            return DereferenceNode(expression=expr)
        elif token.type == 'AMPERSAND': # Address-of: &id
            self.consume('AMPERSAND')
            # As per subtask: AddressOfNode(identifier: IdentifierNode)
            # So, expect an ID here specifically.
            id_token = self.consume('ID')
            return AddressOfNode(identifier=IdentifierNode(name=id_token.value))
        # Add other unary operators like NOT, unary MINUS if they have their own tokens
        # elif token.type == 'NOT':
        #     self.consume('NOT')
        #     expr = self.parse_unary()
        #     return UnaryOpNode(operator='NOT', operand=expr) # Assuming a generic UnaryOpNode
        else: # Not a recognized unary operator, parse as primary
            return self.parse_primary()

    def parse_primary(self) -> ExpressionNode:
        token = self.current_token()
        if token is None:
            raise ParserError("Unexpected end of input, expected a primary expression.")

        if token.type == 'NUMBER':
            self.consume('NUMBER')
            val_str = token.value
            if '.' in val_str or 'e' in val_str or 'E' in val_str:
                return LiteralNode(float(val_str))
            else:
                return LiteralNode(int(val_str))
        elif token.type == 'STRING':
            self.consume('STRING')
            return LiteralNode(token.value[1:-1]) # Remove quotes
        elif token.type == 'TRUE':
            self.consume('TRUE')
            return LiteralNode(True)
        elif token.type == 'FALSE':
            self.consume('FALSE')
            return LiteralNode(False)
        elif token.type == 'NULL':
             self.consume('NULL')
             return LiteralNode(None)
        elif token.type == 'ID':
            id_token = self.consume('ID')
            # Check for function call ID(...) vs simple ID
            if self.current_token() is not None and self.current_token().type == 'LPAREN':
                # This is a function call, handled by parse_function_call_expression
                # To avoid ambiguity if ID can also be start of other primary expressions,
                # this check should be robust. For now, assume ID followed by LPAREN is a call.
                # Rewind and let parse_function_call_expression (if it existed) or general call parsing handle it.
                # For now, simple function call parsing:
                self.consume('LPAREN')
                args = self.parse_argument_list()
                self.consume('RPAREN')
                return FunctionCallNode(callee=IdentifierNode(name=id_token.value), arguments=args)
            else: # Simple identifier
                return IdentifierNode(name=id_token.value)
        elif token.type == 'LPAREN': # Parenthesized expression: (expr)
            self.consume('LPAREN')
            expression = self.parse_expression()
            self.consume('RPAREN')
            return expression
        elif token.type == 'MALLOC': # malloc(Type, Expression)
            self.consume('MALLOC')
            self.consume('LPAREN')
            alloc_type = self.parse_type()
            self.consume('COMMA')
            size_expr = self.parse_expression()
            self.consume('RPAREN')
            return MallocNode(alloc_type=alloc_type, size_expr=size_expr)
        else:
            raise ParserError(f"Unexpected token {token.type} ('{token.value}') at line {token.line}, col {token.column}. Expected a primary expression.")

    def parse_argument_list(self) -> List[ExpressionNode]:
        args: List[ExpressionNode] = []
        if self.current_token() is not None and self.current_token().type == 'RPAREN':
            return args # No arguments
        while True:
            args.append(self.parse_expression())
            if not self.match('COMMA'):
                break
        return args

if __name__ == '__main__':
    # Example Test (assuming tokenizer.py provides STAR, AMPERSAND, MALLOC, FREE)
    sample_code_pointer_stuff = """
    my_ptr AS **int
    
    DEF process_ptr(p AS *int, val AS int) -> void:
        INDENT
        *p = val  // PointerAssignment
        free(p)   // Free statement
        DEDENT

    DEF main() -> int:
        INDENT
        x AS int = 10
        px AS *int
        ppx AS **int
        
        px = &x          // AddressOf
        ppx = &px
        
        y AS int = **ppx // Dereference (nested)
        
        new_ptr AS *int = malloc(int, 1)
        *new_ptr = y + 5
        
        process_ptr(new_ptr, 100)
        // process_ptr(px, 200) // would be use after free if px was malloc'd and freed in process_ptr
        
        RETURN *px       // Dereference
        DEDENT
    """
    print(f"Testing parser with pointer/memory code:\n{sample_code_pointer_stuff}")
    try:
        # Important: Assumes tokenizer.py is updated for STAR, AMPERSAND, MALLOC, FREE
        # and that INDENT/DEDENT are correctly generated for blocks.
        # The main tokenizer.py might need a pre-pass for INDENT/DEDENT or use a different strategy.
        # For this test, one might need to manually create the token stream with these tokens.
        
        # tokens = tokenize(sample_code_pointer_stuff) # This will fail if tokenizer isn't ready for indent/dedent
        
        # Manual token stream example for a snippet like "ptr_var AS *int; *ptr_var = &another_var"
        # This is complex to do for the whole sample_code_pointer_stuff by hand.
        # This __main__ block is more for illustrating potential test cases than a runnable demo without
        # a fully capable tokenizer or hand-crafted token streams.

        print("\nNote: Full test of parser with new features requires tokenizer to support INDENT/DEDENT and new tokens.")
        print("Consider testing specific parsing methods with manually crafted token lists.")

        # Example: Test parse_type
        # parser_test_type = Parser([Token("STAR", "*"), Token("STAR", "*"), Token("ID", "int")])
        # parsed_type = parser_test_type.parse_type() # Expected: PointerTypeNode(PointerTypeNode(BuiltinTypeNode("int")))
        # print(f"Parsed type **int: {parsed_type}")

        # Example: Test parse_unary for &my_var
        # parser_test_addr = Parser([Token("AMPERSAND", "&"), Token("ID", "my_var")])
        # parsed_addr = parser_test_addr.parse_unary() # Expected: AddressOfNode(IdentifierNode("my_var"))
        # print(f"Parsed &my_var: {parsed_addr}")
        
        # Example: Test parse_statement for *p = 10
        # tokens_ptr_assign = [
        #     Token("STAR", "*", 1,0), Token("ID", "p", 1,1), Token("ASSIGN", "=", 1,3), Token("NUMBER", "10",1,5)
        # ]
        # parser_ptr_assign = Parser(tokens_ptr_assign)
        # stmt = parser_ptr_assign.parse_statement() # Expected: PointerAssignmentNode(...)
        # print(f"Parsed *p = 10: {stmt}")


    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
