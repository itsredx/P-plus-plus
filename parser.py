# parser.py
from ast_nodes import *
from tokenizer import tokenize, Token # Assuming tokenize provides INDENT, DEDENT, ARROW, etc.
from typing import List, Optional

# Expected token types from tokenizer (examples, ensure tokenizer.py provides these)
# KEYWORDS: IMPORT, DEF, AS, RETURN, IF, ELSE, WHILE, FOR, IN, CLASS,
#           INT, INT8, ..., FLOAT, ..., BOOL, STRING, VOID, TRUE, FALSE, NULL
# OPERATORS: ASSIGN (=), OP (various, e.g., +, -, *, /, ==, !=, <, >), ARROW (->)
# PUNCTUATION: LPAREN, RPAREN, COLON, COMMA, LBRACE, RBRACE (if/when blocks use them)
# SPECIAL: ID (identifiers), NUMBER (numeric literals), STRING (string literals)
# LAYOUT: INDENT, DEDENT, NEWLINE (though INDENT/DEDENT often make NEWLINE less critical for structure)

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
        while self.current_token() is not None and self.current_token().type != 'DEDENT': # DEDENT check for safety if file ends unexpectedly
            token_type = self.current_token().type
            if token_type == 'IMPORT':
                declarations.append(self.parse_import())
            elif token_type == 'DEF':
                declarations.append(self.parse_function_declaration())
            elif token_type == 'ID': 
                # Need to distinguish GlobalVarDecl from other ID-starting things.
                # GlobalVarDecl: ID AS Type ...
                # If we are at the top level, an ID not followed by AS could be an error or an expression_statement
                # (though ProgramNode doesn't list arbitrary StatementNode).
                # For now, assume top-level IDs must be part of a GlobalVarDecl if not DEF or IMPORT.
                # A more robust way might involve trying to parse and backtracking, or more lookahead.
                if self.peek_token(lookahead=2) and self.peek_token(lookahead=2).type == 'AS':
                     declarations.append(self.parse_global_var_declaration())
                else:
                    # As per ProgramNode, it doesn't take arbitrary statements.
                    # So, an ID at the top level not forming a GlobalVarDecl or FuncDecl is an error.
                    # However, the subtask says "Program ... or StatementNode".
                    # This is ambiguous. For now, I'll allow ExpressionStatementNode at top-level
                    # if it's not a global var decl.
                    # This means `parse_statement` could be called here.
                    # Let's assume for now that only specific declarations are top-level.
                    # If StatementNode is allowed, then parse_statement would be here.
                    # Re-reading: "ProgramNode ... or StatementNode". This implies parse_statement.
                    # This means parse_program should look more like a block's content.
                    # This contradicts ProgramNode's definition in ast_nodes.py.
                    # I will stick to the ast_nodes.py definition of ProgramNode for now.
                    # Which means an ID not forming a global var here is an error.
                    token = self.current_token()
                    raise ParserError(f"Unexpected token '{token.value}' ({token.type}) at top level at line {token.line}, col {token.column}. Expected IMPORT, DEF, or global variable declaration (ID AS Type).")
            else:
                token = self.current_token()
                raise ParserError(f"Unexpected token '{token.value}' ({token.type}) at top level at line {token.line}, col {token.column}. Expected IMPORT, DEF, or global variable declaration.")
        
        # Filter out None if any parsing path could return it (shouldn't for declarations)
        # declarations = [decl for decl in declarations if decl is not None]
        return ProgramNode(declarations)


    def parse_import(self) -> ImportNode:
        self.consume('IMPORT')
        id_token = self.consume('ID')
        return ImportNode(module_name=IdentifierNode(name=id_token.value))

    def parse_type(self) -> TypeNode:
        # Type names are tokenized as ID, or specific types like INT, STRING if tokenizer is set up that way.
        # The tokenizer subtask added types to KEYWORDS, so they should be specific tokens.
        # e.g. INT, STRING, FLOAT32 etc.
        token = self.current_token()
        # List of known type tokens from KEYWORDS.
        # This check ensures we consume something that is actually a type.
        # This list should mirror what's in tokenizer.py's KEYWORDS that are types.
        TYPE_KEYWORDS = {'INT', 'INT8', 'INT16', 'INT32', 'INT64',
                         'UINT8', 'UINT16', 'UINT32', 'UINT64',
                         'FLOAT', 'FLOAT32', 'FLOAT64',
                         'BOOL', 'STRING', 'VOID', 'ID'} # ID for custom types
        
        if token.type in TYPE_KEYWORDS or token.type == 'ID': # Allow general ID for potential custom types
            self.pos +=1 # Consume the type token
            return TypeNode(name=token.value)
        else:
            raise ParserError(f"Expected type identifier at line {token.line}, col {token.column}, found {token.type} ('{token.value}')")

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
        if self.match('ARROW'): # ARROW is '->'
            return_type = self.parse_type()
        
        self.consume('COLON') # Expect colon before block
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
        elif token.type == 'ID':
            # Lookahead needed:
            # ID AS Type ... -> Local Variable Declaration
            # ID ASSIGN ... -> Variable Assignment
            # ID LPAREN ... -> Function Call (ExpressionStatement)
            # ID (other) ... -> Identifier (ExpressionStatement)
            
            # Check for ID AS Type
            if self.peek_token(lookahead=2) and self.peek_token(lookahead=2).type == 'AS':
                return self.parse_local_var_declaration()
            # Check for ID ASSIGN
            elif self.peek_token(lookahead=2) and self.peek_token(lookahead=2).type == 'ASSIGN':
                return self.parse_variable_assignment()
            else: # Fallback to expression statement
                return self.parse_expression_statement()
        # Add parsing for IF, WHILE, FOR etc. here in the future
        # e.g., if token.type == 'IF': return self.parse_if_statement()
        else: # Fallback to expression statement for other cases (e.g. literal, (expr), etc.)
            return self.parse_expression_statement()

    def parse_variable_assignment(self) -> VarAssignNode:
        id_token = self.consume('ID')
        self.consume('ASSIGN')
        expression = self.parse_expression()
        return VarAssignNode(identifier=IdentifierNode(name=id_token.value), expression=expression)

    def parse_return_statement(self) -> ReturnStmtNode:
        self.consume('RETURN')
        expression: Optional[ExpressionNode] = None
        # A return statement might not have an expression.
        # It is followed by DEDENT or NEWLINE (if NEWLINE is significant and not consumed by INDENT/DEDENT logic)
        # Check if the next token can start an expression.
        # For simplicity, if it's not DEDENT (end of block) or other statement keywords, try to parse an expression.
        # This needs to be robust. If the tokenizer guarantees NEWLINEs are significant or INDENT/DEDENT handle it, it's simpler.
        # Assuming INDENT/DEDENT means we don't have NEWLINE tokens to check here.
        if self.current_token() is not None and self.current_token().type != 'DEDENT':
            # This condition is tricky: how do we know if there's an expression or if it's just the end of the line?
            # For now, if it's not DEDENT, we assume an expression.
            # More robust: check if current_token().type can start an expression (ID, Literal, LPAREN, etc.)
            # Python's return can be followed by nothing.
            # A better check: if current token is not one that ends a statement (like DEDENT, or in some languages ';', or NEWLINE)
            # AND it can start an expression, then parse it.
            # The simplest for now: if it's not DEDENT, try parse_expression. parse_expression will fail if it's not an expression.
            # This might be too greedy. A common way is to check if the token is an expression starter.
            # Let's assume for now that if it's not DEDENT, it's an expression.
            # This implies `return` on its own line before DEDENT is `ReturnStmtNode(None)`.
            # If there are other tokens, parse_expression will try.
            # What if `return` is the last statement before DEDENT? current_token becomes DEDENT.
            # So, if current_token is not DEDENT, it must be an expression.
            if self.current_token().type != 'DEDENT': # Check before consuming.
                 # This check is problematic if a statement like `if` can follow `return` without expression (which is invalid P-- but parser might see it)
                 # A simpler check: if the token is one that can start an expression.
                 ct = self.current_token()
                 can_start_expr = ct.type in {'ID', 'NUMBER', 'STRING', 'LPAREN', 'TRUE', 'FALSE', 'NULL'} # Add other expression starters
                 if can_start_expr:
                    expression = self.parse_expression()
        return ReturnStmtNode(expression=expression)

    def parse_expression_statement(self) -> ExpressionStatementNode:
        expression = self.parse_expression()
        return ExpressionStatementNode(expression=expression)

    # Expression parsing (simplified, needs precedence climbing or Pratt parser for full operator precedence)
    def parse_expression(self) -> ExpressionNode:
        # For now, using a simple left-associative binary op parser.
        # This will need to be replaced with a more robust precedence climbing or Pratt parser for real P--.
        # Example: parse_term, parse_factor, etc. or precedence climbing.
        # Current one from input: parse_primary { OP parse_primary }
        
        left = self.parse_primary() # Higher precedence part

        # Simplified: only one level of binary operators for now.
        # This needs to be expanded for proper operator precedence (e.g. * before +)
        while self.current_token() is not None and self.current_token().type == 'OP':
            op_token = self.consume('OP')
            right = self.parse_primary() # Should be a higher precedence expression term
            left = BinaryOpNode(left=left, operator=op_token.value, right=right)
        return left

    def parse_primary(self) -> ExpressionNode:
        token = self.current_token()
        if token is None:
            raise ParserError("Unexpected end of input, expected a primary expression.")

        if token.type == 'NUMBER':
            self.consume('NUMBER')
            if '.' in token.value or 'e' in token.value or 'E' in token.value: # simple float check
                return LiteralNode(float(token.value))
            else:
                return LiteralNode(int(token.value))
        elif token.type == 'STRING':
            self.consume('STRING')
            return LiteralNode(token.value[1:-1]) # Remove quotes
        elif token.type == 'TRUE':
            self.consume('TRUE')
            return LiteralNode(True)
        elif token.type == 'FALSE':
            self.consume('FALSE')
            return LiteralNode(False)
        elif token.type == 'NULL': # Assuming NULL is a keyword for null literal
             self.consume('NULL')
             return LiteralNode(None) # Or a specific NullLiteralNode if defined
        elif token.type == 'ID':
            id_token = self.consume('ID')
            if self.match('LPAREN'): # Function call
                args = self.parse_argument_list()
                self.consume('RPAREN')
                return FunctionCallNode(callee=IdentifierNode(name=id_token.value), arguments=args)
            else: # Simple identifier
                return IdentifierNode(name=id_token.value)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expression = self.parse_expression()
            self.consume('RPAREN')
            return expression
        else:
            raise ParserError(f"Unexpected token {token.type} ('{token.value}') at line {token.line}, col {token.column}. Expected a primary expression.")

    def parse_argument_list(self) -> List[ExpressionNode]:
        args: List[ExpressionNode] = []
        if self.current_token().type == 'RPAREN':
            return args # No arguments

        while True:
            args.append(self.parse_expression())
            if not self.match('COMMA'):
                break
        return args


if __name__ == '__main__':
    # Test cases for the parser
    # Ensure your tokenizer.py is in the same directory or accessible via PYTHONPATH
    # And that it produces INDENT, DEDENT, ARROW, AS, and uppercase keywords.
    
    # Mock Tokenizer and Tokens if direct tokenizer.py is problematic for testing setup
    # For now, assume tokenizer.py works and can be called.

    test_code_samples = {
        "import_simple": "IMPORT my_module",
        "global_var_simple": "my_var AS int",
        "global_var_init": "count AS int = 100",
        "function_simple_void": """
DEF my_func():
    INDENT
    RETURN
    DEDENT
""",
        "function_params_return": """
DEF add(a AS int, b AS int) -> int:
    INDENT
    c AS int = a + b
    RETURN c
    DEDENT
""",
        "function_expr_statement": """
DEF call_print():
    INDENT
    print("hello")
    DEDENT
""",
        "block_with_assign_return": """
DEF test_block() -> int:
    INDENT
    x AS int = 5
    x = x + 10
    RETURN x
    DEDENT
""",
        "program_multiple_decls": """
IMPORT io

gravity AS float = 9.81

DEF main():
    INDENT
    io.print(gravity)
    RETURN
    DEDENT
"""
    }

    # A simplified mock tokenizer for testing if the real one isn't ready for INDENT/DEDENT
    # This is VERY basic and not a real lexer.
    from tokenizer import Token # Import Token class
    
    def mock_tokenize_for_parser(code_str: str) -> List[Token]:
        # This is a placeholder. A real test setup would use the actual tokenizer
        # or a more sophisticated mock. This one will struggle with complex cases.
        # It needs to correctly output INDENT/DEDENT and uppercase keywords.
        # Example: "DEF main(): INDENT RETURN DEDENT"
        # For now, let's assume the actual tokenizer.py is callable and works.
        print(f"\n--- Tokenizing (using actual tokenizer.py) ---\n{code_str}\n------------------------------------")
        try:
            tokens = tokenize(code_str)
            # Filter out SKIP and NEWLINE tokens if parser doesn't expect them post-INDENT/DEDENT stage
            tokens = [t for t in tokens if t.type not in ('SKIP', 'NEWLINE')] # Adjust as needed
            print("Tokens:", [f"{t.type}({t.value})" for t in tokens])
            return tokens
        except Exception as e:
            print(f"Tokenizer error: {e}")
            raise

    for name, code in test_code_samples.items():
        print(f"\n--- Testing: {name} ---")
        print(f"Code:\n{code.strip()}")
        try:
            # tokens = mock_tokenize_for_parser(code.strip()) # Use actual tokenizer
            tokens = tokenize(code) # Use actual tokenizer.
            # Filter tokens typically ignored by parser after block structure is handled by INDENT/DEDENT
            # Parser expects INDENT/DEDENT to structure blocks, not raw newlines usually.
            # However, if tokenizer doesn't emit NEWLINE when INDENT/DEDENT are used, this is fine.
            # The current tokenizer might still emit NEWLINEs. Let's filter them for now.
            # Also SKIP tokens (whitespace).
            tokens = [t for t in tokens if t.type not in ('SKIP', 'NEWLINE')] 
            print("Filtered Tokens for Parser:", [f"{t.type}({t.value})" for t in tokens])

            parser = Parser(tokens)
            ast = parser.parse_program()
            print("AST:", ast)
        except ParserError as e:
            print(f"Parser Error: {e}")
        except SyntaxError as e: # From tokenizer
            print(f"Syntax Error (Tokenizer): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

    # Manual test with a more complex example expecting indent/dedent
    # This requires the tokenizer to correctly emit INDENT/DEDENT.
    # If tokenizer.py is not set up for INDENT/DEDENT, this will fail at tokenization or parsing.
    # The tokenizer from previous steps does not explicitly add INDENT/DEDENT.
    # This parser *requires* INDENT/DEDENT tokens.
    # A note for the user: The tokenizer.py needs to be updated to emit these.
    # For now, the tests above might fail if tokenizer.py isn't providing these.
    # I will assume for this subtask that the tokens are magically provided.
    print("\nNote: The parser relies on the tokenizer to provide INDENT/DEDENT tokens.")
    print("If the tokenizer.py from previous steps has not been updated for this,")
    print("the tests involving blocks might fail at tokenization or parsing stage.")

    # Example:
    # tokens = [
    #     Token('DEF', 'def', 1, 0), Token('ID', 'main', 1, 4), Token('LPAREN', '(', 1, 8), Token('RPAREN', ')', 1, 9), Token('COLON', ':', 1, 10),
    #     Token('INDENT', '<INDENT>', 2, 0),
    #     Token('ID', 'x', 2, 4), Token('AS', 'as', 2, 5), Token('ID', 'int', 2, 8), Token('ASSIGN', '=', 2, 12), Token('NUMBER', '5', 2, 14),
    #     Token('RETURN', 'return', 3, 4), Token('ID', 'x', 3, 11),
    #     Token('DEDENT', '<DEDENT>', 4, 0)
    # ]
    # parser = Parser(tokens)
    # ast = parser.parse_program() # => ProgramNode(declarations=[FunctionDeclNode(...)])
    # print("Manual AST:", ast)
