import re

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"

KEYWORDS = {'def', 'if', 'else', 'while', 'for', 'in', 'return', 
            'not', 'and', 'or', 'null', 'ptr', 'asm', 'true', 'false',
            'as', 'class',  # New keywords
            'int', 'int8', 'int16', 'int32', 'int64',  # Integer types
            'uint8', 'uint16', 'uint32', 'uint64',   # Unsigned integer types
            'float', 'float32', 'float64',             # Float types
            'bool', 'string', 'void',                  # Other built-in types
            'malloc', 'free',                          # Memory management keywords
            'INDENT', 'DEDENT'}                        # Block structure keywords (for current test setup)

# Updated token specification:
# Removed POINTER_OP and let '*' be handled in OP.
token_specification = [
    ('NUMBER',        r'\d+(\.\d+)?'),                      # Integer or decimal number
    ('STRING',        r'"([^"\\]|\\.)*"'),                  # String literal with escape support
    ('ID',            r'[A-Za-z_]\w*'),                     # Identifiers
    ('AMPERSAND',     r'&'),                               # Address-of operator (renamed)
    ('STAR',          r'\*'),                               # Asterisk for pointer ops or multiply
    ('ASSIGN',        r'='),                                # Assignment operator
    ('LPAREN',        r'\('),                               # Left Parenthesis
    ('RPAREN',        r'\)'),                               # Right Parenthesis
    ('COLON',         r':'),                               # Colon
    ('COMMA',         r','),                               # Comma
    ('ARROW',         r'->'),                              # Arrow operator
    ('OP',            r'==|!=|>=|<=|>|<|\+|-|/|%'),         # Operators (multiplication * removed)
    ('NEWLINE',       r'\n'),                               # Newline
    ('SKIP',          r'[ \t]+'),                           # Skip spaces and tabs
    ('MISMATCH',      r'.'),                                # Any other character (error)
]

token_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
get_token = re.compile(token_regex).match

def tokenize(code):
    line_num = 1
    line_start = 0
    pos = 0
    tokens = []
    mo = get_token(code, pos)
    while mo is not None:
        kind = mo.lastgroup
        value = mo.group(kind)
        if kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
        elif kind == 'SKIP':
            pass
        elif kind == 'MISMATCH':
            col = mo.start() - line_start + 1
            raise SyntaxError(f"Unexpected character {value!r} at line {line_num}, column {col}")
        else:
            if kind == 'ID' and value in KEYWORDS:
                kind = value.upper()
            col = mo.start() - line_start + 1
            tokens.append(Token(kind, value, line_num, col))
        pos = mo.end()
        mo = get_token(code, pos)
    return tokens

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tokenizer.py <file.pypp>")
        sys.exit(1)
    
    filename = sys.argv[1]
    try:
        with open(filename, 'r') as f:
            code = f.read()
        tokens = tokenize(code)
        for token in tokens:
            print(token)
    except Exception as e:
        print(e)
        sys.exit(1)
