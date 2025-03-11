# ast_to_ir.py
from ir_nodes import *
from ast_nodes import *

def translate(ast):
    """
    Translates the given AST (of type ProgramNode) into an IRFunction.
    For now, we create an IRFunction named 'main' with a single basic block.
    """
    main_func = IRFunction("main")
    entry_block = IRBasicBlock("entry")
    main_func.add_block(entry_block)
    
    # Process each declaration in the program.
    for decl in ast.declarations:
        ir_instr = translate_node(decl)
        if ir_instr:
            entry_block.add_instruction(ir_instr)
    return main_func

def translate_node(node):
    """
    Recursively translates AST nodes into IR instructions.
    Handles:
      - ExpressionStatementNode
      - VariableDeclarationNode (added)
      - FunctionCallNode
      - BinaryOpNode
      - LiteralNode
      - IdentifierNode
    """
    if isinstance(node, ExpressionStatementNode):
        return translate_node(node.expression)
    elif isinstance(node, VariableDeclarationNode):
        # Translate a variable declaration:
        # 1. Translate the initializer expression.
        expr_ir = translate_node(node.expression)
        # 2. Create a store instruction to store the value into the variable.
        # For simplicity, we ignore type annotation.
        return IRInstruction("store", [node.identifier, expr_ir])
    elif isinstance(node, FunctionCallNode):
        if isinstance(node.callee, IdentifierNode):
            callee = node.callee.name
        else:
            callee = "<unknown>"
        args = [translate_node(arg) for arg in node.arguments]
        return IRInstruction("call", [callee] + args)
    elif isinstance(node, BinaryOpNode):
        left_ir = translate_node(node.left)
        right_ir = translate_node(node.right)
        return IRInstruction(node.operator, [left_ir, right_ir])
    elif isinstance(node, LiteralNode):
        return IRInstruction("const", [node.value])
    elif isinstance(node, IdentifierNode):
        return IRInstruction("load", [node.name])
    else:
        raise Exception("Translation not implemented for node: " + repr(node))

if __name__ == "__main__":
    # For testing purposes:
    from tokenizer import tokenize
    from parser import Parser
    
    # Sample P++ code for testing:
    code = (
        'print("helo world")\n'
        'x = 15\n'
        'print(3 + 9)\n'
        'print(3 + x)'
    )
    
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse_program()
    print("AST:")
    print(ast)
    
    custom_ir = translate(ast)
    print("Custom IR:")
    print(custom_ir)
