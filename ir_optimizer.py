from ir_nodes import *

def fold_constants_in_instruction(instr):
    """
    Checks if the given IR instruction is a binary operation and
    if both operands are constant. If so, computes the result and returns
    a new 'const' instruction. Otherwise, returns the instruction unchanged.
    """
    if instr.op in {"+", "-", "*", "/"} and len(instr.operands) == 2:
        left, right = instr.operands
        if isinstance(left, IRInstruction) and isinstance(right, IRInstruction):
            if left.op == "const" and right.op == "const":
                try:
                    left_val = left.operands[0]
                    right_val = right.operands[0]
                    if instr.op == "+":
                        result = left_val + right_val
                    elif instr.op == "-":
                        result = left_val - right_val
                    elif instr.op == "*":
                        result = left_val * right_val
                    elif instr.op == "/":
                        # Avoid division by zero.
                        if right_val == 0:
                            return instr
                        result = left_val / right_val
                    return IRInstruction("const", [result])
                except Exception as e:
                    return instr
    return instr

def optimize_node(instr):
    """
    Recursively optimizes an IR instruction and its operands.
    """
    if isinstance(instr, IRInstruction):
        # Recursively optimize each operand if it's an IRInstruction.
        optimized_operands = []
        for op in instr.operands:
            if isinstance(op, IRInstruction):
                optimized_operands.append(optimize_node(op))
            else:
                optimized_operands.append(op)
        instr.operands = optimized_operands
        # Attempt to fold constants at this node.
        return fold_constants_in_instruction(instr)
    return instr

def optimize_ir(ir_function):
    """
    Applies IR-level optimizations recursively to each instruction in the IRFunction.
    """
    for block in ir_function.basic_blocks:
        optimized_instructions = []
        for instr in block.instructions:
            optimized_instr = optimize_node(instr)
            optimized_instructions.append(optimized_instr)
        block.instructions = optimized_instructions
    return ir_function

if __name__ == "__main__":
    from ast_to_ir import translate
    from tokenizer import tokenize
    from parser import Parser

    # Sample code: binary expression constant folding.
    code = 'print(21 + 21)'
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse_program()
    
    # Translate AST to IR
    ir = translate(ast)
    print("Before optimization:", ir)
    
    # Optimize IR recursively
    optimized_ir = optimize_ir(ir)
    print("After optimization:", optimized_ir)
