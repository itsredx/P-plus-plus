# ir_nodes.py

class IRNode:
    pass

class IRFunction(IRNode):
    def __init__(self, name):
        self.name = name
        self.basic_blocks = []

    def add_block(self, block):
        self.basic_blocks.append(block)

    def __repr__(self):
        return f"IRFunction({self.name}, blocks={self.basic_blocks})"

class IRBasicBlock(IRNode):
    def __init__(self, label):
        self.label = label
        self.instructions = []

    def add_instruction(self, instr):
        self.instructions.append(instr)

    def __repr__(self):
        return f"IRBasicBlock({self.label}, instrs={self.instructions})"

class IRInstruction(IRNode):
    def __init__(self, op, operands=None):
        self.op = op
        self.operands = operands if operands is not None else []

    def __repr__(self):
        return f"IRInstruction({self.op}, {self.operands})"
