from llvmlite import ir, binding
from ir_nodes import *

# Initialize LLVM
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

def generate_llvm_ir(ir_function):
    module = ir.Module(name=ir_function.name)
    # Set the module triple and data layout based on the default target.
    target_triple = binding.get_default_triple()
    module.triple = target_triple
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    module.data_layout = target_machine.target_data  # Use target_data property

    func_type = ir.FunctionType(ir.IntType(32), [])
    llvm_function = ir.Function(module, func_type, name=ir_function.name)
    entry_block = llvm_function.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_block)
    
    for block in ir_function.basic_blocks:
        for instr in block.instructions:
            translate_instruction(instr, builder, module)
    
    builder.ret(ir.Constant(ir.IntType(32), 0))
    return module


def translate_instruction(instr, builder, module):
    if instr.op == "const":
        try:
            value = float(instr.operands[0])
            return ir.Constant(ir.DoubleType(), value)
        except ValueError:
            string_val = instr.operands[0]
            str_type = ir.ArrayType(ir.IntType(8), len(string_val) + 1)
            global_str = ir.GlobalVariable(module, str_type, name="str_const")
            global_str.linkage = 'internal'
            global_str.global_constant = True
            global_str.initializer = ir.Constant(str_type, bytearray(string_val + "\0", "utf8"))
            ptr = builder.bitcast(global_str, ir.IntType(8).as_pointer(), name="strptr")
            return ptr

    elif instr.op in {"+", "-", "*", "/"}:
        left_val = translate_instruction(instr.operands[0], builder, module)
        right_val = translate_instruction(instr.operands[1], builder, module)
        if instr.op == "+":
            return builder.fadd(left_val, right_val, name="faddtmp")
        elif instr.op == "-":
            return builder.fsub(left_val, right_val, name="fsubtmp")
        elif instr.op == "*":
            return builder.fmul(left_val, right_val, name="fmultmp")
        elif instr.op == "/":
            return builder.fdiv(left_val, right_val, name="fdivtmp")
    
    elif instr.op in {"==", "!=", ">", "<", ">=", "<="}:
        left_val = translate_instruction(instr.operands[0], builder, module)
        right_val = translate_instruction(instr.operands[1], builder, module)
        if instr.op == "==":
            cmp = builder.fcmp_ordered("==", left_val, right_val, name="fcmp_eq")
        elif instr.op == "!=":
            cmp = builder.fcmp_ordered("!=", left_val, right_val, name="fcmp_ne")
        elif instr.op == ">":
            cmp = builder.fcmp_ordered(">", left_val, right_val, name="fcmp_gt")
        elif instr.op == "<":
            cmp = builder.fcmp_ordered("<", left_val, right_val, name="fcmp_lt")
        elif instr.op == ">=":
            cmp = builder.fcmp_ordered(">=", left_val, right_val, name="fcmp_ge")
        elif instr.op == "<=":
            cmp = builder.fcmp_ordered("<=", left_val, right_val, name="fcmp_le")
        bool_to_double = builder.uitofp(cmp, ir.DoubleType(), name="booltodbl")
        return bool_to_double
    
    elif instr.op == "call":
        callee_name = instr.operands[0]
        llvm_args = []
        for arg in instr.operands[1:]:
            if isinstance(arg, IRInstruction):
                llvm_args.append(translate_instruction(arg, builder, module))
            else:
                llvm_args.append(ir.Constant(ir.DoubleType(), float(arg)))
        if callee_name == "print":
            # If the first argument is a pointer to i8, assume it's a string.
            if llvm_args and isinstance(llvm_args[0].type, ir.PointerType) and llvm_args[0].type.pointee == ir.IntType(8):
                func_type = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))])
                target_name = "puts"
            else:
                # Otherwise, assume it's numeric; use printf.
                # Create or retrieve the global format string "%f\n"
                fmt_str = "%f\n"
                fmt_len = len(fmt_str) + 1
                fmt_type = ir.ArrayType(ir.IntType(8), fmt_len)
                global_fmt = module.globals.get("print_fmt")
                if not global_fmt:
                    global_fmt = ir.GlobalVariable(module, fmt_type, name="print_fmt")
                    global_fmt.linkage = 'internal'
                    global_fmt.global_constant = True
                    global_fmt.initializer = ir.Constant(fmt_type, bytearray(fmt_str + "\0", "utf8"))
                fmt_ptr = builder.bitcast(global_fmt, ir.PointerType(ir.IntType(8)), name="fmtptr")
                # Prepend the format string to the arguments.
                llvm_args = [fmt_ptr] + llvm_args
                func_type = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True)
                target_name = "printf"
        else:
            arg_count = len(llvm_args)
            func_type = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()] * arg_count, var_arg=False)
            target_name = callee_name
        callee = module.globals.get(target_name)
        if not callee:
            callee = ir.Function(module, func_type, name=target_name)
        return builder.call(callee, llvm_args, name="calltmp")
    
    elif instr.op == "store":
        var_name = instr.operands[0]
        value_ir = translate_instruction(instr.operands[1], builder, module)
        global_var = module.globals.get(var_name)
        if not global_var:
            global_var = ir.GlobalVariable(module, ir.DoubleType(), var_name)
            global_var.linkage = 'internal'
            global_var.global_constant = False
            global_var.initializer = ir.Constant(ir.DoubleType(), 0.0)
        builder.store(value_ir, global_var)
        return value_ir

    elif instr.op == "load":
        var_name = instr.operands[0]
        global_var = module.globals.get(var_name)
        if not global_var:
            raise Exception(f"Global variable '{var_name}' not defined for load")
        return builder.load(global_var, name="loadtmp")
    
    else:
        raise NotImplementedError(f"Unsupported IR opcode: {instr.op}")

if __name__ == "__main__":
    from ast_to_ir import translate
    from tokenizer import tokenize
    from parser import Parser
    
    test_codes = [
        'print("helo world")',
        'print(21 * 21)',
        'print(21 + 21)',
        'print(21 < 42)',
        'print(21 >= 21)',
        'print((21 + 21) * (42 / 2))',
        'x = 15\nprint(x)',
        'x = 15\nprint(3 + x)'
    ]
    
    for code in test_codes:
        print("======================================")
        print("Source Code:")
        print(code)
        try:
            tokens = tokenize(code)
            parser = Parser(tokens)
            ast = parser.parse_program()
            print("Custom IR:")
            custom_ir = translate(ast)
            print(custom_ir)
            optimized_ir = custom_ir
            llvm_module = generate_llvm_ir(optimized_ir)
            print("Generated LLVM IR:")
            print(llvm_module)
        except Exception as e:
            print("Error:", e)
        print("======================================\n")
