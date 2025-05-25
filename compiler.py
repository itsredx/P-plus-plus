import sys
import subprocess
from tokenizer import tokenize
from parser import Parser
from ast_to_ir import ASTToIRConverter
from ir_optimizer import optimize_ir
from llvm_codegen import LLVMCodeGenerator
from llvmlite import binding
from ctypes import CFUNCTYPE, c_int

def compile_and_run(source_code, aot=False):
    # Tokenize the source code.
    tokens = tokenize(source_code)
    # Parse tokens into an AST.
    parser = Parser(tokens)
    ast = parser.parse_program()
    print("AST:", ast)
    
    # Translate AST to our custom IR.
    converter = ASTToIRConverter()
    custom_ir = converter.translate_program(ast)
    print("Custom IR:", custom_ir)
    
    # Optimize the IR.
    optimized_ir = optimize_ir(custom_ir)
    print("Optimized IR:", optimized_ir)
    
    # Generate LLVM IR from our optimized IR.
    llvm_gen = LLVMCodeGenerator()
    llvm_ir_str = llvm_gen.generate_llvm_ir(optimized_ir)
    print("Generated LLVM IR:\n", llvm_ir_str)
    
    # Initialize LLVM for JIT execution.
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    if aot:
        # AOT: Write LLVM IR to file, compile to object file, and link to create an executable.
        with open("output.ll", "w") as f:
            f.write(llvm_ir_str)
        print("LLVM IR written to output.ll")
        
        # Use llc to compile LLVM IR to an object file.
        llc_cmd = ["llc", "output.ll", "-filetype=obj", "-o", "output.o"]
        print("Running llc:", " ".join(llc_cmd))
        subprocess.run(llc_cmd, check=True)
        
        # Use clang to link the object file into an executable.
        clang_cmd = ["clang", "output.o", "-o", "output_executable"]
        print("Running clang:", " ".join(clang_cmd))
        subprocess.run(clang_cmd, check=True)
        
        print("Executable generated: output_executable")
    else:
        # JIT: Use MCJIT to execute the LLVM IR.
        mod = binding.parse_assembly(llvm_ir_str)
        mod.verify()
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        with binding.create_mcjit_compiler(mod, target_machine) as ee:
            ee.finalize_object()
            ee.run_static_constructors()
            main_ptr = ee.get_function_address("main")
            main_func = CFUNCTYPE(c_int)(main_ptr)
            result = main_func()
            print("Execution result:", result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compiler.py <file.pypp> [--aot]")
        sys.exit(1)
    
    filename = sys.argv[1]
    aot = "--aot" in sys.argv
    with open(filename, "r") as f:
        source_code = f.read()
    compile_and_run(source_code, aot=aot)
