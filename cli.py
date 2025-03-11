import argparse
import sys
import time
from compiler import compile_and_run
from interpreter import Interpreter
from tokenizer import tokenize
from parser import Parser
from runtime import interactive_runtime, CompilationController

def compile_source(file_path):
    """Compile and run the source code."""
    if not file_path.endswith(".pypp"):
        print("[Error] Invalid file extension. Use .pypp for P++ source files.")
        sys.exit(1)

    try:
        with open(file_path, "r") as f:
            code = f.read()
            print(f"[Info] Compiling {file_path}...")

            start_time = time.perf_counter()
            compile_and_run(code, aot=True)  # Call your compiler
            end_time = time.perf_counter()

            print(f"[Success] Compilation and Execution completed in {end_time - start_time:.6f} seconds.")

    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        sys.exit(1)

def interpret_source(file_path):
    """Interpret and execute the source code."""
    if not file_path.endswith(".pypp"):
        print("[Error] Invalid file extension. Use .pypp for P++ source files.")
        sys.exit(1)

    try:
        with open(file_path, "r") as f:
            code = f.read()
            print(f"[Info] Interpreting {file_path}...")

            tokens = tokenize(code)
            parser = Parser(tokens)
            ast = parser.parse_program()

            interpreter = Interpreter()
            start_time = time.perf_counter()
            interpreter.interpret(ast)
            end_time = time.perf_counter()

            print(f"[Success] Interpretation completed in {end_time - start_time:.6f} seconds.")

    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        sys.exit(1)

def run_source(file_path):
    """Run the source code using the P++ runtime."""
    if not file_path.endswith(".pypp"):
        print("[Error] Invalid file extension. Use .pypp for P++ source files.")
        sys.exit(1)

    try:
        with open(file_path, "r") as f:
            code = f.read()
            print(f"[Info] Running {file_path} using runtime...")

            comp_controller = CompilationController()  # Create a CompilationController instance

            start_time = time.perf_counter()
            interactive_runtime(code, comp_controller)  # Pass the required argument
            end_time = time.perf_counter()

            print(f"[Success] Runtime execution completed in {end_time - start_time:.6f} seconds.")

    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(prog="pypp", description="P++ CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile and execute a .pypp source file")
    compile_parser.add_argument("file", type=str, help="The P++ source file to compile")

    # Interpret command
    interpret_parser = subparsers.add_parser("interpret", help="Interpret a .pypp source file")
    interpret_parser.add_argument("file", type=str, help="The P++ source file to interpret")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a .pypp source file using the P++ runtime")
    run_parser.add_argument("file", type=str, help="The P++ source file to run")

    args = parser.parse_args()

    if args.command == "compile":
        compile_source(args.file)
    elif args.command == "interpret":
        interpret_source(args.file)
    elif args.command == "run":
        run_source(args.file)

if __name__ == "__main__":
    main()
