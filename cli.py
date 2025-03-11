import argparse
import sys


def compile_source(file_path):
    if not file_path.endswith(".pypp"):
        print("[Error] Invalid file extension. Use .pypp for P++ source files.")
        sys.exit(1)
    
    try:
        with open(file_path, "r") as f:
            code = f.read()
            print(f"[Info] Compiling {file_path}...")
            # TODO: Pass `code` to the compiler backend
            print("[Success] Compilation completed.")
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="pyppc", description="P++ Compiler CLI")
    parser.add_argument("compile", metavar="file", type=str, help="Compile a .pypp source file")
    args = parser.parse_args()
    
    compile_source(args.compile)


if __name__ == "__main__":
    main()