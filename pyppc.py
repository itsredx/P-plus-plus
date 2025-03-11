import argparse
import os
import sys

VERSION = "0.1.0"

def compile_file(file_path):
    """Placeholder for compiling a .pypp file"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    print(f"Compiling {file_path}...")
    # TODO: Add lexical analysis, parsing, and code generation
    print("Compilation successful (placeholder).")

def main():
    parser = argparse.ArgumentParser(prog="pyppc", description="P++ Compiler")
    parser.add_argument("file", nargs="?", help="The .pypp source file to compile")
    parser.add_argument("--version", action="store_true", help="Show compiler version")

    args = parser.parse_args()

    if args.version:
        print(f"P++ Compiler Version {VERSION}")
        sys.exit(0)

    if args.file:
        if args.file.endswith(".pypp"):
            compile_file(args.file)
        else:
            print("Error: Only .pypp files are supported.")
            sys.exit(1)
    else:
        print("Error: No input file provided. Use --help for usage.")
        sys.exit(1)

if __name__ == "__main__":
    main()
