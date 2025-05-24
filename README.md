# P-- (P Minus Minus)

P-- is a work-in-progress programming language, featuring both a compiler and an interpreter. It is designed to be a learning project exploring various aspects of language design and implementation, including parsing, intermediate representation, code generation (via LLVM), and runtime environments.

## Project Status

The project is developed iteratively. Key completed milestones include:
*   **Lexical Analysis (Tokenization)**
*   **Syntactic Analysis (Parsing to AST)**
*   **Custom Intermediate Representation (IR)**: Translation from AST to IR, including basic error handling.
*   **LLVM Code Generation & Basic Optimization**: Conversion of custom IR to LLVM IR, with initial optimizations like constant folding and dead code elimination.
*   **Basic AOT and JIT Compilation**: Capable of compiling and running simple programs.
*   **AST-Walking Interpreter**: For direct execution of P-- code.
*   **Interactive Shell (REPL)**: Allowing users to experiment with P-- in either interpreted or JIT-compiled mode.
*   **Command-Line Interface (CLI)**: For compiling, interpreting, and running `.pypp` files.

**Current Development Focus (Iteration 4): Memory Management**
*   Designing and implementing manual memory management strategies (e.g., C/C++ style allocation/deallocation, RAII patterns).
*   Developing an optional garbage collector.
*   Ensuring memory safety and defining clear ownership rules within the runtime.

(See `steps.md` for more details on development iterations.)

## Project Structure

The project is organized into several key Python modules:

*   **`tokenizer.py`**: Performs lexical analysis, converting source code into a stream of tokens.
*   **`parser.py`**: Parses the token stream to build an Abstract Syntax Tree (AST).
*   **`ast_nodes.py`**: Defines the node structures for the AST.
*   **`ast_to_ir.py`**: Translates the AST into a custom Intermediate Representation (IR).
*   **`ir_nodes.py`**: Defines the node structures for the custom IR.
*   **`ir_optimizer.py`**: Implements optimization passes on the custom IR.
*   **`compiler.py`**: Orchestrates the compilation pipeline (Tokenize -> Parse -> AST -> IR -> Optimize IR -> LLVM IR). Supports both Ahead-Of-Time (AOT) compilation to executables and Just-In-Time (JIT) execution.
*   **`llvm_codegen.py`**: Generates LLVM IR from the custom IR.
*   **`interpreter.py`**: A tree-walking interpreter that executes P-- code directly from the AST.
*   **`runtime.py`**: Provides the runtime environment for P--, including an `ExecutionOrchestrator` for managing memory, concurrency, and an interactive REPL. It also includes a `CompilationController` for potential dynamic JIT management.
*   **`memory_manager.py`**: Focus of current development for memory allocation, deallocation, and garbage collection.
*   **`concurrency.py`**: Provides building blocks (`ThreadManager`, `AsyncScheduler`) for concurrent execution, utilized by `runtime.py`.
*   **`cli.py`**: The main command-line interface for P++. Supports `compile`, `interpret`, and `run` commands for `.pypp` files.
*   **`shell.py`**: An interactive REPL for P++, allowing code execution in interpreter or JIT compiler mode.
*   **`cpp_adapter.py` / `interop.py`**: Modules likely intended for C++ interoperability (details to be further developed).
*   **`pyppc.py`**: An alternative CLI for the compiler, currently a placeholder.

Other notable files:
*   **`.pypp` files**: Source files written in the P-- language (e.g., `test.pypp`, `myfile.pypp`).
*   **`output.ll`, `output.o`, `output_executable`**: Example output files from the AOT compilation process.

## How to Use P--

P-- source files use the `.pypp` extension.

**1. Command-Line Interface (`cli.py`)**

*   **Compile and Run (AOT)**:
    ```bash
    python cli.py compile <your_file.pypp>
    ```
    This compiles the file to LLVM IR, then to an object file, and finally links it into an executable named `output_executable`. (Note: The current implementation in `compiler.py` builds the executable but does not automatically run it).

*   **Interpret**:
    ```bash
    python cli.py interpret <your_file.pypp>
    ```
    This tokenizes, parses, and directly executes the code using the AST interpreter.

*   **Run via Interactive Runtime**:
    ```bash
    python cli.py run <your_file.pypp>
    ```
    This loads the code into the `interactive_runtime` which can then execute it (defaulting to interpreted mode).

**2. Interactive Shell (`shell.py`)**

*   Start the shell:
    ```bash
    python shell.py
    ```
*   You'll see a `P++>` prompt.
*   **Modes**:
    *   Interpreter mode (default): Executes code directly.
    *   Compiler (JIT) mode: Compiles and runs code on-the-fly using LLVM JIT.
*   **Commands**:
    *   `:mode interpreter` - Switch to interpreter mode.
    *   `:mode compiler` - Switch to JIT compiler mode.
    *   `<P-- code>` - Enter any P-- code to execute.
    *   `:exit` - Quit the shell.

## Compilation and Interpretation Pipeline

**Compilation Pipeline (AOT/JIT):**
1.  **Tokenization** (`tokenizer.py`): Source code -> Tokens
2.  **Parsing** (`parser.py`): Tokens -> Abstract Syntax Tree (AST)
3.  **AST to IR Translation** (`ast_to_ir.py`): AST -> Custom Intermediate Representation (IR)
4.  **IR Optimization** (`ir_optimizer.py`): Custom IR -> Optimized IR
5.  **LLVM IR Generation** (`llvm_codegen.py`): Optimized IR -> LLVM IR
6.  **Execution**:
    *   **AOT**: LLVM IR -> `.ll` file -> `llc` -> `.o` file -> `clang` -> Executable.
    *   **JIT**: LLVM IR -> LLVM MCJIT Engine -> In-memory machine code execution.

**Interpretation Pipeline:**
1.  **Tokenization** (`tokenizer.py`): Source code -> Tokens
2.  **Parsing** (`parser.py`): Tokens -> AST
3.  **AST Execution** (`interpreter.py`): The interpreter walks the AST and executes operations.

The **Interactive Runtime (`runtime.py`)** can use either the interpretation pipeline or the JIT compilation pipeline, managed by an `ExecutionOrchestrator`.

## Future Goals

Beyond the current focus on memory management, potential future directions include:
*   Expanding language features (more data types, control flow, standard library).
*   Enhancing concurrency features.
*   Improving C++ interoperability.
*   Adding more advanced code optimizations.
*   Developing richer developer tooling (debugger, package manager).
