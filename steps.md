🟢 Iteration 2: Intermediate Representation (IR)
Goal: Translate AST into a structured IR
✅ Define the IR structure
✅ Implement an IR generator from the AST
✅ Add error handling for invalid AST inputs
✅ Test the IR generation with sample inputs

🟢 Iteration 3: Code Generation & Optimization
Goal: Convert IR into LLVM IR and optimize
✅ Implement LLVM IR generation
✅ Apply basic optimizations (constant folding, dead code elimination)
✅ Compile and run a simple test program

Iteration 4: Implement Memory Management Strategies

Goals:

    Design Manual Memory Management:
    Implement allocation and deallocation functions that provide fine-grained control over memory (similar to C/C++), including RAII patterns.
    Integrate Optional Garbage Collection:
    Develop a garbage collector (e.g., generational mark-and-sweep or incremental GC) that can manage memory automatically when desired.
    Ensure Correctness and Performance:
    Define clear ownership rules, memory safety checks, and integrate these into the runtime environment.