import time
from memory_manager import MemoryManager
from concurrency import ThreadManager, AsyncScheduler

class ExecutionOrchestrator:
    def __init__(self, memory_manager, thread_manager, async_scheduler):
        self.memory_manager = memory_manager
        self.thread_manager = thread_manager
        self.async_scheduler = async_scheduler
        self.mode = "interpreted"  # Default mode
        self.running_code = None

    def execute(self, code):
        """Execute the given P++ code."""
        print("[ExecutionOrchestrator] Starting execution...")
        self.running_code = code  # Store running code
        print(f"Executing code:\n{code}\n")
        time.sleep(0.5)  # Simulating execution time

    def hot_reload(self):
        """Reloads the currently running code without restarting."""
        if self.running_code:
            print("\n[ExecutionOrchestrator] Hot Reloading...")
            self.execute(self.running_code)
        else:
            print("[Warning] No code to reload.")

    def hot_restart(self):
        """Restarts execution with a fresh state."""
        print("\n[ExecutionOrchestrator] Hot Restarting...")
        self.memory_manager.clear()  # Reset memory
        self.thread_manager.terminate_all()  # Stop all threads
        self.async_scheduler.reset()  # Reset async tasks
        time.sleep(0.5)
        if self.running_code:
            self.execute(self.running_code)

    def switch_execution_mode(self, mode):
        """Switch execution mode dynamically."""
        self.mode = mode
        print(f"[ExecutionOrchestrator] Switched execution mode to: {mode}")

class CompilationController:
    def monitor_performance(self):
        print("[CompilationController] Monitoring performance...")

    def trigger_jit(self, ir):
        print("[CompilationController] Triggering JIT compilation...")
        return "MachineCode"  # Placeholder for generated machine code

    def deoptimize(self, region):
        print(f"[CompilationController] Deoptimizing region: {region}")

    def set_optimization_level(self, level):
        print(f"[CompilationController] Setting optimization level to: {level}")

def interactive_runtime(code=None, comp_controller=None):
    print("\n[P++ Interactive Runtime] Type your commands or enter P++ code directly.")
    print("Commands: run <file>, reload, restart, mode <jit|interpreted>, exit\n")

    exec_orchestrator = ExecutionOrchestrator(
        MemoryManager(), ThreadManager(), AsyncScheduler()
    )

    mode = "interpreted"  # Default execution mode
    interpreter = Interpreter()  # Interpreter instance

    # Execute initial file if provided
    if code:
        print("[Info] Executing initial file in", mode, "mode...\n")
        execute_code(code, mode, interpreter)

    while True:
        user_input = input(f"[{mode.upper()} MODE] >>> ").strip()

        if not user_input:
            continue

        if user_input == "exit":
            print("[Info] Exiting runtime.")
            break
        elif user_input.startswith("run "):
            file_path = user_input[4:].strip()
            try:
                with open(file_path, "r") as f:
                    file_code = f.read()
                    print(f"[Info] Executing {file_path} in {mode} mode...\n")
                    execute_code(file_code, mode, interpreter)
            except FileNotFoundError:
                print(f"[Error] File not found: {file_path}")
        elif user_input == "reload":
            if code:
                print("[Info] Reloading initial file in", mode, "mode...\n")
                execute_code(code, mode, interpreter)
            else:
                print("[Error] No initial file to reload.")
        elif user_input == "restart":
            print("[Info] Restarting runtime...\n")
            interactive_runtime(code, comp_controller)  # Recursively restart
            break
        elif user_input.startswith("mode "):
            new_mode = user_input[5:].strip()
            if new_mode in ["jit", "interpreted"]:
                mode = new_mode
                print(f"[Info] Switched to {mode.upper()} mode.")
            else:
                print("[Error] Invalid mode. Use 'jit' or 'interpreted'.")
        else:
            print(f"[{mode.upper()}] Executing P++ command...")
            execute_code(user_input, mode, interpreter)


def execute_code(code, mode, interpreter):
    """Execute code based on the selected mode."""
    if mode == "interpreted":
        try:
            tokens = tokenize(code)
            parser = Parser(tokens)
            ast = parser.parse_program()
            interpreter.interpret(ast)
        except Exception as e:
            print(f"[Interpreted Error] {e}")
    elif mode == "jit":
        try:
            compile_and_run(code, aot=False)  # Run JIT compilation
        except Exception as e:
            print(f"[JIT Error] {e}")




if __name__ == "__main__":
    # Initialize runtime components.
    mem_manager = MemoryManager()
    thread_manager = ThreadManager()
    async_scheduler = AsyncScheduler()

    # Create the Execution Orchestrator.
    exec_orch = ExecutionOrchestrator(mem_manager, thread_manager, async_scheduler)
    
    # Create a Compilation Controller.
    comp_controller = CompilationController()

    # Start the interactive runtime.
    interactive_runtime(exec_orch, comp_controller)
