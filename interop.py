# interop.py

class PPlusException(Exception):
    def __init__(self, message, exception_type="BindingError"):
        self.message = message
        self.exception_type = exception_type
        super().__init__(f"{exception_type}: {message}")

class PPlusValue:
    def __init__(self, type_info, data):
        self.type_info = type_info
        self.data = data

    def __repr__(self):
        return f"PPlusValue(type_info={self.type_info}, data={self.data})"

class LanguageAdapter:
    """Interface for language adapters."""
    def call_function(self, func_name, args):
        raise NotImplementedError

    def register_function(self, func_name, function):
        raise NotImplementedError

    def translate_exception(self, exception):
        raise NotImplementedError

class PythonAdapter(LanguageAdapter):
    def __init__(self):
        self.functions = {}

    def call_function(self, func_name, args):
        try:
            if func_name in self.functions:
                func = self.functions[func_name]
                # Convert PPlusValue arguments to native Python values
                native_args = [arg.data if isinstance(arg, PPlusValue) else arg for arg in args]
                result = func(*native_args)
                # Wrap the result as a PPlusValue (assuming type is inferred)
                return PPlusValue(type(result).__name__, result)
            else:
                raise PPlusException(f"Function '{func_name}' is not registered", "BindingError")
        except Exception as e:
            raise self.translate_exception(e)

    def register_function(self, func_name, function):
        self.functions[func_name] = function

    def translate_exception(self, exception):
        # Simple translation: wrap in a PPlusException
        return PPlusException(str(exception), exception_type=type(exception).__name__)

class BindingManager:
    def __init__(self):
        self.adapters = {}

    def register_adapter(self, name, adapter: LanguageAdapter):
        self.adapters[name] = adapter
        print(f"[BindingManager] Registered adapter '{name}'.")

    def call_function(self, adapter_name, func_name, args):
        if adapter_name not in self.adapters:
            raise PPlusException(f"Adapter '{adapter_name}' is not registered", "BindingError")
        adapter = self.adapters[adapter_name]
        return adapter.call_function(func_name, args)

    def convert_to_pplus(self, value):
        # For demonstration, wrap value in PPlusValue.
        return PPlusValue(type(value).__name__, value)

    def convert_from_pplus(self, pplus_value: PPlusValue):
        # Extract the underlying value.
        return pplus_value.data

# Testing the interoperability system.
if __name__ == "__main__":
    # Create a binding manager.
    bm = BindingManager()
    
    # Create a Python adapter and register it.
    py_adapter = PythonAdapter()
    bm.register_adapter("python", py_adapter)
    
    # Register a sample function in the Python adapter.
    def greet(name):
        return f"Hello, {name}!"
    
    py_adapter.register_function("greet", greet)
    
    # Call the 'greet' function via the binding manager.
    # Prepare argument as a PPlusValue.
    arg = bm.convert_to_pplus("World")
    result = bm.call_function("python", "greet", [arg])
    print("Function call result:", result)
