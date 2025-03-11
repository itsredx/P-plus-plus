import ctypes
from interop import LanguageAdapter, PPlusException, PPlusValue

class CppAdapter(LanguageAdapter):
    def __init__(self, lib_path):
        try:
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            raise PPlusException(f"Failed to load C++ library: {e}", "LibraryLoadError")
        self.functions = {}

    def call_function(self, func_name, args):
        try:
            # If function is not already cached, try to retrieve it from the shared library.
            if func_name not in self.functions:
                func = getattr(self.lib, func_name, None)
                if not func:
                    raise PPlusException(f"Function '{func_name}' not found in the C++ library", "FunctionNotFound")
                # For demonstration, assume the function returns a double and takes double arguments.
                func.restype = ctypes.c_double
                func.argtypes = [ctypes.c_double] * len(args)
                self.functions[func_name] = func
            else:
                func = self.functions[func_name]
            
            # Convert PPlusValue arguments (or native numbers) to ctypes.c_double.
            native_args = [ctypes.c_double(arg.data) if isinstance(arg, PPlusValue) else ctypes.c_double(arg) for arg in args]
            result = func(*native_args)
            return PPlusValue("double", result)
        except Exception as e:
            raise self.translate_exception(e)

    def register_function(self, func_name, function):
        # For C/C++ adapter, registration is typically not done from Python.
        self.functions[func_name] = function

    def translate_exception(self, exception):
        # Wrap native exceptions as PPlusExceptions.
        return PPlusException(str(exception), exception_type=type(exception).__name__)
