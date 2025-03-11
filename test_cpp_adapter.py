from interop import BindingManager, PPlusValue, PPlusException
from cpp_adapter import CppAdapter

def main():
    bm = BindingManager()
    try:
        # Load the C++ shared library.
        cpp_adapter = CppAdapter("/workspaces/P-/libsample.so")
        bm.register_adapter("cpp", cpp_adapter)
        
        # Prepare arguments as PPlusValue objects.
        arg1 = bm.convert_to_pplus(10.0)
        arg2 = bm.convert_to_pplus(20.0)
        
        # Call the C++ 'add' function via the binding manager.
        result = bm.call_function("cpp", "add", [arg1, arg2])
        print("Result from C++ add function:", result)
    except PPlusException as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
