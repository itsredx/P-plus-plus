import subprocess
import time

# Compile P++ code to machine code
#subprocess.run(["./compiler", "test.pypp", "-o", "output"])

# Run compiled code and measure time
start_time = time.time()
print("helo world")
x = 15
print(3 + 9)
print(3 + x)
y = 6 / 2
print(y) 
end_time = time.time()

print(f"Compiled Execution Time: {end_time - start_time:.6f} seconds")
