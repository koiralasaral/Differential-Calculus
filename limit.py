import sympy as sp

# Define the variable
x = sp.symbols('x')

# Define the function
f = (x * sp.cos(x) - sp.log(1 + x)) / x**2

# Compute the limit
limit_result = sp.limit(f, x, 0)
print("The limit is:", limit_result)