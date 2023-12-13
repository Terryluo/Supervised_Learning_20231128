import numpy as np
from scipy.optimize import fsolve

def equation(n):
    return 260000 * (1.0245)**n - 2500 * 12 * n

# Initial guess for the solution
initial_guess = 1

# Using fsolve to find the root
# solution = fsolve(equation, initial_guess)


while (initial_guess < 40):
    print("The value of n is approximately:", initial_guess, equation(initial_guess))
    initial_guess += 1