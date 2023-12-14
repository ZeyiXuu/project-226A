import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
N = 128  # Number of grid points along each axis
L = 1.0  # Length of the domain [0, L] in both x and y directions
dx = L / (N - 1)  # Grid spacing
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize the error record
error_list = []
error_from_true = []

# Define the source term function f(x, y) = sin(pi * x) * sin(pi * y)
def source_term(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Calculate the source term values
f = source_term(X, Y)

# Initialize the solution matrix u and set boundary conditions to zero
u = np.zeros((N, N))

# Define the true solution for comparison
def true_solution(x, y):
    return -np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**2)

# Implement the Finite Difference method
def solve_poisson_equation(u, f, dx, max_iterations=100000, tolerance=1e-8):
    for iteration in range(max_iterations):
        u_old = u.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                u[i, j] = 0.25 * (u_old[i + 1, j] + u_old[i - 1, j] + u_old[i, j + 1] + u_old[i, j - 1] - dx**2 * f[i, j])
        
        # Record the error
        error_list.append(np.linalg.norm(u - u_old))
        error_from_true.append(np.linalg.norm(u-true_solution_values))
        print("Step",iteration,"error is",np.linalg.norm(u-true_solution_values)/N)

        # Check for convergence
        if np.linalg.norm(u - u_old) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
    
    return u

# Calculate the true solution for comparison
true_solution_values = true_solution(X, Y)

# Solve the Poisson equation
solution = solve_poisson_equation(u, f, dx)

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, y, true_solution_values, cmap='viridis', label='True solution', alpha = 1)
ax = fig.add_subplot(122, projection='3d')
# ax.plot_surface(x, t, U, cmap='viridis')
ax.plot_surface(x, y, solution, cmap='plasma', label='Predicted Solution', alpha = 1)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Plot of the true/numerical solution')
# ax.legend()

# Show the plot
plt.savefig('plotfdsolver-v2.png')


# Plot the training error (loss) over epochs
fig = plt.figure()
# plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
plt.plot(range(1, len(error_list) + 1), error_list, label='L^2 error')
plt.xlabel('Iterates')
plt.ylabel('Error')
plt.title('Error over Iterations')
plt.legend()
plt.grid()
plt.savefig('errorcurve_fdsolver.png')

# Plot the -log of the training error (loss) over epochs
fig = plt.figure()
# plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
plt.plot(range(1, len(error_list) + 1), -np.log(error_list), label='L^2 error')
plt.xlabel('Iterates')
plt.ylabel('-Log Error')
plt.title('-Log of Error over Iterations')
plt.legend()
plt.grid()
plt.savefig('logerrorcurve_fdsolver.png')

# print final error
print('Error:',np.linalg.norm(true_solution_values-solution)/N)

# Plot the training error (loss) over epochs
fig = plt.figure()
# plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
plt.plot(range(1, len(error_from_true) + 1), error_from_true, label='Error from true solution')
plt.xlabel('Iterates')
plt.ylabel('Error')
plt.title('Error from True Solution over Iterations')
plt.legend()
plt.grid()
plt.savefig('trueerrorcurve_fdsolver.png')