import numpy as np
import matplotlib.pyplot as plt
import torch

def restriction(uh,h):
    """
    Input: 2D matrix (uh) of finer grid h, size (1/h)+1, boundary values 0
    Output: 2D matrix (u2h) of coarser grid 2h, size (1/2h)+1, boundary values 0
    """
    n = int(1/(2*h))+1  # size of u2h
    u2h = np.zeros((n,n))
    for i in range(1,n-1):
        for j in range(1,n-1):
            part1 = uh[2*i,2*j]
            part2 = uh[2*i,2*j+1]+uh[2*i,2*j-1]+uh[2*i+1,2*j]+uh[2*i-1,2*j]
            part3 = uh[2*i+1,2*j+1]+uh[2*i+1,2*j-1]+uh[2*i-1,2*j+1]+uh[2*i-1,2*j-1]
            u2h[i,j] = part1/4+part2/8+part3/16

    return u2h

def interpolation(u2h,h):
    """
    Input: 2D matrix (u2h) of coarser grid 2h, size (1/2h)+1, boundary values 0
    Output: 2D matrix (uh) of finer grid h, size (1/h)+1, boundary values 0
    """
    n = int(1/h)+1  # size of uh
    uh = np.zeros((n,n))
    for i in range(2,n,2): # start from 2, even numbers
        for j in range(2,n,2): # start from 2, even numbers
            uh[i,j]=u2h[int(i/2),int(j/2)]
    for i in range(2,n,2): # start from 2, even numbers
        for j in range(1,n,2): # start from 1, odd numbers
            uh[i,j]=(u2h[int(i/2),int(j/2)]+u2h[int(i/2),int(j/2)+1])/2
    for i in range(1,n,2): # start from 1, odd numbers
        for j in range(2,n,2): # start from 2, even numbers
            uh[i,j]=(u2h[int(i/2),int(j/2)]+u2h[int(i/2)+1,int(j/2)])/2
    for i in range(1,n,2): # start from 1, odd numbers
        for j in range(1,n,2): # start from 1, odd numbers
            uh[i,j]=(u2h[int(i/2),int(j/2)]+u2h[int(i/2)+1,int(j/2)]+u2h[int(i/2),int(j/2)+1]+u2h[int(i/2)+1,int(j/2)+1])/4
    return uh

def gsiter(u_init,f,h):
    """
    A Gauss-Seidel iterate step to solve Poisson problem \Delta u=f.
    Input:
        h: grid size
        u: initial guess of the solution, size (1/h)+1, boundary values 0
        f: right-hand-side function, size (1/h)+1
    Output:
        v: the approximated solution after 1 G-S step
    """
    n = int(1/h)+1
    u = u_init.copy()
    for i in range(1,n-1):
        for j in range(1,n-1):
            u[i,j]=(u[i-1,j]+u[i,j-1]+u[i+1,j]+u[i,j+1]-h**2*f[i,j])/4
    return u

def laplacian(u,h):
    """
    Finite difference version of Laplacian operator.
    Input:
        h: grid size
        u: Discretized function, a matrix of size (1/h)+1
    Output:
        v: Discretized function, a matrix of size (1/h)+1
    """
    n = int(1/h)+1
    v = np.zeros((n,n))
    for i in range(1,n-1):
        for j in range(1,n-1):
            v[i,j] = (-4*u[i,j]+u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1])/h**2
    return v

def f_generator(h):
    """
    Generate the discretized function f.
    Input: Grid size h
    Output: A matrix f, size (1/h)+1
    """
    size_h = int(1/h)+1
    x = np.linspace(0, 1, size_h)
    y = np.linspace(0, 1, size_h)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return np.sin(np.pi * xx) * np.sin(np.pi * yy)

def true_solution(h):
    """
    Generate the discretized true solution of Poisson problem.
    Input: Grid size h
    Output: A matrix u_true, size (1/h)+1
    """
    size_h = int(1/h)+1
    x = np.linspace(0, 1, size_h)
    y = np.linspace(0, 1, size_h)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return -np.sin(np.pi * xx) * np.sin(np.pi * yy)/(2*np.pi**2)    

def vcycle(u_init, h):
    """
    One simple v-cycle iterate step, grid h -> 2h -> h.
    Input:
        u_init: initial guess of the solution, size (1/h)+1
        h: grid size
    Output:
        u: updated approximated solution
    """
    f = f_generator(h)
    num_iterate = 3
    uh = u_init.copy()

    # Step 1
    for _ in range(num_iterate):
        uh = gsiter(uh, f, h)

    # Step 2
    resh = f - laplacian(uh,h)
    res2h = restriction(resh,h)

    # Step 3
    size_2h = int(1/(2*h))+1
    e2h_init = np.zeros((size_2h,size_2h))
    e2h = e2h_init.copy()
    for _ in range(num_iterate):
        e2h = gsiter(e2h,res2h,2*h)
    
    # Step 4
    eh = interpolation(e2h,h)
    uh = uh+eh

    # Step 5
    for _ in range(num_iterate):
        uh = gsiter(uh, f, h)

    return uh    


if __name__=='__main__':

    h = 1/128
    size_h = int(1/h)+1
    u = np.zeros((size_h,size_h))
    N_iter = 1000
    u_true = true_solution(h)
    print('Initial error is',np.linalg.norm(u_true)/size_h)
    error_list = []

    for iter in range(N_iter):
        u = vcycle(u, h)
        error = np.linalg.norm(u-u_true)/size_h
        print("Step",iter,"error is",error)
        error_list.append(error)

    # Plot the results
    # Create a 3D surface plot
    x = torch.linspace(0, 1, size_h)
    y = torch.linspace(0, 1, size_h)
    x, y = torch.meshgrid(x, y, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x, y, u_true, cmap='viridis', label='True solution', alpha = 1)
    ax = fig.add_subplot(122, projection='3d')
    # ax.plot_surface(x, t, U, cmap='viridis')
    ax.plot_surface(x, y, u, cmap='plasma', label='Predicted Solution', alpha = 1)

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title('Plot of the true/numerical solution')
    # ax.legend()

    # Show the plot
    plt.savefig('plotmgsolver-v2.png')

    # Plot the training error (loss) over epochs
    fig = plt.figure()
    # plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
    plt.plot(range(1, len(error_list) + 1), error_list, label='L^2 Error')
    plt.xlabel('Iterates')
    plt.ylabel('Error')
    plt.title('Error over Iterations')
    plt.legend()
    plt.grid()
    plt.savefig('errorcurve_mgsolver.png')

    # Plot the log training error (loss) over epochs
    fig = plt.figure()
    # plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
    plt.plot(range(1, len(error_list) + 1), -np.log(error_list), label='-log(L^2 Error)')
    plt.xlabel('Iterates')
    plt.ylabel('-log(Error)')
    plt.title('Log of Error over Iterations')
    plt.legend()
    plt.grid()
    plt.savefig('logerrorcurve_mgsolver.png')