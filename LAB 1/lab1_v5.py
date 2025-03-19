import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def function(x, y):
    return 2*np.sin(x)+3*np.cos(y)

def function_der_fx(x):
    return 2*np.cos(x)

def function_der_fy(y):
    return -3 * np.sin(y) 

def next_x(x,learning_rate):
    return x - function_der_fx(x) * learning_rate

def next_y(y, learning_rate):
    return y - function_der_fy(y) * learning_rate

def next_point(point,learning_rate):
    return [next_x(point[0],learning_rate),next_y(point[1],learning_rate)]

def gradient_descent(initial_guess, learning_rate, tol=1e-6, max_iter=1000):
    """
    Gradient descent algorithm
    
    Parameters:
    - initial_guess: initial 2D coordinate vector
    - learning_rate: learning rate
    - tol: tolerance, convergence criteria
    - max_iter: maximum number of iterations

    """
    iterations = 0
    guess = initial_guess
    diff = np.inf
    path = [initial_guess]
    while iterations < max_iter and diff > tol:
        iterations += 1
        new_guess = next_point(guess,learning_rate)
        diff = abs(function(guess[0],guess[1]) - function(new_guess[0],new_guess[1]))
        path.append(new_guess)
        print(new_guess, diff)
        guess = new_guess

    return guess, iterations, np.array(path)


def visualize(path):
    """
    Visualization function: creates 3D plot of the function. Use colors to show the Z-coordinate
    """
    #ADjust the range to see a bigger chunk of the plot
    range = 4
    x0, y0 = path[0]
    x = np.linspace(x0 - range, x0 + range, 100)
    y = np.linspace(y0 - range, y0 + range, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6)
    x_path = path[:, 0]
    y_path = path[:, 1]
    z_path = function(x_path, y_path)
    
    ax.scatter(x_path, y_path, z_path, color='r', s=50, label="Iteration Points")
    ax.plot(x_path, y_path, z_path, color='r', linestyle='--', label="Descent Path")
    
    ax.set_title("Gradient Descent Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.legend()
    
    plt.show()



initial_guess_1 = [1.0, 2.0]
learning_rate_1 = 0.1
minimum_1, iterations_1, path = gradient_descent(initial_guess_1, learning_rate_1)
visualize(path)

print(f"Minimum approximation with initial guess {initial_guess_1}: {minimum_1}, Iterations: {iterations_1}")
