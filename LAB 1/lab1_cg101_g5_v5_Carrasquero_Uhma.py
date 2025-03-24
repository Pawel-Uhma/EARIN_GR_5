import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def function(x, y):
    return 2*np.sin(x)+3*np.cos(y)

def function_der_fx(x):
    """ 
    derivative of F(X,Y) with respect to Y
    """
    return 2*np.cos(x)

def function_der_fy(y):
    """ 
    derivative of F(X,Y) with respect to X
    """
    return -3 * np.sin(y) 

def next_x(x,learning_rate):
    return x - function_der_fx(x) * learning_rate

def next_y(y, learning_rate):
    return y - function_der_fy(y) * learning_rate

def next_point(point,learning_rate):
    """
    Combining next x guess and next Y guess into a 2d point
    """
    return [next_x(point[0],learning_rate),next_y(point[1],learning_rate)]

def gradient_descent(initial_guess, learning_rate, tol=1e-6, max_iter=1000):
    """
    Gradient descent algorithm
    
    Parameters:
    - initial_guess
    - learning_rate
    - tol: tolerance
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
        guess = new_guess

    return guess, iterations, np.array(path)


def visualize(path):
    """
    Creates 3D plot of the function
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5 , 5, 100)
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

def visualize_multiple(initial_points, learning_rates, tol=1e-6, max_iter=1000):
    """
    Run gradient descent for each (initial_point, learning_rate) pair,
    and visualize all paths on the same 3D grid

    Parameters:
    - initial_points: 2d list of 2d points
    - learning_rates: 2d list corresponding to each initial point
    - tol: tolerance
    - max_iter: maximum number of iterations 
    """
    if len(initial_points) != len(learning_rates):
        raise ValueError("initial_points and learning_rates must have the same length")

    # calculate the grid range 
    xs = [pt[0] for pt in initial_points]
    ys = [pt[1] for pt in initial_points]
    x_min, x_max = min(xs) - 3, max(xs) + 3
    y_min, y_max = min(ys) - 3, max(ys) + 3

    # create grid
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6)

    # colors for different initial points
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:len(initial_points)]

    for idx, (initial, lr) in enumerate(zip(initial_points, learning_rates)):
        minimum, iterations, path = gradient_descent(initial, lr, tol=tol, max_iter=max_iter)
        path = np.array(path)
        x_path = path[:, 0]
        y_path = path[:, 1]
        z_path = function(x_path, y_path)
        
        ax.plot(x_path, y_path, z_path, color=colors[idx], linestyle='--',
                marker='o', label=f"{initial}, lr {lr}")
        
    
    ax.set_title("Gradient Descent Paths")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.legend()
    plt.show()



#MAIN ALGORITHM
initial_point = [1,2]
learning_rate = 0.1
minimum, iterations, path = gradient_descent(initial_point,learning_rate)
print(f"Minimum approximation with initial guess {initial_point}: {minimum}, Iterations: {iterations}")
visualize(path)

# TESTING MULTIPLE POINTS
initial_points = [[1, 0.1], [0, 0], [-2, -1], [0, 3]]
learning_rates = [0.5, 0.2, 0.4, 0.3]
visualize_multiple(initial_points, learning_rates)
