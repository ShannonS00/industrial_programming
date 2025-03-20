import numba
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import time

def get_acceleration(X: np.ndarray) -> np.ndarray:
    """
    Compute gravitational acceleration for a system of N bodies.
    
    Parameters:
        X (np.ndarray): 2D array of shape (N, 3) containing the positions of the bodies.
    
    Returns:
        np.ndarray: 2D array of shape (N, 3) containing the accelerations of the bodies.
    """
    N = X.shape[0] 
    A = np.array([
        sum((X[j] - X[i]) / np.linalg.norm(X[j] - X[i])**3 
            for j in range(N) if i != j and np.linalg.norm(X[j] - X[i]) > 0)
        for i in range(N)
    ])
    return A


# Define the number of bodies
N = 100  # You can change this value

# Generate random positions for the bodies
X = np.random.randn(N, 3)  # Each row is (x, y, z) for a body

# Compute the accelerations and measure the execution time
start_time = time.time()
A = get_acceleration(X)
end_time = time.time()

# Print the results
#print("Positions:\n", X)
#print("Accelerations:\n", A)

# Print the results
print(f"Time taken for N={N}: {end_time - start_time:.6f} seconds")


def measure_execution_time(N_values):
    times = []
    for N in N_values:
        X = np.random.randn(N, 3)  # Generate random positions
        start_time = time.time()
        get_acceleration(X)
        end_time = time.time()
        times.append(end_time - start_time)
    return times

def plot_scaling(N_values, times):
    plt.figure()
    plt.loglog(N_values, times, marker='o', linestyle='-', label='Measured time')
    plt.loglog(N_values, (np.array(N_values)**2) * (times[0] / N_values[0]**2), linestyle='--', label='O(N²) reference')
    plt.xlabel('Number of bodies (N)')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.title('Scaling of Gravitational Acceleration Computation')
    plt.show()

# Run and plot results
N_values = [10, 20, 50, 100, 200, 500]  # Different values of N to test
execution_times = measure_execution_time(N_values)
plot_scaling(N_values, execution_times)