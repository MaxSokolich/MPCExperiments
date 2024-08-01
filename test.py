import numpy as np
import matplotlib.pyplot as plt

def generate_line(start, end, num_points=100):
    x = np.linspace(start[0], end[0], num_points)
    y = np.linspace(start[1], end[1], num_points)
    return np.column_stack((x, y))

def generate_circle(center, radius,theta_start, theta_end, cc, num_points=100):
    if cc :
        t = np.linspace(theta_start, theta_end, num_points)
    else:
        t = np.linspace(theta_end, theta_start, num_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.column_stack((x, y))

def generate_infinity(width, height, center, num_points=1100):
    cx, cy = center[0], center[1]
    r = height/2
    dx = (width-height)/2
 
    theta = np.arccos(r/dx)
    # Left circle
    left_circle_center = [cx - dx, cy]
    left_circle_radius = r
    start_theta_1 = theta
    end_theta_1 = 2*np.pi-theta
    left_circle_points = generate_circle(left_circle_center, left_circle_radius, start_theta_1, end_theta_1, True, num_points // 4)

    # Right circle
    right_circle_center = [cx +dx , cy]
    right_circle_radius = r
    start_theta_2 = np.pi -theta
    end_theta_2 = 2*np.pi-theta
    right_circle_points = generate_circle(right_circle_center, right_circle_radius,start_theta_2, end_theta_2, False, num_points // 4)

    # crossover line1
    line1_start = [left_circle_center[0] + r*np.cos(theta), left_circle_center[1] - r*np.sin(theta)]
    line1_end = [right_circle_center[0]-r*np.cos(theta), right_circle_center[1] + r*np.sin(theta)]
    line1_points = generate_line(line1_start, line1_end, num_points // 4)

    # crossover line2
    line2_start = [right_circle_center[0]-r*np.cos(theta), right_circle_center-r*np.sin(theta)]
    line2_end = [left_circle_center[0] + r*np.cos(theta), left_circle_center[1] + r*np.sin(theta)]
    line2_points = generate_line(line2_start, line2_end, num_points // 4)

    # Combine all segments
    path_points = np.concatenate((left_circle_points, line1_points, right_circle_points,line2_points))

    return path_points

def plot_path(path_points):
    plt.figure(figsize=(10, 6))
    plt.plot(path_points[:, 0], path_points[:, 1], marker='o')
    plt.title('Infinity-like Path Consisting of Lines and Circles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Example usage
width = 10
height = 5
center = (0, 0)
path_points = generate_infinity(width, height, center)
plot_path(path_points)
