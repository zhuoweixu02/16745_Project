import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def extract_path_from_image(image_path, show=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")
    contour = max(contours, key=lambda c: len(c))
    path = contour[:, 0, :]  # shape (N, 2)
    if show:
        display = img.copy()
        for (x, y) in path:
            cv2.circle(display, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("Path", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return np.array(path)

def generate_motion(path, v0=50, t0=2, dt=0.05):
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    total_length = np.sum(distances)
    accel_distance = 0.5 * v0 * t0
    remaining_distance = total_length - accel_distance
    remaining_time = remaining_distance / v0
    total_time = t0 + remaining_time
    steps = int(total_time / dt)

    # Calculate cumulative distances and time array
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    times = np.linspace(0, total_time, steps)

    # Calculate velocity profile
    velocity_profile = np.piecewise(
        times,
        [times < t0, times >= t0],
        [lambda t: (v0 / t0) * t,
         lambda t: v0]
    )

    # Calculate distance profile
    distance_profile = np.piecewise(
        times,
        [times < t0, times >= t0],
        [lambda t: 0.5 * (v0 / t0) * t**2,
         lambda t: accel_distance + v0 * (t - t0)]
    )

    # Calculate positions
    positions = []
    for d in distance_profile:
        idx = np.searchsorted(cumulative_dist, d)
        if idx >= len(path) - 1:
            idx = len(path) - 2
        ratio = (d - cumulative_dist[idx]) / distances[idx]
        point = (1 - ratio) * path[idx] + ratio * path[idx + 1]
        positions.append(point)
    
    return np.array(positions), velocity_profile

def animate_path(path, vehicle_positions):
    fig, ax = plt.subplots()
    ax.plot(path[:, 0], path[:, 1], 'k-', linewidth=1)
    vehicle_dot, = ax.plot([], [], 'ro', markersize=5)
    ax.set_xlim(path[:, 0].min() - 10, path[:, 0].max() + 10)
    ax.set_ylim(path[:, 1].min() - 10, path[:, 1].max() + 10)
    ax.set_aspect('equal')
    ax.set_title("Vehicle Path Animation")

    def init():
        vehicle_dot.set_data([], [])
        return vehicle_dot,

    def update(frame):
        x = vehicle_positions[frame][0]
        y = vehicle_positions[frame][1]
        vehicle_dot.set_data([x], [y])
        return vehicle_dot,

    ani = FuncAnimation(fig, update, init_func=init, 
                       frames=len(vehicle_positions), 
                       interval=50, blit=True)
    plt.show()

# Main
folder = "C:/16745_Optimal_Control_and_Reinforcement_Learning/16745_project/reference_generation/test/"
image_path = folder + "test1.png"
path = extract_path_from_image(image_path, show=True)
# split the path into 2 parts, 0.2 and 0.8
split_index = int(len(path) * 0.1)
path1 = path[:split_index]
path2 = path[split_index:]
# reconstruct the path with path2+path1
path = np.concatenate((path2, path1), axis=0)
# tare the path to start at (0,0)
path -= path[0]
dt = 0.05
vehicle_positions, velocity_profile = generate_motion(path, v0=50, t0=2, dt=0.05)
print("Vehicle positions:", vehicle_positions)
animate_path(path, vehicle_positions)

# based on the x and v, calculate the direction of the vehicle
diections = []
for i in range(len(vehicle_positions)-1):
    dx = vehicle_positions[i+1][0] - vehicle_positions[i][0]
    dy = vehicle_positions[i+1][1] - vehicle_positions[i][1]
    direction = np.arctan2(dy, dx)  # angle in radians
    diections.append(direction)
diections = np.array(diections)
# based on sirection , generate theta dot
theta_dot = []
for i in range(len(diections)-1):
    dtheta = diections[i+1] - diections[i]
    if dtheta > np.pi:
        dtheta -= 2 * np.pi
    elif dtheta < -np.pi:
        dtheta += 2 * np.pi
    theta_dot.append(dtheta / dt)
# save x,v, and direction, theta_dot to a pkl file
import pickle
with open(folder + "reference_data_rec.pkl", "wb") as f:
    pickle.dump({
        "x": vehicle_positions,
        "theta": diections,
        "v": velocity_profile,
        "omega": theta_dot,
    }, f)