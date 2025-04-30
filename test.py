import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

robot_pos = np.array([0.0, 0.0])
obs_pos = np.array([1, -2])
robot_vel = np.array([0.5, -0.5])
obs_vel = np.array([-0.5, 0.5])
robot_radius = 0.5
obs_radius = 0.7
a_max = 3.0
dt = 0.1
Th = 10
action_space = spaces.Box(
            low=-a_max, high=a_max, shape=(2,), dtype=np.float32
        )

combined_radius = robot_radius + obs_radius
rel_pos = obs_pos - robot_pos
angle = np.arctan2(rel_pos[1], rel_pos[0])
VOh = np.linalg.norm(rel_pos) / Th

print(angle)

alphs = np.arcsin(combined_radius / np.linalg.norm(rel_pos))
print(alphs)

print(angle-alphs)
print(angle+alphs)


num_samples = 10000
random_accels = np.random.uniform(
            low=action_space.low, 
            high=action_space.high, 
            size=(num_samples, 2)
        )

reachable_velocities = robot_vel + random_accels * dt
print(reachable_velocities.shape)

# plt.plot(reachable_velocities[:, 0], reachable_velocities[:, 1], 'x', color='purple', markersize=5)
# plt.show()

rel_vel = robot_vel - obs_vel

theta = np.arctan2(rel_vel[1], rel_vel[0])

if theta <= angle+alphs and theta >= angle-alphs:
    print("In the cone")

print("VOh:", VOh)
filtered_velocities = []
not_filtered_velocities = []
for i in range(num_samples):
    theta = np.arctan2(reachable_velocities[i][1], reachable_velocities[i][0])
    
    if theta <= angle+alphs and theta >= angle-alphs and np.linalg.norm(reachable_velocities[i]) >= VOh:
        filtered_velocities.append(reachable_velocities[i])
    else:
        not_filtered_velocities.append(reachable_velocities[i])

filtered_velocities = np.array(filtered_velocities)
not_filtered_velocities = np.array(not_filtered_velocities)
print("Filtered Velocities:", filtered_velocities.shape)
print("Not in Cone Velocities:", not_filtered_velocities.shape)



plt.plot(filtered_velocities[:, 0], filtered_velocities[:, 1], 'x', color='purple', markersize=5)
plt.plot(not_filtered_velocities[:, 0], not_filtered_velocities[:, 1], 'x', color='red', markersize=5)
plt.plot(robot_pos[0], robot_pos[1], 'o', color='blue', markersize=10)
plt.plot(obs_pos[0], obs_pos[1], 'o', color='green', markersize=10)
plt.plot(obs_pos[0] + obs_radius * np.cos(np.linspace(0, 2*np.pi, 100)), 
         obs_pos[1] + obs_radius * np.sin(np.linspace(0, 2*np.pi, 100)), color='green')
plt.plot(robot_pos[0] + robot_radius * np.cos(np.linspace(0, 2*np.pi, 100)),
            robot_pos[1] + robot_radius * np.sin(np.linspace(0, 2*np.pi, 100)), color='blue')
plt.xlim(-1, 2)
plt.ylim(1, -3)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# angle += 2*np.pi
# print(angle)