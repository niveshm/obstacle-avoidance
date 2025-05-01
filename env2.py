import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Tuple, Dict, Optional

class DynamicObstacleEnv(gym.Env):
    """Environment for velocity obstacle-based motion planning in 2D"""
    
    def __init__(self, 
                 num_obstacles: int = 5, 
                 world_size: float = 10.0, 
                 dt: float = 0.1, 
                 a_max: float = 3.0,
                 v_max: float = 3.0,
                 robot_radius: float = 0.5,
                 obstacle_radius: float = 1,
                 Th: float = 10.0):
        super(DynamicObstacleEnv, self).__init__()
        
        # Environment parameters
        self.num_obstacles = num_obstacles
        self.world_size = world_size  # World is [-world_size, world_size] in both x, y
        self.dt = dt  # time step
        self.a_max = a_max  # maximum acceleration (m/s^2)
        self.v_max = v_max  # maximum velocity (m/s)
        self.Th = Th  # time horizon for VO (s)
        self.robot_radius = robot_radius
        self.obstacle_radius = [np.random.uniform(0.1, obstacle_radius) for _ in range(num_obstacles)]
        
        # Visualization
        self.fig = None
        self.ax = None
        self.robot_artist = None
        self.obstacle_artists = []
        self.vel_artists = []
        
        # Define spaces
        # Action = 2D acceleration (ax, ay)
        self.action_space = spaces.Box(
            low=-self.a_max, high=self.a_max, shape=(2,), dtype=np.float32
        )
        
        # Observation = robot (pos, vel) + each obstacle (pos, vel)
        obs_dim = 2 + 2 + num_obstacles * (2 + 2)  # robot pos(2) + vel(2) + obstacles (pos(2)+vel(2))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.robot_pos = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.obstacles_pos = np.zeros((num_obstacles, 2))
        self.obstacles_vel = np.zeros((num_obstacles, 2))
        self.goal_pos = np.zeros(2)
        
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        # Reset robot to center with random velocity
        self.robot_pos = np.random.uniform(-self.world_size/4, self.world_size/4, size=2)
        self.robot_vel = np.random.uniform(-0.5, 0.5, size=2)
        
        # Place goal in opposite quadrant
        self.goal_pos = np.random.uniform(-self.world_size/2, self.world_size/2, size=2)
        while np.linalg.norm(self.goal_pos - self.robot_pos) < self.world_size/2:
            self.goal_pos = np.random.uniform(-self.world_size/2, self.world_size/2, size=2)
        
        # Reset obstacles with random positions and velocities
        self.obstacles_pos = np.zeros((self.num_obstacles, 2))
        self.obstacles_vel = np.zeros((self.num_obstacles, 2))
        
        for i in range(self.num_obstacles):
            # Ensure obstacles don't spawn too close to robot or goal
            valid_position = False
            while not valid_position:
                pos = np.random.uniform(-self.world_size/2, self.world_size/2, size=2)
                if (np.linalg.norm(pos - self.robot_pos) > 2*(self.robot_radius + self.obstacle_radius[i]) and
                    np.linalg.norm(pos - self.goal_pos) > 2*(self.robot_radius + self.obstacle_radius[i])):
                    valid_position = True
                    self.obstacles_pos[i] = pos
                    self.obstacles_vel[i] = np.random.uniform(-self.v_max, self.v_max, size=2)
        
        return self._get_obs()
    
    def set_observation(self, obs: np.ndarray, obstacle_radius: Optional[np.ndarray]):
        """Set the observation of the environment"""
        self.robot_pos = obs[:2]
        self.robot_vel = obs[2:4]
        self.goal_pos = obs[4:6]
        self.obstacles_pos = obs[6:6 + 2 * self.num_obstacles].reshape(self.num_obstacles, 2)
        self.obstacles_vel = obs[6 + 2 * self.num_obstacles:].reshape(self.num_obstacles, 2)
        self.obstacle_radius = obstacle_radius

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        # print(self.robot_pos.shape, self.robot_vel.shape, self.goal_pos.shape, self.obstacles_pos.shape, self.obstacles_vel.shape)
        obs = np.concatenate([
            self.robot_pos, # 2*1
            self.robot_vel, #2*1
            self.goal_pos, # 2*1
            self.obstacles_pos.flatten(), # 2*num_obstacles
            self.obstacles_vel.flatten() # 2*num_obstacles
        ])
        return obs, self.obstacle_radius

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step"""
        # Clip action to max acceleration
        # print(action)
        action = np.clip(action, -self.a_max, self.a_max)
        # print(action)
        
        # Update robot state with acceleration constraints
        new_vel = self.robot_vel + action * self.dt
        vel_norm = np.linalg.norm(new_vel)
        if vel_norm > self.v_max:
            new_vel = new_vel * (self.v_max / vel_norm)
        
        self.robot_pos += new_vel * self.dt
        self.robot_vel = new_vel
        
        # Update obstacles (constant velocity model)
        self.obstacles_pos += self.obstacles_vel * self.dt
        
        # Bounce obstacles off walls with velocity reversal
        for i in range(self.num_obstacles):
            for dim in range(2):
                if abs(self.obstacles_pos[i, dim]) > self.world_size/2:
                    self.obstacles_vel[i, dim] *= -1
                    self.obstacles_pos[i, dim] = np.clip(
                        self.obstacles_pos[i, dim], 
                        -self.world_size/2, 
                        self.world_size/2
                    )
        
        # Check for collisions
        collision = False
        for i in range(self.num_obstacles):
            dist = np.linalg.norm(self.robot_pos - self.obstacles_pos[i])
            if dist < (self.robot_radius + self.obstacle_radius[i]):
                collision = True
                break
        
        # Calculate reward
        goal_dist = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = -goal_dist * 0.1  # reward for getting closer to goal
        if collision:
            reward -= 100.0  # large penalty for collision
        
        # Check termination conditions
        done = False
        if collision:
            print("Collision detected!")
            done = True
        if goal_dist < self.robot_radius:  # reached goal
            reward += 100.0
            done = True
        
        info = {
            'collision': collision,
            'goal_distance': goal_dist,
            'goal_reached': goal_dist < self.robot_radius
        }
        
        return self._get_obs(), reward, done, info

    def reached_goal(self) -> bool:
        """Check if the robot has reached the goal"""
        return np.linalg.norm(self.robot_pos - self.goal_pos) < self.robot_radius

    def render_obs(self, obs):
        for i in range(len(obs)):
            self.set_observation(*obs[i])
            self.render()

    def render(self, feasible_vels=None, mode: str = 'human'):
        """Render environment"""
        if mode == 'human':
            if self.fig is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_xlim(-self.world_size/2, self.world_size/2)
                self.ax.set_ylim(-self.world_size/2, self.world_size/2)
                self.ax.set_aspect('equal')
                self.ax.grid(True)
                
                # Create artists
                self.robot_artist = Circle(
                    (0, 0), self.robot_radius, fill=True, color='blue', alpha=0.8
                )
                self.ax.add_patch(self.robot_artist)
                
                self.goal_artist = Circle(
                    (0, 0), self.robot_radius, fill=True, color='green', alpha=0.5
                )
                self.ax.add_patch(self.goal_artist)
                
                for i in range(self.num_obstacles):
                    obstacle = Circle(
                        (0, 0), self.obstacle_radius[i], fill=True, color='red', alpha=0.6
                    )
                    self.obstacle_artists.append(obstacle)
                    self.ax.add_patch(obstacle)
                    
                    vel = self.ax.arrow(0, 0, 0, 0, color='orange', width=0.05)
                    self.vel_artists.append(vel)
                
                self.robot_vel_artist = self.ax.arrow(
                    0, 0, 0, 0, color='blue', width=0.05
                )
            
            # Update positions
            self.robot_artist.center = self.robot_pos
            self.goal_artist.center = self.goal_pos
            
            for i in range(self.num_obstacles):
                self.obstacle_artists[i].center = self.obstacles_pos[i]
                
                # Update velocity arrows
                self.vel_artists[i].remove()
                self.vel_artists[i] = self.ax.arrow(
                    self.obstacles_pos[i, 0], self.obstacles_pos[i, 1],
                    self.obstacles_vel[i, 0] * 0.5, self.obstacles_vel[i, 1] * 0.5,
                    color='orange', width=0.05
                )
            
            # Update robot velocity arrow
            self.robot_vel_artist.remove()
            self.robot_vel_artist = self.ax.arrow(
                self.robot_pos[0], self.robot_pos[1],
                self.robot_vel[0] * 0.5, self.robot_vel[1] * 0.5,
                color='blue', width=0.05
            )

            if feasible_vels is not None:
                self.ax.plot(feasible_vels, 'x', color='purple', markersize=5)
            
            plt.title(f"Dynamic Obstacles (VO) Environment")
            plt.draw()
            # plt.pause(0.5)
            plt.pause(5)
        
        elif mode == 'rgb_array':
            # For video recording
            pass

    def close(self):
        """Clean up resources"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def compute_velocity_obstacle(self, obstacle_id: int, obstacle_pos: np.ndarray, 
                                 obstacle_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity obstacle for a single obstacle
        Returns:
            Tuple of (apex, left_boundary, right_boundary) of the VO cone
        """
        # Relative position and velocity
        rel_pos = obstacle_pos - self.robot_pos
        rel_vel = obstacle_vel - self.robot_vel
        
        # Combined radius
        combined_radius = self.robot_radius + self.obstacle_radius[obstacle_id]
        
        # Calculate collision cone parameters
        dist = np.linalg.norm(rel_pos)
        if dist < combined_radius:
            # Already in collision - return full plane as VO
            return None, None, None
        
        # Angle to obstacle center
        theta = np.arctan2(rel_pos[1], rel_pos[0])
        # if theta < 0:
        #     theta = -(np.abs(theta) % (2 * np.pi))
        #     theta += 2 * np.pi
        # theta %= 2* np.pi
        
        # Half-angle of the collision cone
        alpha = np.arcsin(combined_radius / dist)
        # if alpha < 0:
        #     alpha = -(np.abs(alpha) % (2 * np.pi))
        #     alpha += 2 * np.pi
        # alpha %= 2 * np.pi
        
        # VO boundaries in relative velocity space
        # left_boundary = np.array([
        #     np.cos(theta + alpha), 
        #     np.sin(theta + alpha)
        # ])
        # right_boundary = np.array([
        #     np.cos(theta - alpha), 
        #     np.sin(theta - alpha)
        # ])
        
        # Translate to absolute velocity space
        apex = obstacle_vel
        # left_boundary = left_boundary * np.linalg.norm(rel_pos) / self.Th + apex
        # right_boundary = right_boundary * np.linalg.norm(rel_pos) / self.Th + apex
        
        return apex, theta + alpha, theta - alpha, np.linalg.norm(rel_pos) / self.Th

    def get_reachable_velocities(self) -> np.ndarray:
        """
        Compute reachable velocities that avoid all velocity obstacles
        Returns:
            Array of collision-free velocity vectors
        """
        
        ## 1. Find a reachable velocity space from the current velocity
        # 1.1. Get all kinematically feasible velocities
        # 1.2. Filter out velocities that intersect with any VO
        # 1.3. Return the set of reachable velocities

        # Please implement the logic here

        # 1. Get all kinematically feasible velocities
        num_samples = 500
        random_accels = np.random.uniform(
            low=self.action_space.low, 
            high=self.action_space.high, 
            size=(num_samples, 2)
        )

        reachable_vel = self.robot_vel + random_accels * self.dt
        
        toward_goal_vel = None
        min_toward_goal_angle = -np.inf

        apex = []
        left_angle = []
        right_angle = []
        VOh = []

        avoiding_velocities = []
        actions = []

        for i in range(self.num_obstacles):
            apex_i, left_angle_i, right_angle_i, VOh_i = self.compute_velocity_obstacle(
                i,
                self.obstacles_pos[i],
                self.obstacles_vel[i]
            )
            apex.append(apex_i)
            left_angle.append(left_angle_i)
            right_angle.append(right_angle_i)
            VOh.append(VOh_i)
        

        for i, vel in enumerate(reachable_vel):
            # Check if the velocity is within the maximum speed
            if np.linalg.norm(vel) > self.v_max:
                continue

            collision = False
            for j in range(self.num_obstacles):
                if apex[j] is None:
                    collision = True
                    break
            
                # Check if velocity is inside VO cone
                rel_vel = vel - apex[j]
                angle = np.arctan2(rel_vel[1], rel_vel[0])
                if self._is_angle_between(angle, left_angle[j], right_angle[j]) and np.linalg.norm(rel_vel) >= VOh[j]:
                    collision = True
                    break
            
            if not collision:
                avoiding_velocities.append(vel)
                actions.append(random_accels[i])

                # find a velocity moving toward the goal
                goal_vector = self.goal_pos - self.robot_pos
                goal_vector /= np.linalg.norm(goal_vector)
                tmp_vel = vel/np.linalg.norm(vel)
                if np.dot(tmp_vel, goal_vector) > min_toward_goal_angle:
                    min_toward_goal_angle = np.dot(tmp_vel, goal_vector)
                    toward_goal_vel = i
        
        avoiding_velocities = np.array(avoiding_velocities)

        

        
        # Enforce maximum speed
        # norms = np.linalg.norm(reachable_vel, axis=1)
        # reachable_vel = reachable_vel[norms <= self.v_max]

        # 2. Filter out velocities that intersect with any VO
        # avoiding_velocities = []
        # actions = []
        # # for ind, vel in enumerate(reachable_vel):
        # for ind in range(len(reachable_vel)):
        #     # Check if the velocity is within the maximum speed
        #     vel = reachable_vel[ind]
        #     if np.linalg.norm(vel) > self.v_max:
        #         continue

        #     collision = False
        #     for i in range(self.num_obstacles):
        #         apex, left_angle, right_angle, VOh = self.compute_velocity_obstacle(
        #             i,
        #             self.obstacles_pos[i],
        #             self.obstacles_vel[i]
        #         )

        #         # VOh = 0
                
        #         if apex is None:
        #             collision = True
        #             break
            
        #         # Check if velocity is inside VO cone
        #         rel_vel = vel - apex
        #         angle = np.arctan2(rel_vel[1], rel_vel[0])
        #         # left_angle = np.arctan2(left_bound[1], left_bound[0])
        #         # right_angle = np.arctan2(right_bound[1], right_bound[0])
        #         if self._is_angle_between(angle, left_angle, right_angle) and np.linalg.norm(rel_vel) >= VOh:
        #             collision = True
        #             break

                
            
        #     if not collision:
        #         avoiding_velocities.append(vel)
        #         actions.append(random_accels[ind])

        #         # find a velocity moving toward the goal
        #         goal_vector = self.goal_pos - self.robot_pos
        #         goal_vector /= np.linalg.norm(goal_vector)
        #         tmp_vel = vel/np.linalg.norm(vel)
        #         if np.dot(tmp_vel, goal_vector) > min_toward_goal_angle:
        #             min_toward_goal_angle = np.dot(tmp_vel, goal_vector)
        #             toward_goal_vel = ind
        
        
        
        if toward_goal_vel is None:

            return np.array(avoiding_velocities), np.array(actions), None, None
            
        return np.array(avoiding_velocities), np.array(actions), reachable_vel[toward_goal_vel], random_accels[toward_goal_vel]


        



        # # 1. Get all kinematically feasible velocities
        # num_samples = 36  # samples per dimension
        # angles = np.linspace(0, 2*np.pi, num_samples)
        # accelerations = np.column_stack([
        #     np.cos(angles) * self.a_max,
        #     np.sin(angles) * self.a_max
        # ])
        
        # reachable_vel = self.robot_vel + accelerations * self.dt
        
        # # Enforce maximum speed
        # norms = np.linalg.norm(reachable_vel, axis=1)
        # reachable_vel = reachable_vel[norms <= self.v_max]
        
        # # 2. Filter out velocities that intersect with any VO
        # avoiding_velocities = []
        # for vel in reachable_vel:
        #     collision = False
        #     for i in range(self.num_obstacles):
        #         apex, left_bound, right_bound = self.compute_velocity_obstacle(
        #             i,
        #             self.obstacles_pos[i],
        #             self.obstacles_vel[i]
        #         )
                
        #         if apex is None:  # Already in collision
        #             collision = True
        #             break
                    
        #         # Check if velocity is inside VO cone
        #         rel_vel = vel - apex
        #         angle = np.arctan2(rel_vel[1], rel_vel[0])
        #         left_angle = np.arctan2(left_bound[1], left_bound[0])
        #         right_angle = np.arctan2(right_bound[1], right_bound[0])
                
        #         if self._is_angle_between(angle, left_angle, right_angle):
        #             collision = True
        #             break
                    
        #     if not collision:
        #         avoiding_velocities.append(vel)
        
        # return np.array(avoiding_velocities)

    def _is_angle_between(self, angle, left, right):
        """Check if angle is between left and right boundaries (handles circular domain)"""
        # Normalize all angles to [0, 2π]
        angle = angle % (2*np.pi)
        left = left % (2*np.pi)
        right = right % (2*np.pi)

        ## assuming left is always greater than right
        return angle <= left and angle >= right
        
        # if left >= right:
        #     return angle <= left and angle >= right
        # else:
        #     return not (angle > left and angle < right)


        # if left < right:
        #     return left <= angle <= right
        # else:
        #     return angle >= left and angle <= right


if __name__ == "__main__":
    # Example usage

    env = DynamicObstacleEnv(num_obstacles=5)

    # Example of using the VO methods
    obs = env.reset()
    reachable_vel = env.get_reachable_velocities()

    # Compute VO for first obstacle
    apex, left, right, VOh = env.compute_velocity_obstacle(
        0,
        env.obstacles_pos[0], 
        env.obstacles_vel[0]
    )

    # Simple random policy
    for _ in range(100):
        action = env.action_space.sample()
        # print("Action:", action)
        # breakpoint()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished")
            obs = env.reset()

    env.close()