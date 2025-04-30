from env2 import DynamicObstacleEnv
import matplotlib.pyplot as plt

env = DynamicObstacleEnv()
obs = env.reset()

# env.render()
print("Initial Agent Position:", env.robot_pos, env.robot_vel)
print("Initial Goal Position:", env.goal_pos, "\n")

done = False

while not done:
    feasible_vels, feasible_actions, toward_goal_vel, toward_goal_action = env.get_reachable_velocities()
    print("Feasible Velocities:", feasible_vels.shape)
    # if feasible_vels.shape[0] != 10000:
    #     plt.plot(feasible_vels[:, 0], feasible_vels[:, 1], 'x', color='purple', markersize=5)
    #     plt.show()
    # obs, reward, done, info = env.step(toward_goal_action)
    # breakpoint()
    # continue
    if len(feasible_vels) == 0 or len(feasible_actions) == 0:
        print("No feasible velocities or actions. Trying again...")
        for _ in range(10):
            feasible_vels, feasible_actions, toward_goal_vel, toward_goal_action = env.get_reachable_velocities()
            if len(feasible_vels) > 0 and len(feasible_actions) > 0:
                break
        


    print(feasible_vels.shape, feasible_actions.shape)
    if toward_goal_vel is None or toward_goal_action is None:
        print("No feasible velocities or actions towards the goal.")
        break


    # goal = env.goal_pos
    # robot = env.robot_pos
    # rel_pos = goal - robot
    # print(rel_pos)
    # print(toward_goal_vel)
    # break

    obs, reward, done, info = env.step(toward_goal_action)
    env.render()
    if done:
        print("Episode finished")
        obs = env.reset()
        break



# for i in range(env.num_obstacles):
#     print(f"Obstacle {i}: Position: {env.obstacles_pos[i]}, Velocity: {env.obstacles_vel[i]}")
#     apex, left, right = env.compute_velocity_obstacle(i, env.obstacles_pos[i], env.obstacles_vel[i])
#     print(f"Obstacle {i}: Apex: {apex}, Left: {left}, Right: {right} \n")


# Step 2: Create Reachable Velocity Set

# Step 3: Select A Velocity from Reachable Velocity Set such that we reach the goal