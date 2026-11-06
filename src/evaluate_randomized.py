from stable_baselines3 import PPO
from randomized_lander import RandomStartLunarLander

model = PPO.load("models/ppo_curriculum_stage1")

env = RandomStartLunarLander(render_mode="human")

for episode in range(30):

    observation, info = env.reset(seed=100 + episode)

    total_reward = 0
    final_step_reward = 0
    steps = 0

    while True:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        final_step_reward = reward
        steps += 1

        if terminated or truncated:
            break

    if terminated and final_step_reward == 100:
        result = "SUCCESSFUL LANDING"

    elif terminated and final_step_reward == -100:
        result = "CRASH OR OUT OF BOUNDS"

    elif truncated:
        result = "TIME LIMIT REACHED"

    else:
        result = "UNKNOWN"

    print(f"\nEPISODE {episode + 1}")
    print("Result:", result)
    print("Steps:", steps)
    print("Total reward:", round(total_reward, 2))
    print("Final-step reward:", final_step_reward)
    print("Final position:", observation[0])
    print("Final angle:", observation[4])
    print("Horizontal velocity:", observation[2])
    print("Vertical velocity:", observation[3])

env.close()