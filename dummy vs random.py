import sys
import time
import random
import matplotlib.pyplot as plt

import numpy as np
import gym
import gym_chess

env = gym.make("ChessVsSelf-v2", log=False)

#
# Play against self
#
num_episodes = 10
num_steps = 500

total_steps = 0
collected_rewards = []
start = time.time()

for i in range(num_episodes):
    state = env.reset()
    print("\n", "=" * 10, "NEW GAME", "=" * 10)
    env.render()
    total_rewards = {"WHITE": 0, "BLACK": 0}

    for j in range(num_steps):
        total_steps += 1
        # white moves
        moves = env.possible_moves
        if len(moves) == 0:
            break
        actions = [env.move_to_action(m) for m in moves]
        rewards = [env.next_state(state, "WHITE" ,m)[1] for m in moves]
        max_r = np.max(rewards)
        max_indices = np.where(rewards == max_r)[0]
        a = actions[np.random.choice(max_indices)]
        # perform action
        state, reward, done, _ = env.step(a)
        total_rewards["WHITE"] += reward
        if done:
            break
        

        # black moves
        moves = env.possible_moves
        if len(moves) == 0:
            break
        m = random.choice(moves)
        a = env.move_to_action(m)
        # perform action
        state, reward, done, _ = env.step(a)
        total_rewards["BLACK"] += reward
        if done:
            break

    print(">" * 5, "GAME", i, "REWARD:", total_rewards)
    collected_rewards.append(total_rewards)

total_white_reward = sum([r["WHITE"] for r in collected_rewards])
total_black_reward = sum([r["BLACK"] for r in collected_rewards])
white_rewards = [r["WHITE"] for r in collected_rewards]
black_rewards = [r["BLACK"] for r in collected_rewards]
plt.plot(range(num_episodes), white_rewards, label="Dummy Agent", color='blue', marker='o')
plt.plot(range(num_episodes), black_rewards, label="Random Agent", color='red', marker='o')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()


print("Average white reward", total_white_reward / num_episodes)
print("Average black reward", total_black_reward / num_episodes)
end = time.time()
diff = end - start

print("Total time (s)", diff)
print("Total episodes", num_episodes)
print("Total steps", total_steps)
print("Time per episode (s)", diff / num_episodes)
print("Time per step (s)", diff / total_steps)
