import sys
import time
import random
import matplotlib.pyplot as plt
import ast
import numpy as np
import gym
import gym_chess
import torch
env = gym.make("ChessVsSelf-v2")

#
# Play against self
#
num_episodes = 10
num_steps = 500

class LogisticRegression:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.params = np.random.rand(state_size, action_size)

    def __call__(self, state):
        q_values = np.dot(state, self.params)
        return np.argmax(q_values)

    def get_params(self):
        return self.params.flatten()

    def set_params(self, params):
        self.params = params.reshape(self.state_size, self.action_size)
    def state_dict(self):
        # Retourne les paramètres sous forme de dictionnaire
        return {'params': self.params}

    def load_state_dict(self, state_dict):
        # Charge les paramètres à partir d'un dictionnaire
        self.params = state_dict['params']

class EpsilonGreedy_cem:
    def __init__(self, epsilon_start, epsilon_min, epsilon_decay, q_network):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = q_network

    def __call__(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.action_size - 1)
        else:
            return self.q_network(state)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

state_size = 64
action_size = 4101
device = torch.device("cpu") 
q_network = LogisticRegression(state_size, action_size)
model_path = r"C:\Users\jlien\Documents\Polytechnique\Autonomous\Project\chess-rl\model_cem_30\_iter_9.pth"
q_network.load_state_dict(torch.load(model_path,weights_only=False))

epsilon_greedy_cem = EpsilonGreedy_cem(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, q_network=q_network)
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
        for move in env.possible_moves:
            print(move)
        move = input("origine de votre coup : ")
        if move == "CASTLE_KING_SIDE_WHITE":
            a= 64 * 64
        elif move == "CASTLE_QUEEN_SIDE_WHITE":
            a= 64 * 64 + 1
        elif move == "CASTLE_KING_SIDE_BLACK":
            a= 64 * 64 + 2
        elif move == "CASTLE_QUEEN_SIDE_BLACK":
            a= 64 * 64 + 3
        else :
            move = ast.literal_eval(move)
            _from = move[0][0] * 8 + move[0][1]
            _to = move[1][0] * 8 + move[1][1]
            a = _from * 64 + _to
        # perform action
        state, reward, done, _ = env.step(a)
        total_rewards["WHITE"] += reward
        if done:
            break
        

        # black moves
        board = np.array(state["board"]).flatten()
        state_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
        action = epsilon_greedy_cem(state_tensor)
        next_state, reward, terminated, info = env.step(action)
        done = terminated
        total_rewards["BLACK"] += reward
        print(reward)
        while reward < 0:
            moves = env.possible_moves
            if len(moves) == 0:
                break
            m = random.choice(moves)
            a = env.move_to_action(m)
            # perform action
            state, reward, done, _ = env.step(a)
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
