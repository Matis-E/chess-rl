import sys
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import gym
import gym_chess

env = gym.make("ChessVsSelf-v2", log=False)

#
# Play against self
#
num_episodes = 10
num_steps = 5000

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

class ChessQNetwork_naive(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ChessQNetwork_naive, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class EpsilonGreedy_dqn:
    """
    An Epsilon-Greedy policy.

    Attributes
    ----------
    epsilon : float
        The initial probability of choosing a random action.
    epsilon_min : float
        The minimum probability of choosing a random action.
    epsilon_decay : float
        The decay rate for the epsilon value after each action.
    env : gym.Env
        The environment in which the agent is acting.
    q_network : torch.nn.Module
        The Q-Network used to estimate action values.

    Methods
    -------
    __call__(state: np.ndarray) -> np.int64
        Select an action for the given state using the epsilon-greedy policy.
    decay_epsilon()
        Decay the epsilon value after each action.
    """

    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        env: gym.Env,
        q_network: torch.nn.Module,
    ):
        """
        Initialize a new instance of EpsilonGreedy.

        Parameters
        ----------
        epsilon_start : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each episode.
        env : gym.Env
            The environment in which the agent is acting.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: np.ndarray) -> np.int64:
        """
        Select an action for the given state using the epsilon-greedy policy.

        If a randomly chosen number is less than epsilon, a random action is chosen.
        Otherwise, the action with the highest estimated action value is chosen.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        np.int64
            The chosen action.
        """
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()  # Select a random action
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_index = torch.argmax(q_values).item()
                action = action_index
        return action

    def map_index_to_action(self, index: int) -> int:
        """
        Map the action index to a valid chess move.

        Parameters
        ----------
        index : int
            The index of the action.

        Returns
        -------
        int
            The valid chess move.
        """
        # Implémentez ici la logique pour mapper l'index à un coup valide
        # Par exemple, utilisez une liste ou un dictionnaire de coups valides
        valid_moves = self.env.possible_moves  # Assurez-vous que cette méthode existe dans votre environnement
        return valid_moves[index]

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



state_size = 64
action_size = 4101
device = torch.device("cpu") 
q_network = LogisticRegression(state_size, action_size)
model_path = "/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/model_cem/_iter_14.pth"
q_network.load_state_dict(torch.load(model_path,weights_only=False))

epsilon_greedy_cem = EpsilonGreedy_cem(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, q_network=q_network)

q_network_white = ChessQNetwork_naive(state_size, action_size)
model_path = "/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/model_dqn2015/chess_white_q_network2015_121.pth"
loaded_model = torch.load(model_path, weights_only=False)
state_dict = loaded_model.state_dict()
q_network_white.load_state_dict(state_dict) 
q_network_white.to(device)
epsilon_greedy_white = EpsilonGreedy_dqn(
        epsilon_start=0.82,
        epsilon_min=0.013,
        epsilon_decay=0.9675,
        env=env,
        q_network=q_network_white,
    )

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
        board = np.array(state["board"]).flatten()
        state_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
        action = epsilon_greedy_white(state_tensor)
        next_state, reward, terminated, info = env.step(action)
        done = terminated
        total_rewards["WHITE"] += reward
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
        

        # black moves
        board = np.array(state["board"]).flatten()
        state_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
        action = epsilon_greedy_cem(state_tensor)
        next_state, reward, terminated, info = env.step(action)
        done = terminated
        total_rewards["BLACK"] += reward
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
    env.render()

    print(">" * 5, "GAME", i, "REWARD:", total_rewards)
    collected_rewards.append(total_rewards)

total_white_reward = sum([r["WHITE"] for r in collected_rewards])
total_black_reward = sum([r["BLACK"] for r in collected_rewards])
white_rewards = [r["WHITE"] for r in collected_rewards]
black_rewards = [r["BLACK"] for r in collected_rewards]



print("Average white reward", total_white_reward / num_episodes)
print("Average black reward", total_black_reward / num_episodes)
end = time.time()
diff = end - start

print("Total time (s)", diff)
print("Total episodes", num_episodes)
print("Total steps", total_steps)
print("Time per episode (s)", diff / num_episodes)
print("Time per step (s)", diff / total_steps)

# plt.plot(range(num_episodes), white_rewards, label="Dummy Agent", color='blue', marker='o')
# plt.plot(range(num_episodes), black_rewards, label="Random Agent", color='red', marker='o')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend()
# plt.show()