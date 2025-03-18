import gym
import gym_chess
import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


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

class EpsilonGreedy:
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

class CEMOptimizer:
    def __init__(self, initial_mean, initial_var, population_size, elite_fraction):
        self.mean = initial_mean
        self.var = initial_var
        self.population_size = population_size
        self.elite_fraction = elite_fraction

    def sample_population(self):
        # Utiliser une matrice de covariance diagonale pour réduire la mémoire
        return [np.random.normal(self.mean, np.sqrt(self.var)) for _ in range(self.population_size)]

    def select_elites(self, rewards, population):
        elite_indices = np.argsort(rewards)[-int(self.elite_fraction * self.population_size):]
        return [population[i] for i in elite_indices]

    def update_distribution(self, elites):
        self.mean = np.mean(elites, axis=0)
        self.var = np.var(elites, axis=0)

def evaluate_population(population, env, q_network, num_episodes_per_eval):
    rewards = []
    passage = []
    for params in population:
        q_network.set_params(params)
        episode_reward = 0
        episode_passage = 0
        for _ in range(num_episodes_per_eval):
            
            try:
                state = env.reset()
                done = False
                while not done:
                    episode_passage +=1
                    board = np.array(state["board"]).flatten()
                    state_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
                    action = epsilon_greedy(state_tensor)
                    next_state, reward, terminated, _ = env.step(action)
                    done = terminated
                    episode_reward += reward
                    state = next_state
            except Exception as e:
                print(f"Exception occurred: {e}. Skipping this episode.")
                episode_reward = 0  # Ignorer cet épisode en attribuant une récompense de 0
                break
        passage.append(episode_passage)
        rewards.append(episode_reward)
    return rewards,passage


def train_cem_agent(
    env,
    q_network,
    cem_optimizer,
    epsilon_greedy,
    num_iterations=50,
    num_episodes_per_eval=5
):
    """
    Train a Q-network using the Cross-Entropy Method (CEM) for a given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : LogisticRegression
        The Q-network to train.
    cem_optimizer : CEMOptimizer
        The CEM optimizer to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    num_iterations : int
        The number of CEM iterations to perform.
    num_episodes_per_eval : int
        The number of episodes to evaluate each set of parameters.

    Returns
    -------
    best_reward : float
        The best reward obtained during training.
    """
    best_reward = float('-inf')
    naive_trains_result_list =  [[], []]
    for iteration in range(num_iterations):
        # Sample a new population of parameters
        population = cem_optimizer.sample_population()

        # Evaluate each set of parameters
        rewards,passage = evaluate_population(population, env, q_network, num_episodes_per_eval)

        # Select the elite parameters
        elites = cem_optimizer.select_elites(rewards, population)

        # Update the distribution parameters
        cem_optimizer.update_distribution(elites)

        # Track the best reward
        max_reward = max(rewards)
        if max_reward > best_reward:
            best_reward = max_reward

        print(f"Iteration {iteration}, Best Reward: {max_reward}")

        # Decay epsilon
        epsilon_greedy.decay_epsilon()
        naive_trains_result_list[0].append(np.mean(passage))
        naive_trains_result_list[1].append(np.mean(rewards))
        naive_trains_result_df = pd.DataFrame(
         np.array(naive_trains_result_list).T,
         columns=["num_episodes", "episode_reward"],
)
        naive_trains_result_df.to_csv("/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/model_cem_30/train.csv")
        model_save_path = "/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/model_cem_30/test" + f"_iter_{iteration + 1}.pth"
        torch.save(q_network.state_dict(), model_save_path)
        print(f"Model saved at iteration {iteration + 1+4} to {model_save_path}")

    return best_reward

# Initialisation
state_size = 64
action_size = 4101
q_network = LogisticRegression(state_size, action_size)
# model_path = "/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/model_cem/model_cem_iter_4.pth"
# q_network.load_state_dict(torch.load(model_path,weights_only=False))
env = gym.make("ChessVsSelf-v2")
# Paramètres initiaux pour CEM
initial_mean = np.random.randn(state_size * action_size)
initial_var = np.ones_like(initial_mean)

cem_optimizer = CEMOptimizer(initial_mean, initial_var, population_size=10, elite_fraction=0.2)
epsilon_greedy = EpsilonGreedy(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, q_network=q_network)

# Entraînement de l'agent
best_reward = train_cem_agent(
    env=env,
    q_network=q_network,
    cem_optimizer=cem_optimizer,
    epsilon_greedy=epsilon_greedy,
    num_iterations=10,
    num_episodes_per_eval=5
)

print(f"Training complete. Best reward obtained: {best_reward}")

