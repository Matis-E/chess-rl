import torch
import torch.nn as nn
import numpy as np
import gym
import gym_chess
import random

class NeuralNetworkPolicy(nn.Module):
    def __init__(self, observations_size: int, actions_size: int, hidden_size: int = 16):
        super(NeuralNetworkPolicy, self).__init__()
        self.hidden_layer = nn.Linear(observations_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, actions_size)
        self.actions_size = actions_size

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden_tensor = torch.nn.functional.relu(self.hidden_layer(observation))
        output_tensor = torch.nn.functional.tanh(self.output_layer(hidden_tensor))
        return output_tensor

    def get_params(self) -> np.ndarray:
        params_tensor = torch.nn.utils.parameters_to_vector(self.parameters())
        return params_tensor.detach().cpu().numpy()

    def set_params(self, params: np.ndarray) -> None:
        params_tensor = torch.tensor(params, dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(params_tensor, self.parameters())

    def state_dict(self):
        return super(NeuralNetworkPolicy, self).state_dict()

    def load_state_dict(self, state_dict):
        super(NeuralNetworkPolicy, self).load_state_dict(state_dict)

class EpsilonGreedy:
    def __init__(self, epsilon_start, epsilon_min, epsilon_decay, q_network):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = q_network

    def __call__(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.actions_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return np.argmax(q_values)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def saes_1_1(
    objective_function,
    mean_array,
    sigma_array,
    max_iterations=500,
    tau=None,
    print_every=10,
    success_score=float("inf"),
    num_evals_for_stop=None,
    hist_dict=None,
):
    n_elite = 1
    d = mean_array.shape[0]

    if tau is None:
        tau = 1.0 / (2.0 * d)

    score = objective_function(mean_array)

    for iteration_index in range(0, max_iterations):
        new_sigma_array = sigma_array * np.exp(tau * np.random.standard_normal(d))
        new_mean_array = mean_array + new_sigma_array * np.random.standard_normal(d)

        new_score = objective_function(new_mean_array)

        if new_score >= score:
            score = new_score
            mean_array = new_mean_array
            sigma_array = new_sigma_array

        if hist_dict is not None:
            hist_dict[iteration_index] = [score] + mean_array.tolist() + sigma_array.tolist()

        if iteration_index % print_every == 0:
            print(f"Iteration {iteration_index}, Score {score}")

        if score >= success_score:
            break

    return mean_array

class ObjectiveFunction:
    def __init__(self, env, policy, num_episodes=1, max_time_steps=float("inf")):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps

    def __call__(self, policy_params):
        self.policy.set_params(policy_params)
        episode_rewards = []

        for _ in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            for t in range(int(self.max_time_steps)):
                board = np.array(state["board"]).flatten()
                state_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
                action = self.policy(state_tensor).argmax().item()
                state, reward, terminated, _ = self.env.step(action)
                total_reward += reward

                if terminated:
                    break

            episode_rewards.append(total_reward)

        return np.mean(episode_rewards)

def train_saes_agent(
    env,
    q_network,
    objective_function,
    max_iterations=500,
    tau=None,
    print_every=10,
    success_score=float("inf"),
    hist_dict=None,
):
    initial_mean_array = q_network.get_params()
    initial_sigma_array = np.ones_like(initial_mean_array) * 0.1

    optimized_policy_params = saes_1_1(
        objective_function=objective_function,
        mean_array=initial_mean_array,
        sigma_array=initial_sigma_array,
        max_iterations=max_iterations,
        tau=tau,
        print_every=print_every,
        success_score=success_score,
        hist_dict=hist_dict,
    )

    q_network.set_params(optimized_policy_params)
    return q_network

# Initialisation
state_size = 64
action_size = 4101
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("ChessVsSelf-v2")
q_network = NeuralNetworkPolicy(state_size, action_size).to(device)

objective_function = ObjectiveFunction(env, q_network, num_episodes=10, max_time_steps=500)

# Entraînement de l'agent
q_network = train_saes_agent(
    env=env,
    q_network=q_network,
    objective_function=objective_function,
    max_iterations=1000,
    tau=0.001,
    print_every=100,
    success_score=1,
    hist_dict=None,
)

# Sauvegarde du modèle
torch.save(q_network.state_dict(), "saes_chess_model.pth")
