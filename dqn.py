import torch
import torch.nn as nn
import random
import numpy as np
import gym_chess
import gym
from typing import List,Callable,Union
from tqdm import tqdm
import pandas as pd
import itertools


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

class EpsilonGreedy:
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

class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_decay: float,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
    ):
        """
        Initialize a new instance of MinimumExponentialLR.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be scheduled.
        lr_decay : float
            The multiplicative factor of learning rate decay.
        last_epoch : int, optional
            The index of the last epoch. Default is -1.
        min_lr : float, optional
            The minimum learning rate. Default is 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        List[float]
            The learning rates of each parameter group.
        """
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]
    
def train_dual_agents(
    env: gym.Env,
    agent_white: EpsilonGreedy,
    q_network_white: torch.nn.Module,
    optimizer_white: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    lr_scheduler_white: torch.optim.lr_scheduler.LRScheduler,
    num_episodes: int,
    gamma: float,
) -> List[float]:
    """
    Train two Q-networks with separate agents for White and Black.
    
    Returns a list of cumulated rewards per episode.
    """
    episode_reward_list = []
    
    for episode_index in tqdm(range(1, num_episodes)):
        episode_reward = 0.0
        state= env.reset()
        agent, q_network, optimizer, lr_scheduler = agent_white, q_network_white, optimizer_white, lr_scheduler_white
        q_network.train()
        for t in itertools.count():
            board = np.array(state["board"]).flatten()
            state_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent(state_tensor)
            next_state, reward, terminated, info = env.step(action)
            done = terminated
            episode_reward += float(reward)

            # Update the corresponding Q-network
            with torch.no_grad():
                next_board = np.array(next_state["board"]).flatten()
                next_state_tensor = torch.tensor(next_board, dtype=torch.float32, device=device).unsqueeze(0)
                target = reward + gamma * (torch.max(q_network(next_state_tensor))) * (1 - done)
            
            q_values = q_network(state_tensor)
            q_value_of_current_action = q_values[0, action]
            loss = loss_fn(q_value_of_current_action, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            while reward<0:
                board = np.array(state["board"]).flatten()
                state_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
                action = agent(state_tensor)
                next_state, reward, terminated, info = env.step(action)
                done = terminated
                episode_reward += float(reward)

                # Update the corresponding Q-network
                with torch.no_grad():
                    next_board = np.array(next_state["board"]).flatten()
                    next_state_tensor = torch.tensor(next_board, dtype=torch.float32, device=device).unsqueeze(0)
                    target = reward + gamma * (torch.max(q_network(next_state_tensor))) * (1 - done)
                board = np.array(state["board"]).flatten()
                
                q_values = q_network(state_tensor)
                q_value_of_current_action = q_values[0, action]
                loss = loss_fn(q_value_of_current_action, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            if done:
                break
            state = next_state
            moves = env.possible_moves
            if len(moves) == 0:
                break
            actions = [env.move_to_action(m) for m in moves]
            rewards = [env.next_state(state, "BLACK" ,m)[1] for m in moves]
            max_r = np.max(rewards)
            max_indices = np.where(rewards == max_r)[0]
            a = actions[np.random.choice(max_indices)]
            # perform action
            state, reward, done, _ = env.step(a)
            if done:
                break
            state = next_state # Switch turn

        episode_reward_list.append(episode_reward)
        agent_white.decay_epsilon()


        torch.save(q_network_white, "Autonomous/chess-rl/model_dqn1/chess_white_q_network1"+str(episode_index+10)+".pth")


    return episode_reward_list

state_size = 64  # Si on utilise une représentation 8x8 pour l'échiquier
action_size = 4101  # Par exemple, si 4672 est le nombre total de coups possibles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

env = gym.make("ChessVsSelf-v2")  # Assurez-vous que gym_chess est bien installé et configuré
env.log = False
NUMBER_OF_TRAININGS = 1
naive_trains_result_list: List[List[Union[int, float]]] = [[], [], []]

for train_index in range(NUMBER_OF_TRAININGS):
    q_network_white = ChessQNetwork_naive(state_size, action_size)
    model_path = "/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/model_dqn2015/chess_white_q_network2015_121.pth"
    loaded_model = torch.load(model_path, weights_only=False)
    state_dict = loaded_model.state_dict()
    q_network_white.load_state_dict(state_dict) 
    q_network_white.to(device)

    
    optimizer_white = torch.optim.AdamW(q_network_white.parameters(), lr=0.004, amsgrad=True)
    
    lr_scheduler_white = MinimumExponentialLR(optimizer_white, lr_decay=0.97, min_lr=0.0001)
    
    loss_fn = torch.nn.MSELoss()
    
    epsilon_greedy_white = EpsilonGreedy(
        epsilon_start=0.82,
        epsilon_min=0.013,
        epsilon_decay=0.9675,
        env=env,
        q_network=q_network_white,
    )
    

    # Entraîner les agents
    episode_reward_list = train_dual_agents(
        env,
        epsilon_greedy_white,
        q_network_white,
        optimizer_white,
        loss_fn,
        device,
        lr_scheduler_white,
        num_episodes=10,
        gamma=0.9,
    )

    naive_trains_result_list[0].extend(range(len(episode_reward_list)))
    naive_trains_result_list[1].extend(episode_reward_list)
    naive_trains_result_list[2].extend([train_index for _ in episode_reward_list])

naive_trains_result_df = pd.DataFrame(
    np.array(naive_trains_result_list).T,
    columns=["num_episodes", "mean_final_episode_reward", "training_index"],
)
naive_trains_result_df["agent"] = "Dual-Agent"

# Sauvegarde des modèles entraînés
torch.save(q_network_white, "chess_white_q_network_vsdummy.pth")
naive_trains_result_df.to_csv("training.csv")
env.close()