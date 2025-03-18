import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym_chess
import gym
from typing import List,Callable,Union,Tuple
from tqdm import tqdm
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import collections


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

class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.

    Methods
    -------
    add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
        Sample a batch of transitions from the buffer.
    __len__()
        Return the current size of the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.int64,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state vector of the added transition.
        action : np.int64
            The action of the added transition.
        reward : float
            The reward of the added transition.
        next_state : np.ndarray
            The next state vector of the added transition.
        done : bool
            The final state of the added transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, Tuple[int], Tuple[float], np.ndarray, Tuple[bool]]:
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, bool]
            A batch of `batch_size` transitions.
        """
        # Here, `random.sample(self.buffer, batch_size)`
        # returns a list of tuples `(state, action, reward, next_state, done)`
        # where:
        # - `state`  and `next_state` are numpy arrays
        # - `action` and `reward` are floats
        # - `done` is a boolean
        #
        # `states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))`
        # generates 5 tuples `state`, `action`, `reward`, `next_state` and `done`, each having `batch_size` elements.
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            The current size of the buffer.
        """
        return len(self.buffer)

def train_dqn2_agent(
    env: gym.Env,
    q_network: torch.nn.Module,
    target_q_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epsilon_greedy: EpsilonGreedy,
    device: torch.device,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_episodes: int,
    gamma: float,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    target_q_network_sync_period: int,
) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    target_q_network : torch.nn.Module
        The target Q-network to use for estimating the target Q-values.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch to use for training.
    replay_buffer : ReplayBuffer
        The replay buffer storing the experiences with their priorities.
    target_q_network_sync_period : int
        The number of episodes after which the target Q-network should be updated with the weights of the Q-network.

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    iteration = 0
    episode_reward_list = []
    naive_trains_result_list =  [[], []]
    for episode_index in tqdm(range(1, num_episodes)):
        state= env.reset()
        episode_reward = 0.0

        for t in itertools.count():
            board = np.array(state["board"]).flatten()
            moves = env.possible_moves
            actions = [env.move_to_action(m) for m in moves]
            bool_array = np.zeros(action_size, dtype=bool)
            # Activation des indices
            bool_array[actions] = True
            board = np.concatenate((board,bool_array))
            state_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
            action = epsilon_greedy(state_tensor)
            next_state, reward, terminated, info = env.step(action)
            done = terminated
            episode_reward += float(reward)
            # Update the q_network weights with a batch of experiences from the buffer

            if len(replay_buffer) > batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

                # Convert to PyTorch tensors
                batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device)

                # Compute the target Q values for the batch
                with torch.no_grad():
                    next_board = np.array(next_state["board"]).flatten()
                    moves = env.possible_moves
                    actions = [env.move_to_action(m) for m in moves]
                    bool_array = np.zeros(action_size, dtype=bool)
                    # Activation des indices
                    bool_array[actions] = True
                    next_board = np.concatenate((next_board,bool_array))
                    next_state_tensor = torch.tensor(next_board, dtype=torch.float32, device=device).unsqueeze(0)
                    targets = reward + gamma * (torch.max(q_network(next_state_tensor))) * (1 - done)

                current_q_values = torch.gather(q_network(batch_states_tensor),1,batch_actions_tensor.unsqueeze(1))

                # Compute loss
                loss = loss_fn(current_q_values, targets)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step()

            # Update the target q-network weights

            # Every episodes (e.g., every `target_q_network_sync_period` episodes), the weights of the target network are updated with the weights of the Q-network
            if iteration % target_q_network_sync_period == 0:
                target_q_network.load_state_dict(q_network.state_dict())

            iteration += 1

            if done:
                break

            state = next_state
            if env.current_player == "BLACK":
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
                state = next_state     
        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()
        naive_trains_result_list[0].append(iteration)
        naive_trains_result_list[1].append(episode_reward)
        naive_trains_result_df = pd.DataFrame(
         np.array(naive_trains_result_list).T,
         columns=["num_episodes", "episode_reward"],
)
        naive_trains_result_df.to_csv("model_dqn2015_gdlr/model_dqn2015_mash_gdlrtrain.csv")
        torch.save(q_network, "model_dqn2015_gdlr/chess_white_q_network2015_test_mash_"+str(episode_index)+".pth")

    return episode_reward_list


state_size = 64+4101  # Si on utilise une représentation 8x8 pour l'échiquier
action_size = 4101  # Par exemple, si 4672 est le nombre total de coups possibles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

env = gym.make("ChessVsSelf-v2")  # Assurez-vous que gym_chess est bien installé et configuré
env.log = False
NUMBER_OF_TRAININGS = 1
naive_trains_result_list: List[List[Union[int, float]]] = [[], [], []]

for train_index in range(NUMBER_OF_TRAININGS):
    # Instancier les agents Blancs et Noirs
    q_network = ChessQNetwork_naive(state_size, action_size)
    # model_path = "/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/model_dqn2015/chess_white_q_network2015_9.pth"
    # loaded_model = torch.load(model_path, weights_only=False)
    # state_dict = loaded_model.state_dict()
    # q_network.load_state_dict(state_dict) 
    q_network.to(device)
    target_q_network = ChessQNetwork_naive(state_size, action_size).to(device)
    target_q_network.load_state_dict(q_network.state_dict())
    
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=0.1, amsgrad=True)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.97, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()
    
    epsilon_greedy_white = EpsilonGreedy(
        epsilon_start=0.82,
        epsilon_min=0.013,
        epsilon_decay=0.9675,
        env=env,
        q_network=q_network,
    )
    
    replay_buffer = ReplayBuffer(2000)

    # Entraîner les agents
    episode_reward_list = train_dqn2_agent(
        env,
        q_network,
        target_q_network,
        optimizer,
        loss_fn,
        epsilon_greedy_white,
        device,
        lr_scheduler,
        num_episodes=150,
        gamma=0.9,
        batch_size=128,
        replay_buffer=replay_buffer,
        target_q_network_sync_period=30,
    )

    naive_trains_result_list[0].extend(range(len(episode_reward_list)))
    naive_trains_result_list[1].extend(episode_reward_list)
    naive_trains_result_list[2].extend([train_index for _ in episode_reward_list])

naive_trains_result_df = pd.DataFrame(
    np.array(naive_trains_result_list).T,
    columns=["num_episodes", "mean_final_episode_reward", "training_index"],
)
naive_trains_result_df["agent"] = "Dual-Agent"
naive_trains_result_df.to_csv("/users/eleves-a/2022/jean.lienhard/Documents/Autonomous/chess-rl/gym-chess/model_dqn201_gdlr/train.csv")
# Sauvegarde des modèles entraînés
torch.save(q_network, "chess_white_q_network.pth")

env.close()