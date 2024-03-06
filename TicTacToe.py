import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class TicTacToeMultiAgent(MultiAgentEnv):
    MAX_STEPS = 9

    def __init__(self, config=None):
        super().__init__()
        self.max_steps = self.MAX_STEPS
        self._agent_ids = {1, 2}


        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(3, 3), dtype=int)
        self.action_space = gym.spaces.Discrete(9)

        self.board = np.zeros((3, 3), dtype=int)

    def reset(self, *, seed=None, options=None):
        self.board.fill(0)
        self.current_player = 1
        self._step = 0
        return {self.current_player: self._get_obs()}, {}

    def _get_obs(self):
        return np.copy(self.board)

    def step(self, action_dict):
        action = action_dict[self.current_player]
        coords = self.place_to_coordinate(action)
        done = False
        if not self.cell_available(coords):
            reward = -10
        else:
            self.board[coords[1]][coords[0]] = self.current_player
            if self.check_win(coords):
                reward = 10
                done = True
            elif np.all(self.board != 0):
                reward = 0
                done = True
            else:
                reward = 0.5
        self.current_player = self.get_next_player(self.current_player)
        
        rewards = {self.current_player: 0, self.get_next_player(self.current_player) : reward}
        terminated = {self.current_player : done, "__all__" : done}
        truncated = {self.current_player : done, "__all__" : done}
        obs = {self.current_player: self._get_obs()}

        self._step += 1
        if self._step >= self.max_steps:
            truncated["__all__"] = True

        return obs, rewards, terminated, truncated, {}
    
    def get_next_player(self, player):
        if player == 1:
            return 2
        else :
            return 1

    def place_to_coordinate(self, place : int):
        counter = 0
        for i in range(3):
            for j in range(3):
                if counter == place:
                    return (j, i)
                counter+=1

    def cell_available(self, coords):
        return self.board[coords[1]][coords[0]] == 0

    def check_win(self, coords):
        player = self.board[coords[1]][coords[0]] 
        win_conditions = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]
        
        for condition in win_conditions:
            if all(self.board[row][col] == player for row, col in condition):
                return True  
        return False  

        