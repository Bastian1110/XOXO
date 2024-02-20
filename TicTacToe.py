import gymnasium as gym 
import numpy as np

class TicTacToe(gym.Env):
    def __init__(self):
        super(TicTacToe, self).__init__()
        self.board = np.zeros((3, 3), dtype=int)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(3,3), dtype=int)
        self.current_player = 1
        self.n_actions = 9

    def reset(self, seed=None, options=None):
        self.board.fill(0)
        self.current_player = 1
        self.n_actions = 9
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.copy(self.board)

    def get_next_player(self, player):
        if player == 1:
            return 2
        else:
            return 1
        
    def step(self, action):
        coords = self.place_to_coordinate(action)
        self.current_player = self.get_next_player(self.current_player)
        if not self.cell_available(coords):
            return self._get_obs(), -5, False, False, {"message" : "Invalid action"}
        self.board[coords[1]][coords[0]] = self.get_next_player(self.current_player)
        self.n_actions-=1
        if self.check_win(coords):    
            return self._get_obs(), 5, True, False, {"message" : "Player won!"}
        if self.n_actions == 0:
            return self._get_obs(), 0, True, False, {"message" :  "Draw"}
        return self._get_obs(), 1, False, False, {"message ": "Normal action"}

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

        