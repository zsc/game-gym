import gym
from gym import spaces
import numpy as np

class FifteenPuzzleEnv(gym.Env):
    """
    Fifteen Puzzle environment following gym interface.
    Adapted from the Haskell implementation by bradrn.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(FifteenPuzzleEnv, self).__init__()
        
        # Define action space (4 possible moves: up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (4x4 grid with numbers 0-15, where 0 represents empty space)
        self.observation_space = spaces.Box(
            low=0, 
            high=15,
            shape=(4, 4), 
            dtype=np.int32
        )
        
        # Initialize state
        self.reset()

    def _string_to_board(self, state_str):
        """Convert string representation to numpy array."""
        board = np.zeros((4, 4), dtype=np.int32)
        rows = state_str.strip().split('\n')
        for i, row in enumerate(rows):
            for j, char in enumerate(row):
                if char == ' ':
                    board[i, j] = 0
                else:
                    # Convert hex to integer (a->10, b->11, etc.)
                    board[i, j] = int(char, 16)
        return board

    def _board_to_string(self, board):
        """Convert numpy array to string representation."""
        result = []
        for row in board:
            row_str = ''
            for val in row:
                if val == 0:
                    row_str += ' '
                else:
                    # Convert to hex representation
                    row_str += hex(val)[2] if val < 16 else hex(val)[2:]
            result.append(row_str)
        return '\n'.join(result) + '\n'

    def _find_empty(self, board):
        """Find the empty space (0) in the board."""
        pos = np.where(board == 0)
        return pos[0][0], pos[1][0]

    def _is_valid_move(self, action, empty_pos):
        """Check if the move is valid."""
        i, j = empty_pos
        if action == 0:  # up
            return i > 0
        elif action == 1:  # right
            return j < 3
        elif action == 2:  # down
            return i < 3
        elif action == 3:  # left
            return j > 0
        return False

    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action: int
                0: up
                1: right
                2: down
                3: left
        Returns:
            observation: numpy array
            reward: float
            done: bool
            info: dict
        """
        empty_pos = self._find_empty(self.state)
        
        if not self._is_valid_move(action, empty_pos):
            return self.state.copy(), -1, False, {}

        # Calculate new position
        i, j = empty_pos
        if action == 0:  # up
            new_i, new_j = i - 1, j
        elif action == 1:  # right
            new_i, new_j = i, j + 1
        elif action == 2:  # down
            new_i, new_j = i + 1, j
        else:  # left
            new_i, new_j = i, j - 1

        # Swap empty space with number
        self.state[i, j], self.state[new_i, new_j] = \
            self.state[new_i, new_j], self.state[i, j]

        # Check if puzzle is solved
        target = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ])
        done = np.array_equal(self.state, target)
        
        # Calculate reward
        reward = 100 if done else 0

        return self.state.copy(), reward, done, {}

    def reset(self):
        """Reset the environment to initial state."""
        initial_state = "1234\n5678\n9abc\nfde \n"
        self.state = self._string_to_board(initial_state)
        return self.state.copy()

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print("+----+")
            print(self._board_to_string(self.state).replace('\n', '|\n|')[:-1])
            print("+----+")
            print("[wasd]")

# Example usage:
if __name__ == "__main__":
    env = FifteenPuzzleEnv()
    obs = env.reset()
    env.render()
    
    # Manual play loop
    while True:
        try:
            action = input("Move (w/a/s/d): ").lower()
            if action == 'w':
                a = 0
            elif action == 'd':
                a = 1
            elif action == 's':
                a = 2
            elif action == 'a':
                a = 3
            else:
                continue
                
            obs, reward, done, _ = env.step(a)
            env.render()
            
            if done:
                print("Puzzle solved! Congratulations!")
                break
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
