import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

class PureDoorsEnv(gym.Env):
    """
    Pure Doors game environment following OpenAI Gym interface.
    
    State space:
    - 8x6 grid where:
        - 0: empty space
        - 1: door
        - 2: player
    
    Action space:
    - 0: move left (j)
    - 1: stay (k)
    - 2: move right (l)
    """
    
    def __init__(self):
        super(PureDoorsEnv, self).__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # left, stay, right
        
        # 8x6 grid representation
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(6, 8), dtype=np.uint8
        )
        
        # Game state variables
        self.player_pos = None
        self.doors = None
        self.score = None
        self.fib_sequence = None
        self.current_step = None
        
    def _generate_fib_sequence(self, n: int) -> list:
        """Generate first n numbers of Fibonacci sequence."""
        fib = [1, 2]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
        
    def _get_door_positions(self, n: int) -> list:
        """Get door positions using Fibonacci sequence mod 8."""
        return [x % 8 for x in self._generate_fib_sequence(n)]
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize game state
        self.player_pos = 3  # Start in middle position
        self.score = 0
        self.current_step = 0
        
        # Initialize doors (distance, position)
        door_positions = self._get_door_positions(10)  # Get first 10 door positions
        self.doors = [(5, pos) for pos in door_positions]
        
        # Create initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def _get_observation(self) -> np.ndarray:
        """Convert current game state to observation matrix."""
        obs = np.zeros((6, 8), dtype=np.uint8)
        
        # Add doors
        for distance, pos in self.doors:
            if 0 <= distance < 5:  # Only show doors within view
                obs[5-distance-1, pos] = 1
        
        # Add player
        obs[5, self.player_pos] = 2
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (left), 1 (stay), or 2 (right)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Update player position
        if action == 0 and self.player_pos > 0:  # left
            self.player_pos -= 1
        elif action == 2 and self.player_pos < 7:  # right
            self.player_pos += 1
        
        # Update doors
        new_doors = []
        crashed = False
        passed_door = False
        
        for distance, pos in self.doors:
            new_distance = distance - 1
            if new_distance >= 0:
                new_doors.append((new_distance, pos))
                if new_distance == 0 and pos != self.player_pos:
                    crashed = True
                elif new_distance == 0:
                    passed_door = True
        
        # Add new door if needed
        if passed_door:
            next_pos = self._get_door_positions(len(self.doors) + 1)[-1]
            new_doors.append((5, next_pos))
        
        self.doors = new_doors
        self.current_step += 1
        
        # Calculate reward and check terminal conditions
        reward = 1.0 if passed_door else 0.0
        if passed_door:
            self.score += 1
        
        terminated = crashed or self.current_step >= 1000
        truncated = False
        
        observation = self._get_observation()
        info = {
            'score': self.score,
            'crashed': crashed,
            'player_position': self.player_pos
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render the current game state."""
        display = []
        
        # Render each row
        for row in range(6):
            line = "|"
            for col in range(8):
                if row == 5 and col == self.player_pos:
                    line += "^"
                elif any(d[0] == 5-row-1 and d[1] == col for d in self.doors):
                    line += "-"
                else:
                    line += " "
            line += "|"
            display.append(line)
        
        # Add control hint
        display.append("[jkl]> Score: " + str(self.score))
        
        # Print the display
        print("\n".join(display))

# Example usage:
if __name__ == "__main__":
    env = PureDoorsEnv()
    obs, _ = env.reset()
    env.render()
    
    # Manual play loop
    while True:
        try:
            # Get keyboard input
            action = input("Action (j/k/l): ")
            if action == "j":
                a = 0
            elif action == "k":
                a = 1
            elif action == "l":
                a = 2
            else:
                continue
                
            # Take step
            obs, reward, terminated, truncated, info = env.step(a)
            env.render()
            
            if terminated:
                print(f"Game Over! Final score: {info['score']}")
                break
                
        except KeyboardInterrupt:
            break
