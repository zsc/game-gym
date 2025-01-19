import gym
from gym import spaces
import numpy as np

class Binary2BCDEnv(gym.Env):
    """
    A gym environment that simulates a Binary to BCD converter circuit.
    The agent needs to predict the values on different wires given inputs.
    """
    
    def __init__(self, width=13, digits=4):
        super(Binary2BCDEnv, self).__init__()
        
        self.width = width
        self.digits = digits
        self.width_c = int(np.ceil(np.log2(width)))  # Counter width
        
        # Define action space: predictions for each wire/register
        # Format: [binaryReg, counterReg, bcd registers, valid, state]
        self.action_space = spaces.Dict({
            'binaryReg': spaces.MultiBinary(width),
            'counterReg': spaces.MultiBinary(self.width_c),
            'bcd': spaces.MultiBinary(4 * digits),
            'valid': spaces.Discrete(2),
            'state': spaces.Discrete(2)  # IDLE=0, OPERATION=1
        })
        
        # Define observation space: circuit inputs
        self.observation_space = spaces.Dict({
            'clk': spaces.Discrete(2),
            'reset': spaces.Discrete(2),
            'start': spaces.Discrete(2),
            'binary': spaces.MultiBinary(width)
        })
        
        self.reset()
        
    def step(self, action):
        """
        Execute one timestep within the environment
        """
        # Verify action format
        assert isinstance(action, dict), "Action must be a dictionary"
        
        # Calculate the correct values based on circuit logic
        correct_values = self._calculate_correct_values()
        
        # Calculate reward based on matching predictions
        reward = self._calculate_reward(action, correct_values)
        
        # Update internal state
        self._update_state()
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'correct_values': correct_values,
            'accuracy': reward
        }
        
        self.current_step += 1
        return observation, reward, done, info
    
    def reset(self):
        """
        Reset the environment to initial state
        """
        self.current_step = 0
        self.max_steps = self.width + 2  # Width + start + end cycles
        
        # Initialize circuit state
        self.clk = 0
        self.reset_signal = 0
        self.start = 0
        self.binary_input = np.random.randint(0, 2, size=self.width)
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get current observation of the environment
        """
        return {
            'clk': self.clk,
            'reset': self.reset_signal,
            'start': self.start,
            'binary': self.binary_input
        }
    
    def _calculate_correct_values(self):
        """
        Calculate correct values for all wires based on current inputs
        """
        # Simulate one clock cycle of the Binary2BCD converter
        binaryReg = np.zeros(self.width)
        counterReg = np.zeros(self.width_c)
        bcd = np.zeros(4 * self.digits)
        valid = 0
        state = 0  # IDLE
        
        # Implementation of Binary2BCD logic here
        if self.reset_signal:
            pass  # Everything stays at 0
        elif self.start:
            binaryReg = self.binary_input
            state = 1  # OPERATION
        elif state == 1:  # OPERATION
            # Implement double-dabble algorithm
            pass
            
        return {
            'binaryReg': binaryReg,
            'counterReg': counterReg,
            'bcd': bcd,
            'valid': valid,
            'state': state
        }
    
    def _calculate_reward(self, action, correct_values):
        """
        Calculate reward based on how well the predictions match correct values
        """
        reward = 0
        
        # Check each component
        components = ['binaryReg', 'counterReg', 'bcd', 'valid', 'state']
        for comp in components:
            if np.array_equal(action[comp], correct_values[comp]):
                reward += 1
                
        return reward / len(components)  # Normalize to [0,1]
    
    def _update_state(self):
        """
        Update internal state (clock, etc.)
        """
        self.clk = 1 - self.clk  # Toggle clock
        
    def render(self, mode='human'):
        """
        Render the environment to the screen
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Clock: {self.clk}")
            print(f"Reset: {self.reset_signal}")
            print(f"Start: {self.start}")
            print(f"Binary Input: {self.binary_input}")
            
    def close(self):
        pass

# Example usage
if __name__ == "__main__":
    env = Binary2BCDEnv(width=13, digits=4)
    
    # Run one episode
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Random action for demonstration
        action = env.action_space.sample()
        
        # Take step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Render environment
        env.render()
        
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print("-------------------")
    
    print(f"Episode finished with total reward: {total_reward}")
