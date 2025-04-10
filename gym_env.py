import gym
import numpy as np
from collections import deque
from gym import spaces

class QuizEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, student_profile='beginner', seed=None):
        super(QuizEnvironment, self).__init__()

        # Set the random seed
        # self.seed = seed
        # if self.seed is not None:
        #     np.random.seed(self.seed)

        # Define difficulty levels and student profiles
        self.difficulty_levels = ['Easy', 'Medium', 'Hard']
        self.student_profiles = {
            'beginner': {'Easy': 0.7, 'Medium': 0.4, 'Hard': 0.1},
            'intermediate': {'Easy': 0.9, 'Medium': 0.6, 'Hard': 0.3},
            'advanced': {'Easy': 0.95, 'Medium': 0.8, 'Hard': 0.6}
        }
        self.student_profile = self.student_profiles[student_profile]

        # Define observation space: (10-history binary, accuracy_rate (discrete), difficulty)
        self.observation_space = spaces.MultiDiscrete([2] * 10 + [11] + [3])

        # Define action space: choosing difficulty (0=Easy, 1=Medium, 2=Hard)
        self.action_space = spaces.Discrete(len(self.difficulty_levels))

        # Initialize learning factor
        self.learning_factor = 1.0  # Start with no learning
        self.learning_rate = 0.005  # Rate at which the student learns

        self.last_difficulty = None  # Keep track of last difficulty chosen

        self.reset()

    def reset(self):
        """Reset the environment to an initial state."""
        # Reset the random seed if specified
        # if self.seed is not None:
        #     np.random.seed(self.seed)

        self.performance_history = deque([0] * 10, maxlen=10)
        self.accuracy_rate = 0.0
        self.current_difficulty = 'Easy'
        self.learning_factor = 1.0  # Reset learning factor
        self.step_count = 0  # Reset step counter
        return self._get_state()

    def _get_state(self):
        """Convert the environment state to a gym-compatible observation."""
        performance_history = list(self.performance_history)
        accuracy_rate = int(self.accuracy_rate * 10)  # Convert to discrete value (0-10)
        difficulty_level = [1 if d == self.current_difficulty else 0 for d in self.difficulty_levels]
        return np.array(performance_history + [accuracy_rate] + difficulty_level, dtype=np.int32)

    def step(self, action):
        """Apply an action to the environment and return (next_state, reward, done, info)."""
        self.step_count += 1

        chosen_difficulty = self.difficulty_levels[action]

        # Adjust correct_prob based on learning factor
        correct_prob = self.student_profile[chosen_difficulty] * self.learning_factor
        correct_prob = max(0.0, min(1.0, correct_prob))  # Ensure probability stays within bounds

        # Simulate the student answering the question
        correct = np.random.random() < correct_prob
        self.performance_history.append(1 if correct else 0)

        # Update accuracy rate using weighted moving average
        alpha = 0.2  
        self.accuracy_rate = alpha * (1 if correct else 0) + (1 - alpha) * self.accuracy_rate # More weight to recent answers

        # Update learning factor based on performance
        if correct:
            self.learning_factor += self.learning_rate  # Student learns and improves
        else:
            self.learning_factor -= self.learning_rate * 0.5  # Student struggles slightly

        # Ensure learning factor stays within bounds
        self.learning_factor = max(0.5, min(2.0, self.learning_factor))

        # Save the current difficulty before computing the reward
        previous_difficulty = self.current_difficulty
        # Calculate reward
        reward = self._calculate_reward(correct, chosen_difficulty)

        # Penalize rapid oscillations between difficulties
        if previous_difficulty != chosen_difficulty:
            reward -= 5  # Can adjust

        self.current_difficulty = chosen_difficulty # Update current difficulty

        # Check termination condition (when student reaches 90% accuracy)
        done = self.accuracy_rate >= 0.90

        # max_steps = 100
        # if self.step_count >= max_steps:
        #     done = True

        return self._get_state(), reward, done, {}
    
    def _calculate_reward(self, correct, difficulty):
        reward = 0
        base_rewards = {'Easy': 5, 'Medium': 10, 'Hard': 20}
        penalties = {'Easy': 2, 'Medium': 5, 'Hard': 10}

        # Base reward/penalty for correct/incorrect answers
        if correct:
            reward += base_rewards[difficulty]
        else:
            reward -= penalties[difficulty]

        # Encourage Progression to Harder Difficulties
        if self.accuracy_rate > 0.7:
            if difficulty == 'Easy':
                reward -= 15
            elif difficulty == 'Medium' and self.current_difficulty == 'Easy':
                reward += 10 

        if self.accuracy_rate > 0.8:
            if difficulty == 'Medium':
                reward -= 10
            elif difficulty == 'Hard' and self.current_difficulty == 'Medium':
                reward += 15 

        # Encourage moving to easier difficulties if struggling
        if self.accuracy_rate < 0.5:
            if difficulty == 'Hard':
                reward -= 20
            elif difficulty == 'Medium':
                reward -= 10
            elif difficulty == 'Easy':
                reward += 15

        return reward

    def render(self, mode='human'):
        """Render the environment state. Supports 'human' (text)."""
        
        if mode == 'human':
            print(f"Performance History: {list(self.performance_history)}")
            print(f"Accuracy Rate: {self.accuracy_rate:.2f}")
            print(f"Current Difficulty: {self.current_difficulty}")
            print(f"Learning Factor: {self.learning_factor:.2f}")
            print()