import numpy as np
import collections

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Using a dictionary instead of a fixed-size table
        self.q_table = collections.defaultdict(lambda: np.zeros(action_size))

    def encode_state(self, state):
        """Convert state array to a tuple (hashable key)"""
        return tuple(state)  # Converts NumPy array to hashable format

    def act(self, state):
        state = self.encode_state(state)  # Ensure state is hashable
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        state = self.encode_state(state)
        next_state = self.encode_state(next_state)

        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])  # Get best action for next state
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] # * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error  # Update Q-table

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay