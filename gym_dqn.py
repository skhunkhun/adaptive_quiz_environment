import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size      # Size of the input state vector
        self.action_size = action_size    # Number of actions available
        self.memory = deque(maxlen=2000)  # Experience replay memory

        # Hyperparameters
        self.gamma = 0.95              # Discount factor for future rewards
        self.epsilon = 1.0             # Exploration rate (initially 1.0)
        self.epsilon_min = 0.01        # Minimum exploration rate
        self.epsilon_decay = 0.995     # Decay factor for exploration rate
        self.learning_rate = 0.001     # Learning rate for our optimizer
        self.batch_size = 32           # Mini-batch size for replay

        # Build the main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Builds a simple fully connected neural network for DQN."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Update the target network weights from the main network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Return an action using epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a minibatch of transitions
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Unpack the minibatch
        states = np.array([s for s, a, r, s_next, done in minibatch])
        actions = np.array([a for s, a, r, s_next, done in minibatch])
        rewards = np.array([r for s, a, r, s_next, done in minibatch])
        next_states = np.array([s_next for s, a, r, s_next, done in minibatch])
        dones = np.array([done for s, a, r, s_next, done in minibatch])
        
        # Predict Q-values for the next states (batch)
        q_next = self.target_model.predict(next_states, verbose=0)
        # Compute target values using vectorized operations
        target_values = rewards + self.gamma * np.amax(q_next, axis=1) * (1 - dones)

        # Predict Q-values for current states (batch)
        target_batch = self.model.predict(states, verbose=0)
        
        # Update the Q-values for the actions taken in the batch
        target_batch[np.arange(self.batch_size), actions] = target_values

        # Train on the entire batch in one call
        self.model.fit(states, target_batch, epochs=1, verbose=0)

        # Update epsilon (decay the exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay