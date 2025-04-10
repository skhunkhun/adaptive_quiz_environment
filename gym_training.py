# train_agents.py
import numpy as np
import matplotlib.pyplot as plt
from gym_env import QuizEnvironment
from gym_qlearning import QLearningAgent
from gym_dqn import DQNAgent

def train_agents(student_profile='beginner', episodes=100, seed=50):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Create the environment with the seed
    env = QuizEnvironment(student_profile=student_profile, seed=seed)
    state = env.reset()
    state_size = state.shape[0]  # The length of the state vector
    action_size = env.action_space.n

    # Initialize agents
    q_agent = QLearningAgent(state_size, action_size)
    dqn_agent = DQNAgent(state_size, action_size)

    # Initialize reward lists for different policies
    easy_only_rewards = []
    q_rewards = []
    dqn_rewards = []
    
    update_target_frequency = 10  # Frequency (in episodes) for updating DQN's target network

    for episode in range(episodes):
        # ----- Easy-Only Policy -----
        state = env.reset()
        total_easy_reward = 0
        done = False
        while not done:
            action = 0  # Always choose 'Easy' (action index 0)
            next_state, reward, done, _ = env.step(action)
            total_easy_reward += reward
            state = next_state
        easy_only_rewards.append(total_easy_reward)
        
        # ----- Q-Learning Policy -----
        state = env.reset()
        total_q_reward = 0
        done = False
        while not done:
            action = q_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            q_agent.learn(state, action, reward, next_state, done)
            total_q_reward += reward
            state = next_state
        q_rewards.append(total_q_reward)
        # Decay the exploration rate for Q-learning agent
        if q_agent.epsilon > q_agent.epsilon_min:
            q_agent.epsilon *= q_agent.epsilon_decay
        
        # ----- DQN Policy -----
        state = env.reset()
        total_dqn_reward = 0
        done = False
        step_counter = 0
        while not done:
            action = dqn_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            dqn_agent.remember(state, action, reward, next_state, done)
            total_dqn_reward += reward
            state = next_state
            
            step_counter += 1
            # Call replay every 4 steps (adjust as needed)
            # if step_counter % 4 == 0:
                # print("REPLAYING")
            dqn_agent.replay()
        dqn_rewards.append(total_dqn_reward)
        # Update target network every few episodes
        if episode % update_target_frequency == 0:
            dqn_agent.update_target_model()
        
        # Print progress every 5 episodes
        if (episode + 1) % 5 == 0:
            print(f"Episode: {episode + 1}")
            print(f"  Easy-Only Reward: {total_easy_reward}")
            print(f"  Q-Learning Reward: {total_q_reward}")
            print(f"  DQN Reward: {total_dqn_reward}")
            env.render('human')
    
    # Plot the learning curves for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(easy_only_rewards, label='Easy-Only Policy', color='blue')
    plt.plot(q_rewards, label='Q-Learning Policy', color='green')
    plt.plot(dqn_rewards, label='DQN Policy', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Episode Rewards Comparison ({student_profile.capitalize()} Student)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # Run training for each student profile
    train_agents(student_profile='beginner', seed=1)
    train_agents(student_profile='intermediate', seed=1)
    train_agents(student_profile='advanced', seed=1)