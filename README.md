# Adaptive Quiz Environment

This project implements an adaptive quiz environment using reinforcement learning. The environment simulates a student's learning process by adjusting the difficulty of quiz questions based on the student's performance. It serves as a hands-on educational tool for studying core RL concepts such as state representation, reward shaping, and policy optimization.

## Features

- **Custom Environment:** Simulates a student's quiz performance with dynamic difficulty using different student profiles (Beginner, Intermediate, Advanced).
- **Multiple RL Approaches:**  
  - A Baseline Easy-Only policy that always selects the easiest questions.  
  - A table-based Q-learning agent.  
  - A Deep Q-Network (DQN) agent with experience replay and target network updates.
- **Reward Structure:** Balances rewards and penalties to encourage correct responses and appropriate progression while discouraging exploitation.

## Getting Started

### Prerequisites

- Python 3.6+
- [OpenAI Gym](https://github.com/openai/gym)
- [TensorFlow](https://www.tensorflow.org/) (for DQN)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/skhunkhun/adaptive_quiz_environment.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd adaptive_quiz_environment
   ```

### Running the Project

To train the agents and compare their performance, run the training script:
```bash
python gym_training.py
```

This script outputs learning curves and logs that compare the performance of the Easy-Only baseline, Q-learning, and DQN agents across student profiles.

## Project Structure

- **gym_env.py** – Contains the custom `QuizEnvironment` implementation.
- **gym_qlearning.py** – Implements the table-based Q-learning agent.
- **gym_dqn.py** – Implements the Deep Q-Network (DQN) agent.
- **gym_training.py** – Script to train and compare the baseline, Q-learning, and DQN policies.
- **README.md** – This file.
