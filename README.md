# Flappy Bird DQN Agent

## 1. Short Description
This project implements Deep Q-Learning (DQN) to train an agent to play Flappy Bird. The agent is designed to learn through reinforcement learning techniques, navigating a dynamic environment to maximize its survival and score. Two models are implemented to compare performance and adaptability.

---

## 2. Types of Models Implemented
### 1. **Positional Model**  
   - Utilizes positional data from the game environment.  
   - Reward system: Positive for survival, negative for collisions.  
   - Optimization: Uses RMSProp with MSE loss.  

### 2. **RGB Model**  
   - Processes RGB frames of the game environment.  
   - Reward system: Positive for survival, negative for collisions.  
   - Optimization: Updates target and policy networks every 10 episodes.  

---

## 3. Setup

1. **Create and activate a virtual environment**  
   ```bash
   python -m venv flappy_env_3
   source flappy_env_3/bin/activate  # For Linux/Mac
   flappy_env_3\Scripts\activate     # For Windows
   ```

2. **Add the project path to PYTHONPATH**  
<mark>To ensure the project is accessible as a module, add the project path to the PYTHONPATH environment variable. Replace ...\5100_final_project\flappy_bird_gym with your project's complete path.</mark>

   For example:
   ```bash
   set PYTHONPATH=...\5100_final_project\flappy_bird_gym  # For Windows
   export PYTHONPATH=...\5100_final_project\flappy_bird_gym  # For Linux/Mac
   ```

3. **Run the Positional Model**  
   ```bash
   python flappy_dqn_agent.py
   ```

4. **Run the RGB Model**  
   ```bash
   python flappy_dqn_rgb.py
   ```

---

## 4. Authors
- Mehul  
- Likhith  
- Gabriel  
- Navya  
- Fuhan  
