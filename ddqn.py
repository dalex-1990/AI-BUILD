import sys
import gym
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
import matplotlib.pyplot as plt

# Configure GPU - GPU REQUIRED
print("TensorFlow version:", tf.__version__)

# Check for GPU availability - MANDATORY
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("ERROR: No GPU found!")
    print("This script requires a GPU to run. Please ensure:")
    print("1. NVIDIA GPU is installed")
    print("2. CUDA drivers are installed")
    print("3. TensorFlow-GPU is properly installed")
    sys.exit(1)

try:
    # Enable GPU memory growth to avoid allocating all GPU memory at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set GPU as the logical device
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"GPU configuration successful. Using GPU: {gpus[0]}")
    
    # Verify GPU is being used
    with tf.device('/GPU:0'):
        test_tensor = tf.constant([1.0, 2.0, 3.0])
        print(f"Test tensor device: {test_tensor.device}")
        
        # Ensure the tensor is actually on GPU
        if '/GPU:0' not in test_tensor.device:
            print("ERROR: GPU not properly configured!")
            sys.exit(1)
        
except RuntimeError as e:
    print(f"FATAL: GPU configuration error: {e}")
    print("Cannot proceed without GPU. Exiting...")
    sys.exit(1)

# Enable mixed precision for faster training on GPU
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled for faster GPU training")
except Exception as e:
    print(f"Warning: Mixed precision setup failed: {e}")
    print("Continuing with default precision...")

print(f"Available devices: {tf.config.list_logical_devices()}")
print("GPU-only mode: All operations will run on GPU")


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # minimum epsilon
        self.epsilon_decay = 0.995  # rate of epsilon decay
        self.learning_rate = 0.001
        
        # Build model with GPU placement - MANDATORY
        with tf.device('/GPU:0'):
            self.model = self._build_model()

    def _build_model(self):
        # Build the MLP model with relu activations in 2 hidden layers
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        # Use appropriate optimizer and loss
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        
        print(f"Model built on device: {model.layers[0].weights[0].device if model.layers else 'Unknown'}")
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        
        # Limit memory size to prevent excessive RAM usage
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def act(self, state):
        # Epsilon-greedy exploration - GPU ONLY
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        # Ensure state prediction runs on GPU
        with tf.device('/GPU:0'):
            act_values = self.model.predict(state, verbose=0)
            
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        # Sample minibatch
        minibatch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        # Prepare batch data
        states = np.array([self.memory[i][0].flatten() for i in minibatch_indices])
        actions = np.array([self.memory[i][1] for i in minibatch_indices])
        rewards = np.array([self.memory[i][2] for i in minibatch_indices], dtype=np.float32)
        next_states = np.array([self.memory[i][3].flatten() for i in minibatch_indices])
        dones = np.array([self.memory[i][4] for i in minibatch_indices], dtype=np.float32)

        # Perform training on GPU - MANDATORY
        with tf.device('/GPU:0'):
            # Convert to tensors for GPU processing
            states = tf.constant(states, dtype=tf.float32)
            next_states = tf.constant(next_states, dtype=tf.float32)
            
            # Compute targets
            next_q_values = self.model.predict_on_batch(next_states)
            targets = rewards + self.gamma * np.amax(next_q_values, axis=1) * (1 - dones)
            
            # Get current Q values
            target_f = self.model.predict_on_batch(states)
            target_f[np.arange(batch_size), actions] = targets
            
            # Train the model
            self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        
        # Build target model on GPU - MANDATORY
        with tf.device('/GPU:0'):
            self.target_model = self._build_model()
            
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = np.array([self.memory[i][0].flatten() for i in minibatch_indices])
        actions = np.array([self.memory[i][1] for i in minibatch_indices])
        rewards = np.array([self.memory[i][2] for i in minibatch_indices], dtype=np.float32)
        next_states = np.array([self.memory[i][3].flatten() for i in minibatch_indices])
        dones = np.array([self.memory[i][4] for i in minibatch_indices], dtype=np.float32)

        # Double DQN computation on GPU - MANDATORY
        with tf.device('/GPU:0'):
            states = tf.constant(states, dtype=tf.float32)
            next_states = tf.constant(next_states, dtype=tf.float32)
            
            # Double DQN: use main network to select actions, target network to evaluate
            next_actions = np.argmax(self.model.predict_on_batch(next_states), axis=1)
            target_q_values = self.target_model.predict_on_batch(next_states)
            targets = rewards + self.gamma * target_q_values[np.arange(batch_size), next_actions] * (1 - dones)

            target_f = self.model.predict_on_batch(states)
            target_f[np.arange(batch_size), actions] = targets

            self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(agent_class, seed, episodes=20, batch_size=32):
    # Create environment
    env = gym.make('CartPole-v0')
    env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent (model creation handles device placement internally)
    agent = agent_class(state_size, action_size)

    rewards = []
    steps = []
    global_step = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(200):
            global_step += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Train agent with sufficient experience
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            if done:
                break
                
        if e % 5 == 0:
            print(f"Episode {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Global Step: {global_step}")
            
        # Update target model for Double DQN
        if hasattr(agent, 'update_target_model') and e % 2 == 0:
            agent.update_target_model()
            
        rewards.append(total_reward)
        steps.append(global_step)
        
    env.close()
    return steps, rewards


# Main training loop
if __name__ == "__main__":
    seeds = [0]
    dqn_steps_all = []
    dqn_rewards_all = []
    ddqn_steps_all = []
    ddqn_rewards_all = []

    for seed in seeds:
        print(f"\nTraining with seed {seed}...")
        print("Training DQN...")
        steps_dqn, rewards_dqn = train_agent(DQNAgent, seed)
        print("Training Double DQN...")
        steps_ddqn, rewards_ddqn = train_agent(DoubleDQNAgent, seed)
        
        dqn_steps_all.append(steps_dqn)
        dqn_rewards_all.append(rewards_dqn)
        ddqn_steps_all.append(steps_ddqn)
        ddqn_rewards_all.append(rewards_ddqn)

    # Calculate means and standard errors
    dqn_rewards_mean = np.mean(dqn_rewards_all, axis=0)
    dqn_rewards_std = np.std(dqn_rewards_all, axis=0) / np.sqrt(len(seeds))
    dqn_steps_mean = np.mean(dqn_steps_all, axis=0)

    ddqn_rewards_mean = np.mean(ddqn_rewards_all, axis=0)
    ddqn_rewards_std = np.std(ddqn_rewards_all, axis=0) / np.sqrt(len(seeds))
    ddqn_steps_mean = np.mean(ddqn_steps_all, axis=0)

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_steps_mean, dqn_rewards_mean, label='DQN', linewidth=2)
    plt.fill_between(dqn_steps_mean, dqn_rewards_mean - dqn_rewards_std, 
                     dqn_rewards_mean + dqn_rewards_std, alpha=0.2)

    plt.plot(ddqn_steps_mean, ddqn_rewards_mean, label='Double DQN', linewidth=2)
    plt.fill_between(ddqn_steps_mean, ddqn_rewards_mean - ddqn_rewards_std, 
                     ddqn_rewards_mean + ddqn_rewards_std, alpha=0.2)

    plt.title('DQN vs Double DQN on CartPole-v0')
    plt.xlabel('Learning Steps')  
    plt.ylabel('Total Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_vs_double_dqn_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'dqn_vs_double_dqn_comparison.png'")
    plt.show()