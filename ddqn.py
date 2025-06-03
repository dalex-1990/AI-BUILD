import sys
import gym
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
import matplotlib.pyplot as plt

# Configure GPU and enable mixed precision for faster training
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configuration successful. Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

# Enable mixed precision for faster training on modern GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled for faster training")


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # minimum epsilon
        self.epsilon_decay = 0.995 #rate of epsilon decay
        self.learning_rate = 0.001
        self.model = self._build_model()#build the model

    def _build_model(self):
        #this function builds the MLP model with relu activations in 2 the hidden layers, and linear activation in the output layer
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear', dtype='float32'))  # Ensure float32 output for mixed precision
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))#specify the type of loss and the learning rate
        return model

    def remember(self, state, action, reward, next_state, done):
        #This function pushes instances of experience into the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #epsilon-greedy exploration to select actions
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state,verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)#select a minibatch of size batch_size
        states = np.array([self.memory[i][0].flatten() for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3].flatten() for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        targets = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)#form the learning targets
        target_f = self.model.predict_on_batch(states)#current estimate of Q values
        target_f[np.arange(batch_size), actions] = targets#update only the Q values corresponding to the action taken. Leave the others unchanged.

        self.model.fit(states, target_f, epochs=1, verbose=0)#train the model

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.target_model = self._build_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0].flatten() for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3].flatten() for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        next_actions = np.argmax(self.model.predict_on_batch(next_states), axis=1)
        target_q_values = self.target_model.predict_on_batch(next_states)
        targets = rewards + self.gamma * target_q_values[np.arange(batch_size), next_actions] * (1 - dones)

        target_f = self.model.predict_on_batch(states)
        target_f[np.arange(batch_size), actions] = targets

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(agent_class, seed, episodes=200, batch_size=192):
    env = gym.make('CartPole-v0')
    env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent on GPU if available
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
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
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done:
                break
        if hasattr(agent, 'update_target_model') and e % 2 == 0:
            agent.update_target_model()
        rewards.append(total_reward)
        steps.append(global_step)
    return steps, rewards

seeds = [0]
dqn_steps_all = []
dqn_rewards_all = []
ddqn_steps_all = []
ddqn_rewards_all = []

for seed in seeds:
    print(f"Training with seed {seed}...")
    steps_dqn, rewards_dqn = train_agent(DQNAgent, seed)
    steps_ddqn, rewards_ddqn = train_agent(DoubleDQNAgent, seed)
    dqn_steps_all.append(steps_dqn)
    dqn_rewards_all.append(rewards_dqn)
    ddqn_steps_all.append(steps_ddqn)
    ddqn_rewards_all.append(rewards_ddqn)

dqn_rewards_mean = np.mean(dqn_rewards_all, axis=0)
dqn_rewards_std = np.std(dqn_rewards_all, axis=0) / np.sqrt(len(seeds))
dqn_steps_mean = np.mean(dqn_steps_all, axis=0)

ddqn_rewards_mean = np.mean(ddqn_rewards_all, axis=0)
ddqn_rewards_std = np.std(ddqn_rewards_all, axis=0) / np.sqrt(len(seeds))
ddqn_steps_mean = np.mean(ddqn_steps_all, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(dqn_steps_mean, dqn_rewards_mean, label='DQN')
plt.fill_between(dqn_steps_mean, dqn_rewards_mean - dqn_rewards_std, dqn_rewards_mean + dqn_rewards_std, alpha=0.2)

plt.plot(ddqn_steps_mean, ddqn_rewards_mean, label='Double DQN')
plt.fill_between(ddqn_steps_mean, ddqn_rewards_mean - ddqn_rewards_std, ddqn_rewards_mean + ddqn_rewards_std, alpha=0.2)

plt.title('DQN vs Double DQN on CartPole-v0')
plt.xlabel('Learning Steps')  
plt.ylabel('Total Reward per Episode')
plt.legend()
plt.grid()
plt.savefig('dqn_vs_double_dqn_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'dqn_vs_double_dqn_comparison.png'")
plt.close()  # Close the figure to free memory
