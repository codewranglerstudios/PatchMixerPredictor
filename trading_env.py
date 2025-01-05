import numpy as np
import gym
from gym import spaces
from keras import keras.models
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import random

# Step 1: Feature Integration
class TradingEnvironment(gym.Env):
    def __init__(self, market_data, initial_balance=10000):
        super(TradingEnv, self).__init__()

        # Market data and features
        self.market_data = market_data
        self.current_step = 0
        self.balance = initial_balance
        self.holdings = 0
        self.net_worth = initial_balance

        # State space: [CNN prediction, account balance, holdings, net worth]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        # Get current market data point
        current_price = self.market_data[self.current_step]

        # Execute action
        if action == 1:  # Buy
            self.holdings += self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.holdings * current_price
            self.holdings = 0

        # Update net worth
        self.net_worth = self.balance + self.holdings * current_price

        # Increment step
        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1

        # Reward: Net worth increase
        reward = self.net_worth - self.balance

        # State
        state = np.array([
            current_price, self.balance, self.holdings, self.net_worth
        ])

        return state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = initial_balance
        self.holdings = 0
        self.net_worth = initial_balance
        return np.array([
            self.market_data[self.current_step], self.balance, self.holdings, self.net_worth
        ])

# Step 2: RL Agent with Double Q-Learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Step 3: Training the Agent
def train_agent(env, agent, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e+1}/{episodes} - Time: {time} - Net Worth: {env.net_worth}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Example usage
if __name__ == "__main__":
    market_data = np.random.rand(1000)  # Simulated market data
    env = TradingEnvironment(market_data)
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    train_agent(env, agent)
