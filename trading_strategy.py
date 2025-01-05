import numpy as np


class TradingStrategy:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def execute_strategy(self, episodes=10, render=False):
        total_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                if render:
                    self.env.render()

                # Use the agent's policy to select an action
                action = self.agent.act(state, epsilon=0.0)  # Exploitation only
                next_state, reward, done, info = self.env.step(action)

                # Update state and accumulate rewards
                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{episodes}: Total Reward: {episode_reward}")

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward Over {episodes} Episodes: {avg_reward}")
        return total_rewards
