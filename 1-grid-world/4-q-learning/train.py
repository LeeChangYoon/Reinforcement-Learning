import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# Frozen Lake 에서의 Q-Learning Agent
class Agent:
    def __init__(self, _env: gym.make):
        # Agent 변수 초기화
        self.env = _env
        self.q_table = np.zeros((_env.observation_space.n, _env.action_space.n))

        # Q-Learning Hyperparameter
        self.epsilon = 0.3
        self.learning_rate = 0.01
        self.discount_factor = 0.9

    # Q-Table 과 Epsilon-Greedy Policy 에 따라 Action 반환
    def get_action(self, _state: int):
        if np.random.rand() < self.epsilon:
            # Random Action 반환
            return self.env.action_space.sample()
        else:
            # Q-Table 에 따흔 Action 반환
            max_idx_list = np.argwhere(self.q_table[_state] == np.amax(self.q_table[_state]))
            max_idx_list = max_idx_list.flatten().tolist()
            return random.choice(max_idx_list)

    # Sample 에 따라 Q-Table 갱신
    def update_q_table(self, _state: int, _action: 1, _reward: float, _next_state: int):
        q_value = self.q_table[_state, _action]
        max_q_value = np.max(self.q_table[_next_state])

        # Bellman Optimality Equation 을 사용한 Q-Table 갱신
        new_q_value = (1 - self.learning_rate) * q_value + self.learning_rate * (_reward + self.discount_factor * max_q_value)
        self.q_table[_state, _action] = new_q_value


if __name__ == "__main__":
    # FrozenLake Environment 생성
    env = gym.make('FrozenLake-v1', is_slippery=False,  render_mode="rgb_array")

    # Q-Learning Agent 객체 생성
    agent = Agent(env)

    # Q-Learning 수행
    num_episodes = 1000
    scores, episodes = [], []

    for episode in range(num_episodes):
        score = 0
        done = False
        state = agent.env.reset()[0]

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 Action 으로 Environment 에서 한 Step 진행 후 Sample 수집
            next_state, reward, done, _, _ = agent.env.step(action)

            # Sample 로 Model 학습
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            score += reward

            if done:
                # 각 Episode 학습 결과 출력
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(episode, int(score), agent.epsilon))

                scores.append(score)
                episodes.append(episode)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("episode")
                plt.ylabel("score")
                plt.savefig("./save_graph/q-learning-graph.png")

                # 목적지 도달 시 종료
                np.save("save_model/q-learning-model.npy", agent.q_table)
                break
