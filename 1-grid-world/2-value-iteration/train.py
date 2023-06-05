import random
import numpy as np
import gymnasium as gym


# Frozen Lake 에서의 Value Iteration Agent
class Agent:
    def __init__(self, _env: gym.make):
        # Agent 변수 초기화
        self.env = _env
        self.value_table = np.zeros(self.env.observation_space.n)

        # Agent Hyperparameter
        self.threshold = 1e-20
        self.discount_factor = 0.9

    # Bellman Optimality Equation 을 통해 Optimal Policy 계산
    def value_iteration(self):
        while True:
            # 다음 State-Value Table 초기화
            next_value_table = np.zeros(self.env.observation_space.n)

            # 모든 State 에 대한 Bellman Optimality Equation 계산
            for s in range(self.env.observation_space.n):
                # Bellman Optimality Equation
                values = []
                for a in range(self.env.action_space.n):
                    total_reward = 0
                    for p, ns, r, _ in self.env.P[s][a]:
                        total_reward += p * (r + self.discount_factor * self.value_table[ns])

                    values.append(total_reward)

                # Maximum 값을 다음 State-Value Function 으로 대입
                next_value_table[s] = max(values)

            if np.max(np.abs(self.value_table - next_value_table)) < self.threshold:
                break
            self.value_table = next_value_table

    # 현재 State-Value Function 에 따라 Action 반환
    def get_action(self, _state: int):
        if _state == self.env.observation_space.n - 1:
            return []

        # 모든 Action 에 대해 Q Function 을 계산
        value_list = []
        for a in range(self.env.action_space.n):
            _, ns, r, _ = self.env.P[_state][a][0]
            value_list.append(r + self.discount_factor * self.value_table[_state])

        # Maximum Q Function 을 가진 Action 을 반환
        max_idx_list = np.argwhere(value_list == np.amax(value_list))
        action_list = max_idx_list.flatten().tolist()
        return random.choice(action_list)

    # Optimal Policy 연산
    def get_optimal_policy(self):
        new_optimal_policy = [0 for _ in range(self.env.observation_space.n)]
        for s in range(self.env.observation_space.n):
            values = np.zeros(self.env.observation_space.n)
            for a in range(self.env.action_space.n):
                for p, ns, r, _ in self.env.P[s][a]:
                    values[a] += p * (r + self.discount_factor * self.value_table[ns])

            new_optimal_policy[s] = int(np.argmax(values))

        return new_optimal_policy


if __name__ == "__main__":
    # FrozenLake Environment 생성
    env = gym.make('FrozenLake-v1', is_slippery=False,  render_mode="rgb_array")

    # Value-Iteration Agent 객체 생성
    agent = Agent(env)

    # Value Iteration 수행
    agent.value_iteration()

    # Optimal Policy 구하기
    optimal_policy = agent.get_optimal_policy()

    # Episode 수행
    num_episodes = 1
    for episode in range(num_episodes):
        done = False
        state = agent.env.reset()[0]

        while not done:
            # 현재 상태에 대한 행동 선택
            action = optimal_policy[state]

            # 선택한 Action 으로 Environment 에서 한 Step 진행 후 Sample 수집
            next_state, _, done, _, _ = agent.env.step(action)
            state = next_state

            if done:
                # Episode 학습 결과 출력
                print("Episode finished")
                np.save("save_model/value-iteration-model.npy", np.array(optimal_policy))
                break
