import numpy as np
import gymnasium as gym


# Frozen Lake 에서의 Policy Iteration Agent
class Agent:
    def __init__(self, _env: gym.make):
        # Agent 변수 초기화
        self.env = _env
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n

        # Agent Value 와 Policy Table
        self.value_table = np.zeros(self.num_states)
        self.policy_table = np.zeros([self.num_states, self.num_actions]) / self.num_actions

        # Agent Hyperparameter
        self.threshold = 1e-20
        self.discount_factor = 0.9

    def policy_evaluation(self):
        while True:
            next_value_table = np.zeros(self.num_states)
            for s in range(self.num_states):
                value = 0
                policy = self.policy_table[s]
                for a, ap in enumerate(policy):
                    for p, ns, r, _ in self.env.P[s][a]:
                        value += ap * p * (r + self.discount_factor * self.value_table[ns])

                next_value_table[s] = value

            if np.max(np.abs(self.value_table - next_value_table)) < self.threshold:
                break
            self.value_table = next_value_table

    def policy_improvement(self):
        next_policy_table = np.zeros([self.num_states, self.num_actions])
        for s in range(self.num_states):
            values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                for p, ns, r, _ in self.env.P[s][a]:
                    values[a] += p * (r + self.discount_factor * self.value_table[ns])

            optimal_action = np.argmax(values)
            next_policy_table[s, optimal_action] = 1

        self.policy_table = next_policy_table

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            prev_policy_table = np.copy(self.policy_table)
            self.policy_improvement()
            if np.array_equal(prev_policy_table, self.policy_table):
                break

    def get_optimal_policy(self):
        return np.argmax(self.policy_table, axis=1)


if __name__ == "__main__":
    # FrozenLake Environment 생성
    env = gym.make('FrozenLake-v1', is_slippery=False,  render_mode="rgb_array")

    # Policy Iteration Agent 객체 생성
    agent = Agent(env)

    # Policy Iteration 수행
    agent.policy_iteration()

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
                np.save("save_model/policy-iteration-model.npy", np.array(optimal_policy))
                break
