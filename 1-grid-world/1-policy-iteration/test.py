import numpy as np
import gymnasium as gym


# Frozen Lake 에서의 Value Iteration Agent
class Agent:
    def __init__(self, _env: gym.make):
        # Agent 변수 초기화
        self.env = _env
        self.policy_table = np.load("save_model/policy-iteration-model.npy")

    # Optimal Policy 에 따라 Action 반환
    def get_action(self, _state: int):
        return self.policy_table[_state]


if __name__ == "__main__":
    # FrozenLake Environment 생성
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")

    # Policy Iteration Agent 객체 생성
    agent = Agent(env)
    print(agent.policy_table)

    # Episode 수행
    num_episodes = 1
    for episode in range(num_episodes):
        done = False
        state = agent.env.reset()[0]

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 Action 으로 Environment 에서 한 Step 진행 후 Sample 수집
            next_state, _, done, _, _ = agent.env.step(action)
            state = next_state

            if done:
                # Episode 학습 결과 출력
                print("Episode finished")
                break
