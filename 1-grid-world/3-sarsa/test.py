import random
import numpy as np
import gymnasium as gym


# Frozen Lake 에서의 SARSA Agent
class Agent:
    def __init__(self, _env: gym.make):
        # Agent 변수 초기화
        self.env = _env
        self.q_table = np.load("save_model/sarsa-model.npy")

    # Q-Table 과 Epsilon-Greedy Policy 에 따라 Action 반환
    def get_action(self, _state: int):
        max_idx_list = np.argwhere(self.q_table[_state] == np.amax(self.q_table[_state]))
        max_idx_list = max_idx_list.flatten().tolist()
        return random.choice(max_idx_list)


if __name__ == "__main__":
    # FrozenLake Environment 생성
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")

    # SARSA Agent 객체 생성
    agent = Agent(env)

    # SARSA 수행
    num_episodes = 10

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
            state = next_state
            score += reward

            if done:
                # 각 Episode 학습 결과 출력
                print("episode: {:3d} | score: {:3d}".format(episode, int(score)))
                break
