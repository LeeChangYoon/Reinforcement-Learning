import numpy as np
import gymnasium as gym
from gym.wrappers import RecordVideo

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform


# State 가 입력, Q Function 이 출력인 Neural Network 생성
class DQN(tf.keras.Model):
    def __init__(self, _num_actions: int):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(_num_actions, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


# CartPole 의 DQN Agent
class Agent:
    def __init__(self, _env: gym.make):
        # Action 과 State 의 크기 정의
        self.env = _env
        self.episode = 1
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]

        # Model 과 Target Model 생성
        self.model = DQN(self.num_actions)
        self.model.load_weights("./save_model/cartpole-dqn-model/model")

    # Epsilon Greedy Policy 로 행동 선택
    def get_action(self, _state: np.ndarray):
        values = self.model(_state)
        return np.argmax(values[0])


if __name__ == "__main__":
    # CartPole-v1 State, 최대 Time-Step 수가 500
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(env, './save_video/cartpole-dqn-video')

    # DQN Agent 생성
    agent = Agent(env)

    # Episode 수행
    score_avg = 0
    scores, episodes = [], []

    for episode in range(agent.episode):
        done = False
        score = 0

        # env 초기화
        state = env.reset()[0]
        state = np.reshape(state, [1, agent.num_states])

        while not done:
            # 현재 State 로 Action 을 선택
            action = agent.get_action(state)

            # 선택한 State 으로 Environment 에서 한 Time-Step 진행
            next_state, reward, done, info, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.num_states])

            score += reward
            state = next_state

            if done:
                # Episode 마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score: {:.3f} ".format(episode, score))
                break
