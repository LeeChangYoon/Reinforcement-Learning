import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform


# Policy Neural Network 와 Value Neural Network 생성
class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.actor_fc = Dense(24, activation='tanh')
        self.actor_out = Dense(action_size, activation='softmax', kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return policy, value


# CartPole 의 A2C Agent
class Agent:
    def __init__(self, _env: gym.make):
        # Action 과 State 의 크기 정의
        self.env = _env
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]

        # Policy Neural Network 와 Value Neural Network 생성
        self.model = A2C(self.num_actions)
        self.model.load_weights("./save_model/cartpole-a2c-model/model")

    # Policy Neural Network 의 출력을 받아 확률적 행동을 선택
    def get_action(self, _state: np.ndarray):
        policy, _ = self.model(_state)
        policy = np.array(policy[0])
        return np.random.choice(self.num_actions, 1, p=policy)[0]


if __name__ == "__main__":
    # CartPole-v1 State, 최대 Time-Step 수가 500
    env = gym.make('CartPole-v1', render_mode="human")

    # A2C Agent 생성
    agent = Agent(env)

    # Episode 수행
    score_avg = 0
    num_episode = 30
    scores, episodes = [], []

    for episode in range(num_episode):
        done = False
        score = 0
        state = env.reset()[0]
        state = np.reshape(state, [1, agent.num_states])

        while not done:
            env.render()

            action = agent.get_action(state)
            next_state, reward, done, info, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.num_states])

            score += reward
            state = next_state

            if done:
                # Episode 마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score: {:3d}".format(episode, int(score)))

                # Episode 마다 학습 결과 Graph 로 저장
                scores.append(score_avg)
                episodes.append(episode)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("episode")
                plt.ylabel("average score")
                plt.savefig("./save_graph/cartpole-a2c-graph-test.png")

                break
