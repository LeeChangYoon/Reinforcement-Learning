import sys
import random
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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
        # 렌더링 여부 정의
        self.render = False

        # Action 과 State 의 크기 정의
        self.env = _env
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]

        # DQN Hyperparameter
        self.epsilon = 1.0
        self.episode = 300
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.train_start = 1000
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.discount_factor = 0.99

        # Replay Memory, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # Model 과 Target Model 생성
        self.model = DQN(self.num_actions)
        self.target_model = DQN(self.num_actions)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # Target Model 초기화
        self.update_target_model()

    # Target Model 을 Model 의 Weight 로 갱신
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Epsilon Greedy Policy 로 행동 선택
    def get_action(self, _state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            values = self.model(_state)
            return np.argmax(values[0])

    # Sample 을 Replay Memory 에 저장
    def append_sample(self, _state: np.ndarray, _action: int, _reward: int, _next_state: np.ndarray, _done: bool):
        self.memory.append((_state, _action, _reward, _next_state, _done))

    # Replay Memory 에서 Random 추출한 Batch 로 Model 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Memory 에서 Batch Size 만큼 Random Sample 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # Train Parameter
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 State 에 대한 Model 의 Q Function
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.num_actions)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 State 에 대한 Target Model 의 Q Function
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            # Bellman Optimality Function 을 이용한 Update Target
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # Loss Function 를 줄이는 Model 갱신
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    # CartPole-v1 State, 최대 Time-Step 수가 500
    env = gym.make('CartPole-v1', render_mode="rgb_array")

    # DQN Agent 생성
    agent = Agent(env)

    # Episode 수행
    score_avg = 0
    scores, episodes = [], []

    for episode in range(agent.episode):
        score = 0
        done = False

        # env 초기화
        state = env.reset()[0]
        state = np.reshape(state, [1, agent.num_states])

        while not done:
            if agent.render:
                env.render()

            # 현재 State 로 Action 을 선택
            action = agent.get_action(state)

            # 선택한 State 으로 Environment 에서 한 Time-Step 진행
            next_state, reward, done, info, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.num_states])

            # Time-Step 마다 Reward 0.1, Episode 가 중간에 끝나면 -1 Reward
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # Replay Memory 에 Sample 저장
            agent.append_sample(state, action, reward, next_state, done)

            # 매 Time-Step 마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                # 각 Episode 마다 Target Model 을 Model 의 Weight 로 갱신
                agent.update_target_model()

                # Episode 마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | memory length: {:4d} | epsilon: {:.4f}".format(episode, score_avg, len(agent.memory), agent.epsilon))

                # Episode 마다 학습 결과 Graph 로 저장
                scores.append(score_avg)
                episodes.append(episode)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("episode")
                plt.ylabel("average score")
                plt.savefig("./save_graph/cartpole-dqn-graph-train.png")

                # Moving Average 가 400 이상일 때 종료
                if score_avg > 400:
                    agent.model.save_weights("./save_model/cartpole-dqn-model/model", save_format="tf")
                    sys.exit()
                break
