import os
import random
import numpy as np
import gymnasium as gym
from collections import deque

from skimage.color import rgb2gray
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Dense, Flatten


# State 가 Input, Q Function 이 Output 인 Neural Network 생성
class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size)
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.fc = Dense(512, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q


# Breakout 에서의 DQN Agent
class DQNAgent:
    def __init__(self, _action_size: int, _state_size=(84, 84, 4)):
        # 렌더링 여부 걸정
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = _state_size
        self.action_size = _action_size

        # DQN Hyperparameter
        self.epsilon = 1.
        self.batch_size = 32
        self.train_start = 1000
        self.learning_rate = 1e-4
        self.discount_factor = 0.99
        self.update_target_rate = 10000
        self.exploration_steps = 1000000.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps

        # Replay Memory <= 100,000
        self.memory = deque(maxlen=100000)

        # 게임 시작 후 Random Action 옵션
        self.no_op_steps = 30

        # Model 과 Target Model 생성
        self.model = DQN(_action_size, _state_size)
        self.target_model = DQN(_action_size, _state_size)
        self.optimizer = Adam(self.learning_rate, clipnorm=10.)

        # Target Model 초기화
        self.update_target_model()

        self.avg_q_max, self.avg_loss = 0, 0

        self.writer = tf.summary.create_file_writer('summary/breakout_dqn')
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # Target Model 을 Model 의 Weight 로 갱신
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Epsilon-Greedy Policy 로 Action 선택
    def get_action(self, _history: np.ndarray):
        _history = np.float32(_history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(_history)
            return np.argmax(q_value[0])

    # Sample 을 Replay Memory 에 저장
    def append_sample(self, _history: np.ndarray, _action: int, _reward: np.ndarray, _next_history: np.ndarray, _dead: bool):
        self.memory.append((_history, _action, _reward, _next_history, _dead))

    # Tensorboard 에 학습 정보를 기록
    def draw_tensorboard(self, _score: int, _step: int, _episode: int):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', _score, step=_episode)
            tf.summary.scalar('Average Max Q/Episode', self.avg_q_max / float(_step), step=_episode)
            tf.summary.scalar('Duration/Episode', _step, step=_episode)
            tf.summary.scalar('Average Loss/Episode', self.avg_loss / float(_step), step=_episode)

    # Replay Memory 에서 Random 추출한 Batch 로 Model 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # Memory 서 Batch Size 만큼 Random 샘플 추출
        batch = random.sample(self.memory, self.batch_size)

        _history = np.array([sample[0][0] / 255. for sample in batch], dtype=np.float32)
        _actions = np.array([sample[1] for sample in batch])
        _rewards = np.array([sample[2] for sample in batch])
        _next_history = np.array([sample[3][0] / 255. for sample in batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in batch])

        # Training Parameters
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 State 에 대한 Model 의 Q Function
            predicts = self.model(_history)
            one_hot_action = tf.one_hot(_actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 State 에 대한 Target Model 의 Q Function
            target_predicts = self.target_model(_next_history)

            # Bellman Optimality Equation 을 위한 Target 과 Q Function 의 최대 값 계산
            max_q = np.amax(target_predicts, axis=1)
            targets = _rewards + (1 - dones) * self.discount_factor * max_q

            # Huber Loss 계산
            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            self.avg_loss += loss.numpy()

        # Loss Function 를 줄이는 Model 갱신
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


# Learning Rate 을 높이기 위해 흑백 전처리
def pre_processing(_observe: np.ndarray):
    processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # DQN Agent 생성
    agent = DQNAgent(_action_size=3)
    env = gym.make('BreakoutDeterministic-v4')

    score_avg = 0
    score_max = 0
    global_step = 0

    # 불 필요한 행동을 없애기 위한 Dictionary 선언
    num_episode = 1000
    action_dict = {0: 1, 1: 2, 2: 3, 3: 3}

    for e in range(num_episode):
        done = False
        dead = False
        step, score, start_life = 0, 0, 5

        # env 초기화
        observe = env.reset()[0]

        # Random 값 만큼의 Frame 동안 Action 없음
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _, _ = env.step(1)

        # Frame 을 전처리 한 후 4개의 상태를 쌓아서 Input 으로 사용.
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # 바로 전 history 를 Input 으로 받아 행동을 선택
            action = agent.get_action(history)
            
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            real_action = action_dict[action]

            # 죽었을 때 시작을 위해 발사 행동을 함
            if dead:
                action, real_action, dead = 0, 1, False

            # 선택한 Action 으로 Environment 에서 한 Time-Step 진행
            observe, reward, done, _, info = env.step(real_action)

            # 각 Time-Step State 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(agent.model(np.float32(history / 255.))[0])

            if start_life > info['lives']:
                dead = True
                start_life = info['lives']

            score += reward
            reward = np.clip(reward, -1., 1.)

            # Sample 을 Replay Memory 에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)

            # Replay Memory 크기가 정한 수치에 도달한 시점에 Model Training 시작
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

                # 일정 Episode 마다 Target Model 을 Model 의 Weight 로 갱신
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

            if dead:
                history = np.stack((next_state, next_state,
                                    next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            if done:
                # 각 Episode 당 Training Information 을 기록
                if global_step > agent.train_start:
                    agent.draw_tensorboard(score, step, e)

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                score_max = score if score > score_max else score_max

                log = "episode: {:5d} | ".format(e)
                log += "score: {:4.1f} | ".format(score)
                log += "score max : {:4.1f} | ".format(score_max)
                log += "score avg: {:4.1f} | ".format(score_avg)
                log += "memory length: {:5d} | ".format(len(agent.memory))
                log += "epsilon: {:.3f} | ".format(agent.epsilon)
                log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                log += "avg loss : {:3.2f}".format(agent.avg_loss / float(step))
                print(log)

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 Episode 마다 Model 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/model", save_format="tf")
