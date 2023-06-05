import gym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform


# Policy Neural Network 와 Value Neural Network 생성
class A2C(tf.keras.Model):
    def __init__(self, _num_actions: int):
        super(A2C, self).__init__()
        self.actor_fc = Dense(24, activation='tanh')
        self.actor_out = Dense(_num_actions, activation='softmax', kernel_initializer=RandomUniform(-1e-3, 1e-3))

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
        # 렌더링 여부 정의
        self.render = False

        # Action 과 State 의 크기 정의
        self.env = _env
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.shape[0]

        # A2C Hyperparameter
        self.learning_rate = 0.001
        self.discount_factor = 0.99

        # Policy Neural Network 와 Value Neural Network 생성
        self.model = A2C(self.num_actions)

        # Optimizer 설정, Differentiation 이 너무 커지는 현상을 막기 위해 Clip-Norm 설정
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=5.0)

    # Policy Neural Network 의 출력을 받아 확률적 행동을 선택
    def get_action(self, _state: np.ndarray):
        policy, _ = self.model(_state)
        policy = np.array(policy[0])
        return np.random.choice(self.num_actions, 1, p=policy)[0]

    # 각 Time-Step 마다 Policy Neural Network 과 Value Neural Network 을 갱신
    def train_model(self, _state: np.ndarray, _action: int, _reward: int, _next_state: np.ndarray, _done: bool):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            policy, value = self.model(_state)
            _, next_value = self.model(_next_state)
            target = _reward + (1 - _done) * self.discount_factor * next_value[0]

            # Policy Neural Network 의 Loss Function 구하기
            one_hot_action = tf.one_hot([_action], self.num_actions)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            advantage = tf.stop_gradient(target - value[0])
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # Value Neural Network 의 Loss Function 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 Loss Function 으로 만들기
            new_loss = 0.2 * actor_loss + critic_loss

        # Loss Function 을 줄이는 Model 갱신
        grads = tape.gradient(new_loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(new_loss)


if __name__ == "__main__":
    # CartPole-v1 State, 최대 Time-Step 수가 500
    env = gym.make('CartPole-v1', render_mode="rgb_array")

    # A2C Agent 생성
    agent = Agent(env)

    # Episode 수행
    score_avg = 0
    scores, episodes = [], []

    num_episode = 1000
    for episode in range(num_episode):
        done = False
        score = 0
        loss_list = []
        state = env.reset()[0]
        state = np.reshape(state, [1, agent.num_states])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.num_states])

            # Time-Step 마다 Reward 0.1, Episode 가 중간에 끝나면 -1 Reward
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # 매 Time-Step 마다 학습
            loss = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            state = next_state

            if done:
                # Episode 마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f}".format(episode, score_avg, np.mean(loss_list)))

                # Episode 마다 학습 결과 Graph 로 저장
                scores.append(score_avg)
                episodes.append(episode)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("episode")
                plt.ylabel("average score")
                plt.savefig("./save_graph/cartpole-a2c-graph-train.png")

                # Moving Average 가 400 이상일 때 종료
                if score_avg > 400:
                    agent.model.save_weights("./save_model/cartpole-a2c-model/model", save_format="tf")
                    sys.exit()
                break
