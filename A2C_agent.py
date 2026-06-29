# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def build_actor_network(state_shape, action_dim):
    inputs = tf.keras.Input(shape=state_shape)

    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    outputs = tf.keras.layers.Dense(action_dim, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

def build_critic_network(state_shape):
    inputs = tf.keras.Input(shape=state_shape)

    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    return tf.keras.Model(inputs, outputs)
    
# A2C Agent Implementation
class A2CAgent:
    def __init__(self, state_dim, action_dim, player_id):
        self.player_id = player_id
        self.actor = build_actor_network(state_dim, action_dim)
        self.critic = build_critic_network(state_dim)
        self.optimizer = tf.keras.optimizers.Adam()
        
    def select_action(self, state):
        # Policy-based actie selectie
        state = np.expand_dims(state, axis=0) # Belangrijk voor Atari
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs=probs[0])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.numpy()[0]), log_prob
        
    def update(self, states, actions, rewards, dones, next_states):

        gamma = 0.99

        # 1. critic values
        values = tf.squeeze(self.critic(np.array(states)))
        values = tf.cast(values, tf.float32)

        # 2. compute returns
        returns = []
        R = 0

        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)

        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # 3. advantage
        advantages = returns - values

        with tf.GradientTape() as tape:

            probs = self.actor(np.array(states))
            dist = tfp.distributions.Categorical(probs=probs)
            
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            log_probs = dist.log_prob(actions)

            actor_loss = -tf.reduce_mean(log_probs * advantages)
            critic_loss = tf.reduce_mean(tf.square(advantages))
            entropy = tf.reduce_mean(dist.entropy())

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        grads = tape.gradient(loss,
            self.actor.trainable_variables + self.critic.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(grads,
                self.actor.trainable_variables + self.critic.trainable_variables)
        )

        grads, _ = tf.clip_by_global_norm(grads, 0.5)