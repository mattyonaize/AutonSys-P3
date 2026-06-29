# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def build_actor_network(state_shape, action_dim, hidden_layers=[256, 128]):
    inputs = tf.keras.Input(shape=state_shape)
    x = inputs
    
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
    outputs = tf.keras.layers.Dense(action_dim, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def build_critic_network(state_shape, hidden_layers=[256, 128]):
    inputs = tf.keras.Input(shape=state_shape)
    x = inputs
    
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    return tf.keras.Model(inputs, outputs)
    
# A2C Agent Implementation
class A2CAgent:
    def __init__(self, state_dim, action_dim, player_id, 
             learning_rate=0.001, gamma=0.99, entropy_coef=0.01, 
             value_coef=0.5, max_grad_norm=0.5, hidden_layers=[256, 128]):
        self.player_id = player_id
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
    
        self.actor = build_actor_network(state_dim, action_dim, hidden_layers)
        self.critic = build_critic_network(state_dim, hidden_layers)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
    
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        
    def select_action(self, state):
        # Policy-based actie selectie
        state = np.expand_dims(state, axis=0) # Belangrijk voor Atari
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs=probs[0])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.numpy()[0]), log_prob
        
    def update(self, states, actions, rewards, dones, next_states):
        states = np.array(states)
        actions = np.array(actions)
    
        # Compute returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
    
        # Compute values and advantages
        values = tf.squeeze(self.critic(states))
        values = tf.cast(values, tf.float32)
        advantages = returns - values
    
        # Update actor
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            dist = tfp.distributions.Categorical(probs=probs)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
            log_probs = dist.log_prob(actions_tensor)
        
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
            entropy = tf.reduce_mean(dist.entropy())
            total_actor_loss = actor_loss - self.entropy_coef * entropy
    
        actor_grads = tape.gradient(total_actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
        # Update critic
        with tf.GradientTape() as tape:
            values_pred = tf.squeeze(self.critic(states))
            critic_loss = tf.reduce_mean(tf.square(returns - values_pred))
    
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
      
        return total_actor_loss.numpy(), critic_loss.numpy()

    def train_episode(self, env, agent_name):
        states, actions, rewards, dones = [], [], [], []
        total_reward = 0
    
        obs, _ = env.reset()
        done = False
    
        while not done:
            action, _ = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.last()
            done = terminated or truncated
        
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
            total_reward += reward
            obs = next_obs
    
        # Update networks
        if len(states) > 0:
            actor_loss, critic_loss = self.update(states, actions, rewards, dones, states)
            self.episode_rewards.append(total_reward)
            self.episode_losses.append((actor_loss, critic_loss))
    
        return total_reward

    def evaluate(self, env, n_episodes=10):
        eval_rewards = []
    
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
        
            while not done:
                action, _ = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.last()
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs
        
            eval_rewards.append(episode_reward)
    
        return np.mean(eval_rewards), np.std(eval_rewards)