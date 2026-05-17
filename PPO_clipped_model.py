import tensorflow as tf
from tensorflow.keras import layers


class PPOClipped(tf.keras.Model):
    def __init__(self, num_actions, num_hidden_units):
        super().__init__()

        self.actor_dense = layers.Dense(num_hidden_units, activation='relu')
        self.actor = layers.Dense(num_actions)

        self.critic_dense = layers.Dense(num_hidden_units, activation='relu')
        # V-network: outputs a single scalar V(s) instead of Q(s,a) per action
        self.critic = layers.Dense(1)

    def call(self, inputs):
        actor = self.actor_dense(inputs)
        critic = self.critic_dense(inputs)
        return self.actor(actor), self.critic(critic)
