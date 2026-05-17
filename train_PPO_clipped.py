import csv

from PPO_clipped_model import PPOClipped
import gymnasium as gym
import numpy as np
import tensorflow as tf


def train_episode(env, model, optimizer, gamma=0.99, lam=0.97, n_epochs=4, clip_epsilon=0.2, entropy_coef=0.01, critic_coef=0.5, target_kl=0.05, minibatch_size=64):
    states, actions, rewards, log_probs, values_list, dones = [], [], [], [], [], []
    state, _ = env.reset()
    done = False

    # collect full episode without tape, same collection strategy as REINFORCE
    while not done:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_logits, value = model(state_tensor)
        action = int(tf.random.categorical(action_logits, 1)[0, 0].numpy())

        # calculate the log probability of the action chosen
        neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=[action],
            logits=action_logits
        )

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(np.array(state, dtype=np.float32).flatten())
        actions.append(action)
        rewards.append(float(reward))
        # neg_log_prob is negative, so it has to be negated
        log_probs.append(float(-tf.squeeze(neg_log_probs).numpy()))
        values_list.append(float(tf.squeeze(value).numpy()))
        dones.append(float(done))


        state = next_state

    T = len(rewards)

    # GAE advantage estimation
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values_list[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values_list[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + np.array(values_list, dtype=np.float32)

    states_tensor = tf.constant(states, dtype=tf.float32)
    actions_tensor = tf.constant(actions, dtype=tf.int32)
    returns_tensor = tf.constant(returns, dtype=tf.float32)
    log_probs_tensor = tf.constant(log_probs, dtype=tf.float32)

    # calculate advantages
    advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages_tensor = tf.constant(advantages_norm, dtype=tf.float32)

    indices = np.arange(T)

    # multiple epochs of minibatch SGD
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        early_stop = False

        for start in range(0, T, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]

            mb_states = tf.gather(states_tensor, mb_idx)
            mb_actions = tf.gather(actions_tensor, mb_idx)
            mb_returns = tf.gather(returns_tensor, mb_idx)
            mb_old_log_probs = tf.gather(log_probs_tensor, mb_idx)
            mb_advantages = tf.gather(advantages_tensor, mb_idx)

            with tf.GradientTape() as tape:
                action_logits, values = model(mb_states)
                # squeeze critic output from (T, 1) to (T,)
                values = tf.squeeze(values, axis=1)

                # critic loss
                critic_loss = tf.reduce_mean(tf.square(mb_returns - values))

                # actor loss
                neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=mb_actions,
                    logits=action_logits
                )
                log_probs_new = -neg_log_probs

                log_ratio = log_probs_new - mb_old_log_probs
                ratio = tf.exp(log_ratio)

                unclipped = ratio * mb_advantages
                clipped = tf.clip_by_value(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages

                actor_loss = tf.reduce_mean(tf.minimum(unclipped, clipped))

                probs = tf.nn.softmax(action_logits)
                log_prob = tf.nn.log_softmax(action_logits)
                entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_prob, axis=1))

                # calculate loss
                total_loss = critic_loss * critic_coef - actor_loss - entropy * entropy_coef

            grads = tape.gradient(total_loss, model.trainable_variables)
            clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

            approximate_kl = tf.reduce_mean(-log_ratio).numpy()
            if approximate_kl > target_kl:
                early_stop = True
                break

        if early_stop:
            break

    return sum(rewards)


def run_episodes(env, model, optimizer, n_episodes):
    rewards_list = []
    for ep in range(n_episodes):
        rewards_list.append(train_episode(env, model, optimizer))
        if (ep + 1) % 100 == 0:
            recent = np.mean(rewards_list[-100:])
            print(f"  episode {ep + 1}/{n_episodes}  avg return (last 100): {recent:.1f}")
    return rewards_list


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    seeds = [69, 420, 666, 69420, 6669]

    num_actions = int(env.action_space.n)
    num_hidden_units = 128

    with open('results_ppo.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Config', 'Seed', 'Step', 'Score'])

        for seed in seeds:

            tf.random.set_seed(seed)
            np.random.seed(seed)

            model = PPOClipped(num_actions, num_hidden_units)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            rewards_list = run_episodes(env, model, optimizer, 800)

            cumsteps = np.cumsum(rewards_list).astype(int)
            for step, score in zip(cumsteps, rewards_list):
                writer.writerow(['PPO', seed, step, score])

