"""Training loop for signal weighting RL agent using PPO."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from environment import SignalWeightingEnv
from agent import PPOAgent
from data_generator import generate_synthetic_data


def train(n_episodes: int = 1000, episode_length: int = 252, lr: float = 3e-4,
          gamma: float = 0.99, checkpoint_freq: int = 200,
          data_path: str = "data/synthetic_data.csv", output_dir: str = "checkpoints",
          ppo_epochs: int = 10, batch_size: int = 64) -> tuple[list[float], list[np.ndarray]]:
    """Train the PPO agent."""
    if not Path(data_path).exists():
        print("Generating synthetic data...")
        generate_synthetic_data(output_path=data_path)

    env = SignalWeightingEnv(data_path, episode_length)
    obs_dim = env.observation_space.shape[0]
    agent = PPOAgent(obs_dim, env.n_signals, lr=lr, gamma=gamma)

    print(f"Training PPO: {env.n_signals} signals, obs_dim={obs_dim}, {n_episodes} episodes")
    print(f"PPO epochs={ppo_epochs}, batch_size={batch_size}, lr={lr}")

    episode_returns, mean_weights = [], []
    best_avg_return = -np.inf

    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward, ep_weights = 0.0, []

        for step in range(episode_length):
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store_reward(reward, done)
            ep_reward += reward
            ep_weights.append(info["weights"])
            state = next_state
            if done:
                break

        # PPO update at end of episode
        stats = agent.update(n_epochs=ppo_epochs, batch_size=batch_size)
        episode_returns.append(ep_reward)
        mean_w = np.mean(ep_weights, axis=0)
        mean_weights.append(mean_w)

        # Logging
        if (ep + 1) % 10 == 0:
            avg10 = np.mean(episode_returns[-10:])
            avg100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else avg10
            print(f"Ep {ep+1:4d} | Ret: {ep_reward:8.1f} | Avg10: {avg10:8.1f} | Avg100: {avg100:8.1f} | W: {mean_w.round(2)}")

            # Track best
            if avg100 > best_avg_return and len(episode_returns) >= 100:
                best_avg_return = avg100
                agent.save(f"{output_dir}/policy_best.pt")

        if (ep + 1) % checkpoint_freq == 0:
            agent.save(f"{output_dir}/policy_ep{ep+1}.pt")

    agent.save(f"{output_dir}/policy_final.pt")
    print(f"\nDone. Best Avg100: {best_avg_return:.1f}")
    print(f"Models saved to {output_dir}/")
    return episode_returns, mean_weights


def plot_learning_curve(returns: list[float], weights: list[np.ndarray], path: str = "learning_curve.png"):
    """Plot training progress."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Returns with multiple smoothing windows
    ax1 = axes[0]
    ax1.plot(returns, alpha=0.2, color='blue', label="Episode")
    for w, color in [(10, 'orange'), (50, 'green'), (100, 'red')]:
        if len(returns) >= w:
            smoothed = np.convolve(returns, np.ones(w)/w, "valid")
            ax1.plot(range(w-1, len(returns)), smoothed, label=f"MA({w})", color=color)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set(xlabel="Episode", ylabel="Return", title="Learning Curve")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Weight evolution
    ax2 = axes[1]
    weights_arr = np.array(weights)
    colors = plt.cm.viridis(np.linspace(0, 1, weights_arr.shape[1]))
    for i in range(weights_arr.shape[1]):
        # Smooth weights for cleaner visualization
        if len(weights_arr) >= 20:
            smoothed_w = np.convolve(weights_arr[:, i], np.ones(20)/20, "valid")
            ax2.plot(range(19, len(weights_arr)), smoothed_w, label=f"Signal {i+1}", color=colors[i], linewidth=2)
        else:
            ax2.plot(weights_arr[:, i], label=f"Signal {i+1}", color=colors[i])
    ax2.axhline(y=0.2, color='k', linestyle='--', alpha=0.3, label="Uniform")
    ax2.set(xlabel="Episode", ylabel="Weight", title="Signal Weight Evolution (smoothed)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved: {path}")


def main():
    p = argparse.ArgumentParser(description="Train PPO agent for signal weighting")
    p.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    p.add_argument("--episode-length", type=int, default=252, help="Steps per episode")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--ppo-epochs", type=int, default=10, help="PPO update epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--checkpoint-freq", type=int, default=200, help="Checkpoint frequency")
    p.add_argument("--data-path", default="data/synthetic_data.csv")
    p.add_argument("--output-dir", default="checkpoints")
    a = p.parse_args()

    returns, weights = train(
        n_episodes=a.episodes, episode_length=a.episode_length, lr=a.lr,
        gamma=a.gamma, checkpoint_freq=a.checkpoint_freq,
        data_path=a.data_path, output_dir=a.output_dir,
        ppo_epochs=a.ppo_epochs, batch_size=a.batch_size
    )
    plot_learning_curve(returns, weights)


if __name__ == "__main__":
    main()
