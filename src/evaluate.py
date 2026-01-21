"""Evaluation and backtesting for trained signal weighting agent."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from environment import SignalWeightingEnv
from agent import PPOAgent


def compute_metrics(returns: np.ndarray) -> dict[str, float]:
    """Compute trading performance metrics."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return {
        "total_return": cum[-1] - 1,
        "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
        "max_dd": np.max((peak - cum) / peak),
        "vol": np.std(returns) * np.sqrt(252),
        "win_rate": np.mean(returns > 0),
    }


def evaluate(model_path: str = "checkpoints/policy_best.pt",
             data_path: str = "data/synthetic_data.csv", output_path: str = "evaluation.png",
             eval_length: int = 500):
    """Run deterministic evaluation of trained policy."""
    if not Path(model_path).exists():
        # Try final if best doesn't exist
        model_path = model_path.replace("_best", "_final")
        if not Path(model_path).exists():
            print(f"Model not found. Run train.py first.")
            return

    env = SignalWeightingEnv(data_path, episode_length=eval_length)
    obs_dim = env.observation_space.shape[0]
    agent = PPOAgent(obs_dim, env.n_signals)
    agent.load(model_path)
    print(f"Loaded: {model_path}")

    # Deterministic evaluation from start of data
    state, _ = env.reset(seed=42)
    env.start_idx = env.lookback  # Start after lookback period

    trade_ret, price_ret, weights_hist, positions = [], [], [], []

    for _ in range(min(eval_length, len(env.returns) - env.lookback - 1)):
        action = agent.select_action(state, deterministic=True)
        state, _, done, _, info = env.step(action)
        trade_ret.append(info["trading_return"])
        price_ret.append(info["price_return"])
        weights_hist.append(info["weights"])
        positions.append(info["position"])
        if done:
            break

    trade_ret = np.array(trade_ret)
    price_ret = np.array(price_ret)
    weights_hist = np.array(weights_hist)

    sm = compute_metrics(trade_ret)
    bm = compute_metrics(price_ret)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'RL Strategy':>15} {'Buy & Hold':>15}")
    print("-" * 60)
    print(f"{'Total Return':<20} {sm['total_return']:>14.2%} {bm['total_return']:>14.2%}")
    print(f"{'Sharpe Ratio':<20} {sm['sharpe']:>15.2f} {bm['sharpe']:>15.2f}")
    print(f"{'Max Drawdown':<20} {sm['max_dd']:>14.2%} {bm['max_dd']:>14.2%}")
    print(f"{'Volatility (Ann.)':<20} {sm['vol']:>14.2%} {bm['vol']:>14.2%}")
    print(f"{'Win Rate':<20} {sm['win_rate']:>14.2%} {bm['win_rate']:>14.2%}")

    print(f"\nLearned Signal Weights (mean):")
    for i, w in enumerate(weights_hist.mean(0)):
        bar = "█" * int(w * 40)
        print(f"  Signal {i+1}: {w:.3f} {bar}")

    # Check if weights are differentiated
    weight_std = weights_hist.mean(0).std()
    print(f"\nWeight differentiation (std): {weight_std:.3f}")
    if weight_std < 0.02:
        print("  ⚠ Weights are still near-uniform. Consider more training.")
    else:
        print("  ✓ Agent learned differentiated weights.")

    # Plot
    plot_evaluation(trade_ret, price_ret, weights_hist, positions, output_path)


def plot_evaluation(trade_ret, price_ret, weights_hist, positions, output_path):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Cumulative returns
    ax1 = axes[0]
    strat_cum = np.cumprod(1 + trade_ret)
    bench_cum = np.cumprod(1 + price_ret)
    ax1.plot(strat_cum, label="RL Strategy", linewidth=2, color='blue')
    ax1.plot(bench_cum, label="Buy & Hold", linewidth=2, alpha=0.7, color='gray')
    ax1.fill_between(range(len(strat_cum)), 1, strat_cum, alpha=0.3, color='blue')
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax1.set(ylabel="Cumulative Return", title="Strategy vs Buy & Hold")
    ax1.legend(loc='upper left'); ax1.grid(alpha=0.3)

    # Add outperformance annotation
    final_outperf = (strat_cum[-1] / bench_cum[-1] - 1) * 100
    ax1.annotate(f"Outperformance: {final_outperf:+.1f}%",
                 xy=(len(strat_cum)-1, strat_cum[-1]), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Weight evolution
    ax2 = axes[1]
    colors = plt.cm.tab10(np.arange(weights_hist.shape[1]))
    for i in range(weights_hist.shape[1]):
        ax2.plot(weights_hist[:, i], label=f"Signal {i+1}", alpha=0.8, color=colors[i])
    ax2.axhline(y=0.2, color='k', linestyle='--', alpha=0.3, label="Uniform (0.2)")
    ax2.set(ylabel="Weight", title="Signal Weights Over Time")
    ax2.legend(loc='upper right'); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, max(0.5, weights_hist.max() * 1.1))

    # Position
    ax3 = axes[2]
    ax3.fill_between(range(len(positions)), 0, positions, alpha=0.5,
                     where=np.array(positions) > 0, color='green', label='Long')
    ax3.fill_between(range(len(positions)), 0, positions, alpha=0.5,
                     where=np.array(positions) < 0, color='red', label='Short')
    ax3.plot(positions, color='black', linewidth=0.5)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax3.set(xlabel="Step", ylabel="Position", title="Trading Position", ylim=(-1.1, 1.1))
    ax3.legend(); ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    p.add_argument("--model", default="checkpoints/policy_best.pt")
    p.add_argument("--data", default="data/synthetic_data.csv")
    p.add_argument("--output", default="evaluation.png")
    p.add_argument("--length", type=int, default=500, help="Evaluation length")
    a = p.parse_args()
    evaluate(a.model, a.data, a.output, a.length)


if __name__ == "__main__":
    main()
