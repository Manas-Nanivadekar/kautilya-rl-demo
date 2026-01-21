"""Compare RL agent performance against baseline strategies."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from environment import SignalWeightingEnv
from agent import PPOAgent
from baselines import BaselineStrategy, get_all_baselines


def compute_metrics(returns: np.ndarray) -> dict[str, float]:
    """Compute trading performance metrics."""
    if len(returns) == 0:
        return {"total_return": 0, "sharpe": 0, "max_dd": 0, "vol": 0, "win_rate": 0}
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return {
        "total_return": cum[-1] - 1,
        "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
        "max_dd": np.max((peak - cum) / peak),
        "vol": np.std(returns) * np.sqrt(252),
        "win_rate": np.mean(returns > 0),
    }


def run_baseline_strategy(
    strategy: BaselineStrategy,
    data_path: str,
    eval_length: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a baseline strategy and return trading returns, price returns, and weights."""
    env = SignalWeightingEnv(data_path, episode_length=eval_length)
    state, _ = env.reset(seed=seed)
    env.start_idx = env.lookback
    strategy.reset()

    trade_returns, price_returns, weights_hist = [], [], []
    prev_return = 0.0

    for step in range(min(eval_length, len(env.returns) - env.lookback - 1)):
        idx = env.start_idx + env.current_step
        signals = env.signals[idx]

        weights = strategy.get_weights(signals, price_return=prev_return)
        state, _, done, _, info = env.step(weights)

        trade_returns.append(info["trading_return"])
        price_returns.append(info["price_return"])
        weights_hist.append(info["weights"])
        prev_return = info["price_return"]

        if done:
            break

    return np.array(trade_returns), np.array(price_returns), np.array(weights_hist)


def run_rl_strategy(
    model_path: str,
    data_path: str,
    eval_length: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Run the trained RL agent and return trading returns, price returns, and weights."""
    if not Path(model_path).exists():
        alt_path = model_path.replace("_best", "_final")
        if Path(alt_path).exists():
            model_path = alt_path
        else:
            return None

    env = SignalWeightingEnv(data_path, episode_length=eval_length)
    obs_dim = env.observation_space.shape[0]
    agent = PPOAgent(obs_dim, env.n_signals)
    agent.load(model_path)

    state, _ = env.reset(seed=seed)
    env.start_idx = env.lookback

    trade_returns, price_returns, weights_hist = [], [], []

    for _ in range(min(eval_length, len(env.returns) - env.lookback - 1)):
        action = agent.select_action(state, deterministic=True)
        state, _, done, _, info = env.step(action)

        trade_returns.append(info["trading_return"])
        price_returns.append(info["price_return"])
        weights_hist.append(info["weights"])

        if done:
            break

    return np.array(trade_returns), np.array(price_returns), np.array(weights_hist)


def compare_strategies(
    model_path: str = "src/checkpoints/policy_best.pt",
    data_path: str = "src/data/synthetic_data.csv",
    eval_length: int = 500,
    output_path: str = "comparison.png",
    seed: int = 42,
):
    """Compare RL strategy against all baselines."""
    results = {}

    # Run RL strategy
    rl_result = run_rl_strategy(model_path, data_path, eval_length, seed)
    if rl_result is not None:
        trade_ret, price_ret, weights = rl_result
        results["RL Agent (PPO)"] = {
            "returns": trade_ret,
            "weights": weights,
            "metrics": compute_metrics(trade_ret),
        }
        # Store price returns for Buy & Hold baseline
        bh_metrics = compute_metrics(price_ret)
    else:
        print("Warning: No trained RL model found. Running baselines only.")
        # Still need price returns for comparison
        env = SignalWeightingEnv(data_path, episode_length=eval_length)
        state, _ = env.reset(seed=seed)
        env.start_idx = env.lookback
        price_ret = []
        for _ in range(min(eval_length, len(env.returns) - env.lookback - 1)):
            idx = env.start_idx + env.current_step
            price_ret.append(env.returns[idx])
            env.current_step += 1
        price_ret = np.array(price_ret)
        bh_metrics = compute_metrics(price_ret)

    # Add Buy & Hold
    results["Buy & Hold"] = {
        "returns": price_ret,
        "weights": None,
        "metrics": bh_metrics,
    }

    # Run all baseline strategies
    baselines = get_all_baselines(seed=seed)
    for strategy in baselines:
        trade_ret, _, weights = run_baseline_strategy(
            strategy, data_path, eval_length, seed
        )
        results[strategy.name] = {
            "returns": trade_ret,
            "weights": weights,
            "metrics": compute_metrics(trade_ret),
        }

    # Print comparison table
    print_comparison_table(results)

    # Generate plots
    plot_comparison(results, output_path)

    return results


def print_comparison_table(results: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("STRATEGY COMPARISON")
    print("=" * 90)
    print(
        f"{'Strategy':<20} {'Return':>12} {'Sharpe':>10} {'Max DD':>10} {'Volatility':>12} {'Win Rate':>10}"
    )
    print("-" * 90)

    # Sort by Sharpe ratio
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True
    )

    for name, data in sorted_results:
        m = data["metrics"]
        print(
            f"{name:<20} {m['total_return']:>11.2%} {m['sharpe']:>10.2f} {m['max_dd']:>9.2%} {m['vol']:>11.2%} {m['win_rate']:>9.2%}"
        )

    print("=" * 90)

    # Highlight RL performance
    if "RL Agent (PPO)" in results:
        rl_sharpe = results["RL Agent (PPO)"]["metrics"]["sharpe"]
        baseline_sharpes = [
            v["metrics"]["sharpe"]
            for k, v in results.items()
            if k not in ["RL Agent (PPO)", "Buy & Hold"]
        ]
        if baseline_sharpes:
            best_baseline = max(baseline_sharpes)
            improvement = (
                (rl_sharpe - best_baseline) / abs(best_baseline + 1e-8)
            ) * 100
            print(f"\nRL vs Best Baseline: {improvement:+.1f}% Sharpe improvement")


def plot_comparison(results: dict, output_path: str):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Cumulative returns comparison
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        cum_ret = np.cumprod(1 + data["returns"])
        linewidth = 3 if name == "RL Agent (PPO)" else 1.5
        linestyle = "-" if name == "RL Agent (PPO)" else "--"
        alpha = 1.0 if name == "RL Agent (PPO)" else 0.7
        ax1.plot(
            cum_ret,
            label=name,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )

    ax1.axhline(y=1, color="k", linestyle=":", alpha=0.3)
    ax1.set(
        xlabel="Step", ylabel="Cumulative Return", title="Cumulative Returns Comparison"
    )
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. Sharpe ratio bar chart
    ax2 = axes[0, 1]
    names = list(results.keys())
    sharpes = [results[n]["metrics"]["sharpe"] for n in names]
    bar_colors = ["green" if n == "RL Agent (PPO)" else "steelblue" for n in names]
    bars = ax2.barh(names, sharpes, color=bar_colors, alpha=0.8)
    ax2.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    ax2.set(xlabel="Sharpe Ratio", title="Sharpe Ratio Comparison")
    ax2.grid(alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, sharpes):
        ax2.text(
            val + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )

    # 3. Risk-Return scatter
    ax3 = axes[1, 0]
    for (name, data), color in zip(results.items(), colors):
        m = data["metrics"]
        marker = "*" if name == "RL Agent (PPO)" else "o"
        size = 300 if name == "RL Agent (PPO)" else 100
        ax3.scatter(
            m["vol"],
            m["total_return"],
            label=name,
            color=color,
            s=size,
            marker=marker,
            alpha=0.8,
            edgecolors="black",
        )
        ax3.annotate(
            name,
            (m["vol"], m["total_return"]),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax3.axhline(y=0, color="k", linestyle=":", alpha=0.3)
    ax3.set(
        xlabel="Volatility (Annualized)",
        ylabel="Total Return",
        title="Risk-Return Profile",
    )
    ax3.grid(alpha=0.3)

    # 4. Metrics heatmap
    ax4 = axes[1, 1]
    metrics_names = ["total_return", "sharpe", "max_dd", "win_rate"]
    metrics_labels = ["Return", "Sharpe", "Max DD", "Win Rate"]

    # Normalize metrics for heatmap
    data_matrix = []
    for name in names:
        row = [results[name]["metrics"][m] for m in metrics_names]
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)

    # Normalize each column (higher is better, except max_dd)
    normalized = np.zeros_like(data_matrix)
    for i in range(data_matrix.shape[1]):
        col = data_matrix[:, i]
        if metrics_names[i] == "max_dd":
            # Lower is better for max drawdown
            normalized[:, i] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-8)
        else:
            normalized[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    im = ax4.imshow(normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax4.set_xticks(range(len(metrics_labels)))
    ax4.set_xticklabels(metrics_labels)
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names)
    ax4.set_title("Normalized Performance Metrics (Green = Better)")

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(metrics_names)):
            val = data_matrix[i, j]
            if metrics_names[j] in ["total_return", "max_dd", "win_rate", "vol"]:
                text = f"{val:.1%}"
            else:
                text = f"{val:.2f}"
            ax4.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if normalized[i, j] < 0.3 or normalized[i, j] > 0.7
                    else "black"
                ),
            )

    plt.colorbar(im, ax=ax4, label="Normalized Score")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Compare RL agent vs baseline strategies")
    p.add_argument(
        "--model", default="checkpoints/policy_best.pt", help="Path to trained model"
    )
    p.add_argument(
        "--data", default="data/synthetic_data.csv", help="Path to data file"
    )
    p.add_argument("--length", type=int, default=500, help="Evaluation length")
    p.add_argument("--output", default="comparison.png", help="Output plot path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    compare_strategies(
        model_path=args.model,
        data_path=args.data,
        eval_length=args.length,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
