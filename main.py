"""RL Signal Weighting Demo - Main entry point."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="RL Signal Weighting Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train              # Train the RL agent
  python main.py evaluate           # Evaluate trained agent
  python main.py compare            # Compare RL vs baselines
  python main.py all                # Train, evaluate, and compare

  python main.py train --episodes 500 --lr 1e-4
  python main.py compare --length 1000
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_p = subparsers.add_parser("train", help="Train the PPO agent")
    train_p.add_argument("--episodes", type=int, default=1000)
    train_p.add_argument("--episode-length", type=int, default=252)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--gamma", type=float, default=0.99)
    train_p.add_argument("--ppo-epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--checkpoint-freq", type=int, default=200)
    train_p.add_argument("--data-path", default="data/synthetic_data.csv")
    train_p.add_argument("--output-dir", default="checkpoints")

    # Evaluate command
    eval_p = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_p.add_argument("--model", default="checkpoints/policy_best.pt")
    eval_p.add_argument("--data", default="data/synthetic_data.csv")
    eval_p.add_argument("--output", default="evaluation.png")
    eval_p.add_argument("--length", type=int, default=500)

    # Compare command
    comp_p = subparsers.add_parser(
        "compare", help="Compare RL agent vs baseline strategies"
    )
    comp_p.add_argument("--model", default="checkpoints/policy_best.pt")
    comp_p.add_argument("--data", default="data/synthetic_data.csv")
    comp_p.add_argument("--output", default="comparison.png")
    comp_p.add_argument("--length", type=int, default=500)
    comp_p.add_argument("--seed", type=int, default=42)

    # All command (train + evaluate + compare)
    all_p = subparsers.add_parser(
        "all", help="Run full pipeline: train, evaluate, compare"
    )
    all_p.add_argument("--episodes", type=int, default=1000)
    all_p.add_argument("--eval-length", type=int, default=500)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "train":
        from train import train, plot_learning_curve

        returns, weights = train(
            n_episodes=args.episodes,
            episode_length=args.episode_length,
            lr=args.lr,
            gamma=args.gamma,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            checkpoint_freq=args.checkpoint_freq,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )
        plot_learning_curve(returns, weights)

    elif args.command == "evaluate":
        from evaluate import evaluate

        evaluate(args.model, args.data, args.output, args.length)

    elif args.command == "compare":
        from compare import compare_strategies

        compare_strategies(
            model_path="src/checkpoints/policy_best.pt",
            data_path="src/data/synthetic_data.csv",
            eval_length=args.length,
            output_path=args.output,
            seed=args.seed,
        )

    elif args.command == "all":
        from train import train, plot_learning_curve
        from evaluate import evaluate
        from compare import compare_strategies

        print("=" * 60)
        print("STEP 1: TRAINING")
        print("=" * 60)
        returns, weights = train(n_episodes=args.episodes)
        plot_learning_curve(returns, weights)

        print("\n" + "=" * 60)
        print("STEP 2: EVALUATION")
        print("=" * 60)
        evaluate(eval_length=args.eval_length)

        print("\n" + "=" * 60)
        print("STEP 3: COMPARISON (RL vs Baselines)")
        print("=" * 60)
        compare_strategies(eval_length=args.eval_length)


if __name__ == "__main__":
    main()
