
import argparse
import torch
import mptorch.quant as qpt

from error_analyzer import ErrorAnalyzer
from builder import build_models_uniform, build_models_mixed_block, build_models_mhsa, build_models_similarity, build_models_mlp

def parse_args():
    parser = argparse.ArgumentParser(description="GPT Shakespeare precision analysis")
    parser.add_argument("--model_path", type=str, default="../script_test/gpt_shakespeare.pt",
                        help="path to the trained model checkpoint")
    parser.add_argument("--data_path", type=str, default="../data/shakespeare_input.txt",
                        help="path to the raw text dataset")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for evaluation (default: 64)")
    parser.add_argument("--num_batches", type=int, default=500,
                        help="number of batches to evaluate (default: 500)")
    parser.add_argument("--output", type=str, default=None,
                        help="path to save the CSV results (default: test_results/<experiment>.csv)")
    parser.add_argument("--seed", type=int, default=1337, metavar="S",
                        help="random seed (default: 1337)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA")
    parser.add_argument("--experiment", type=str, default="uniform",
                        choices=["uniform", "mixed_block", "mhsa", "similarity", "mlp"],
                        help="which experiment to run (default: uniform)")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def get_device(args) -> torch.device:
    return torch.device("cuda" if args.cuda else "cpu")


def load_checkpoint(model_path: str, device: torch.device) -> dict:
    return torch.load(model_path, map_location=device)


EXPERIMENTS = {
    "uniform": (build_models_uniform, "test_results/uniform_precision.csv"),
    "mixed_block":  (build_models_mixed_block,   "test_results/mixed_block_fp8_fp16.csv"),
    "mhsa": (build_models_mhsa, "test_results/mhsa_mp_fp16.csv"),
    "similarity" : {build_models_similarity, "test_results/similarity_mp.csv"},
    "mlp" : {build_models_mlp, "test_results/mlp_mp.csv"}
}

def main():
    args = parse_args()
    device = get_device(args)
    torch.manual_seed(args.seed)

    # Load checkpoint once, share across all models
    checkpoint = load_checkpoint(args.model_path, device)
    cfg = checkpoint["config"]

    build_fn, default_output = EXPERIMENTS[args.experiment]
    output_path = args.output or default_output

    models = build_fn(cfg, checkpoint["model_state_dict"], device)

    analyzer = ErrorAnalyzer(
        models=models,
        data_path=args.data_path,
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"],
        block_size=cfg["block_size"],
        batch_size=args.batch_size,
        device=device,
    )

    results = analyzer.compare(num_batches=args.num_batches)

    print("\n=== Results ===")
    for name, metrics in results.items():
        print(f"  {name:<20}: avg_loss = {metrics['avg_loss']:.6f}")

    analyzer.save_results(results, output_path)


if __name__ == "__main__":
    main()