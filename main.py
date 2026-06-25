"""
main.py — Entry point for precision comparison experiments.

Usage:
    python main.py --experiment uniform --model_path gpt_shakespeare.pt
    python main.py --experiment mixed_block   --model_path gpt_shakespeare.pt

To add a new experiment:
    1. Define a build_models_<exp>() function below
    2. Register it in EXPERIMENTS

To test a new model architecture:
    1. Create model_<name>.py with your model class and config dataclass
    2. Import and use it in the relevant build_models_<exp>() function
"""

import argparse
import torch
import mptorch.quant as qpt

from formats import create_layer_format, create_softmax_format, create_LN_format
from model_up import GPTModelUP, ModelUPConfig
from model_mp_block import ModelBlockMPConfig, GPTModelBlockMP
from model_mp_mhsa import ModelMHSAConfig, GPTModelMHSA
from error_analyzer import ErrorAnalyzer

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
                        choices=["uniform", "mixed_block", "mhsa"],
                        help="which experiment to run (default: uniform)")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def get_device(args) -> torch.device:
    return torch.device("cuda" if args.cuda else "cpu")


def load_checkpoint(model_path: str, device: torch.device) -> dict:
    return torch.load(model_path, map_location=device)



def build_model(model_class, config, checkpoint_state: dict, device: torch.device):
    """Instantiate a model, load weights from checkpoint, move to device."""
    model = model_class(config)
    state = model.state_dict()
    for name in state.keys():
        if name in checkpoint_state:
            state[name].copy_(checkpoint_state[name])
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def build_models_uniform(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:
    """
    Three models with the same format applied to every component.
    cfg: the 'config' dict from the checkpoint.
    """
    precisions = ["fp32", "fp16", "fp8"]
    models = {}
    for precision in precisions:
        model_config = ModelUPConfig(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
            n_layer=cfg["n_layer"],
            layer_format=create_layer_format(precision),
            softmax_format=create_softmax_format(precision),
            LN_format=create_LN_format(precision),
            name=precision,
        )
        models[precision] = build_model(GPTModelUP, model_config, checkpoint_state, device)
    return models

def build_model_mixed_block(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:
    """
    Several models where attention, feed-forward and layer norm can each have
    a different precision. Add or remove entries to define new combinations.
    """
 
    # Define the combinations to compare:
    # each entry is (name, attn, ffwd, ln)
    combinations = [
        ("attention_fp8","fp16", "fp8", "fp8"),  # attention fp8
        ("ffwd_fp8",     "fp8", "fp16", "fp8"),  # feedforward fp8
        ("LN_fp8",       "fp8", "fp8", "fp16")  # Layer Norm fp8
    ]
 
    models = {}
    for name, attn, ffwd, ln in combinations:
        model_config = ModelBlockMPConfig(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
            n_layer=cfg["n_layer"],
            attn_layer_format=create_layer_format(attn),
            attn_softmax_format=create_softmax_format(attn),
            ffwd_layer_format=create_layer_format(ffwd),
            ln_format=create_LN_format(ln),
            name=name,
        )
        models[name] = build_model(GPTModelBlockMP, model_config, checkpoint_state, device)
    return models


def build_models_mhsa(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:

    combinations = {
        ("QKV_fp8",         "fp8", "fp16", "fp16", "fp16"),
        ("attention_fp8",   "fp8", "fp8", "fp16", "fp16"),
        ("softmax_fp8",     "fp8", "fp8", "fp8", "fp16"),
        ("head_fp8",        "fp8", "fp8", "fp8", "fp8")
    }
    models = {}
    for name, QKV, attn, softmax, head in combinations:
        model_config = ModelMHSAConfig(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
            n_layer=cfg["n_layer"],
            QKV_format=create_layer_format(QKV),
            attention_format=create_layer_format(attn),
            softmax_format=create_softmax_format(softmax),
            head_format=create_layer_format(head),
            layer_format=create_layer_format("fp16"),
            LN_format=create_LN_format("fp16"),
            name=name
        )
        models[name] = build_model(GPTModelMHSA, model_config, checkpoint_state, device)
    return models



EXPERIMENTS = {
    "uniform": (build_models_uniform, "test_results/uniform_precision.csv"),
    "mixed_block":  (build_model_mixed_block,   "test_results/mixed_block_fp8_fp16.csv"),
    "mhsa": (build_models_mhsa, "test_results/mhsa_mp_fp16.csv"),
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