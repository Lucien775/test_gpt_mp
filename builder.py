import torch
from formats import create_layer_format, create_softmax_format, create_LN_format
from model_up import GPTModelUP, ModelUPConfig
from model_mp_block import ModelBlockMPConfig, GPTModelBlockMP
from model_mp_mhsa import ModelMHSAConfig, GPTModelMHSA
from model_similarity_mp import ModelSimilarityMPConfig, GPTModelSimilarityMP

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

def build_models_mixed_block(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:
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

def build_models_similarity(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:

    experiments = {
        ("tau = 0.5",   0.5)
    } 

    models = {}
    for name, tau, dtype in experiments:
        model_config = ModelSimilarityMPConfig(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
            n_layer=cfg["n_layer"],
            layer_format = create_layer_format("fp8"),
            LN_format = create_LN_format("fp8"),
            matmul_format_low = create_layer_format("fp8"),
            matmul_format_high = create_layer_format("fp16"),
            softmax_format_low = create_softmax_format("fp8"),
            softmax_format_high = create_softmax_format("fp16"),
            tau=tau,
            name=name
        )
        models[name] = build_model(GPTModelSimilarityMP, model_config, checkpoint_state, device)
    return models