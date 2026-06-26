"""
Helpers to build and instantiate the different GPT precision experiments.

Each builder function creates one or several models based on the checkpoint
configuration and returns them as a dictionary keyed by experiment name.
"""


import torch
from formats import create_layer_format, create_softmax_format, create_LN_format
from model_up import GPTModelUP, ModelUPConfig
from model_mp_block import ModelBlockMPConfig, GPTModelBlockMP
from model_mp_mhsa import ModelMHSAConfig, GPTModelMHSA
from model_mp_ln import ModelLNMPConfig, GPTModelLNMP
from model_similarity_mp import ModelSimilarityMPConfig, GPTModelSimilarityMP
from model_mlp_mp import ModelMLPMPConfig, GPTModelMLPMP

def build_model(model_class, config, checkpoint_state: dict, device: torch.device):
    """Instantiate a model, load checkpoint weights, move it to device and set eval mode."""
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
    Build a set of models with the same precision for all components.

    Args:
        cfg: Configuration dictionary from the checkpoint.
        checkpoint_state: Weight state dictionary from the checkpoint.
        device: Target device for the created models.

    Returns:
        dict: Mapping of model names (``fp32``, ``fp16``, ``fp8``) to model instances.
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
    Build models where attention, feed-forward and layer norm use different precisions.

    Args:
        cfg: Configuration dictionary from the checkpoint.
        checkpoint_state: Weight state dictionary from the checkpoint.
        device: Target device for the created models.

    Returns:
        dict: Mapping of experiment names to instantiated mixed-precision models.
    """
 
    # Define the combinations to compare:
    # each entry is (name, attn, ffwd, ln)
    combinations = [
        ("attention_fp8","fp16", "fp8", "fp8"),  
        ("ffwd_fp8",     "fp8", "fp16", "fp8"),  
        ("LN_fp8",       "fp8", "fp8", "fp16")  
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
    """
    Build variants focused on the multi-head self-attention sub-block.

    Args:
        cfg: Configuration dictionary from the checkpoint.
        checkpoint_state: Weight state dictionary from the checkpoint.
        device: Target device for the created models.

    Returns:
        dict: Mapping of MHSA experiment names to model instances.
    """
    # Define the combinations to compare:
    # each entry is (name, QKV_format, Q*K^T format, softmax format, format of each head)
    combinations = [
        ("QKV_fp8",         "fp8", "fp16", "fp16", "fp16"),
        ("attention_fp8",   "fp8", "fp8", "fp16", "fp16"),
        ("softmax_fp8",     "fp8", "fp8", "fp8", "fp16"),
        ("head_fp8",        "fp8", "fp8", "fp8", "fp8")
    ]
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

def build_models_ln(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:
    """
    Build the adaptive LayerNorm experiment using the mixed LayerNorm model.

    Args:
        cfg: Configuration dictionary from the checkpoint.
        checkpoint_state: Weight state dictionary from the checkpoint.
        device: Target device for the created models.

    Returns:
        dict: Mapping of LN experiment names to model instances.
    """
    # Define the experiments:
    # each entry is (name, low format, high format, threshold)
    experiments = [
        ("LN_mp_fp8", "fp8", "fp16", 0.1)
    ]
    models = {}
    for name, ln_low, ln_high, threshold in experiments:
        model_config = ModelLNMPConfig(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
            n_layer=cfg["n_layer"],
            layer_format=create_layer_format("fp8"),
            softmax_format=create_softmax_format("fp8"),
            LN_format=create_LN_format(ln_low),
            LN_high_format=create_LN_format(ln_high),
            proximity_threshold=threshold,
            name=name,
        )
        models[name] = build_model(GPTModelLNMP, model_config, checkpoint_state, device)
    return models


def build_models_similarity(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:
    """
    Build experiments for similarity in mixed precision

    Args:
        cfg: Configuration dictionary from the checkpoint.
        checkpoint_state: Weight state dictionary from the checkpoint.
        device: Target device for the created models.

    Returns:
        dict: Mapping of similarity experiment names to model instances.
    """
    # Define the experiments:
    # each entry is (name, tau)
    experiments = [
        ("tau = 0.5",   0.5)
    ]
    models = {}
    for name, tau in experiments:
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

def build_models_mlp(cfg: dict, checkpoint_state: dict, device: torch.device) -> dict:
    """
    Build experiments for MLP in mixed precision

    Args:
        cfg: Configuration dictionary from the checkpoint.
        checkpoint_state: Weight state dictionary from the checkpoint.
        device: Target device for the created models.

    Returns:
        dict: Mapping of MLP experiment names to model instances.
    """
    # Define the experiments:
    # each entry is (name, tau)
    experiments = [
        ("tau = 0.5", 0.5)
    ]
    models = {}
    for name, tau in experiments:
        model_config = ModelMLPMPConfig(
            vocab_size=cfg["vocab_size"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
            n_layer=cfg["n_layer"],
            layer_format=create_layer_format("fp16"),
            softmax_format=create_softmax_format("fp16"),
            LN_format=create_LN_format("fp16"),
            ffwd_layer_format=create_layer_format("fp16"),
            tau = tau,
            name = name
        )
        models[name] = build_model(GPTModelMLPMP, model_config, checkpoint_state, device)
    return models