try:
    import torch
    from yacs.config import CfgNode
    # Allow OmegaConf objects inside checkpoints when using weights_only=True (PyTorch>=2.6)
    try:
        from omegaconf import DictConfig, ListConfig
    except Exception:
        DictConfig, ListConfig = None, None

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        safe_types = [CfgNode]
        if DictConfig is not None and ListConfig is not None:
            safe_types += [DictConfig, ListConfig]
        torch.serialization.add_safe_globals(safe_types)
except Exception:
    # Best-effort registration; proceed if unavailable (older torch) or if yacs isn't installed yet
    pass




