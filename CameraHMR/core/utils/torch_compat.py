try:
    import torch
    from yacs.config import CfgNode

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([CfgNode])
except Exception:
    # Best-effort registration; proceed if unavailable (older torch) or if yacs isn't installed yet
    pass




