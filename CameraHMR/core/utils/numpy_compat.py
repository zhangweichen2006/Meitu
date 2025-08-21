def ensure_numpy_legacy_aliases():
    try:
        import numpy as np
    except Exception:
        return

    # Re-create removed aliases for libraries that still import them (e.g., chumpy)
    alias_map = {
        'bool': bool,
        'int': int,
        'float': float,
        'complex': complex,
        'object': object,
        'unicode': str,
        'str': str,
    }
    for name, target in alias_map.items():
        if not hasattr(np, name):
            try:
                setattr(np, name, target)
            except Exception:
                pass


