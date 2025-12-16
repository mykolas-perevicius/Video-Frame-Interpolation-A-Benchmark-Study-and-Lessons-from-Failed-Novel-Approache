# SOTA model wrappers
# These require external repositories to be cloned to external/

# Available models:
# - RIFEModel: Fast VFI (speed champion)
# - VFIMambaModel: Quality VFI SOTA
# - SAFAModel: Joint VFI+SR
# - SPANModel: Efficient SR (NTIRE 2024 winner)
# - TwoStageModel: Combine any VFI + SR models

__all__ = [
    'RIFEModel',
    'RIFELiteModel',
    'VFIMambaModel', 
    'VFIMambaLite',
    'SAFAModel',
    'SPANModel',
    'TwoStageModel',
]

# Lazy imports to avoid import errors when external repos aren't available
def __getattr__(name):
    if name == 'RIFEModel':
        from .rife_wrapper import RIFEModel
        return RIFEModel
    elif name == 'RIFELiteModel':
        from .rife_wrapper import RIFELiteModel
        return RIFELiteModel
    elif name == 'VFIMambaModel':
        from .vfimamba_wrapper import VFIMambaModel
        return VFIMambaModel
    elif name == 'VFIMambaLite':
        from .vfimamba_wrapper import VFIMambaLite
        return VFIMambaLite
    elif name == 'SAFAModel':
        from .safa_wrapper import SAFAModel
        return SAFAModel
    elif name == 'SPANModel':
        from .span_wrapper import SPANModel
        return SPANModel
    elif name == 'TwoStageModel':
        from .span_wrapper import TwoStageModel
        return TwoStageModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
