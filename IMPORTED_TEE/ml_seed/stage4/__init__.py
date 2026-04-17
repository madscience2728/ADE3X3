from .generator import generate_batch, generate_seed
from .invert import image_to_C, round_trip_error


def inject(*args, **kwargs):
    from .inject import inject as _inject

    return _inject(*args, **kwargs)

__all__ = [
    "generate_batch",
    "generate_seed",
    "image_to_C",
    "inject",
    "round_trip_error",
]