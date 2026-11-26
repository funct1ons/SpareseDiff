"""Lazy CFA adapters: selecting metadata does not initialize a model."""
from training.cfa.base import CFAAdapter


def get_cfa_adapter(mode: str) -> CFAAdapter:
    if mode == 'sparse':
        from training.cfa.sparse import SparseAdapter
        return SparseAdapter()
    if mode == 'bayer':
        from training.cfa.bayer import BayerAdapter
        return BayerAdapter()
    raise ValueError(f'Unsupported CFA mode: {mode!r}; expected sparse or bayer')
