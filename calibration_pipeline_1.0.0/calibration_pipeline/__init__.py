"""Portable scene-to-calibration-to-BEV pipeline; CPU imports never load CUDA."""
from .pipeline import VERSION as __version__, Pipeline, run_pipeline

__all__ = ["Pipeline", "run_pipeline", "__version__"]
