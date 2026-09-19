"""Renderer boundary: RGB uint8, ray-distance depth meters, alpha and valid mask."""
from dataclasses import dataclass
import numpy as np


@dataclass
class RenderResult:
    rgb: np.ndarray
    depth: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray
