from .depth import DepthLoss
from .normal import NormalLoss
from .pose import PoseLoss
from .total import MultiTaskLoss

__all__ = ["DepthLoss", "NormalLoss", "PoseLoss", "MultiTaskLoss"]
