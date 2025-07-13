import torch
from mmdet.models import HEADS
from mmcv.runner.base_module import BaseModule
import numpy as np


@HEADS.register_module()
class BaseRenderModel(BaseModule):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)