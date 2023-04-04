# @Time    : 2023/3/14 下午2:18
# @Author  : Boyang
# @Site    : 
# @File    : config.py
# @Software: PyCharm
from detectron2.config import CfgNode


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detector2 CfgNode instance.
    """
    from .defaults import _C
    return _C.clone()
