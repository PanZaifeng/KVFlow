"""
策略包初始化文件
"""
from .base_policy import BasePolicy, DefaultPolicy
# from .DistPolicy import DistPolicy
# from .JumpPolicy import JumpPolicy
# from .NeedPolicy import NeedPolicy
# from .PlanPolicy import PlanPolicy

# 策略映射字典，用于根据策略名称获取对应的策略类
POLICY_MAP = {
    'default': DefaultPolicy,
    # 'dist': DistPolicy,
    # 'distance': DistPolicy,
    # 'jump': JumpPolicy,
    # 'need': NeedPolicy,
    # 'plan': PlanPolicy,
    # 'schedule': PlanPolicy,
}

__all__ = [
    'BasePolicy',
    'DefaultPolicy',
    'POLICY_MAP'
]
