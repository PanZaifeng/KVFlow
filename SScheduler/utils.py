from SScheduler.timestepManager import BaseManager, PlanManager


MANAGER_REGISTRY = {
    'base': BaseManager,
    'plan': PlanManager
}