import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from abc import abstractmethod
from SScheduler.timestepManager.BaseManager import BaseManager

# Import logger
from SScheduler.logger import get_logger

logger = get_logger()

class PlanManager(BaseManager):
    """
    PlanManager base class: maintain the datastructures for agents and their plans

    Features:
    Input the plan_lasting_time of agent
    """

    def __init__(self, agent_num: int, timestep_ticks: int, timestep_max: int = 100, priority: Optional[int] = 1):
        """
        Initialize PlanManager
        
        Args:
            agent_num: agent count
            timestep_max: maximum timestep value
            timestep_ticks: ticks per timestep
        """
        super().__init__(agent_num, timestep_max)
        self.timestep_ticks = timestep_ticks
        self.type = self.__class__.__name__
        self.priority = priority

    def update_agent_timestep(self, agent_id: int, obj: Any, emergency: bool = False) -> None:
        """
        Update agent timestep based on object
        
        Args:
            agent_id: agent's ID
            obj: object containing plan duration information
            emergency: if True, set timestep to 0 immediately
        """
        logger.debug(f"[PFEngine][update_agent_timestep][status]: agent_{agent_id}, obj={obj}, emergency={emergency}")
        
        if emergency:
            self.modify_agent_timestep(agent_id=agent_id, new_timestep=0)
        else:
            new_timestep = self.dependency_translation(obj)
            self.modify_agent_timestep(agent_id=agent_id, new_timestep=new_timestep)

    def dependency_check(self, obj) -> bool:
        """Check if object is valid for timestep calculation"""
        result = isinstance(obj, int)
        logger.debug(f"[PFEngine][dependency_check][result]: obj={obj}, valid={result}")
        return result

    def dependency_translation(self, obj) -> int:
        """Translate object to timestep value"""
        logger.debug(f"[PFEngine][dependency_translation][status]: translating obj={obj}")
        
        if not self.dependency_check(obj):
            raise ValueError(f"Invalid timestep value: {obj}")
        else:
            if obj <= 0:
                logger.error(f"[PFEngine][dependency_translation][error]: invalid timestep value = {obj}")
                result = 0
            else:
                # Ensure integer timestep even if timestep_ticks is a float
                result = int(max(0, obj - 1) // self.timestep_ticks)

            logger.debug(f"[PFEngine][dependency_translation][result]: {obj} -> {result} (ticks={self.timestep_ticks})")
            return result


    def dependency_check(self, obj):

        return isinstance(obj, int)

    # def dependency_translation(self, obj):

    #     if not self.dependency_check(obj):
    #         raise ValueError(f"Invalid timestep value: {obj}")
    #     else:
    #         if obj <= 0:
    #             return 0
    #         else:
    #             return obj // self.timestep_ticks

