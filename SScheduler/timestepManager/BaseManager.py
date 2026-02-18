import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
from SScheduler.logger import get_logger

logger = get_logger()

class BaseManager(ABC):
    """
    BaseManager base class: maintain the datastructures for agents and their timesteps    
    
    Features:
    1. Maintain the number of timesteps remaining for each agent
    2. Quickly query agents with timesteps in the range of 0 to n
    3. Provide abstract interfaces for subclasses to implement specific query logic
    """
    
    def __init__(self, agent_num: int, timestep_max: int = 100):
        """
        Initialize BaseManager
        
        Args:
            agent_num: agent count
            timestep_max: maximum timestep value
        """
        self.agent_num = agent_num
        self.timestep_max = timestep_max
        self.lock_a2t = threading.RLock()
        self.lock_t2a = threading.RLock()
        self.priority = 1
        self.type = self.__class__.__name__
        self.activate = True  # Will be set when registered to scheduler

        # agent_id -> timestep mapping
        self.agent_timestep: List[int] = [-1] * agent_num

        # timestep -> agent_ids_set mapping
        self.timestep_agent: List[Set[int]] = [set() for _ in range(timestep_max)]
        
    def get_type(self) -> str:
        """
        Get the type of the manager

        Returns:
            The type of the manager as a string
        """
        return self.type

    def get_priority(self) -> int:
        """
        Get the priority of the manager

        Returns:
            The priority of the manager as an integer
        """
        return self.priority

    def get_agent_timestep(self, agent_id: int) -> int:
        """
        Get the current timestep of the specified agent
        
        Args:
            agent_id: agent's ID
        Returns:
            The current timestep of the agent
        """
        if not self.activate:
            logger.debug(f"[PFEngine][get_agent_timestep][disabled]: manager not activated")
            return -1
            
        if agent_id < 0 or agent_id >= self.agent_num:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        with self.lock_a2t:
            timestep = self.agent_timestep[agent_id]
            logger.debug(f"[PFEngine][get_agent_timestep][result]: agent_{agent_id} -> timestep_{timestep}")
            return timestep

    def modify_agent_timestep(self, agent_id: int, new_timestep: int) -> None:
        """
        Modify the timestep of the specified agent

        Args:
            agent_id: agent's ID
            new_timestep: new timestep value
        """
        if not self.activate:
            logger.debug(f"[PFEngine][modify_agent_timestep][disabled]: manager not activated")
            return

        if agent_id >= self.agent_num or agent_id < 0:
            raise ValueError(f"Invalid agent_id: {agent_id}")
        
        if new_timestep < 0:
            raise ValueError(f"Invalid timestep that smaller than 0: {new_timestep}")
    
        with self.lock_a2t:
            old_timestep = self.agent_timestep[agent_id]
            # logger.debug(f"[PFEngine][modify_agent_timestep][status]: agent_{agent_id} timestep changing: {old_timestep} -> {new_timestep}")
            if old_timestep != new_timestep:
                self.agent_timestep[agent_id] = new_timestep
                with self.lock_t2a:
                    if 0 <= old_timestep < self.timestep_max:
                        self.timestep_agent[old_timestep].discard(agent_id)
                    else:
                        logger.debug(
                            f"[PFEngine][modify_agent_timestep][skip-discard]: agent_{agent_id}, old_timestep={old_timestep} out of range"
                        )
                    if 0 <= new_timestep < self.timestep_max:
                        self.timestep_agent[new_timestep].add(agent_id)
                    else:
                        logger.warning(
                            f"[PFEngine][modify_agent_timestep][warning]: New timestep {new_timestep} exceeds maximum {self.timestep_max}"
                        )

        logger.debug(f"[PFEngine][modify_agent_timestep][result]: successfully modified agent_{agent_id}, timestep changing: {old_timestep} -> {new_timestep}")

    def get_agent_n_timesteps(self, n: int) -> Dict[int, Set[int]]:
        """
        Get all agents in the nearest n timesteps
        
        Args:
            n: nearest timesteps to retrieve
            
        Returns:
            Dictionary where key is timestep and value is set of agent_ids
        """
        if not self.activate:
            logger.debug(f"[PFEngine][get_agent_n_timesteps][disabled]: manager not activated")
            return {}
            
        if n <= 0:
            raise ValueError(f"Invalid n: {n}, must be greater than 0")

        with self.lock_t2a:
            result = {}
            for timestep in range(n):
                # print(f"Checking timestep {timestep}: agents {self.timestep_agent[timestep]}")
                if self.timestep_agent[timestep]:
                    result[timestep] = self.timestep_agent[timestep].copy()
        logger.debug(f"[PFEngine][get_agent_n_timesteps][result]: {len(result)} timesteps with agents")
        return result
    
    def get_agent_x_timesteps(self, x: int) -> Set[int]:
        """
        Get all agents with a specific timestep
        
        Args:
            x: the timestep to query
        Returns:
            Set of agent_ids with the specified timestep
        """
        if x < 0 or x >= self.timestep_max:
            raise ValueError(f"Invalid timestep x: {x}, must be in range [0, {self.timestep_max})")

        with self.lock_t2a:
            return self.timestep_agent[x].copy()

    def update_per_timestep(self) -> None:
        """
        Decrease all agents' timestep by 1 and update the timestep_agent structure.
        Agents with timestep 0 will remain at timestep 0.
        
        This function is typically called at each simulation tick to advance the timeline.
        """
        if not self.activate:
            logger.debug(f"[PFEngine][update_per_timestep][disabled]: manager not activated")
            return
            
        with self.lock_a2t:
            with self.lock_t2a:
                new_timestep_agent: List[Set[int]] = [set() for _ in range(self.timestep_max)]
                
                for agent_id in range(self.agent_num):
                    current_timestep = self.agent_timestep[agent_id]
                    # print(f"Updating agent {agent_id}: current timestep {current_timestep}")
                    if current_timestep >= 0:
                        new_timestep = current_timestep - 1
                        self.agent_timestep[agent_id] = new_timestep
                        # print(f"Agent {agent_id} new timestep: {new_timestep}")
                        if new_timestep >= 0:
                            if new_timestep < self.timestep_max:
                                new_timestep_agent[new_timestep].add(agent_id)
                            else:
                                logger.info(f"[PFEngine][update_per_timestep][warning]: New timestep {new_timestep} exceeds maximum {self.timestep_max}")

                self.timestep_agent = new_timestep_agent
                logger.debug(f"[PFEngine][update_per_timestep][result]: updated all agent timesteps")
    
    def check_illegal_timesteps(self) -> None:
        """
        Check if any agent has a negative timestep value.
        """
        with self.lock_a2t:
            negative_agents = []
            for agent_id in range(self.agent_num):
                if self.agent_timestep[agent_id] < -1:
                    negative_agents.append(agent_id)
            if negative_agents:
                raise RuntimeError(f"Agents with negative timesteps detected: {negative_agents}")    # def get_stats(self) -> Dict[str, Any]:
    #     """
    #     BaseManager's statistic information

    #     Returns:
    #         Dictionary with statistic information
    #     """
    #     with self.lock_a2t:
    #         with self.lock_t2a:
    #             # Count non-empty timesteps
    #             unique_timesteps = sum(1 for agents in self.timestep_agent if agents)
                
    #             # Find min and max timesteps
    #             min_timestep = None
    #             max_timestep = None
    #             total_agents_tracked = 0
                
    #             for timestep, agents in enumerate(self.timestep_agent):
    #                 if agents:
    #                     if min_timestep is None:
    #                         min_timestep = timestep
    #                     max_timestep = timestep
    #                     total_agents_tracked += len(agents)
                
    #             return {
    #                 "agent_num": self.agent_num,
    #                 "timestep_max": self.timestep_max,
    #                 "unique_timesteps": unique_timesteps,
    #                 "min_timestep": min_timestep if min_timestep is not None else 0,
    #                 "max_timestep": max_timestep if max_timestep is not None else 0,
    #                 "total_agents_tracked": total_agents_tracked
    #             }


    @abstractmethod
    def update_agent_timestep(self, agent_id: int, obj: Any, emergency: bool=False) -> None:
        """
        Abstract method: query and update the agent's timestep

        Subclasses must implement this method to process the object based on its data type,
        calculate the new timestep for the agent, and then call modify_agent_timestep to update it.

        Args:
            agent_id: agent's ID
            obj: object to be processed
        """
        pass

    @abstractmethod
    def dependency_check(self, obj: Any) -> bool:
        """
        Abstract method: check if the object has dependencies that need to be resolved

        Subclasses must implement this method to determine if the object requires any dependencies
        before its timestep can be updated.

        Args:
            obj: object to be checked
        Returns:
            True if dependencies exist, False otherwise
        """
        pass

    @abstractmethod
    def dependency_translation(self, obj: Any) -> int:
        """
        Abstract method: translate the object to a timestep value

        Subclasses must implement this method to convert the object into a valid timestep value.

        Args:
            obj: object to be translated
        Returns:
            Translated timestep value
        """
        pass

    
