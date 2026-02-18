from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from SScheduler.logger import get_logger
logger = get_logger()

class BasePolicy(ABC):
    """
    Abstract base class for scheduler policies
    """
    
    @abstractmethod
    def integrate_agents(self, manager_data: Dict[str, Dict]) -> tuple[Dict[str, Any], Dict[int, List[str]]]:
        """
        Integrate agent information from different managers

        Args:
            manager_data: Dictionary with structure:
                {
                    'name': manager_name,
                    'type': type,
                    'priority': priority,
                    'data': {timestep: Set[agent_ids]}
                }
        Returns:
            Tuple containing:
                - Dictionary mapping agent_id to its minimum timestep
                - Dictionary mapping timestep to list of agent_ids
        """
        pass

class DefaultPolicy(BasePolicy):
    """
    Default policy implementation: satisfied all the requirements
    """

    def integrate_agents(self, manager_data: Dict[str, Dict]) -> tuple[Dict[str, Any], Dict[int, List[str]]]:
        """
        Simple integration: combine all agents from all managers
        For each agent, find the minimum timestep across all managers
        
        Args:
            manager_data: Dictionary
        Returns:
            Dictionary mapping agent_id to its minimum timestep:
            {agent_id(str): timestep(int)}
            {timestep_agents: {timestep(int): List[agent_id(str)]}}
        """
        if not manager_data:
            logger.debug("[PFEngine][integrate_agents][result]: no manager data found")
            return {}, {}

        logger.debug(f"[PFEngine][integrate_agents][status]: processing {len(manager_data)} managers")

        updict = defaultdict(list)

        for manager_name, timestep_data in manager_data['data'].items():
            for timestep, agent_ids in timestep_data.items():
                for agent_id in agent_ids:
                    updict[str(agent_id)].append(timestep)

        agent_timestep = {
            agent_id: int(min(timesteps))
            for agent_id, timesteps in updict.items()
        }
        timestep_agents: Dict[int, List[str]] = defaultdict(list)
        for agent_id, timestep in agent_timestep.items():
            timestep_agents[timestep].append(agent_id)
        timestep_agents = dict(timestep_agents)
            
        logger.debug(f"[PFEngine][integrate_agents][result]: integration completed, {len(agent_timestep)} agents processed")
        return agent_timestep, timestep_agents
