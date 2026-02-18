import threading
from collections import defaultdict
from typing import Dict, Set, Any, Optional, List
# from sglang.srt.mem_cache.radix_cache import TreeNode
import logging

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, hold_step: int, prefetch_step: int):

        self.hold_step = hold_step
        self.prefetch_step = prefetch_step
        self.agent_to_last_nodes: Dict[str, Set[Any]] = defaultdict(set)
        self.agent_last_node_id: int = None
        self.update_dict_agent = dict[str, Any]()
        self.update_dict_timestep = dict[int, List[str]]()
        self.update_version: int = 0
        self.lock = threading.RLock()

    def update_agent_timestep(self, update_dict_agent: dict[str, Any], update_dict_timestep: dict[int, List[str]]):
        with self.lock:
            self.update_dict_agent = update_dict_agent
            self.update_dict_timestep = update_dict_timestep
            self.update_version = (self.update_version + 1) % (1 << 63)
        if self.update_dict_agent is None or len(self.update_dict_agent) == 0:
            logger.error("\033[91mupdate_dict_agent is None or empty\033[0m")
        if self.update_dict_timestep is None or len(self.update_dict_timestep) == 0:
            logger.warning("\033[91mupdate_dict_timestep is None or empty\033[0m")

    def get_agent_hold_priority(self, agent_id: str) -> int:
        if agent_id is None:
            raise ValueError("agent_id cannot be None")
        if agent_id == '-1':
            logger.error(f"\033[91m hold agent_id: {agent_id} is invalid!!! \033[0m")
            return self.hold_step + 2
        if agent_id not in self.update_dict_agent:
            logger.debug(f"\033[91m hold agent_id: {agent_id} not in update_dict_agent\033[0m")
            return self.hold_step + 1
        if self.update_dict_agent[agent_id] is None:
            logger.error(f"\033[91m hold agent_id: {agent_id} 's value is invalid!!! \033[0m")
            return self.hold_step + 2
        return min(self.hold_step + 1, self.update_dict_agent[agent_id])

    def get_agents_hold_priority(self, agents: list[str]):
        max_priority = self.hold_step + 2
        max_agent_id = None
        for agent_id in agents:
            priority = self.get_agent_hold_priority(agent_id)
            if priority < max_priority:
                max_priority = priority
                max_agent_id = agent_id
        return max_agent_id, max_priority

    def get_node_hold_priority(self, node) -> tuple[Optional[str], int]:
        if node is None:
            return None, self.hold_step + 2
        if node.hold_priority_version == self.update_version:
            return None, node.hold_priority

        agent_ids = list(node.agents.keys())
        agent_id, priority = self.get_agents_hold_priority(agent_ids)
        node.hold_priority_version = self.update_version
        node.hold_priority = priority
        return agent_id, priority

    def get_agent_prefetch_priority(self, agent_id: str) -> int:
        if self.update_dict_agent[agent_id] is None:
            logger.error(f"\033[91m prefetch agent_id: {agent_id} 's value is invalid!!! \033[0m")
            return self.prefetch_step + 2
        return min(self.prefetch_step + 1, self.update_dict_agent[agent_id])

    def get_agents_prefetch_priority(self, agents: list[str]):
        max_priority = self.prefetch_step + 2
        max_agent_id = None
        for agent_id in agents:
            priority = self.get_agent_prefetch_priority(agent_id)
            if priority < max_priority:
                max_priority = priority
                max_agent_id = agent_id
        return max_agent_id, max_priority

    def get_update_dict_agent(self):
        return self.update_dict_agent
    
    def get_update_dict_timestep(self):
        return self.update_dict_timestep
    
    def get_agent_to_last_nodes(self):
        return self.agent_to_last_nodes

    def debug_print(self):
        for agent_id, last_nodes in self.agent_to_last_nodes.items():
            logger.info(f"- Agent ID: {agent_id}, Last Nodes: {[node.id for node in last_nodes]}")
        logger.info(f"- Agent Last Node ID: {self.agent_last_node_id}")
