import argparse
from typing import Dict, List, Any, Optional, Set, Type
from abc import ABC, abstractmethod
import importlib
import time

# Import available managers
from SScheduler.policy.base_policy import BasePolicy, DefaultPolicy
from SScheduler.timestepManager.BaseManager import BaseManager
from SScheduler.timestepManager.PlanManager import PlanManager
from SScheduler.logger import get_logger, set_logger_level
from SScheduler.utils import MANAGER_REGISTRY
import requests

# Configure logging using PFEngine logger
logger = get_logger()
set_logger_level('INFO', logger)

class Scheduler:
    """
    Scheduler class for managing multiple TimestepManagers and integrating agent information
    """

    def __init__(self, prefetch_step: int = 1, policy: Optional[BasePolicy] = None, activate: bool = True):
        """
        Initialize Scheduler
        
        Args:
            prefetch_step: Number of timesteps to prefetch
            policy: Policy for integrating agent information
            activate: Whether the scheduler is activated
        """
        self.prefetch_step = prefetch_step
        self.policy = policy if policy else DefaultPolicy()
        self.activate = activate
        
        # Dictionary to store registered managers
        self.managers: Dict[str, BaseManager] = {}
        
        # Track which managers are active
        self.active_managers: Set[str] = set()

        self.last_update_time = time.time()

        self.timestep_cnt = 0
        self.activate_agent = []

        logger.info(f"Scheduler initialized with prefetch_step={prefetch_step}, activate={activate}")
    
    def register_manager(self, manager_instance: BaseManager, manager_name: Optional[str] = None, activate: bool = True) -> bool:
        """
        Register a manager instance
        
        Args:
            manager_instance: An instance of a manager that inherits from BaseManager
            manager_name: Optional custom name for the manager. If not provided, will use class name
            activate: Whether to activate this manager
        Returns:
            True if successful, False otherwise
        """
        try:
            if not isinstance(manager_instance, BaseManager):
                logger.error(f"Manager instance must inherit from BaseManager, got {type(manager_instance)}")
                return False

            if manager_name is None:
                manager_name = manager_instance.__class__.__name__
            
            if manager_name in self.managers:
                logger.warning(f"Manager '{manager_name}' already exists, replacing it")
                return False

            # Set the manager's activate state
            manager_instance.activate = activate
            
            self.managers[manager_name] = manager_instance
            self.active_managers.add(manager_name)
            
            logger.debug(f"[PFEngine][register_manager][result]: {manager_name} registered with activate={activate}, active_managers: {self.active_managers}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register manager: {e}")
            return False
    
    def unregister_manager(self, manager_name: str) -> bool:
        """
        Unregister a manager
        
        Args:
            manager_name: Name of the manager to unregister
        Returns:
            True if successful, False otherwise
        """
        if manager_name not in self.managers:
            logger.warning(f"Manager '{manager_name}' not found")
            return False
        
        try:
            del self.managers[manager_name]
            self.active_managers.discard(manager_name)
            logger.info(f"Successfully unregistered manager: {manager_name}")
            return True
        except Exception as e:
            logger.error(f"Error unregistering manager {manager_name}: {e}")
            return False
    
    def list_managers(self) -> Dict[str, str]:
        """
        List all registered managers
        
        Returns:
            Dictionary mapping manager names to their class names
        """
        return {name: type(manager).__name__ for name, manager in self.managers.items()}
    
    def get_manager(self, manager_name: str) -> Optional[BaseManager]:
        """
        Get a registered manager by name
        
        Args:
            manager_name: Name of the manager
            
        Returns:
            Manager instance or None if not found
        """
        return self.managers.get(manager_name)

    def collect_and_merge(self, steps: Optional[int], policy: Optional[BasePolicy]) -> tuple[Dict[str, Any], Dict[int, List[str]]]:
        """
        Collect agent data, apply policy, merge
        
        Returns:
            dictionaries of agent with their timesteps
            {agent_id: int, timestep: int}
        """
        if not self.activate:
            logger.debug(f"[PFEngine][collect_and_merge][disabled]: scheduler not activated")
            return {}

        manager_data = {'name': {}, 'type': {}, 'priority': {}, 'data': {}}
        steps = steps if steps else self.prefetch_step
        policy = policy if policy else self.policy

        for manager_name in self.active_managers:
            manager = self.managers[manager_name]
            try:
                timestep_data = manager.get_agent_n_timesteps(steps)
                manager_data['name'][manager_name] = manager_name
                manager_data['data'][manager_name] = timestep_data
                manager_data['type'][manager_name] = manager.get_type()
                manager_data['priority'][manager_name] = manager.get_priority() 
            except Exception as e:
                logger.error(f"Error collecting data from manager {manager_name}: {e}")
                manager_data['data'][manager_name] = {}
                manager_data['priority'][manager_name] = None

        for mname, data in manager_data['data'].items():
            logger.info(f"[PFEngine][collect_and_merge][collected]: manager={mname}, data={data}")
        
        a2t, t2a = policy.integrate_agents(manager_data)

        logger.info(f"[PFEngine][collect_and_merge][result][{self.timestep_cnt} ts]: integration completed, {len(a2t)} agents processed")
    
        return a2t, t2a

    def send_result(self, agent_data: Dict[str, Any], timestep_agents: Dict[int, List[str]]) -> bool:
        """
        Send processed result to backend
        
        Args:
            agent_data: Processed agent information
            timestep_agents: Mapping of timesteps to agent IDs
            
        Returns:
            True if successful, False otherwise
        """
        if not self.activate:
            logger.debug(f"[PFEngine][send_result][disabled]: scheduler not activated")
            return True

        try:
            payload = {
                "agent_data": agent_data,
                "timestep_data": timestep_agents,
                "timestep_cnt": self.timestep_cnt
            }
            response = requests.post(f"http://localhost:8001/v1/update", json=payload, timeout=2)
            if response.status_code != 200:
                logger.error(f"Failed to send agent data: {response.status_code}, {response.text}")
                return False
            logger.critical(f"\033[31m[PFEngine][send_result][statistic][{self.timestep_cnt} ts]: sending {len(agent_data)} agent successfully, with [{len(self.activate_agent)}] timestep=0 agents: {self.activate_agent}\033[0m")
            self.activate_agent = timestep_agents.get(0, [])
            self.timestep_cnt += 1
        except Exception as e:
            logger.error(f"Exception while sending agent data: {e}")
            return False
        logger.info(f"Sending {len(agent_data)} agents to sglang backend")

        return True
    
    def do_schedule(self, steps: Optional[int] = None, policy: Optional[BasePolicy] = None) -> bool:
        """
        Run a complete scheduling cycle
        
        Args:
            steps: Number of steps to prefetch (optional)
            policy: Policy to use for integration (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.activate:
            logger.debug(f"[PFEngine][do_schedule][disabled]: scheduler not activated")
            return True
            
        try:
            agent_data, timestep_data = self.collect_and_merge(steps, policy)
            success = self.send_result(agent_data, timestep_data)
            # 保留time的精度到1ms
            last_update_time = self.last_update_time
            self.last_update_time = time.time()
            elapsed = self.last_update_time - last_update_time
            logger.critical(f"[PFEngine][do_schedule][result]: scheduling cycle completed, success={success}, time taken={elapsed:.3f} seconds")
            # logger.critical(".")
            # logger.critical(".")
            # logger.critical(".")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in scheduling cycle: {e}")
            return False
    

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status information
        
        Returns:
            Dictionary with status information
        """
        manager_stats = {}
        for name, manager in self.managers.items():
            try:
                manager_stats[name] = manager.get_stats()
                manager_stats[name]['activate'] = manager.activate
            except Exception as e:
                manager_stats[name] = {'error': str(e), 'activate': getattr(manager, 'activate', None)}
        
        return {
            'prefetch_step': self.prefetch_step,
            'scheduler_activate': self.activate,
            'active_managers': list(self.active_managers),
            'total_managers': len(self.managers),
            'registered_managers': self.list_managers(),
            'manager_stats': manager_stats
        }
    
