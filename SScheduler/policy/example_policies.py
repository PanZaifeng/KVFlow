from typing import Dict, List, Any, Set
from SScheduler.Scheduler import SchedulerPolicy


class PriorityPolicy(SchedulerPolicy):
    """
    Priority-based scheduling policy
    Agents with lower timesteps get higher priority
    """
    
    def __init__(self, max_agents_per_batch: int = 50):
        self.max_agents_per_batch = max_agents_per_batch
    
    def integrate_agents(self, manager_data: Dict[str, Dict[int, Set[int]]]) -> List[Dict[str, Any]]:
        """
        Integrate agents with priority-based scheduling
        """
        all_agents = []
        
        # Collect all agents with their priorities
        for manager_name, timestep_data in manager_data.items():
            for timestep, agent_ids in timestep_data.items():
                for agent_id in agent_ids:
                    all_agents.append({
                        'agent_id': agent_id,
                        'timestep': timestep,
                        'manager': manager_name,
                        'priority': timestep,  # Lower timestep = higher priority
                        'urgency': 1.0 / (timestep + 1)  # Higher urgency for lower timesteps
                    })
        
        # Sort by priority (timestep) and limit batch size
        all_agents.sort(key=lambda x: (x['priority'], x['agent_id']))
        return all_agents[:self.max_agents_per_batch]


class RoundRobinPolicy(SchedulerPolicy):
    """
    Round-robin scheduling policy
    Fairly distribute agents across managers
    """
    
    def __init__(self, max_agents_per_manager: int = 20):
        self.max_agents_per_manager = max_agents_per_manager
    
    def integrate_agents(self, manager_data: Dict[str, Dict[int, Set[int]]]) -> List[Dict[str, Any]]:
        """
        Integrate agents using round-robin scheduling
        """
        integrated_agents = []
        manager_counts = {}
        
        # Process each manager's data
        for manager_name, timestep_data in manager_data.items():
            manager_agents = []
            
            # Collect agents from this manager
            for timestep, agent_ids in sorted(timestep_data.items()):
                for agent_id in agent_ids:
                    manager_agents.append({
                        'agent_id': agent_id,
                        'timestep': timestep,
                        'manager': manager_name,
                        'priority': timestep
                    })
            
            # Limit agents per manager
            manager_agents = manager_agents[:self.max_agents_per_manager]
            integrated_agents.extend(manager_agents)
            manager_counts[manager_name] = len(manager_agents)
        
        return integrated_agents


class WeightedPolicy(SchedulerPolicy):
    """
    Weighted scheduling policy
    Different managers have different weights
    """
    
    def __init__(self, manager_weights: Dict[str, float] = None):
        self.manager_weights = manager_weights or {'base': 1.0, 'plan': 2.0}
    
    def integrate_agents(self, manager_data: Dict[str, Dict[int, Set[int]]]) -> List[Dict[str, Any]]:
        """
        Integrate agents using weighted scheduling
        """
        all_agents = []
        
        for manager_name, timestep_data in manager_data.items():
            weight = self.manager_weights.get(manager_name, 1.0)
            
            for timestep, agent_ids in timestep_data.items():
                for agent_id in agent_ids:
                    # Calculate weighted priority
                    weighted_priority = timestep / weight
                    
                    all_agents.append({
                        'agent_id': agent_id,
                        'timestep': timestep,
                        'manager': manager_name,
                        'priority': timestep,
                        'weighted_priority': weighted_priority,
                        'weight': weight
                    })
        
        # Sort by weighted priority
        all_agents.sort(key=lambda x: (x['weighted_priority'], x['agent_id']))
        return all_agents


class BatchPolicy(SchedulerPolicy):
    """
    Batch scheduling policy
    Group agents into batches for efficient processing
    """
    
    def __init__(self, batch_size: int = 32, max_batches: int = 10):
        self.batch_size = batch_size
        self.max_batches = max_batches
    
    def integrate_agents(self, manager_data: Dict[str, Dict[int, Set[int]]]) -> List[Dict[str, Any]]:
        """
        Integrate agents into batches
        """
        all_agents = []
        
        # Collect all agents
        for manager_name, timestep_data in manager_data.items():
            for timestep, agent_ids in timestep_data.items():
                for agent_id in agent_ids:
                    all_agents.append({
                        'agent_id': agent_id,
                        'timestep': timestep,
                        'manager': manager_name,
                        'priority': timestep
                    })
        
        # Sort by priority
        all_agents.sort(key=lambda x: (x['priority'], x['agent_id']))
        
        # Group into batches
        batched_agents = []
        total_agents = min(len(all_agents), self.batch_size * self.max_batches)
        
        for i in range(0, total_agents, self.batch_size):
            batch = all_agents[i:i + self.batch_size]
            batch_id = i // self.batch_size
            
            for agent in batch:
                agent['batch_id'] = batch_id
                agent['batch_size'] = len(batch)
                batched_agents.append(agent)
        
        return batched_agents
