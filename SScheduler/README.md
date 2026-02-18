# PFEngine

A lightweight scheduling and timestep management engine for agent simulations.

## Features
- Core timestep managers (BaseManager, PlanManager, SpaceManager) to track agents and their interaction cadence.
- Scheduler to register managers and integrate agent timesteps.
- Optional LLM helper (llm/) for language-model-driven behaviors; not required for core engine use.

## Project Layout
- [Scheduler.py](Scheduler.py): orchestrates managers and integrates results.
- [timestepManager/BaseManager.py](timestepManager/BaseManager.py): abstract manager with shared timestep storage.
- [timestepManager/PlanManager.py](timestepManager/PlanManager.py): plan-based timestep logic.
- [timestepManager/SpaceManager.py](timestepManager/SpaceManager.py): spatial/interaction timestep logic.
- [policy/](policy/): policies for merging manager outputs.
- [llm/](llm/): optional LLM utilities (can be ignored if not using LLM features).

## Quick Start
1) Instantiate managers and register with the scheduler:
```python
from PFEngine.Scheduler import Scheduler
from PFEngine.timestepManager import PlanManager, SpaceManager

scheduler = Scheduler(prefetch_step=1)
plan_mgr = PlanManager(agent_num=10, timestep_max=100)
space_mgr = SpaceManager(agent_num=10, timestep_ticks=1, timestep_max=100)

scheduler.register_manager(plan_mgr, activate=True)
scheduler.register_manager(space_mgr, activate=True)
```

2) Update agent timesteps through the managers, then run scheduling:
```python
space_mgr.update_agent_timestep(
    agent_id=0,
    obj={
        "moving_speed": 1.0,
        "current_timestep": 10,
        "interact_threshold": 3,
        "last_interact_timestep": 5,
        "is_moving": [True, True],
        "agent_place": ["A", "A"],
        "corresponding_distances": [2.0, 5.0],
    },
)

# scheduler.do_schedule()  # integrate and send results
```

## Notes
- `SpaceManager` computes the next interaction timestep based on movement speed, interaction cooldown, destination matching, and arrival time.
- you can develop your own `Manager` followed the `BaseManager`.
- `llm/` is optional; you might config your llm in your frontend code.
