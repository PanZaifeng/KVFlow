import math
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from abc import abstractmethod
from SScheduler.timestepManager.BaseManager import BaseManager

# Import logger
from SScheduler.logger import get_logger

logger = get_logger()

class SpaceManager(BaseManager):
    """
    SpaceManager base class: maintain the datastructures for agents and their plans

    Features:
    Input the distance and other correspond information of the agent, calculate the timestep value for the agent
    """

    def __init__(self, agent_num: int, timestep_ticks: int, timestep_max: int = 100, priority: Optional[int] = 1):
        """
        Initialize SpaceManager
        
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
        Update agent timestep based on movement and interaction context.

        `obj` should carry:
        - moving_speed: float
        - current_timestep: int
        - interact_threshold: int
        - last_interact_timestep: int
        - is_moving: List[bool]
        - agent_place: List[Any]
        - corresponding_distances: List[Any]

        Args:
            agent_id: agent's ID
            obj: mapping/object containing the fields above
            emergency: if True, set timestep to 0 immediately
        """
        (
            moving_speed,
            current_timestep,
            interact_threshold,
            last_interact_timestep,
            is_moving,
            agent_place,
            corresponding_distances,
        ) = self._parse_obj(obj)

        logger.debug(
            "[PFEngine][update_agent_timestep][status]: "
            f"agent_{agent_id}, moving_speed={moving_speed}, current_timestep={current_timestep}, "
            f"interact_threshold={interact_threshold}, last_interact_timestep={last_interact_timestep}, "
            f"is_moving={is_moving}, agent_place={agent_place}, corresponding_distances={corresponding_distances}, "
            f"emergency={emergency}"
        )

        if emergency:
            self.modify_agent_timestep(agent_id=agent_id, new_timestep=0)
            return

        new_timestep = self.dependency_translation(obj)
        self.modify_agent_timestep(agent_id=agent_id, new_timestep=new_timestep)

    def dependency_check(self, obj: Any) -> bool:
        """Light validation for timestep calculation inputs."""
        (
            moving_speed,
            current_timestep,
            interact_threshold,
            last_interact_timestep,
            is_moving,
            agent_place,
            corresponding_distances,
        ) = self._parse_obj(obj)

        if moving_speed <= 0:
            raise ValueError("moving_speed must be positive to compute arrival time.")

        if not (len(is_moving) == len(agent_place) == len(corresponding_distances)):
            raise ValueError(
                "is_moving, agent_place, and corresponding_distances must share the same length (agents count)."
            )

        if len(is_moving) == 0:
            raise ValueError("Agent lists cannot be empty.")

        if current_timestep < 0 or interact_threshold < 0 or last_interact_timestep < 0:
            raise ValueError("Timesteps and thresholds must be non-negative.")

        return True

    def _coerce_distance(self, distance_entry: Any) -> float:
        """Extract a numeric distance from allowed formats (scalar or singleton list/tuple)."""
        if isinstance(distance_entry, (int, float)):
            return float(distance_entry)
        if isinstance(distance_entry, (list, tuple)) and len(distance_entry) > 0:
            inner = distance_entry[0]
            if isinstance(inner, (int, float)):
                return float(inner)
        raise ValueError(f"Unsupported distance entry: {distance_entry}")

    def _parse_obj(self, obj: Any) -> Tuple[float, int, int, int, List[bool], List[Any], List[Any]]:
        """Extract expected fields from a mapping or simple object."""
        if isinstance(obj, dict):
            getter = obj.get
        else:
            getter = lambda k: getattr(obj, k, None)  # type: ignore

        moving_speed = getter("moving_speed")
        current_timestep = getter("current_timestep")
        interact_threshold = getter("interact_threshold")
        last_interact_timestep = getter("last_interact_timestep")
        is_moving = getter("is_moving")
        agent_place = getter("agent_place")
        corresponding_distances = getter("corresponding_distances")

        required = {
            "moving_speed": moving_speed,
            "current_timestep": current_timestep,
            "interact_threshold": interact_threshold,
            "last_interact_timestep": last_interact_timestep,
            "is_moving": is_moving,
            "agent_place": agent_place,
            "corresponding_distances": corresponding_distances,
        }

        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"Missing fields in obj: {missing}")

        return (
            float(moving_speed),
            int(current_timestep),
            int(interact_threshold),
            int(last_interact_timestep),
            list(is_moving),
            list(agent_place),
            list(corresponding_distances),
        )

    def dependency_translation(self, obj: Any) -> int:
        """
        Translate context to the timestep until the next interaction.

        Logic:
        1) If the gap since last interaction is below the threshold, return the remaining gap.
        2) Filter to agents (including self) sharing the same destination as the current agent.
        3) For those agents, compute remaining timesteps to arrive: 0 if already there, else ceil(distance / speed).
        4) Return the minimum of those remaining arrival times.
        """
        (
            moving_speed,
            current_timestep,
            interact_threshold,
            last_interact_timestep,
            is_moving,
            agent_place,
            corresponding_distances,
        ) = self._parse_obj(obj)

        logger.debug(
            f"[PFEngine][dependency_translation][status]: moving_speed={moving_speed}, "
            f"current_timestep={current_timestep}, interact_threshold={interact_threshold}, "
            f"last_interact_timestep={last_interact_timestep}, is_moving={is_moving}, "
            f"agent_place={agent_place}, corresponding_distances={corresponding_distances}"
        )

        self.dependency_check(obj)

        # enforce interaction threshold gap
        elapsed_since_last = current_timestep - last_interact_timestep
        if elapsed_since_last < interact_threshold:
            remaining_gap = interact_threshold - elapsed_since_last
            logger.debug(
                f"[PFEngine][dependency_translation][threshold]: elapsed={elapsed_since_last}, remaining_gap={remaining_gap}"
            )
            return max(remaining_gap, 0)

        # keep only agents with the same destination as the current agent
        self_destination = agent_place[0]
        candidate_indices = [idx for idx, place in enumerate(agent_place) if place == self_destination]

        if not candidate_indices:
            logger.debug("[PFEngine][dependency_translation][dest]: no candidates share destination; returning 0")
            return 0

        # compute remaining timesteps to arrival for candidates
        remaining_timesteps: List[int] = []
        for idx in candidate_indices:
            moving_flag = is_moving[idx]
            distance_value = self._coerce_distance(corresponding_distances[idx])

            if not moving_flag or distance_value <= 0:
                remaining_timesteps.append(0)
                continue

            timesteps_needed = math.ceil(distance_value / moving_speed)
            remaining_timesteps.append(int(max(timesteps_needed, 0)))

        if not remaining_timesteps:
            logger.debug("[PFEngine][dependency_translation][compute]: no remaining candidates; returning 0")
            return 0

        result = min(remaining_timesteps)
        logger.debug(f"[PFEngine][dependency_translation][result]: remaining_timesteps={remaining_timesteps}, result={result}")
        return result

