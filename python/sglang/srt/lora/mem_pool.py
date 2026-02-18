import logging
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import time
import torch
import threading
from queue import Empty, Full, PriorityQueue, Queue

from sglang.srt.distributed import divide
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.utils import (
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    get_hidden_dim,
    get_normalized_target_modules,
    get_stacked_multiply,
    get_target_module_name,
)

logger = logging.getLogger(__name__)

class EmptySlot:
    """
    Singleton class to represent an empty slot in the memory pool.
    This is used to improve readability by not using special str as a placeholder.
    """

    __slots__ = ()

    def __repr__(self):
        return "|EMPTY|"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


EMPTY_SLOT = EmptySlot()
LowestPriority = 1024

class BufferSlot:
    
    READY = 0
    LOADING = 1

    def __init__(self, uid: Union[str, None, EmptySlot], priority: Optional[int] = 0, status: int = READY, pinned: bool = False):
        self.uid = uid
        self.priority = priority
        self.status = status
        self.pinned = pinned


class LoRAOperation:

    counter = 0

    def __init__(
        self,
        uid: Optional[str],
        buffer_id: int,
        lora_adapter: Optional[LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        priority: Optional[int] = None,
    ):

        self.uid = uid
        self.buffer_id = buffer_id
        self.lora_adapter = lora_adapter
        self.lora_modules = lora_modules
        self.priority = priority


class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_lora_rank: int,
        target_modules: Set[str],
        base_model: torch.nn.Module,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.max_lora_rank: int = max_lora_rank
        self.target_modules: Set[str] = target_modules

        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # A_buffer contains num_layer number of row-major tensors with shape
        #   (max_loras_per_batch, stacked_num * max_lora_dim, input_dim)
        # B_buffer contains num_layer number of column-major tensors with shape
        #   (stacked_num, max_loras_per_batch, output_dim, max_lora_dim)
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}

        # Buffer idx -> lora uid in memory pool
        # All uids are initialized as `EmptySlot` for empty buffer slots
        # Here we don't initialize to None since None is a valid uid
        # self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [
        #     EMPTY_SLOT
        # ] * self.max_loras_per_batch
        self.buffer_id_to_uid: Dict[int, BufferSlot] = {
            i: BufferSlot(uid=EMPTY_SLOT, priority=LowestPriority, status=BufferSlot.READY, pinned=False) for i in range(self.max_loras_per_batch)
        }


        self.init_buffers(base_model)
        
        self.load_lora_thread = threading.Thread(target=self.load_lora_cpu_to_gpu, daemon=True)
        self.load_lora_stream = torch.cuda.Stream()
        self.stop_lora_event = threading.Event()
        self.load_lora_queue = Queue()
        self._lora_registry = None

        self.load_lora_thread.start()


    def can_support(self, config: Union[LoRAConfig, Iterable[LoRAConfig]]) -> bool:
        """
        Check if the memory pool can support the given LoRA adapters.
        """

        def _can_support(config: LoRAConfig) -> bool:
            """
            Check if the memory pool can support a single LoRA adapter.
            """
            if config.r > self.max_lora_rank:
                return False
            target_module_names = get_normalized_target_modules(config.target_modules)
            return target_module_names.issubset(self.target_modules)

        if isinstance(config, LoRAConfig):
            return _can_support(config)
        else:
            return all(_can_support(x) for x in config)

    def get_lora_A_shape(
        self, module_name: str, base_model: torch.nn.Module, max_lora_dim: int
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        input_dim, _ = get_hidden_dim(module_name, self.base_hf_config, base_model)
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            input_dim = divide(input_dim, self.tp_size)
        return (
            self.max_loras_per_batch,
            max_lora_dim * c,
            input_dim,
        )

    def get_lora_B_shape(
        self, module_name: str, base_model: torch.nn.Module, max_lora_dim: int
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        _, output_dim = get_hidden_dim(module_name, self.base_hf_config, base_model)
        if self.tp_size > 1 and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            output_dim = divide(output_dim, self.tp_size)
        return (
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def init_buffers(self, base_model: torch.nn.Module):
        device = next(base_model.parameters()).device

        def init_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[str, torch.nn.Module, int], Tuple[int]],
        ):
            for module_name in target_modules:
                lora_shape = get_lora_shape_fn(
                    module_name, base_model, self.max_lora_rank
                )
                buffer[module_name] = [
                    torch.empty(
                        lora_shape,
                        dtype=self.dtype,
                        device=device,
                    )
                    for _ in range(self.num_layer)
                ]

        init_buffer(
            self.A_buffer,
            self.target_modules,
            self.get_lora_A_shape,
        )

        init_buffer(
            self.B_buffer,
            self.target_modules,
            self.get_lora_B_shape,
        )

    # [SS][Prefetch LoRA]
    def prefetch_lora_weights(
        self,
        lora_id: str,
        priority: int,
        step_lora_ids: List[str],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        lora_refs: Dict[str, LoRARef],
    ) -> bool:
        
        def get_available_buffer_slot():
            for buffer_id in range(self.max_loras_per_batch):
                if self.buffer_id_to_uid[buffer_id].uid == EMPTY_SLOT:
                    return buffer_id
            logger.debug("[lora][prefetch]  no enough slot for lora, need to evict")
            target_priority, target_buffer_id, target_slot_id, target_slot_status = -1, -1, -1, 0
            try:
                for buffer_id in range(0, self.max_loras_per_batch):
                    slot = self.buffer_id_to_uid[buffer_id]
                    if priority is not None and priority > slot.priority:
                        continue
                    if slot.uid in step_lora_ids or slot.pinned is True:
                        continue
                    if slot.uid is not None:
                        lora_ref = lora_refs.get(slot.uid)
                        if lora_ref is not None and lora_ref.pinned:
                            continue
                    if target_priority < slot.priority:
                        target_priority = slot.priority
                        target_buffer_id = buffer_id
                        target_slot_id = slot.uid
                        target_slot_status = slot.status
                
                if target_slot_id != -1:
                    self.uid_to_buffer_id.pop(target_slot_id)
                    if target_slot_status == BufferSlot.LOADING:
                        self._drop_pending_load(target_slot_id)
                    self.buffer_id_to_uid[target_buffer_id].uid = EMPTY_SLOT
                    return target_buffer_id

                return -1

            except Exception as e:
                logger.error(f"[lora][prefetch][Evict]  Error during eviction: {e}")
                return -1

        try:
            if lora_id not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                name = self._lora_registry.get(lora_id, lora_id) if self._lora_registry else lora_id
                if buffer_id == -1:
                    logger.info(f"[lora][prefetch]  no available slot for prefetching LoRA {name} now.")
                    return False
                logger.debug(f"[lora][prefetch]  Assigning LoRA {name} to slot {buffer_id}")
                lora_adapter = lora_adapters.get(lora_id, None)
                operation = LoRAOperation(uid=lora_id, buffer_id=buffer_id, lora_adapter=lora_adapter, lora_modules=lora_modules, priority=priority)
                try:
                    self.load_lora_queue.put(operation, block=False)
                    self.uid_to_buffer_id[lora_id] = buffer_id
                    self.buffer_id_to_uid[buffer_id].uid = lora_id
                    self.buffer_id_to_uid[buffer_id].priority = priority
                    self.buffer_id_to_uid[buffer_id].status = BufferSlot.LOADING
                except Full:
                    logger.info(f"[lora][prefetch]  Load queue is full, cannot prefetch LoRA {name} now.")
                    return False
            else:
                buffer_id = self.uid_to_buffer_id[lora_id]
                self.buffer_id_to_uid[buffer_id].priority = priority
        
            return True

        except Exception as e:
            logger.error(f"[lora][prefetch]  Error during prefetching LoRA {lora_id}: {e}")
            return False

    # [SS][Prepare LoRA]
    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        lora_refs: Dict[str, LoRARef],
    ):
        def get_available_buffer_slot():
            for buffer_id in range(self.max_loras_per_batch):
                if self.buffer_id_to_uid[buffer_id].uid == EMPTY_SLOT:
                    return buffer_id
            target_priority, target_buffer_id, target_slot_id, target_slot_status = -1, -1, -1, BufferSlot.READY
            try:
                for buffer_id in range(0, self.max_loras_per_batch):
                    slot = self.buffer_id_to_uid[buffer_id]
                    if slot.pinned is True or slot.uid in cur_uids:
                        continue
                    if slot.uid is not None:
                        lora_ref = lora_refs.get(slot.uid)
                        if lora_ref is not None and lora_ref.pinned:
                            continue
                    if (
                        slot.priority > target_priority
                        or (
                            slot.priority == target_priority
                            and slot.status == BufferSlot.LOADING
                        )
                    ):
                        target_priority = slot.priority
                        target_buffer_id = buffer_id
                        target_slot_id = slot.uid
                        target_slot_status = slot.status
                        
                if target_slot_id != -1:
                    self.uid_to_buffer_id.pop(target_slot_id)
                    if target_slot_status == BufferSlot.LOADING:
                        self._drop_pending_load(target_slot_id)
                    name = self._lora_registry.get(target_slot_id) if self._lora_registry else target_slot_id
                    logger.debug(f"[lora][prepare]  Evicting LoRA {name} from buffer slot {target_buffer_id}.")
                    self.buffer_id_to_uid[target_buffer_id].uid = EMPTY_SLOT
                    return target_buffer_id
                
                raise ValueError("No available buffer slots found after eviction attempt.")
            
            except Exception as e:
                logger.error(f"[lora][prepare][Evict]  Error during eviction: {e}")
                return -1

        try:
            for uid in cur_uids:
                if uid not in self.uid_to_buffer_id:
                    buffer_id = get_available_buffer_slot()
                    name = self._lora_registry.get(uid, uid) if self._lora_registry else uid
                    logger.debug(f"[lora][prepare]  Assigning LoRA {name} to slot {buffer_id}")
                    lora_adapter = lora_adapters.get(uid, None)
                    self.load_lora_weight_to_buffer(
                        uid, buffer_id, lora_adapter, lora_modules
                    )

                    # if name == "lora0":
                    #     self.buffer_id_to_uid[buffer_id].pinned = True
                    #     logger.info(f"[lora][pinned]  lora0 = none-agent-lora is pinned to slot {buffer_id}")
                        
                    self.uid_to_buffer_id[uid] = buffer_id
                    self.buffer_id_to_uid[buffer_id].uid = uid
                    self.buffer_id_to_uid[buffer_id].priority = 0
                    self.buffer_id_to_uid[buffer_id].status = BufferSlot.READY
                    logger.info(f"\033[91m [lora][prepare][Immediate]  Loaded LoRA {name} seconds\033[0m")
            
            break_flag = False
            
            while break_flag == False:
                for uid in cur_uids:
                    buffer_id = self.uid_to_buffer_id[uid]
                    if self.buffer_id_to_uid[buffer_id].status == BufferSlot.LOADING:
                        time.sleep(0.001)
                    else:
                        break_flag = True
                        break

        except Exception as e:
            logger.error(f"[lora][prepare]  Error during preparing LoRA batch: {e}")

    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
    ):
        def load_lora_weight_tensor(
            buffer_view: torch.Tensor, weight: Optional[torch.Tensor]
        ):
            if weight is None:
                # If the particular weight is not present in the adapter, we initialize the buffer to zero
                # to avoid contamination from the residual weight of the evicted adapters.
                buffer_view.zero_()
            else:
                assert (
                    buffer_view.shape == weight.shape
                ), f"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}."
                buffer_view.copy_(weight)

        t0 = time.perf_counter()
        if uid is None:
            for i in range(self.num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] = 0
            return

        assert lora_adapter is not None
        lora_rank = lora_adapter.config.r
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            temp_A_buffer: Dict[str, Optional[torch.Tensor]] = {
                target_module: None for target_module in self.A_buffer
            }
            temp_B_buffer: Dict[str, Optional[torch.Tensor]] = {
                target_module: None for target_module in self.B_buffer
            }
            for name, weights in layer_weights.items():
                target_module = get_target_module_name(name, self.target_modules)
                if "lora_A" in name:
                    temp_A_buffer[target_module] = weights
                else:
                    temp_B_buffer[target_module] = weights

            if self.tp_size > 1:
                cur_layer_modules = lora_modules[layer_id]
                for module_name, module in cur_layer_modules.items():
                    target_module = get_target_module_name(
                        module_name, self.target_modules
                    )

                    if temp_A_buffer[target_module] is None:
                        # Skip weight slicing if the weight is not present in the adapter
                        continue

                    temp_A_buffer[target_module] = module.slice_lora_a_weights(
                        temp_A_buffer[target_module], self.tp_rank
                    )
                    temp_B_buffer[target_module] = module.slice_lora_b_weights(
                        temp_B_buffer[target_module], self.tp_rank
                    )

            for name, weights in temp_A_buffer.items():
                c = get_stacked_multiply(name)
                target_buffer = self.A_buffer[name][layer_id]
                buffer_view = target_buffer[buffer_id, : lora_rank * c, :]
                load_lora_weight_tensor(buffer_view, weights)

            for name, weights in temp_B_buffer.items():
                target_buffer = self.B_buffer[name][layer_id]
                buffer_view = target_buffer[buffer_id, :, :lora_rank]
                load_lora_weight_tensor(buffer_view, weights)

    def get_tensor(
        self, target_module: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:
        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[target_module][layer_id]

        return self.B_buffer[target_module][layer_id]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]

    # [SS][Drop Pending Load]
    def _drop_pending_load(self, lora_uid: Optional[str]):
        if lora_uid is None:
            return
        with self.load_lora_queue.mutex:
            q = self.load_lora_queue.queue
            orig_len = len(q)
            filtered = [op for op in q if op.uid != lora_uid]
            removed = orig_len - len(filtered)
            if removed == 0:
                return
            q.clear()
            q.extend(filtered)
            logger.info(f"[lora][prefetch][drop]  Dropped {removed} pending load(s) for LoRA {lora_uid} from the load queue.")
            # Keep unfinished_tasks consistent so task_done() calls remain valid.
            self.load_lora_queue.unfinished_tasks = max(
                0, self.load_lora_queue.unfinished_tasks - removed
            )

    # [SS][Load LoRA CPU to GPU]
    def load_lora_cpu_to_gpu(self):
        with self.load_lora_stream:
            while not self.stop_lora_event.is_set():
                try:
                    operation = self.load_lora_queue.get(block=True, timeout=1)
                except Empty:
                    operation = None
                try:
                    if operation is not None:
                        self.load_lora_weight_to_buffer(
                            operation.uid,
                            operation.buffer_id,
                            operation.lora_adapter,
                            operation.lora_modules,
                        )
                        self.buffer_id_to_uid[operation.buffer_id].status = BufferSlot.READY
                        # name = self._lora_registry.get(operation.uid, operation.uid) if self._lora_registry else operation.uid
                        # logger.debug(f"\033[94m [lora][prefetch][finish]  LoRA {name} is ready in slot {operation.buffer_id} \033[0m")
                except Exception as e:
                    print(f"[SYP][lora]  Error loading LoRA: {e}")

    # [SS][Reset LoRA Priority]       
    def update_lora_priority(self):
        for buffer_id in range(self.max_loras_per_batch):
            slot = self.buffer_id_to_uid[buffer_id]
            slot.priority = LowestPriority
            # if slot.uid is EMPTY_SLOT:
            #     slot.priority = 1000
            # if slot.uid is not EMPTY_SLOT and slot.status == BufferSlot.READY:
            #     slot.priority -= 1
            #     if slot.priority < 0:
            #         slot.priority = 1000

    # [SS][Mark LoRA for Eviction]
    def mark_lora_for_eviction(self, lora_id: Optional[str], priority: int = LowestPriority):
        """Raise the priority of a finished request's LoRA to make it the first eviction candidate."""
        if lora_id is None:
            return False
        if lora_id not in self.uid_to_buffer_id:
            return False
        buffer_id = self.uid_to_buffer_id[lora_id]
        self.buffer_id_to_uid[buffer_id].priority = priority
        logger.info(f"[lora][priority] Marking LoRA {lora_id} in buffer slot {buffer_id} for eviction with priority {priority}")
        return True

    # [SS][List Loaded LoRA Names]
    def list_loaded_lora_names(self) -> List[str]:
        """Return the names of all LoRA adapters currently occupying memory pool slots."""
        names: List[str] = []
        seen: Set[Optional[str]] = set()
        for buffer_id in sorted(self.buffer_id_to_uid.keys()):
            uid = self.buffer_id_to_uid[buffer_id].uid
            if uid in (EMPTY_SLOT, None):
                continue
            if uid in seen:
                continue
            seen.add(uid)
            names.append(self._lora_registry.get(uid, uid) if self._lora_registry else uid)
        return names
