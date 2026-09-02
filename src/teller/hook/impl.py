#!/usr/bin/env python3
# vllm_hook.py

from __future__ import annotations

import ctypes
import functools
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys

# import atexit
from types import ModuleType
from typing import Any, Dict, List

from teller import log

# import nvtx  # pip install nvidia-extensions


# Load the native NVTX library.
try:
    if os.name == "nt":
        # Windows
        libnvtx = ctypes.WinDLL("nvToolsExt64_1.dll")
    elif os.name == "posix" and sys.platform == "darwin":
        # macOS
        libnvtx = ctypes.CDLL("libnvToolsExt.dylib")
    else:
        # Linux
        libnvtx = ctypes.CDLL("libnvToolsExt.so")
except OSError as e:
    raise RuntimeError(f"Failed to load NVTX library: {e}")

# NVTX function signatures
_nvtxRangePushA = libnvtx.nvtxRangePushA
_nvtxRangePushA.argtypes = [ctypes.c_char_p]
_nvtxRangePushA.restype = ctypes.c_int

_nvtxRangePop = libnvtx.nvtxRangePop
_nvtxRangePop.argtypes = []
_nvtxRangePop.restype = ctypes.c_int

# --------------------------------------------------------------------------- #
# Meta path finder for configured Torch and vLLM modules.
# --------------------------------------------------------------------------- #
_VLLM_MODULES = frozenset({
    "vllm.executor.executor_base",
    "vllm.engine.async_llm_engine",
    "vllm.engine.llm_engine",
    "vllm.executor.ray_distributed_executor",
    "vllm.distributed.communication_op",
})


class TorchNVTXTracer(importlib.abc.MetaPathFinder):
    def __init__(self, json_path: str = "pytorch.json") -> None:
        self.json_path = json_path
        self.trace_entries: List[Dict[str, Any]] = []
        self.loader_instance: TorchNVTXLoader | None = None
        self._intercepted_modules: set[str] = {"torch"} | set(_VLLM_MODULES)
        self._parse_json()

    
    # Load configured Torch functions.
    def _parse_json(self) -> None:
        if not os.path.isfile(self.json_path):
            log.warn(f"config missing: {self.json_path} (NVTX tracking will be skipped)")
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                configs = json.load(f)
        except Exception as e:
            log.error(f"config parse failed: {self.json_path}: {e}")
            return

        count = 0
        for entry in configs:
            module = entry.get("module")
            domain = entry.get("domain", "Default")
            color = entry.get("color")
            funcs = entry.get("functions", [])
            if not module or not funcs:
                continue
            for fpath in funcs:
                self.trace_entries.append(
                    {"module": module, "func_path": fpath, "domain": domain, "color": color}
                )
                count += 1
            if module:
                self._intercepted_modules.add(module)
        log.info(f"config loaded: {count} entries from {self.json_path}")

    # Inject NVTX when an intercepted module is loaded.
    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname not in self._intercepted_modules:
            return None

        sys.meta_path = [h for h in sys.meta_path if h is not self]
        try:
            spec = importlib.util.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)

        if spec and not isinstance(spec.loader, TorchNVTXLoader):
            spec.loader = TorchNVTXLoader(spec.loader, self, fullname)
            self.loader_instance = spec.loader
        return spec


# --------------------------------------------------------------------------- #
# Loader that wraps selected Torch and vLLM functions.
# --------------------------------------------------------------------------- #
class TorchNVTXLoader(importlib.abc.Loader):
    def __init__(self, base_loader: importlib.abc.Loader, tracer: TorchNVTXTracer, fullname: str) -> None:
        self.base_loader = base_loader
        self.tracer = tracer
        self.fullname = fullname
        self.emit_nvtx_cm = None

    def create_module(self, spec):  # type: ignore[override]
        return self.base_loader.create_module(spec)

    def exec_module(self, module):  # type: ignore[override]
        self.base_loader.exec_module(module)

        if self.fullname == "torch":
            torch = module
            orig_set_device = torch.cuda.set_device

            # Start backend NVTX emission after device selection.
            @functools.wraps(orig_set_device)
            def hooked_set_device(device):
                result = orig_set_device(device)
                if not hasattr(hooked_set_device, "_nvtx_started"):
                    try:
                        cm = torch.autograd.profiler.emit_nvtx()
                        cm.__enter__()
                        hooked_set_device._nvtx_cm = cm
                        hooked_set_device._nvtx_started = True
                        log.info(f"emit_nvtx started, device={device}")
                    except Exception as e:
                        log.error(f"emit_nvtx failed: {e}")
                return result

            torch.cuda.set_device = hooked_set_device

            self._patch_functions(only_module="torch")
        elif self.fullname in self.tracer._intercepted_modules and self.fullname not in _VLLM_MODULES:
            # Patch the loaded PyTorch submodule.
            self._patch_functions_for_module(module, self.fullname)
        # Wrap communication operators.
        elif self.fullname == "vllm.distributed.communication_op":
            try:
                nvtx = importlib.import_module("torch.cuda.nvtx")
                
                # Selected communication functions
                func_names = [
                    "tensor_model_parallel_all_reduce",
                    "tensor_model_parallel_all_gather",
                    "tensor_model_parallel_reduce_scatter",
                    "tensor_model_parallel_gather",
                    "broadcast_tensor_dict"
                ]
                
                # Wrapper factory
                def create_wrapped_function(original_func, func_name):
                    @functools.wraps(original_func)
                    def wrapped_function(*args, **kwargs):
                        nvtx.range_push(f"communication_op::{func_name}")
                        try:
                            return original_func(*args, **kwargs)
                        finally:
                            nvtx.range_pop()
                    return wrapped_function
                
                # Apply wrappers
                for func_name in func_names:
                    if hasattr(module, func_name):
                        original_func = getattr(module, func_name)
                        wrapped_func = create_wrapped_function(original_func, func_name)
                        setattr(module, func_name, wrapped_func)
                        log.info(f"nvtx inject ok: {func_name}")
                    else:
                        log.warn(f"nvtx inject skip: {func_name} (not found)")
            except Exception as e:
                log.error(f"nvtx inject fail (communication_op): {e}")
                
        # Wrap inference-engine step functions.
        elif self.fullname == "vllm.engine.llm_engine":
            try:
                LLMEngine = module.LLMEngine
                original_step = LLMEngine.step
                # print(LLMEngine.step)
                @functools.wraps(original_step)
                def wrapped_step(self, *args, **kwargs):
                    return original_step(self, *args, **kwargs)

                LLMEngine.step = wrapped_step
                log.info("nvtx inject ok: LLMEngine.step")
            except Exception as e:
                log.error(f"nvtx inject fail (LLMEngine.step): {e}")
        elif self.fullname == "vllm.engine.async_llm_engine":
            try:
                LLMEngine = module._AsyncLLMEngine
                original_step = LLMEngine.step_async
                # print(LLMEngine.step_async)
                @functools.wraps(original_step)
                def wrapped_step(self, *args, **kwargs):
                    return original_step(self, *args, **kwargs)

                LLMEngine.step_async = wrapped_step
                log.info("nvtx inject ok: AsyncLLMEngine.step_async")
            except Exception as e:
                log.error(f"nvtx inject fail (AsyncLLMEngine.step_async): {e}")
        elif self.fullname == "vllm.executor.executor_base":
            try:
                ExecutorBase = module.DistributedExecutorBase
                original_execute_model = ExecutorBase.execute_model
                # print(ExecutorBase.execute_model)
                
                @functools.wraps(original_execute_model)
                def wrapped_step(self, *args, **kwargs):
                    nvtx = importlib.import_module("torch.cuda.nvtx")
                    if 'execute_model_req' in kwargs:
                        execute_model_req = kwargs['execute_model_req']
                    elif args:
                        execute_model_req = args[0]
                    else:
                        execute_model_req = None

                    request_ids = "unknown"
                    
                    if execute_model_req:
                        request_ids_list = []
                        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
                            request_id = getattr(seq_group_metadata, 'request_id', None)
                            if request_id:
                                request_ids_list.append(str(request_id))
                        
                        request_ids = "_".join(request_ids_list)
                    
                    message = f"step_execute_model[request_ids:{request_ids}]"
                    nvtx.range_push(message)
                    try:
                        return original_execute_model(self, *args, **kwargs)
                    finally:
                        nvtx.range_pop()
                ExecutorBase.execute_model = wrapped_step
                log.info("nvtx inject ok: ExecutorBase.execute_model")
            except Exception as e:
                log.error(f"nvtx inject fail (ExecutorBase.execute_model): {e}") 
        elif self.fullname == "vllm.executor.ray_distributed_executor":
            try:
                ExecutorBase = module.RayDistributedExecutor
                original_execute_model = ExecutorBase.execute_model_async
                # print(ExecutorBase.execute_model)
                
                @functools.wraps(original_execute_model)
                def wrapped_step(self, *args, **kwargs):
                    nvtx = importlib.import_module("torch.cuda.nvtx")
                    if 'execute_model_req' in kwargs:
                        execute_model_req = kwargs['execute_model_req']
                    elif args:
                        execute_model_req = args[0]
                    else:
                        execute_model_req = None
                    request_ids = "unknown"
                    if execute_model_req:
                        request_ids_list = []
                        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
                            request_id = getattr(seq_group_metadata, 'request_id', None)
                            if request_id:
                                request_ids_list.append(str(request_id))
                        
                        request_ids = "_".join(request_ids_list)
                    
                    message = f"step_execute_model[request_ids:{request_ids}]"
                    nvtx.range_push(message)
                    try:
                        return original_execute_model(self, *args, **kwargs)
                    finally:
                        nvtx.range_pop()
                ExecutorBase.execute_model_async = wrapped_step
                log.info("nvtx inject ok: RayExecutorBase.execute_model_async")
            except Exception as e:
                log.error(f"nvtx inject fail (RayExecutorBase.execute_model_async): {e}")
    # Patch configured Torch functions.
    def _make_wrapper(self, orig_fn, domain: str, color: Any, fullname: str):
        file = (
            getattr(orig_fn, "__code__", None).co_filename
            if hasattr(orig_fn, "__code__")
            else ""
        )
        lino = (
            getattr(orig_fn, "__code__", None).co_firstlineno
            if hasattr(orig_fn, "__code__")
            else 0
        )
        message = f"{fullname} at {file}:{lino}"

        @functools.wraps(orig_fn)
        def _wrapped(*args, **kwargs):
            msg_bytes = message.encode('utf-8')
            _nvtxRangePushA(msg_bytes)
            try:
                return orig_fn(*args, **kwargs)
            finally:
                _nvtxRangePop()

        _wrapped.__nvtx_wrapped__ = True
        return _wrapped

    def _patch_functions(self, only_module: str | None = None) -> None:
        """Patch functions with NVTX. If only_module is set, only patch entries for that module."""
        ok, fail = 0, 0
        for ent in self.tracer.trace_entries:
            module_name: str = ent["module"]
            if only_module is not None and module_name != only_module:
                continue
            func_path: str = ent["func_path"]
            domain: str = ent["domain"]
            color = ent["color"]

            try:
                mod: ModuleType = importlib.import_module(module_name)
                parent_obj = mod
                attrs = func_path.split(".")
                for attr in attrs[:-1]:
                    parent_obj = getattr(parent_obj, attr)
                fn_name = attrs[-1]
                orig_fn = getattr(parent_obj, fn_name)

                if getattr(orig_fn, "__nvtx_wrapped__", False):
                    continue

                new_fn = self._make_wrapper(
                    orig_fn, domain, color, f"{module_name}.{func_path}"
                )
                setattr(parent_obj, fn_name, new_fn)
                ok += 1
            except Exception as e:
                log.error(f"wrap fail {module_name}.{func_path}: {e}")
                fail += 1
        if ok or fail:
            log.info(f"wrap ok: {ok} ok, {fail} fail")

    def _patch_functions_for_module(self, module: ModuleType, module_name: str) -> None:
        """Patch only the entries for the given module (already loaded)."""
        ok, fail = 0, 0
        for ent in self.tracer.trace_entries:
            if ent["module"] != module_name:
                continue
            func_path: str = ent["func_path"]
            domain: str = ent["domain"]
            color = ent["color"]
            try:
                parent_obj = module
                attrs = func_path.split(".")
                for attr in attrs[:-1]:
                    parent_obj = getattr(parent_obj, attr)
                fn_name = attrs[-1]
                orig_fn = getattr(parent_obj, fn_name)
                if getattr(orig_fn, "__nvtx_wrapped__", False):
                    continue
                new_fn = self._make_wrapper(
                    orig_fn, domain, color, f"{module_name}.{func_path}"
                )
                setattr(parent_obj, fn_name, new_fn)
                ok += 1
            except Exception as e:
                log.error(f"wrap fail {module_name}.{func_path}: {e}")
                fail += 1
        if ok or fail:
            log.info(f"wrap ok [{module_name}]: {ok} ok, {fail} fail")







