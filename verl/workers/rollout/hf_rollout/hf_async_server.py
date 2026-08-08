# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HuggingFace `.generate()` rollout server on the V1 AgentLoop/TransferQueue path.

Motivation (issue #54, maze). For a *tiny* model (~4M params) vLLM's per-step
engine/scheduler/KV-paging overhead dwarfs the actual matmul, so HF batched
`.generate()` is ~3.6x faster end-to-end (benchmark: 5s vs 18s/step at the real
train shape n=128 x 256 prompts). But our verl fork deleted the legacy synchronous
fsdp_workers rollout path and migrated fully to the V1 AgentLoop, where rollout is
a Ray server actor reached per-request via
`server.generate.remote(request_id, prompt_ids, sampling_params) -> TokenOutput`.
`rollout.name=hf` was therefore unregistered on this path.

This module registers an HF backend that plugs into that path with NO fork-wide
changes — only config (`rollout.name=hf`) + two registry entries (see
`hf_rollout/__init__` and `base._ROLLOUT_REGISTRY`). The design reuses every
existing V1 seam:

- Actor-side client: the stock vLLM `ServerAdapter` is backend-agnostic — it sends
  weights over a generic ZMQ/CUDA-IPC `BucketedWeightSender` and resolves the
  server actor by `f"{rollout.name}_server_{r}_{n}"`. With `rollout.name=hf` it
  resolves `hf_server_*` and reuses the "naive" weight-sync path verbatim. So we
  register `("hf","async") -> ServerAdapter` and add nothing on the client side.
- Weight sync: `HFHttpServer.update_weights_from_ipc` mirrors the vLLM colocate
  worker extension — `BucketedWeightReceiver` -> `load_state_dict` into the HF
  model. The weights arrive as HF `state_dict()` keys (see fsdp
  `get_per_tensor_param` -> `convert_weight_keys`), directly loadable.
- Slot injection (coupled_maxrl `<IDk>` prefix) is backend-independent: it happens
  in `agent_loop_tq._coupled_session_prompt` BEFORE tokenization, at the AgentLoop
  worker level, so it is preserved for free by any backend honoring the per-request
  `generate` contract.

Speed recovery: the AgentLoop fires one `generate` call per sample (up to
train_batch_size*n concurrent tasks funneled through the load balancer to this one
server on the maze single-GPU setup). Naive per-request `.generate()` would forfeit
HF's batched-generation win, so this server runs an internal async MICRO-BATCHER:
concurrent requests with identical sampling params are coalesced into a single
left-padded `model.generate(...)` wave, then demultiplexed back to each awaiting
caller. This reproduces the batched shape the benchmark measured.
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import sys
from typing import Any, Callable, Optional

import ray
import torch
from ray.actor import ActorHandle

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_resource_name,
    get_torch_device,
    get_visible_devices_keyword,
)
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.plugin.platform import get_platform

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# How long the micro-batcher waits to accumulate concurrent requests before
# dispatching a generate wave. The AgentLoop submits all samples up front as
# asyncio tasks, so the very first coalescing window already sees the full batch;
# this small linger only smooths late arrivals. Overridable via env for tuning.
_HF_BATCH_LINGER_S = float(os.getenv("VERL_HF_BATCH_LINGER_S", "0.005"))
# Cap the per-wave sample count so a huge step (e.g. 32768 seqs) is chunked rather
# than attempted in one generate call. Mirrors maxrl's sample-level micro batch.
_HF_MAX_MICRO_BATCH = int(os.getenv("VERL_HF_MAX_MICRO_BATCH", "4096"))
# Watchdog: max wall-clock for a single model.generate wave. A tiny maze model
# generating <=180 tokens for a <=4096-seq wave completes in seconds, so a wave
# still running after this long is wedged (CUDA sync / transformers-internal). We
# turn that silent hang into a LOUD, fast failure with a full stack dump instead of
# letting the sole batcher block forever. 0 disables. Generous default (real waves
# finish in seconds; this only fires on a genuine stall).
_HF_GENERATE_TIMEOUT_S = float(os.getenv("VERL_HF_GENERATE_TIMEOUT_S", "600"))


def _dump_all_stacks(reason: str) -> None:
    """Dump every thread's Python stack to stderr — makes an invisible wedge visible.

    Called on any batcher death or generate-wave timeout so the log carries the
    exact frame the process is stuck in (event loop, to_thread worker, or a CUDA
    sync inside model.generate) instead of a silent freeze. Best-effort: never
    raises out of the diagnostics path.
    """
    try:
        sys.stderr.write(f"\n===== HF rollout stack dump: {reason} =====\n")
        sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 — diagnostics must never mask the real failure
        logger.exception("HF rollout: stack dump itself failed")


class _PendingRequest:
    """One awaiting generate() call, parked until the batcher dispatches its wave."""

    __slots__ = ("prompt_ids", "sampling_params", "max_tokens", "want_logprobs", "future")

    def __init__(
        self,
        prompt_ids: list[int],
        sampling_params: dict,
        max_tokens: int,
        want_logprobs: bool,
        future: asyncio.Future,
    ):
        self.prompt_ids = prompt_ids
        self.sampling_params = sampling_params
        self.max_tokens = max_tokens
        self.want_logprobs = want_logprobs
        self.future = future


class HFHttpServer:
    """HuggingFace generate() rollout server actor (one per node/replica).

    Mirrors the load-bearing surface of `vLLMHttpServer` that the V1 rollout stack
    calls, but backed by a plain `AutoModelForCausalLM` instead of a vLLM engine.
    Because there is no separate inference-engine subprocess, the weight-transfer
    receiver runs in THIS process and `collective_rpc` dispatches methods on `self`.
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices
        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)
        # Forward the Ray job id so the colocated weight-transfer IPC socket path is
        # unique per Ray job (matches ServerAdapter's sender side); see vLLM server.
        os.environ["VERL_RAY_JOB_ID"] = ray.get_runtime_context().get_job_id()

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        # model weights version, set by ServerAdapter after each weight update.
        self.global_steps = None

        # Resolve max_model_len like vLLMHttpServer._validate_configs.
        from verl.workers.rollout.utils import get_max_position_embeddings

        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings

        # HTTP server address is unused on the token-in-token-out path (the AgentLoop
        # calls server.generate.remote directly), but the load balancer keys on a
        # unique address string, so synthesize one.
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address, with_alive_sock=True)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None

        self.device = torch.device(f"{get_device_name()}:{get_device_id()}")
        self._load_model()

        # Micro-batcher state (initialized lazily on the server's event loop).
        self._queue: Optional[asyncio.Queue] = None
        self._batcher_task: Optional[asyncio.Task] = None
        self._inflight = 0
        # Every request currently parked on a future, so a batcher death can fail
        # them ALL (not just the one guarded group) — a hung caller becomes a loud
        # per-request exception the AgentLoop turns into a `failure` tag + traceback.
        self._pending: set[_PendingRequest] = set()
        # Sticky death: once the batcher dies, new generate() calls fail-fast with
        # this instead of parking on a future that will never resolve.
        self._batcher_dead: Optional[BaseException] = None
        # Dump C-level stacks too (CUDA/transformers stalls the event loop can't see).
        try:
            faulthandler.enable()
        except Exception:  # noqa: BLE001 — some sandboxes lack a real stderr fd
            pass

        logger.info(
            f"HFHttpServer replica_rank={replica_rank} node_rank={node_rank} "
            f"{get_visible_devices_keyword()}={cuda_visible_devices} device={self.device} "
            f"max_model_len={self.config.max_model_len}"
        )

    # ------------------------------------------------------------------ model

    def _load_model(self):
        from transformers import AutoModelForCausalLM

        tokenizer = self.model_config.tokenizer
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        self.eos_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.local_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.model_config.trust_remote_code,
            attn_implementation=getattr(self.model_config, "attn_implementation", None) or "sdpa",
        )
        model.to(self.device)
        model.eval()
        self.model = model

    # -------------------------------------------------------------- addresses

    def get_master_address(self):
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        # Synthesize a stable, unique port so the load balancer's server_id is unique
        # across replicas/nodes. No socket is actually bound on the TiT path.
        if self._server_port is None:
            self._server_port = 10000 + self.replica_rank * 100 + self.node_rank
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        """No HTTP server to launch for the HF backend; just finalize the address."""
        if self.node_rank != 0:
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port
        self.get_server_address()

    # ------------------------------------------------------------ weight sync

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        """Dispatch an RPC on this process (no separate inference workers for HF).

        ServerAdapter.update_weights fires collective_rpc("update_weights_from_ipc").
        """
        kwargs = kwargs or {}
        if isinstance(method, str):
            fn = getattr(self, method)
        else:
            fn = method
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done: bool = False, use_shm: bool = False):
        """Receive bucketed weights over ZMQ/CUDA-IPC and load into the HF model.

        Mirrors vLLMColocateWorkerExtension.update_weights_from_ipc: build a
        BucketedWeightReceiver on the same socket path the sender (ServerAdapter)
        binds, and load each bucket into the model as it arrives. Weights arrive as
        HF state_dict() keys (fsdp get_per_tensor_param -> convert_weight_keys), so a
        plain load_state_dict(strict=False) per bucket is exactly right.
        """
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if peft_config is not None:
            raise NotImplementedError("HF rollout backend does not support LoRA weight sync yet.")

        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )

        def on_bucket_received(weights: list[tuple[str, torch.Tensor]]):
            # Clone out of the receiver's reused IPC bucket buffer before load — the
            # buffer is overwritten by the next bucket, and load_state_dict copies
            # in-place into the (persistent) model params so a view is unsafe to keep.
            sd = {name: tensor for name, tensor in weights}
            missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=False)
            if unexpected:
                logger.warning(f"HF weight sync: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

        receiver.receive_weights(on_bucket_received=on_bucket_received)
        get_torch_device().synchronize()

    def _get_zmq_handle(self) -> str:
        """Socket path matching ServerAdapter's sender side (Ray job id + ranks)."""
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        # HF backend is single-GPU per server (tp=dp=pp=1), so local rank is 0.
        local_rank = 0
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock"

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps

    # ---------------------------------------------------------------- sleep/wake

    # There is no vLLM engine to sleep/wake; the HF model simply stays resident.
    # These are no-ops so the trainer's resume/release/clear_kv_cache calls succeed.
    async def wake_up(self, tags: list[str] | None = None):
        return

    async def sleep(self):
        return

    async def clear_kv_cache(self):
        return

    async def release_kv_cache(self):
        return

    async def resume_kv_cache(self):
        return

    async def wait_for_requests_to_drain(self):
        # Yield until no generate wave is in flight. Surface a dead batcher instead
        # of "succeeding" while parked callers hang forever (which reads as a clean
        # drain but is really a deadlock).
        while self._inflight > 0:
            if self._batcher_dead is not None:
                raise RuntimeError(
                    "HF rollout batcher died while draining"
                ) from self._batcher_dead
            await asyncio.sleep(0.01)

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        return

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        return {"aborted": False, "request_id": request_id, "error": "HF backend does not support abort"}

    async def start_profile(self, **kwargs):
        return

    async def stop_profile(self):
        return

    # --------------------------------------------------------------- generate

    def _resolve_max_tokens(self, prompt_ids: list[int], sampling_params: dict) -> int:
        max_possible = self.config.max_model_len - len(prompt_ids)
        if max_possible < 1:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) leaves no room to generate within max_model_len "
                f"({self.config.max_model_len})."
            )
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = min(
                self.config.response_length,
                self.config.prompt_length + self.config.response_length - len(prompt_ids),
            )
        return max(1, min(max_tokens, max_possible))

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        """Token-in-token-out generation, coalesced into batched waves internally."""
        if image_data or video_data or audio_data:
            raise NotImplementedError("HF rollout backend is text-only.")

        sampling_params = dict(sampling_params)
        # verl's agent loop passes `logprobs` truthy when rollout.calculate_log_probs
        # is set (the trainer then reads `rollout_log_probs` for the on-policy debug
        # metric / bypass mode). vLLM sets it to 0 to mean "top-0 = the sampled token
        # only"; any non-None value here means the caller wants per-token logprobs.
        want_logprobs = sampling_params.pop("logprobs", None) is not None
        max_tokens = self._resolve_max_tokens(prompt_ids, sampling_params)

        # Fail-fast if the batcher has already died — never park on a future that
        # can never resolve (that is the silent hang we are eliminating).
        if self._batcher_dead is not None:
            raise RuntimeError(
                "HF rollout batcher is dead; refusing new generate()"
            ) from self._batcher_dead

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._batcher_task = asyncio.create_task(self._batcher_loop())
        req = _PendingRequest(prompt_ids, sampling_params, max_tokens, want_logprobs, future)
        self._pending.add(req)
        await self._queue.put(req)
        try:
            token_ids, log_probs = await future
        finally:
            self._pending.discard(req)

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=None,
            stop_reason="completed",
            num_preempted=None,
            extra_fields={"global_steps": self.global_steps},
        )

    async def _batcher_loop(self):
        """Coalesce concurrent generate() requests into batched model.generate waves.

        Requests are grouped by a sampling-params key (temperature/top_p/top_k/
        max_tokens) so a single generate() call is valid for the whole group.

        This is the SOLE task every generate() future depends on, so it must never
        die silently: the whole body is guarded so any escape (a raise in queue.get/
        partitioning, or a wave that fails past its own guard) is logged with a stack
        dump, marks the batcher dead, and fails EVERY parked future — turning what was
        a 75-min invisible wedge into a loud, immediate per-request crash.
        """
        assert self._queue is not None
        try:
            await self._batcher_loop_body()
        except BaseException as e:  # noqa: BLE001 — the one task all futures depend on
            logger.exception(f"HF rollout batcher_loop died: {e!r}")
            _dump_all_stacks(f"batcher_loop died: {e!r}")
            self._batcher_dead = e
            self._fail_all_pending(e)
            raise

    def _fail_all_pending(self, exc: BaseException) -> None:
        """Propagate `exc` to every parked future so no caller hangs forever."""
        wrapped = RuntimeError(f"HF rollout batcher died: {exc!r}")
        wrapped.__cause__ = exc
        for req in list(self._pending):
            if not req.future.done():
                req.future.set_exception(wrapped)

    async def _batcher_loop_body(self):
        while True:
            first: _PendingRequest = await self._queue.get()
            batch = [first]
            # Brief linger to accumulate the burst the AgentLoop submitted at once.
            if _HF_BATCH_LINGER_S > 0:
                await asyncio.sleep(_HF_BATCH_LINGER_S)
            while not self._queue.empty() and len(batch) < _HF_MAX_MICRO_BATCH:
                batch.append(self._queue.get_nowait())

            # Partition the wave by compatible sampling params.
            groups: dict[tuple, list[_PendingRequest]] = {}
            for req in batch:
                key = self._sampling_key(req)
                groups.setdefault(key, []).append(req)

            for reqs in groups.values():
                try:
                    await self._run_generate_group(reqs)
                except Exception as e:  # noqa: BLE001 — surface to every awaiting caller
                    logger.exception(f"HF generate group failed: {e}")
                    for req in reqs:
                        if not req.future.done():
                            req.future.set_exception(e)

    @staticmethod
    def _sampling_key(req: _PendingRequest) -> tuple:
        sp = req.sampling_params
        temperature = float(sp.get("temperature", 1.0))
        greedy = temperature == 0.0
        return (
            greedy,
            round(temperature, 6),
            round(float(sp.get("top_p", 1.0)), 6),
            int(sp.get("top_k", -1) or -1),
            round(float(sp.get("repetition_penalty", 1.0)), 6),
            req.max_tokens,
            req.want_logprobs,
        )

    @torch.no_grad()
    async def _run_generate_group(self, reqs: list[_PendingRequest]):
        """Left-pad a group of prompts, run one model.generate, demux results."""
        from transformers import GenerationConfig

        sp = reqs[0].sampling_params
        max_tokens = reqs[0].max_tokens
        want_logprobs = reqs[0].want_logprobs  # uniform within a group (part of the key)
        temperature = float(sp.get("temperature", 1.0))
        greedy = temperature == 0.0

        pad_id = self.pad_token_id
        max_len = max(len(r.prompt_ids) for r in reqs)
        input_ids = torch.full((len(reqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(reqs), max_len), dtype=torch.long)
        for i, r in enumerate(reqs):
            L = len(r.prompt_ids)
            input_ids[i, max_len - L :] = torch.tensor(r.prompt_ids, dtype=torch.long)
            attn[i, max_len - L :] = 1
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)

        if greedy:
            gen_kwargs = dict(do_sample=False, num_beams=1)
        else:
            gen_kwargs = dict(
                do_sample=True,
                num_beams=1,
                temperature=temperature,
                top_p=float(sp.get("top_p", 1.0)),
                top_k=max(0, int(sp.get("top_k", 0) or 0)),  # HF: 0 disables; vLLM uses -1
            )
        gen_kwargs["repetition_penalty"] = float(sp.get("repetition_penalty", 1.0))
        generation_config = GenerationConfig(**gen_kwargs)

        # Run the (blocking) generate off the event loop so concurrent Ray calls and
        # the batcher stay responsive.
        def _do_generate():
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                return self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_tokens,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=pad_id,
                    generation_config=generation_config,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=want_logprobs,
                )

        self._inflight += 1
        try:
            gen_coro = asyncio.to_thread(_do_generate)
            if _HF_GENERATE_TIMEOUT_S > 0:
                try:
                    output = await asyncio.wait_for(gen_coro, timeout=_HF_GENERATE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    # A wave that never returns is the wedge we are hunting. Make it
                    # LOUD: dump every stack (the to_thread worker's frame shows the
                    # exact spot inside model.generate / CUDA) and fail the group. The
                    # orphaned worker thread cannot be cancelled, but the batcher no
                    # longer blocks on it and the step crashes with a real traceback.
                    _dump_all_stacks(
                        f"model.generate wave of {len(reqs)} seqs exceeded "
                        f"{_HF_GENERATE_TIMEOUT_S}s (max_tokens={max_tokens})"
                    )
                    raise TimeoutError(
                        f"HF model.generate wave ({len(reqs)} seqs, max_tokens={max_tokens}) "
                        f"exceeded {_HF_GENERATE_TIMEOUT_S}s — likely a CUDA/transformers stall"
                    )
            else:
                output = await gen_coro
        finally:
            self._inflight -= 1

        seq = output.sequences  # (B, max_len + generated)
        responses = seq[:, max_len:].tolist()

        # Per-token log-probs of the SAMPLED tokens. compute_transition_scores maps
        # the generation `scores` (already temperature/top_p/top_k-warped logits ->
        # log-softmax when normalize_logits=True) to the log-prob of each chosen
        # token. This matches vLLM's rollout_log_probs semantics (log-prob under the
        # sampling distribution) and the actor's temperature-scaled recomputation.
        transition = None
        if want_logprobs:
            transition = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            ).tolist()  # (B, generated)

        # Truncate each response at the first EOS (inclusive) to mirror vLLM's
        # token-in-token-out semantics (the AgentLoop builds its own response_mask).
        for i, r in enumerate(reqs):
            toks = responses[i]
            cut = len(toks)
            if self.eos_token_id is not None and self.eos_token_id in toks:
                cut = toks.index(self.eos_token_id) + 1
                toks = toks[:cut]
            lps = transition[i][:cut] if transition is not None else None
            if not r.future.done():
                r.future.set_result((toks, lps))


class HFReplica(RolloutReplica):
    """RolloutReplica backed by HFHttpServer actors (one per node).

    Mirrors vLLMReplica.launch_servers: query each fused worker for its
    (node_id, CUDA_VISIBLE_DEVICES), then create one HFHttpServer per node with
    NodeAffinity + the node's visible devices, named `hf_server_{r}_{n}` so
    ServerAdapter (the reused vLLM client) can resolve it by actor name.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(HFHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "hf_"

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            prefix = self._get_server_name_prefix()
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}{self.name_suffix}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}{self.name_suffix}"
            else:
                name = f"{prefix}server_{self.replica_rank}_{node_rank}{self.name_suffix}"
            env_vars = {
                **{var: "1" for var in get_platform().ray_noset_envvars()},
                **get_platform().rollout_env_vars(),
            }

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
