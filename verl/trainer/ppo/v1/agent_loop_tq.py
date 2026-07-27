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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# AGRO teacher/student regression (issue #43): model_type selecting the AGRO engine+loss, and the
# passthrough column carrying the unprivileged student prompt (mirrors the contextdistillation
# TeacherStudentDataset STUDENT_PROMPT_KEY). Kept as literals here so this verl file has no import
# dependency on the contextdistillation package (which is injected via model.external_lib).
AGRO_MODEL_TYPE = "agro_teacher_student_language_model"
# Off-policy GRPO (issue #48) reuses the SAME teacher/student mixture rollouts as AGRO (student_rollout_ratio
# split + rollout_source stamp); it just selects a different engine/loss downstream. So the mixture gate
# fires for both model_types.
OFFPOLICY_GRPO_MODEL_TYPE = "offpolicy_grpo_language_model"
_MIXTURE_MODEL_TYPES = (AGRO_MODEL_TYPE, OFFPOLICY_GRPO_MODEL_TYPE)
AGRO_STUDENT_PROMPT_KEY = "student_prompt"


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            # AGRO teacher/student mixture rollouts (issue #43): route the first n_student sessions to
            # the UNPRIVILEGED student prompt π_θ(·|x) and the rest to the (privileged) teacher prompt
            # π_θ(·|x,z) — the mixture sampler μ = sg(½π_θ(·|x) + ½π_θ(·|x,z)). Each session is stamped a
            # rollout_source (1.0 student / 0.0 teacher) that rides through into the stored trajectory
            # for diagnostics; the AGRO loss uses teacher/student/ref logprobs for EVERY rollout
            # regardless of source, so the source does NOT change the loss. Only on TRAIN rollouts;
            # validation always uses the (student) prompt untouched. Off (non-AGRO model_type) ⇒
            # _agro_session_prompt returns the prompt untouched (byte-for-byte pre-#43).
            actor_cfg = self.config.actor_rollout_ref.actor
            model_type = self.config.actor_rollout_ref.model.get("model_type", "language_model")
            agro = (not trajectory["validate"]) and model_type in _MIXTURE_MODEL_TYPES
            n_student = self._agro_n_student(n, actor_cfg) if agro else 0

            tasks = []
            for i in range(n):
                session_prompt = self._agro_session_prompt(prompt, session_id=i, n_student=n_student, agro=agro)
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **session_prompt
                    )
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    @staticmethod
    def _agro_n_student(n: int, actor_cfg) -> int:
        """Number of STUDENT-prompt sessions in a prompt's n rollouts (AGRO, issue #43).

        ``round(n · student_rollout_ratio)``, asserted to split n into whole student/teacher halves
        for an interior ratio (0 < ratio < 1). Endpoints are allowed and skip the assert: ratio 0.0
        ⇒ 0 student (all-teacher), 1.0 ⇒ n student (all-student)."""
        ratio = float(actor_cfg.get("student_rollout_ratio", 0.5))
        assert 0.0 <= ratio <= 1.0, f"student_rollout_ratio must be in [0,1], got {ratio}"
        n_student = int(round(n * ratio))
        if 0.0 < ratio < 1.0:
            assert abs(n * ratio - n_student) < 1e-6 and 0 < n_student < n, (
                f"student_rollout_ratio={ratio} does not split rollout.n={n} into whole "
                f"student/teacher halves (got n_student={n * ratio}). Choose a ratio with "
                f"round(n·ratio) == n·ratio and 0 < n_student < n."
            )
        return n_student

    @staticmethod
    def _agro_session_prompt(prompt: dict, *, session_id: int, n_student: int, agro: bool) -> dict:
        """Per-session prompt for the AGRO teacher/student mixture split (issue #43).

        Sessions ``[0, n_student)`` are STUDENT (swap ``raw_prompt`` for the passthrough
        ``student_prompt`` column, the unprivileged prompt); sessions ``[n_student, n)`` are TEACHER
        (keep the rollout ``raw_prompt``, which TeacherStudentDataset forced to the privileged teacher
        prompt). Every AGRO session gets a ``rollout_source`` (1.0 student / 0.0 teacher) stored for
        diagnostics.

        CRITICAL: the ``student_prompt`` passthrough column is kept in the spawned kwargs (NOT popped)
        so it commits to TransferQueue — ``_add_agro_context`` reads it back to build the student
        context sequence. It's the same per-sample dataset column on every session (both student and
        teacher rollouts carry it), so the AGRO advantage stage can rebuild BOTH contexts regardless of
        which prompt actually generated the response. ``run()`` tolerates the extra kwarg (the base M4
        path carries ``student_prompt`` through identically).

        With ``agro=False`` this is the IDENTITY (returns ``prompt`` untouched — no copy, no
        ``rollout_source`` stamp), so non-AGRO runs are byte-for-byte pre-#43."""
        if not agro:
            return prompt
        session_prompt = dict(prompt)
        if session_id < n_student:
            student = prompt.get(AGRO_STUDENT_PROMPT_KEY)
            assert student is not None, (
                "student_rollout_ratio implies student rollouts but the batch has no student_prompt "
                "column — is data.custom_cls pointing at TeacherStudentDataset?"
            )
            session_prompt["raw_prompt"] = student
            session_prompt["rollout_source"] = 1.0
        else:
            session_prompt["rollout_source"] = 0.0
        return session_prompt

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
