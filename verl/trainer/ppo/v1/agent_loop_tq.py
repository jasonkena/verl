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

            # Contrastive teacher (issue #36): route the first n_pos sessions to the POSITIVE
            # teacher prompt (the rollout raw_prompt, already the teacher prompt) and the rest to
            # the NEGATIVE arm (contrastive_neg_source). Each session is stamped with a
            # contrastive_arm tag ("pos"/"neg", for the separate advantage baseline) and a
            # teacher_sign (+1 / −1, for reward negation). Only on TRAIN rollouts; validation always
            # uses the (student) prompt untouched. Off ⇒ every session keeps the original prompt,
            # arm="pos", teacher_sign=+1 (byte-for-byte pre-#36).
            actor_cfg = self.config.actor_rollout_ref.actor
            contrastive = (not trajectory["validate"]) and actor_cfg.get("contrastive_teacher", False)
            n_pos = self._contrastive_n_pos(n, actor_cfg) if contrastive else n
            neg_source = str(actor_cfg.get("contrastive_neg_source", "student"))

            tasks = []
            for i in range(n):
                session_prompt = self._session_prompt(
                    prompt, session_id=i, n_pos=n_pos, contrastive=contrastive, neg_source=neg_source
                )
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
    def _contrastive_n_pos(n: int, actor_cfg) -> int:
        """Number of POSITIVE-teacher sessions in a prompt's n rollouts (issue #36).

        ``round(n · contrastive_pos_ratio)``, asserted to split n evenly into a positive
        and a negative half (0 < n_pos < n unless the ratio is exactly 0 or 1) so both
        teachers get a well-defined, baseline-able group. A ratio of 1.0 (pos-only) or 0.0
        (neg-only) is allowed and skips the assert."""
        ratio = float(actor_cfg.get("contrastive_pos_ratio", 0.5))
        assert 0.0 <= ratio <= 1.0, f"contrastive_pos_ratio must be in [0,1], got {ratio}"
        n_pos = int(round(n * ratio))
        if 0.0 < ratio < 1.0:
            assert abs(n * ratio - n_pos) < 1e-6 and 0 < n_pos < n, (
                f"contrastive_pos_ratio={ratio} does not split rollout.n={n} into whole "
                f"positive/negative halves (got n_pos={n * ratio}). Choose a ratio with "
                f"round(n·ratio) == n·ratio and 0 < n_pos < n."
            )
        return n_pos

    @staticmethod
    def _session_prompt(prompt: dict, *, session_id: int, n_pos: int, contrastive: bool,
                        neg_source: str = "student") -> dict:
        """Per-session prompt for the contrastive split (issue #36).

        Sessions ``[0, n_pos)`` are POSITIVE (keep the rollout ``raw_prompt``, the positive teacher
        prompt): ``contrastive_arm="pos"``, ``teacher_sign=+1``. Sessions ``[n_pos, n)`` are the
        NEGATIVE arm; what they sample and how their reward is treated depends on ``neg_source``:

        - ``"neg_teacher"``: swap ``raw_prompt`` ← the ``neg_teacher_prompt`` passthrough column;
          ``teacher_sign=−1`` (reward negated downstream — the neg teacher maximizes ``E[−R]``).
          This is the original objective; it DIVERGES (unbounded ``KL(neg‖student)``).
        - ``"student"`` (default): swap ``raw_prompt`` ← the ``student_prompt`` passthrough column;
          ``teacher_sign=+1`` (reward NOT negated). The teacher and student forwards are then
          identical, so the local KL is a no-op — the neg arm is just extra on-policy RLOO from the
          student prompt (Jason's fix for the neg-teacher divergence).

        Every session is tagged ``contrastive_arm`` ("pos"/"neg") so the trainer can baseline the
        two arms separately regardless of ``teacher_sign``. Returns a shallow copy; the passthrough
        prompt columns not used for generation are dropped from the spawned kwargs. With
        ``contrastive=False`` this is the identity apart from arm="pos"/teacher_sign=+1."""
        session_prompt = dict(prompt)
        # These passthrough columns are only used to SELECT the neg-arm prompt; drop both from the
        # kwargs the agent loop actually generates from (raw_prompt is what it tokenizes).
        neg_teacher = session_prompt.pop("neg_teacher_prompt", None)
        student = session_prompt.get("student_prompt")  # keep: the trainer still needs it (M3)
        is_neg = contrastive and session_id >= n_pos
        if is_neg and neg_source == "neg_teacher":
            assert neg_teacher is not None, (
                "contrastive_teacher=true, contrastive_neg_source=neg_teacher, but the batch has no "
                "neg_teacher_prompt column — rebuild the parquet with the #36 dataset builder."
            )
            session_prompt["raw_prompt"] = neg_teacher
            session_prompt["contrastive_arm"] = "neg"
            session_prompt["teacher_sign"] = -1.0
        elif is_neg:  # neg_source == "student"
            assert student is not None, (
                "contrastive_teacher=true, contrastive_neg_source=student, but the batch has no "
                "student_prompt column — is data.custom_cls TeacherStudentDataset?"
            )
            session_prompt["raw_prompt"] = student
            session_prompt["contrastive_arm"] = "neg"
            session_prompt["teacher_sign"] = 1.0  # student-arm reward is NOT negated
        else:
            session_prompt["contrastive_arm"] = "pos"
            session_prompt["teacher_sign"] = 1.0
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
