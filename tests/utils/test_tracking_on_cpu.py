# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import sys
import types
from unittest.mock import MagicMock, patch

from verl.utils.tracking import ValidationGenerationsLogger


def test_validation_generations_logger_logs_trackio_traces():
    mock_trackio = MagicMock()
    mock_trackio.context_vars = types.SimpleNamespace(current_run=MagicMock())
    mock_trackio.context_vars.current_run.get.return_value = None
    mock_trackio.Trace.side_effect = lambda messages, metadata=None: {
        "_type": "trackio.trace",
        "messages": messages,
        "metadata": metadata or {},
    }

    with patch.dict(sys.modules, {"trackio": mock_trackio}):
        ValidationGenerationsLogger().log(
            ["trackio"],
            samples=[["question", "answer", 0.5]],
            step=7,
        )

    mock_trackio.Trace.assert_called_once()
    trace_kwargs = mock_trackio.Trace.call_args.kwargs
    assert trace_kwargs["messages"] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert trace_kwargs["metadata"]["source"] == "validation_generations"
    assert trace_kwargs["metadata"]["score"] == 0.5
    mock_trackio.log.assert_called_once()
    assert mock_trackio.log.call_args.kwargs["step"] == 7


class _FakeTable:
    """Minimal wandb.Table stand-in: records columns + accumulates rows (data)."""

    def __init__(self, columns=None, data=None):
        self.columns = columns
        self.data = list(data) if data is not None else []

    def add_data(self, *row):
        self.data.append(list(row))


def _fake_wandb():
    w = MagicMock()
    w.Table.side_effect = lambda columns=None, data=None: _FakeTable(columns=columns, data=data)
    w.run = object()  # truthy so wandb.log is exercised
    return w


def test_wandb_generations_default_table_name_is_val():
    """Default table_name preserves the historical 'val/generations' key (byte-compatible)."""
    w = _fake_wandb()
    logger = ValidationGenerationsLogger()
    # log() dispatches to the wandb path; verify via the public API with a faked module.
    with patch.dict(sys.modules, {"wandb": w}):
        logger.log(["wandb"], samples=[["q", "a", 1.0]], step=3)
    # wandb.log is called as wandb.log({table_name: table}, step=...)
    key = next(iter(w.log.call_args.args[0].keys()))
    assert key == "val/generations"


def test_wandb_generations_separate_tables_do_not_clobber():
    """Student and teacher train tables use distinct keys AND distinct cached tables.

    Logging twice to the SAME name accumulates rows; logging to a DIFFERENT name starts fresh — so the
    two train-generation tables (student vs teacher) never share/overwrite each other's rows.
    """
    w = _fake_wandb()
    logger = ValidationGenerationsLogger()
    with patch.dict(sys.modules, {"wandb": w}):
        # Two student logs (should accumulate to 2 rows) + one teacher log (independent, 1 row).
        logger.log(["wandb"], [["qs1", "as1", 1.0]], step=1, table_name="train/generations_student")
        logger.log(["wandb"], [["qt1", "at1", 0.0]], step=1, table_name="train/generations_teacher")
        logger.log(["wandb"], [["qs2", "as2", 1.0]], step=2, table_name="train/generations_student")

    keys_logged = [c.args[0] and next(iter(c.args[0].keys())) for c in w.log.call_args_list]
    assert keys_logged == [
        "train/generations_student",
        "train/generations_teacher",
        "train/generations_student",
    ]
    # Cached tables are keyed per-name → student has 2 accumulated rows, teacher has 1.
    student_tbl = getattr(logger, "_gen_table__train__generations_student")
    teacher_tbl = getattr(logger, "_gen_table__train__generations_teacher")
    assert len(student_tbl.data) == 2, student_tbl.data
    assert len(teacher_tbl.data) == 1, teacher_tbl.data
    # No cross-contamination: teacher row is not in the student table.
    assert all("qt1" not in row for row in student_tbl.data)
