# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "cxl"] = "nccl"
    """The backend to use for weight transfer. "cxl" reads each TP worker's weights
    directly from a shared canonical weight store (mmap/CXL), bypassing the NCCL
    broadcast and the per-GPU bucket IPC hop."""
