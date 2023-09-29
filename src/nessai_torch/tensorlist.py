from collections.abc import Sequence
import logging
import math
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class TensorList(Sequence):
    def __init__(
        self,
        *,
        device: torch.DeviceObjType = "cpu",
        size: Tuple[int] = None,
        buffer_size: int = 1000,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = device
        self.buffer_size = buffer_size
        if size is None:
            size = ()
        self._size = size
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.buffer = self._new_buffer()
        self.current_buffer_size = len(self.buffer)
        self._count = 0

    @property
    def data(self) -> torch.Tensor:
        return self.buffer[: self._count]

    _torch_fn_data: torch.Tensor = data
    """Attribute used for torch functions."""

    def append(self, x: torch.Tensor) -> None:
        self.buffer[self._count] = x
        self._count += 1
        if self._count >= self.current_buffer_size:
            self._extend_buffer()

    @property
    def _remaining(self):
        return len(self.buffer) - self._count

    def extend(self, x: torch.Tensor) -> None:
        n = len(x)
        if n > self._remaining:
            extra = n - self._remaining
            n_buffers = math.ceil(extra / self.buffer_size)
            self._extend_buffer(n_buffers)
        self.buffer[self._count : (self._count + n)] = x
        self._count += n
        if self._count >= self.current_buffer_size:
            self._extend_buffer()

    def _new_buffer(self) -> torch.Tensor():
        return torch.empty(
            self.buffer_size,
            *self._size,
            device=self.device,
            dtype=self.dtype,
        )

    def _extend_buffer(self, n: int = 1) -> None:
        logger.debug("Extending buffer")
        self.buffer = torch.cat(
            [self.buffer] + [self._new_buffer() for _ in range(n)], dim=0
        )
        self.current_buffer_size = len(self.buffer)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.buffer[index]

    def __len__(self) -> int:
        return self._count

    def __str__(self) -> str:
        return self.data.__str__()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [
            a._torch_fn_data if hasattr(a, "_torch_fn_data") else a
            for a in args
        ]
        ret = func(*args, **kwargs)
        return ret
