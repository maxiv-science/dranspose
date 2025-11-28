import numpy as np
from typing import Optional, Union


class IncrementalBuffer:
    """
    Buffer incoming lines (scalars or 1-D rows) that may arrive out-of-order.
    """

    def __init__(
        self,
        initial_rows: int = 16,
        grow_factor: float = 2,
        filler_value: Union[int, float, np.nan] = 0,
    ) -> None:
        self._capacity: int = max(1, int(initial_rows))
        self._dtype: Optional[np.dtype] = None
        self._arr: Optional[np.ndarray] = None
        self._max_filled_index: int = -1
        self._filler_value = filler_value
        assert grow_factor > 1, "The grow factor must be > 1"
        self._user_grow_factor: float = grow_factor

    def _init_buffer(self, first_entry: np.ndarray) -> None:
        self._dtype = first_entry.dtype
        self._arr = np.full(
            (self._capacity, *first_entry.shape), self._filler_value, dtype=self._dtype
        )

    def _ensure_capacity(self, req_size: int) -> None:
        if self._arr is None or self._dtype is None:
            raise RuntimeError("Buffer not initialized")
        if req_size <= self._arr.shape[0]:
            return
        new_shape = list(self._arr.shape)
        while new_shape[0] < req_size:
            new_shape[0] = new_shape[0] * self._user_grow_factor
        new_arr = np.full(new_shape, self._filler_value, dtype=self._dtype)
        new_arr[: self._arr.shape[0]] = self._arr
        self._arr = new_arr

    def add_entry(self, index: int, data: np._typing.ArrayLike) -> None:
        """
        Insert a scalar or array at the specified index (0-based).
        """
        entry = np.array(data)
        if self._arr is None:
            self._init_buffer(entry)
        if entry.shape != self._arr.shape[1:]:
            raise ValueError("Incoming shape does not match buffer shape")

        if entry.dtype != self._dtype:
            entry = entry.astype(self._dtype, copy=False)

        self._ensure_capacity(
            index + 1
        )  # idx is 0 based -> to fit entry #3 the array needs to size 4
        self._arr[index] = entry
        if index > self._max_filled_index:
            self._max_filled_index = index

    def is_empty(self):
        return self._arr is None

    def preview(self, copy: bool = True) -> np.ndarray:
        if self._arr is None:
            return np.zeros((0,))
        curr_view = self._arr[: self._max_filled_index + 1]
        return curr_view.copy() if copy else curr_view

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        out = self.preview(copy=False)
        if dtype is not None:
            return out.astype(dtype, copy=False)
        return out

    def __getitem__(self, key):
        view = self.preview(copy=False)
        return view[key]

    @property
    def shape(self) -> tuple:
        view = self.preview(copy=False)
        return view.shape

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            raise ValueError("Buffer not initialized")
        return self._dtype
