from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = [0] * (4 * size)

    def update(self, idx: int, diff: int) -> None:
        """
        idx에 diff만큼 변경
        """
        self._update(1, 1, self.size, idx, diff)

    def kth(self, k: int) -> int:
        """
        전체에서 누적합이 k가 되는 인덱스를 반환
        """
        return self._kth(1, 1, self.size, k)

    def _update(self, node: int, start: int, end: int, idx: int, diff: int) -> None:
        if idx < start or idx > end:
            return
        self.tree[node] += diff
        if start != end:
            mid = (start + end) // 2
            self._update(node*2,     start, mid,    idx, diff)
            self._update(node*2+1, mid+1,   end,    idx, diff)

    def _kth(self, node: int, start: int, end: int, k: int) -> int:
        if start == end:
            return start
        mid = (start + end) // 2
        left_sum = self.tree[node*2]
        if left_sum >= k:
            return self._kth(node*2, start, mid, k)
        else:
            return self._kth(node*2+1, mid+1, end, k - left_sum)