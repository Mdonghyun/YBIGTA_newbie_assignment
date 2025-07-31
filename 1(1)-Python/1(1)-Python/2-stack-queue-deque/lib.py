from __future__ import annotations
from collections import deque


"""
TODO:
- rotate_and_remove 구현하기 
"""


def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성합니다."""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.
    큐를 k-1번 회전시킨 후 맨 앞 원소를 제거합니다.
    Args:
        queue: 원소가 있는 큐
        k: 제거할 원소의 순서 (1부터 시작)
    Returns:
        제거된 원소
    """
    # 구현하세요!
    queue.rotate(-(k - 1))  # k번째 원소를 맨 앞에 오도록 회전
    return queue.popleft()  # 맨 앞 원소를 제거하고 반환