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




"""
TODO:
- josephus_problem 구현하기
    # 요세푸스 문제 구현
        # 1. 큐 생성
        # 2. 큐가 빌 때까지 반복
        # 3. 제거 순서 리스트 반환
"""


def josephus_problem(n: int, k: int) -> list[int]:
    """
    요세푸스 문제 해결
    n명 중 k번째마다 제거하는 순서를 반환
    Args:
        n: 사람의 수
        k: 제거할 순서 (1부터 시작)
    Returns:
        제거된 순서의 리스트
    """
    # 구현하세요!
    queue = create_circular_queue(n)
    result = []
    while queue:
        removed = rotate_and_remove(queue, k)
        result.append(removed)
    return result

def solve_josephus() -> None:
    """입, 출력 format"""
    n: int
    k: int
    n, k = map(int, input().split())
    result: list[int] = josephus_problem(n, k)
    
    # 출력 형식: <3, 6, 2, 7, 5, 1, 4>
    print("<" + ", ".join(map(str, result)) + ">")

if __name__ == "__main__":
    solve_josephus()