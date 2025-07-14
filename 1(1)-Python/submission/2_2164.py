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
- simulate_card_game 구현하기
    # 카드 게임 시뮬레이션 구현
        # 1. 큐 생성
        # 2. 카드가 1장 남을 때까지 반복
        # 3. 마지막 남은 카드 반환
"""


def simulate_card_game(n: int) -> int:
    """
    카드2 문제의 시뮬레이션
    맨 위 카드를 버리고, 그 다음 카드를 맨 아래로 이동
    카드가 하나 남을 때까지 반복

    Args:
        n: 초기 카드 개수

    Returns:
        마지막에 남은 카드의 번호
    """
    # 구현하세요!
    queue = create_circular_queue(n)
    while len(queue) > 1:
        queue.popleft()
        queue.append(queue.popleft())
    return queue[0]

def solve_card2() -> None:
    """입, 출력 format"""
    n: int = int(input())
    result: int = simulate_card_game(n)
    print(result)

if __name__ == "__main__":
    solve_card2()