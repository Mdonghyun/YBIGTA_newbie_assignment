from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    input = sys.stdin.readline
    n = int(input())
    MAX_TASTE = 1_000_000

    seg_tree: SegmentTree = SegmentTree(MAX_TASTE)
    results: list[int] = []

    for _ in range(n):
        data = list(map(int, input().split()))
        if data[0] == 1:
            k = data[1]
            taste = seg_tree.kth(k)
            results.append(taste)
            seg_tree.update(taste, -1)
        else:
            taste, cnt = data[1], data[2]
            seg_tree.update(taste, cnt)

    sys.stdout.write("\n".join(map(str, results)))


if __name__ == "__main__":
    main()