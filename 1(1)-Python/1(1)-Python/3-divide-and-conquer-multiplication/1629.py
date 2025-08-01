# lib.py의 Matrix 클래스를 참조하지 않음
import sys


"""
TODO:
- fast_power 구현하기 
"""


def fast_power(base: int, exp: int, mod: int) -> int:
    """
    빠른 거듭제곱 알고리즘 구현
    분할 정복을 이용, 시간복잡도 고민!

    재귀함수로 구현

    Args:
        base (int): 밑(base) 값
        exp (int): 지수(exp) 값
        mod (int): 모듈로 연산을 위한 값    

    Returns:
        int: (base ** exp) % mod의 결과값
    """
    # 구현하세요!
    if exp == 0:
        return 1
    elif exp % 2 == 0:
        half = fast_power(base, exp // 2, mod)
        return (half * half) % mod
    else:
        half = fast_power(base, (exp - 1) // 2, mod)
        return (half * half * base) % mod

def main() -> None:
    A: int
    B: int
    C: int
    A, B, C = map(int, input().split()) # 입력 고정
    
    result: int = fast_power(A, B, C) # 출력 형식
    print(result) 

if __name__ == "__main__":
    main()
