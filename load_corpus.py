from datasets import load_dataset
from typing import List

def load_corpus() -> List[str]:
    """
    Word2Vec 학습용 corpus를 반환합니다.
    - Hugging Face 'poem_sentiment' 데이터셋의 train split 사용
    - verse_text 필드를 소문자로 변환하고 양쪽 공백 제거
    """
    corpus: List[str] = []
    # train split에서 모든 verse_text를 가져옵니다 (총 892개 예제)
    ds = load_dataset("poem_sentiment", split="train")
    for ex in ds:
        verse = ex["verse_text"].strip()
        if verse:
            corpus.append(verse.lower())
    return corpus


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"{len(corpus)} sentences loaded.")
    print("샘플:", corpus[:5])
