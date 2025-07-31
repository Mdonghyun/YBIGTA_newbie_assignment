import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!
from typing import List
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss



class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        """
        1) corpus: raw 문장 리스트
        2) tokenizer: HuggingFace 토크나이저
        """
        # 1) Tokenize (special token 제외)
        tokenized = tokenizer(
            corpus,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )["input_ids"]  # List[List[int]]

        # 2) 필터링: 길이 > 1, 패딩/unk 토큰(id)가 있으면 제외할 수도 있음
        sequences: list[list[int]] = []
        pad_id = tokenizer.pad_token_id or -1
        for seq in tokenized:
            # special token ID가 없도록
            clean = [tok for tok in seq if tok != pad_id]
            if len(clean) > 1:
                sequences.append(clean)

        optimizer = Adam(self.parameters(), lr=lr)
        
        if self.method == "skipgram":
            criterion = nn.BCEWithLogitsLoss()
            self._train_skipgram(
                sequences, optimizer, criterion, num_epochs
            )
        else:  # "cbow"
            criterion = nn.CrossEntropyLoss()
            self._train_cbow(
                sequences, optimizer, criterion, num_epochs
            )

    def _train_cbow(
        self,
        sequences: List[List[int]],
        optimizer: Optimizer,
        criterion: CrossEntropyLoss,
        num_epochs: int
    ) -> None:
        """
        CBOW 학습 루프
        - sequences: 토큰 ID로 이루어진 문장 리스트
        - optimizer: Adam 등
        - criterion: CrossEntropyLoss
        - num_epochs: 전체 에폭 수
        """
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0

            # 1) 한 문장씩
            for seq in sequences:
                # 2) 문장 내에서 각 단어를 중심 단어로 삼아
                for idx, center_id in enumerate(seq):
                    # 3) 주변 단어 ID 수집 (window_size 만큼 좌우)
                    context_ids = []
                    for w in range(1, self.window_size + 1):
                        if idx - w >= 0:
                            context_ids.append(seq[idx - w])
                        if idx + w < len(seq):
                            context_ids.append(seq[idx + w])
                    if not context_ids:
                        continue  # 주변이 하나도 없으면 skip

                    # 4) 텐서로 변환
                    #    contexts: (1, num_contexts), center: (1,)
                    contexts = torch.LongTensor(context_ids).unsqueeze(0)
                    center = torch.LongTensor([center_id])

                    # 5) 순전파 + 손실 계산
                    #    a) 주변 단어들 임베딩 → (1, num_contexts, d_model)
                    emb = self.embeddings(contexts)
                    #    b) num_contexts 축 평균 → (1, d_model)
                    v = emb.mean(dim=1)
                    #    c) 선형층 통과 → (1, vocab_size)
                    logits = self.weight(v)
                    loss = criterion(logits, center)

                    # 6) 역전파
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            avg_loss = total_loss / sum(len(seq) for seq in sequences)
            print(f"[CBOW] Epoch {epoch}/{num_epochs} — avg loss: {avg_loss:.4f}")

    def _train_skipgram(
        self,
        sequences: List[List[int]],
        optimizer: Optimizer,
        criterion: BCEWithLogitsLoss,
        num_epochs: int
    ) -> None:
        """
        Skip gram with negative sampling
        - sequences: List of token ID sequences (각 문장)
        - optimizer: Adam 등 optimizer 인스턴스
        - criterion: BCEWithLogitsLoss
        - num_epochs: 전체 학습 epoch 수
        """
        neg_k = 5
        vocab_size = self.embeddings.num_embeddings

        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_pairs = 0

            for seq in sequences:
                for idx, center_id in enumerate(seq):
                    for w in range(1, self.window_size + 1):
                        for pos in (idx - w, idx + w):
                            if pos < 0 or pos >= len(seq):
                                continue

                            context_id = seq[pos]

                            # positive + negative 샘플
                            center_tensor = torch.LongTensor([center_id]).to(self.embeddings.weight.device)
                            pos_tensor    = torch.LongTensor([context_id]).to(self.embeddings.weight.device)
                            neg_tensors   = torch.randint(0, vocab_size, (neg_k,), device=center_tensor.device)

                            sample_ids = torch.cat([pos_tensor, neg_tensors], dim=0)      # (1+neg_k,)
                            labels     = torch.tensor([1] + [0]*neg_k, dtype=torch.float, device=sample_ids.device)

                            # 임베딩 조회
                            h = self.embeddings(center_tensor)      # (1, d_model)
                            u = self.embeddings(sample_ids)         # (1+neg_k, d_model)

                            # score 계산
                            scores = u @ h.squeeze(0)               # (1+neg_k,)

                            # 손실 계산 (BCEWithLogitsLoss)
                            loss = criterion(scores, labels)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()
                            total_pairs += 1

            avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
            print(f"[Skip‑gram NS] Epoch {epoch}/{num_epochs} — avg loss: {avg_loss:.4f}")

    # 구현하세요!
    pass