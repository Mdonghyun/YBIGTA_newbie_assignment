import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 업데이트 게이트 z
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        # 리셋 게이트 r
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        # 후보 은닉 상태 h_tilde
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        x: (batch, input_size)
        h: (batch, hidden_size)
        returns: (batch, hidden_size)
        """
        # 1) 업데이트 게이트 z
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        # 2) 리셋 게이트 r
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        # 3) 후보 은닉 상태
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h))
        # 4) 최종 은닉 상태
        h_new = (1 - z) * h_tilde + z * h
        return h_new


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        """
        inputs: (batch, seq_len, input_size)
        returns: outputs tensor of shape (batch, hidden_size),
                    which is the last hidden state after processing the sequence
        """
        batch_size, seq_len, _ = inputs.size()
        # 초기 은닉 상태는 0으로
        h = inputs.new_zeros(batch_size, self.hidden_size)

        # 시퀀스 길이만큼 순환
        for t in range(seq_len):
            # inputs[:, t, :] shape = (batch, input_size)
            h = self.cell(inputs[:, t, :], h)

        # (batch, hidden_size)
        return h
