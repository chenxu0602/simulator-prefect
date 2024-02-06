import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

class ProbSparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, topk_ratio=0.25):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.topk_ratio = topk_ratio

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        return x.view(x.size(0), -1, self.n_heads, self.d_k)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_scores = torch.einsum("nqhd,nkhd->nhqk", [Q, K]) / (self.d_k ** 0.5)

        top_val, top_idx = attn_scores.topk(int(self.topk_ratio * attn_scores.shape[-1]), sorted=False, dim=-1)
        attn_scores_topk = torch.zeros_like(attn_scores).fill_(-1e20).scatter_(dim=-1, index=top_idx, src=top_val)

        if mask is not None: attn_scores_topk += (mask * -1e9)

        weights = F.softmax(attn_scores_topk, dim=-1)
        attn = torch.einsum("nhql,nlhd->nqhd", [weights, V]).contiguous().view(query.size(0), -1, self.d_model)

        return self.W_o(attn), weights


class TestProbSparseMultiheadAttention(unittest.TestCase):

    def test_output_shape(self):
        d_model = 128
        n_heads = 8
        batch_size = 32
        seq_length = 50

        attention = ProbSparseMultiheadAttention(d_model, n_heads)
        query = torch.rand(batch_size, seq_length, d_model)
        key = torch.rand(batch_size, seq_length, d_model)
        value = torch.rand(batch_size, seq_length, d_model)

        output, weights = attention(query, key, value)
        self.assertEqual(output.shape, (batch_size, seq_length, d_model))
        self.assertEqual(weights.shape, (batch_size, n_heads, seq_length, seq_length))

    def test_mask_application(self):
        d_model = 64
        n_heads = 4
        batch_size = 16
        seq_length = 30

        attention = ProbSparseMultiheadAttention(d_model, n_heads)
        query = torch.rand(batch_size, seq_length, d_model)
        key = torch.rand(batch_size, seq_length, d_model)
        value = torch.rand(batch_size, seq_length, d_model)
        mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
        # mask = torch.ones(batch_size, 1, seq_length).fill_(-1e9)  # Simulating a mask

        output, weights = attention(query, key, value, mask=mask)
        self.assertFalse(torch.isinf(weights).any(), "Weights should not contain inf values after applying mask")

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()