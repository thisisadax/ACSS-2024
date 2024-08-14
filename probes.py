from abc import ABC, abstractmethod
import torch
from torch.nn import Module, Linear, Sequential, Tanh, LayerNorm, Dropout, ReLU


class AttentionPooler(Module):
    '''Attention pooling as described in https://arxiv.org/pdf/1905.06316.pdf (page 14, C).
    '''

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.in_projections = Linear(input_dim, hidden_dim)
        self.attention_scorers = Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, return_att_vectors=False):
        '''
        :param hidden_states: the hidden states of the subject model with shape (N, L, D), i.e. batch size,
        sequence length and hidden dimension.
        span's position in the batch, and [a, b) the span interval along the sequence dimension.
        :return: a tensor of pooled span embeddings of dimension `hidden_dim`.
        '''
        return self._pool(hidden_states, return_att_vectors=return_att_vectors)

    def _pool(self, hidden_states, return_att_vectors=False):
        '''
        :param hidden_states: the hidden states of the subject model with shape (N, L, D), i.e. batch size,
        sequence length and hidden dimension.
        :return: a tensor of pooled span embeddings.
        '''

        # apply projections with parameters for span target k
        embed_spans = [self.in_projections(span) for span in hidden_states]
        att_vectors = [self.attention_scorers(span).softmax(0)
                       for span in embed_spans]

        pooled_spans = [att_vec.T @ embed_span
                        for att_vec, embed_span in zip(embed_spans, att_vectors)]
        
        if return_att_vectors:
            return torch.stack(pooled_spans).squeeze(-1), torch.stack(att_vectors).squeeze(-1)
        else:
            return torch.stack(pooled_spans).squeeze(-1)


class MLP(Module, ABC):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2, single_span=True):
        super().__init__()

        if not single_span:
            input_dim *= 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout_prob

        self.classifier = self._build_mlp()

    def forward(self, inputs):
        return self.classifier(inputs)

    @abstractmethod
    def _build_mlp(self):
        '''build the mlp classifier

        :rtype: Module
        :return: the mlp module.
        '''


class CampbellMLP(MLP):

    def _build_mlp(self):
        return Sequential(
            LayerNorm(self.input_dim),
            Dropout(self.dropout),
            Linear(self.input_dim, self.output_dim)
        )

class TenneyMLP(MLP):
    '''The 2 layer MLP used by Tenney et al. in https://arxiv.org/abs/1905.06316.

    https://github.com/nyu-mll/jiant/blob/ead63af002e0f755c6418478ec3cabb4062a601e/jiant/modules/simple_modules.py#L49
    '''

    def _build_mlp(self):
        return Sequential(
            Linear(self.input_dim, self.hidden_dim),
            Tanh(),
            LayerNorm(self.hidden_dim),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.output_dim)
        )


class HewittMLP(MLP):
    '''MLP-2 from Hewitt and Liang: https://arxiv.org/abs/1909.03368.
    '''

    def _build_mlp(self):
        return Sequential(
            Linear(self.input_dim, self.hidden_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.output_dim)
        )