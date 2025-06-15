import mlx.core as mx

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    t = (mx.rsqrt(query.shape[-1]) if scale is None else scale) * (query @ key.swapaxes(-2, -1))
    if mask is not None:
        t += mask
    return softmax(t, -1) @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        assert hidden_size % num_heads == 0
        assert wq.shape == wk.shape == wv.shape == wo.shape == (hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        assert query.shape == key.shape == value.shape and query.shape[-1] == self.hidden_size
        assert mask is None or mask.shape == (query.shape[-2], key.shape[-2])
        q = linear(query, self.wq)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).swapaxes(-3, -2)
        k = linear(key, self.wk)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).swapaxes(-3, -2)
        v = linear(value, self.wv)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).swapaxes(-3, -2)
        t = scaled_dot_product_attention_simple(q, k, v, mask=mask).swapaxes(-3, -2)
        t = t.reshape(*t.shape[:-2], self.hidden_size)
        return linear(t, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
