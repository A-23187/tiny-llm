import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.half_dims = dims // 2
        theta = mx.power(base, mx.arange(0, self.half_dims, dtype=mx.float32) / -self.half_dims)
        theta_mat = mx.outer(mx.arange(seq_len, dtype=mx.float32), theta)
        self.cos_mat = mx.cos(theta_mat)
        self.sin_mat = mx.sin(theta_mat)

    def __call__(self, x: mx.array, offset: list[slice] | slice | None = None) -> mx.array:
        assert x.shape[-1] == self.dims
        assert not isinstance(offset, slice) or 0 <= offset.start and offset.stop <= self.seq_len
        if offset is None:
            offset = slice(0, self.seq_len)
        if not self.traditional:
            raise RuntimeError("Non-traditional form is not supported")
        x = x.reshape(*x.shape[:-1], self.half_dims, 2)
        y = x * self.cos_mat.reshape(self.seq_len, 1, self.half_dims, 1)
        y += mx.stack([-x[:, 1], x[:, 0]], axis=-1) * self.sin_mat.reshape(self.seq_len, 1, self.half_dims, 1)
        return y.reshape(*y.shape[:-2], self.dims)
