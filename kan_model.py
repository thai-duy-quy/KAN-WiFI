"""
kan_model.py - Ten Kolmogorov-Arnold Network (KAN) variants for Wi-Fi CSI classification.

All models share one interface (same as FC_KAN-main/run.py):

    model = SomeKAN(layers_hidden=[n_input, n_hidden, ..., n_output], **kwargs)
    logits = model(x)          # x: (batch, n_input)  -> (batch, n_output)

or through the factory:

    model = get_model("chebykan", [n_input, 64, n_output], degree=4)

Implemented variants (basis function used on every edge):
    1. EfficientKAN  - B-splines + SiLU residual            (Blealtan, efficient-kan)
    2. FastKAN       - Gaussian RBF                          (Li, 2024)
    3. FasterKAN     - Reflectional switch (sech^2) RBF      (Delis, 2024)
    4. BSRBF_KAN     - B-splines + Gaussian RBF combined     (Ta, 2024)
    5. ChebyKAN      - Chebyshev polynomials                 (SynodicMonth, 2024)
    6. JacobiKAN     - Jacobi polynomials P^(a,b)            (SpaceLearner, 2024)
    7. FourierKAN    - Fourier series (sin/cos)              (GistNoesis, 2024)
    8. WavKAN        - Mexican-hat wavelets                  (Bozorgasl & Chen, 2024)
    9. ReLUKAN       - Squared-ReLU bump functions           (Qiu et al., 2024)
   10. RationalKAN   - Safe Pade rational functions          (Aghaei, 2024 / KAT)

An MLP baseline is included for comparison (not counted among the ten).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Generic network wrapper
# ---------------------------------------------------------------------------
class _KANNet(nn.Module):
    """Stacks KAN layers built by `layer_fn(in_features, out_features)`.

    If `norm_between` is True a LayerNorm is inserted between layers, which keeps
    inputs in a well-behaved range for bounded bases (polynomials, Fourier, ...).
    """

    def __init__(self, layers_hidden, layer_fn, norm_between=False):
        super().__init__()
        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden needs at least [n_input, n_output]")
        self.layers_hidden = list(layers_hidden)
        blocks = []
        n_layers = len(layers_hidden) - 1
        for i, (d_in, d_out) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            blocks.append(layer_fn(d_in, d_out))
            if norm_between and i < n_layers - 1:
                blocks.append(nn.LayerNorm(d_out))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _make_grid(in_features, grid_size, spline_order, grid_range):
    """Extended uniform knot vector, shape (in_features, grid_size + 2*spline_order + 1)."""
    h = (grid_range[1] - grid_range[0]) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
    return grid.expand(in_features, -1).contiguous()


def _b_splines(x, grid, spline_order):
    """Cox-de Boor recursion. x: (B, in) -> bases: (B, in, grid_size + spline_order)."""
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[..., :-1]
            + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[..., 1:]
        )
    return bases.contiguous()


# ---------------------------------------------------------------------------
# 1. EfficientKAN (B-spline)
# ---------------------------------------------------------------------------
class EfficientKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        self.register_buffer("grid", _make_grid(in_features, grid_size, spline_order, grid_range))

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.base_activation = base_activation()

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        nn.init.normal_(self.spline_weight, std=scale_noise / grid_size)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * scale_spline)

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        base = F.linear(self.base_activation(x), self.base_weight)
        spline = F.linear(
            _b_splines(x, self.grid, self.spline_order).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base + spline

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1 = self.spline_weight.abs().mean(-1)
        act = l1.sum()
        p = l1 / (act + 1e-8)
        ent = -torch.sum(p * torch.log(p + 1e-8))
        return regularize_activation * act + regularize_entropy * ent


class EfficientKAN(_KANNet):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0), **kw):
        super().__init__(layers_hidden, lambda i, o: EfficientKANLinear(
            i, o, grid_size=grid_size, spline_order=spline_order, grid_range=grid_range, **kw))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(l.regularization_loss(regularize_activation, regularize_entropy)
                   for l in self.layers if isinstance(l, EfficientKANLinear))


# ---------------------------------------------------------------------------
# 2. FastKAN (Gaussian RBF)
# ---------------------------------------------------------------------------
class FastKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_grids=8, grid_min=-2.0, grid_max=2.0,
                 use_base_update=True, spline_weight_init_scale=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids))
        self.denominator = (grid_max - grid_min) / (num_grids - 1)
        self.spline_linear = nn.Linear(in_features * num_grids, out_features, bias=False)
        nn.init.trunc_normal_(self.spline_linear.weight, std=spline_weight_init_scale)
        self.base_linear = nn.Linear(in_features, out_features) if use_base_update else None

    def forward(self, x):
        rbf = torch.exp(-(((self.layernorm(x).unsqueeze(-1) - self.grid) / self.denominator) ** 2))
        out = self.spline_linear(rbf.view(x.size(0), -1))
        if self.base_linear is not None:
            out = out + self.base_linear(F.silu(x))
        return out


class FastKAN(_KANNet):
    def __init__(self, layers_hidden, num_grids=8, grid_min=-2.0, grid_max=2.0, **kw):
        super().__init__(layers_hidden, lambda i, o: FastKANLayer(
            i, o, num_grids=num_grids, grid_min=grid_min, grid_max=grid_max, **kw))


# ---------------------------------------------------------------------------
# 3. FasterKAN (Reflectional Switch Activation Function, RSWAF)
# ---------------------------------------------------------------------------
class FasterKANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_grids=8, grid_min=-2.0, grid_max=2.0,
                 inv_denominator=1.5, spline_weight_init_scale=0.667):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids))
        self.inv_denominator = inv_denominator
        self.spline_linear = nn.Linear(in_features * num_grids, out_features, bias=False)
        nn.init.xavier_uniform_(self.spline_linear.weight, gain=spline_weight_init_scale)

    def forward(self, x):
        t = torch.tanh((self.layernorm(x).unsqueeze(-1) - self.grid) * self.inv_denominator)
        basis = 1.0 - t * t  # sech^2
        return self.spline_linear(basis.view(x.size(0), -1))


class FasterKAN(_KANNet):
    def __init__(self, layers_hidden, num_grids=8, grid_min=-2.0, grid_max=2.0, inv_denominator=1.5, **kw):
        super().__init__(layers_hidden, lambda i, o: FasterKANLayer(
            i, o, num_grids=num_grids, grid_min=grid_min, grid_max=grid_max,
            inv_denominator=inv_denominator, **kw))


# ---------------------------------------------------------------------------
# 4. BSRBF_KAN (B-spline + Gaussian RBF)
# ---------------------------------------------------------------------------
class BSRBFKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 base_activation=nn.SiLU, grid_range=(-1.5, 1.5)):
        super().__init__()
        self.spline_order = spline_order
        self.num_bases = grid_size + spline_order
        self.layernorm = nn.LayerNorm(in_features)
        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.spline_linear = nn.Linear(in_features * self.num_bases, out_features, bias=False)
        nn.init.trunc_normal_(self.spline_linear.weight, std=0.1)

        self.register_buffer("grid", _make_grid(in_features, grid_size, spline_order, grid_range))
        self.register_buffer("rbf_grid", torch.linspace(grid_range[0], grid_range[1], self.num_bases))
        self.rbf_denominator = (grid_range[1] - grid_range[0]) / (self.num_bases - 1)

    def forward(self, x):
        x = self.layernorm(x)
        base = F.linear(self.base_activation(x), self.base_weight)
        bs = _b_splines(x, self.grid, self.spline_order)
        rbf = torch.exp(-(((x.unsqueeze(-1) - self.rbf_grid) / self.rbf_denominator) ** 2))
        return base + self.spline_linear((bs + rbf).view(x.size(0), -1))


class BSRBF_KAN(_KANNet):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, **kw):
        super().__init__(layers_hidden, lambda i, o: BSRBFKANLayer(
            i, o, grid_size=grid_size, spline_order=spline_order, **kw))


# ---------------------------------------------------------------------------
# 5. ChebyKAN (Chebyshev polynomials of the first kind)
# ---------------------------------------------------------------------------
class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=4):
        super().__init__()
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(in_features, out_features, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, dtype=torch.float32))

    def forward(self, x):
        # T_n(x) = cos(n * arccos(x)), x squashed into (-1, 1)
        x = torch.tanh(x).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(x).unsqueeze(-1)                  # (B, in, 1)
        basis = torch.cos(theta * self.arange)                # (B, in, deg+1)
        return torch.einsum("bid,iod->bo", basis, self.cheby_coeffs)


class ChebyKAN(_KANNet):
    def __init__(self, layers_hidden, degree=4):
        super().__init__(layers_hidden, lambda i, o: ChebyKANLayer(i, o, degree), norm_between=True)


# ---------------------------------------------------------------------------
# 6. JacobiKAN (Jacobi polynomials P_n^(a, b))
# ---------------------------------------------------------------------------
class JacobiKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=4, a=1.0, b=1.0):
        super().__init__()
        self.degree, self.a, self.b = degree, a, b
        self.jacobi_coeffs = nn.Parameter(torch.empty(in_features, out_features, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

    def forward(self, x):
        x = torch.tanh(x)
        a, b = self.a, self.b
        polys = [torch.ones_like(x)]
        if self.degree > 0:
            polys.append(((a - b) + (a + b + 2) * x) / 2)
        for n in range(2, self.degree + 1):
            k1 = (2 * n + a + b) * (2 * n + a + b - 1) / (2 * n * (n + a + b))
            k2 = (2 * n + a + b - 1) * (a * a - b * b) / (2 * n * (n + a + b) * (2 * n + a + b - 2))
            k3 = (n + a - 1) * (n + b - 1) * (2 * n + a + b) / (n * (n + a + b) * (2 * n + a + b - 2))
            polys.append((k1 * x + k2) * polys[-1] - k3 * polys[-2])
        basis = torch.stack(polys, dim=-1)                    # (B, in, deg+1)
        return torch.einsum("bid,iod->bo", basis, self.jacobi_coeffs)


class JacobiKAN(_KANNet):
    def __init__(self, layers_hidden, degree=4, a=1.0, b=1.0):
        super().__init__(layers_hidden, lambda i, o: JacobiKANLayer(i, o, degree, a, b), norm_between=True)


# ---------------------------------------------------------------------------
# 7. FourierKAN (truncated Fourier series)
# ---------------------------------------------------------------------------
class FourierKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, add_bias=True):
        super().__init__()
        self.grid_size = grid_size
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, out_features, in_features, grid_size) / (math.sqrt(in_features) * math.sqrt(grid_size))
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if add_bias else None
        self.register_buffer("k", torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x):
        kx = x.unsqueeze(-1) * self.k                         # (B, in, G)
        out = torch.einsum("big,oig->bo", torch.cos(kx), self.fourier_coeffs[0]) \
            + torch.einsum("big,oig->bo", torch.sin(kx), self.fourier_coeffs[1])
        if self.bias is not None:
            out = out + self.bias
        return out


class FourierKAN(_KANNet):
    def __init__(self, layers_hidden, grid_size=5, add_bias=True):
        super().__init__(layers_hidden, lambda i, o: FourierKANLayer(i, o, grid_size, add_bias), norm_between=True)


# ---------------------------------------------------------------------------
# 8. WavKAN (Mexican-hat wavelet)
# ---------------------------------------------------------------------------
class WavKANLayer(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type="mexican_hat"):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.empty(out_features, in_features))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.bn = nn.BatchNorm1d(out_features)

    def _wavelet(self, t):
        if self.wavelet_type == "mexican_hat":
            c = 2 / (math.sqrt(3) * math.pi ** 0.25)
            return c * (t ** 2 - 1) * torch.exp(-0.5 * t ** 2)
        if self.wavelet_type == "morlet":
            return torch.exp(-0.5 * t ** 2) * torch.cos(5.0 * t)
        if self.wavelet_type == "dog":
            return -t * torch.exp(-0.5 * t ** 2)
        raise ValueError(f"Unknown wavelet_type: {self.wavelet_type}")

    def forward(self, x):
        t = (x.unsqueeze(1) - self.translation) / self.scale  # (B, out, in)
        wav = (self._wavelet(t) * self.wavelet_weights).sum(dim=2)
        base = F.linear(F.silu(x), self.base_weight)
        return self.bn(wav + base)


class WavKAN(_KANNet):
    def __init__(self, layers_hidden, wavelet_type="mexican_hat"):
        super().__init__(layers_hidden, lambda i, o: WavKANLayer(i, o, wavelet_type))


# ---------------------------------------------------------------------------
# 9. ReLUKAN (squared ReLU bumps)
# ---------------------------------------------------------------------------
class ReLUKANLayer(nn.Module):
    """Basis R_i(x) = [ReLU(x - s_i) * ReLU(e_i - x)]^2 * r, with trainable s_i, e_i.

    Inputs are mapped to (0, 1) with a sigmoid, the domain ReLU-KAN assumes.
    """

    def __init__(self, in_features, out_features, grid_size=5, k=3, train_ab=True):
        super().__init__()
        self.r = 4 * grid_size * grid_size / ((k + 1) * (k + 1))
        low = torch.arange(-k, grid_size, dtype=torch.float32) / grid_size
        high = low + (k + 1) / grid_size
        self.phase_low = nn.Parameter(low.expand(in_features, -1).clone(), requires_grad=train_ab)
        self.phase_high = nn.Parameter(high.expand(in_features, -1).clone(), requires_grad=train_ab)
        self.linear = nn.Linear(in_features * (grid_size + k), out_features)

    def forward(self, x):
        x = torch.sigmoid(x).unsqueeze(-1)                    # (B, in, 1)
        basis = (F.relu(x - self.phase_low) * F.relu(self.phase_high - x) * self.r) ** 2
        return self.linear(basis.view(basis.size(0), -1))


class ReLUKAN(_KANNet):
    def __init__(self, layers_hidden, grid_size=5, k=3, train_ab=True):
        super().__init__(layers_hidden, lambda i, o: ReLUKANLayer(i, o, grid_size, k, train_ab), norm_between=True)


# ---------------------------------------------------------------------------
# 10. RationalKAN (safe Pade approximant per edge)
# ---------------------------------------------------------------------------
class RationalKANLayer(nn.Module):
    """phi_ij(x) = P_ij(x) / (1 + |Q_ij(x)|), P of order m, Q of order n (no constant term)."""

    def __init__(self, in_features, out_features, num_order=3, den_order=2):
        super().__init__()
        self.num_order, self.den_order = num_order, den_order
        self.p_coeffs = nn.Parameter(torch.empty(in_features, out_features, num_order + 1))
        self.q_coeffs = nn.Parameter(torch.empty(in_features, out_features, den_order))
        nn.init.normal_(self.p_coeffs, std=1.0 / math.sqrt(in_features * (num_order + 1)))
        nn.init.normal_(self.q_coeffs, std=0.1)
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        xt = torch.tanh(x)
        max_order = max(self.num_order, self.den_order)
        powers = torch.stack([xt ** d for d in range(max_order + 1)], dim=-1)  # (B, in, max+1)
        p = torch.einsum("bid,iod->bio", powers[..., : self.num_order + 1], self.p_coeffs)
        q = torch.einsum("bid,iod->bio", powers[..., 1: self.den_order + 1], self.q_coeffs)
        rational = (p / (1.0 + q.abs())).sum(dim=1)           # (B, out)
        return rational + self.base_linear(F.silu(x))


class RationalKAN(_KANNet):
    def __init__(self, layers_hidden, num_order=3, den_order=2):
        super().__init__(layers_hidden, lambda i, o: RationalKANLayer(i, o, num_order, den_order), norm_between=True)


# ---------------------------------------------------------------------------
# Baseline: MLP
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, layers_hidden, dropout=0.0):
        super().__init__()
        blocks = []
        for i, (d_in, d_out) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            blocks.append(nn.Linear(d_in, d_out))
            if i < len(layers_hidden) - 2:
                blocks += [nn.LayerNorm(d_out), nn.SiLU(), nn.Dropout(dropout)]
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x.reshape(x.shape[0], -1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
MODELS = {
    "efficientkan": EfficientKAN,
    "fastkan": FastKAN,
    "fasterkan": FasterKAN,
    "bsrbf_kan": BSRBF_KAN,
    "chebykan": ChebyKAN,
    "jacobikan": JacobiKAN,
    "fourierkan": FourierKAN,
    "wavkan": WavKAN,
    "relukan": ReLUKAN,
    "rationalkan": RationalKAN,
    "mlp": MLP,
}


def get_model(name, layers_hidden, **kwargs):
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return MODELS[key](layers_hidden, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smoke_test():
    """Forward + backward pass of every model on random CSI-shaped input."""
    # Smoke test: forward + backward for every model on random "CSI" input.
    torch.manual_seed(0)
    n_input, n_hidden, n_output, batch = 270, 64, 7, 16
    x = torch.randn(batch, n_input)
    y = torch.randint(0, n_output, (batch,))
    print(f"{'model':<14}{'params':>10}  output")
    for name in MODELS:
        model = get_model(name, [n_input, n_hidden, n_output])
        out = model(x)
        F.cross_entropy(out, y).backward()
        assert out.shape == (batch, n_output) and torch.isfinite(out).all(), name
        print(f"{name:<14}{count_parameters(model):>10,}  {tuple(out.shape)}")


# if __name__ == "__main__":
#     smoke_test()
