"""Executable derivations, checked against float64 autograd and finite differences."""
import math
import torch
from torch.nn import functional as F


def close(actual, expected, name):
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    print(f"PASS {name}")


def main():
    torch.manual_seed(7)
    torch.set_default_dtype(torch.float64)
    z = torch.randn(3, 5, requires_grad=True)
    y = torch.tensor([0, 3, 2])
    loss = F.cross_entropy(z, y)
    loss.backward()
    close(z.grad, (z.softmax(-1) - F.one_hot(y, 5)) / 3, "cross entropy gradient")

    x = torch.randn(3, 4, requires_grad=True)
    w = torch.randn(4, 2, requires_grad=True)
    g = torch.randn(3, 2)
    ((x @ w) * g).sum().backward()
    close(x.grad, g @ w.detach().T, "linear input gradient")
    close(w.grad, x.detach().T @ g, "linear weight gradient")

    u = torch.tensor([-3., -1., 0., 1., 3.], requires_grad=True)
    F.gelu(u, approximate="none").sum().backward()
    with torch.no_grad():
        cdf = 0.5 * (1 + torch.erf(u / math.sqrt(2)))
        pdf = torch.exp(-u.square() / 2) / math.sqrt(2 * math.pi)
        close(u.grad, cdf + u * pdf, "exact GELU derivative")

    q, k, v = [torch.randn(4, 3, requires_grad=True) for _ in range(3)]
    mask = torch.ones(4, 4, dtype=torch.bool).tril()
    def attention(q, k, v):
        p = (q @ k.T / math.sqrt(3)).masked_fill(~mask, -float("inf")).softmax(-1)
        return p @ v, p
    o, p = attention(q, k, v)
    upstream = torch.randn_like(o)
    (o * upstream).sum().backward()
    with torch.no_grad():
        dp = upstream @ v.T
        ds = p * (dp - (dp * p).sum(-1, keepdim=True))
        close(v.grad, p.T @ upstream, "attention dV")
        close(q.grad, ds @ k / math.sqrt(3), "attention dQ")
        close(k.grad, ds.T @ q / math.sqrt(3), "attention dK")
        plus, minus = q.clone(), q.clone()
        plus[2, 1] += 1e-6
        minus[2, 1] -= 1e-6
        numeric = ((attention(plus, k, v)[0] - attention(minus, k, v)[0]) * upstream).sum() / 2e-6
        close(numeric, q.grad[2, 1], "attention finite difference")

    x = torch.randn(2, 4, requires_grad=True)
    gamma = torch.randn(4)
    g = torch.randn_like(x)
    (F.layer_norm(x, (4,), gamma, eps=1e-5) * g).sum().backward()
    with torch.no_grad():
        inv = torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
        xhat = (x - x.mean(-1, keepdim=True)) * inv
        gx = g * gamma
        dx = inv * (gx - gx.mean(-1, keepdim=True) - xhat * (gx * xhat).mean(-1, keepdim=True))
        close(x.grad, dx, "LayerNorm gradient with epsilon")

    z = torch.randn(5, 4, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 0])
    full = torch.autograd.grad(F.cross_entropy(z, y), z, retain_graph=True)[0]
    weighted = torch.autograd.grad((F.cross_entropy(z[:1], y[:1], reduction="sum") +
                                   F.cross_entropy(z[1:], y[1:], reduction="sum")) / 5, z, retain_graph=True)[0]
    wrong = torch.autograd.grad((F.cross_entropy(z[:1], y[:1]) + F.cross_entropy(z[1:], y[1:])) / 2, z)[0]
    close(weighted, full, "unequal microbatch token weighting")
    assert not torch.allclose(wrong, full), "The wrong formula must produce a counterexample"

    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    gradient = torch.tensor([0.5, -0.2])
    opt = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, foreach=False)
    before = parameter.detach().clone()
    parameter.grad = gradient
    opt.step()
    close(parameter.detach(), before * (1 - 0.01 * 0.1) - 0.01 * gradient / (gradient.abs() + 1e-8), "AdamW first update")

    def rotate(vector, angle):
        c, s = math.cos(angle), math.sin(angle)
        return torch.stack((c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]))
    q, k = torch.randn(2), torch.randn(2)
    close(rotate(q, 0.7) @ rotate(k, 1.1), q @ rotate(k, 0.4), "RoPE relative position")

    chosen = torch.tensor(-2.0, requires_grad=True)
    rejected = torch.tensor(-3.0, requires_grad=True)
    beta = 0.2
    margin = beta * ((chosen - rejected) - (-2.5 + 2.7))
    F.softplus(-margin).backward()
    close(chosen.grad, -beta * torch.sigmoid(-margin.detach()), "DPO chosen derivative")
    close(rejected.grad, beta * torch.sigmoid(-margin.detach()), "DPO rejected derivative")
    print("All mathematical checks passed (float64, fixed seed).")


if __name__ == "__main__":
    main()
