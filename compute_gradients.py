"""
Exercise 2: Exponential Decay (and transient growth) of Gradients in RNNs

Reproduces every numeric result used in Exercise2_Vanishing_Gradient_RNN.md:
  - the 100-row hidden-state / activation-derivative table (Part B)
  - the backpropagated gradient dL/dh_1 via the chain rule (Part C)
  - the first-5-step comparison between a vanishing (W=0.9) and an
    exploding (W=1.5) recurrent weight, in an isolated near-zero regime (Part D)

Run:  python3 compute_gradients.py
"""

import math


def tanh(z: float) -> float:
    return math.tanh(z)


def forward_pass(W, U, b, x, h0, T):
    """Vanilla RNN forward pass: z_t = W*h_{t-1} + U*x_t + b, h_t = tanh(z_t)."""
    h = [h0] + [0.0] * T
    z = [None] + [0.0] * T
    dsig = [None] + [0.0] * T  # sech^2(z_t) = 1 - h_t^2
    for t in range(1, T + 1):
        z[t] = W * h[t - 1] + U * x + b
        h[t] = tanh(z[t])
        dsig[t] = 1 - h[t] ** 2
    return z, h, dsig


def main():
    # ---------------- Part A / B / C : main system ----------------
    W, U, b, x, h0, T, y_target = 0.9, 0.5, 0.1, 1.0, 0.0, 100, 0.8

    z, h, dsig = forward_pass(W, U, b, x, h0, T)

    with open("exercise2_table_100rows.md", "w") as f:
        f.write("| t | z_t | h_t = tanh(z_t) | sech^2(z_t) = 1 - h_t^2 |\n")
        f.write("|---|---|---|---|\n")
        for t in range(1, T + 1):
            f.write(f"| {t} | {z[t]:.6f} | {h[t]:.6f} | {dsig[t]:.6f} |\n")

    dLdhT = h[T] - y_target  # dL/dh_T for L = 1/2 (y - h_T)^2

    prod = 1.0
    for t in range(2, T + 1):
        prod *= W * dsig[t]
    dLdh1 = dLdhT * prod

    print("=== Main system (W=0.9, U=0.5, b=0.1, x_t=1, h_0=0, T=100) ===")
    print(f"h_100            = {h[T]:.10f}")
    print(f"dL/dh_100        = {dLdhT:.10f}")
    print(f"prod_{{k=2..100}} = {prod:.6e}")
    print(f"dL/dh_1          = {dLdh1:.6e}")

    # ---------------- Part D : vanishing vs exploding, first 5 steps ----------------
    print("\n=== Part D: isolated near-zero regime (x_t=0, b=0, h_0=0.1) ===")
    for Wd in (0.9, 1.5):
        zd, hd, dsigd = forward_pass(Wd, 0.0, 0.0, 0.0, 0.1, 5)
        prod5 = 1.0
        print(f"\n-- W' = {Wd} --")
        for t in range(1, 6):
            term = Wd * dsigd[t]
            prod5 *= term
            print(
                f" t={t}  z_t={zd[t]: .6f}  h_t={hd[t]: .6f}  "
                f"sech^2(z_t)={dsigd[t]:.6f}  term=W'*sech^2={term:.6f}  "
                f"running product={prod5:.6f}"
            )


if __name__ == "__main__":
    main()
