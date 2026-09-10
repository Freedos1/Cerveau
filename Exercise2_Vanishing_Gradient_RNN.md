# Exercise 2: Exponential Decay of Gradients in RNNs

This exercise demonstrates, with a fully worked numerical example, why the
gradient of the loss with respect to an early hidden state in a
vanilla (Elman) RNN decays exponentially with the number of time steps
between that state and the loss — the *vanishing gradient problem* — and
what has to change for the same recursion to *explode* instead.

All numbers below are reproducible by running `compute_gradients.py`
(included in this repository) with Python 3; it regenerates the 100-row
table in `exercise2_table_100rows.md` and every numeric result quoted here.

---

## Part A: System Parameters

We use a standard single-unit (scalar) vanilla RNN cell, which is the
simplest setting in which the recursive Jacobian structure that causes
vanishing/exploding gradients is fully visible:

**Model equations**

```
z_t = W * h_(t-1) + U * x_t + b      (pre-activation)
h_t = tanh(z_t)                       (hidden state)
```

**Parameters chosen for this exercise**

| Symbol   | Meaning                               | Value |
|----------|----------------------------------------|-------|
| `W`      | recurrent (hidden-to-hidden) weight    | 0.9   |
| `U`      | input (input-to-hidden) weight         | 0.5   |
| `b`      | bias                                    | 0.1   |
| `x_t`    | input at every time step               | 1.0 (constant) |
| `h_0`    | initial hidden state                    | 0.0   |
| `T`      | number of time steps                    | 100   |
| `y`      | target value at the final time step     | 0.8   |

**Loss function** — mean-squared error evaluated only at the final time step
`T = 100` (a common choice for sequence-classification/regression RNNs):

```
L = 1/2 * (y - h_T)^2
```

so that

```
dL/dh_T = -(y - h_T) = h_T - y
```

We will show that `dL/dh_1`, the gradient of this single final loss with
respect to the *first* hidden state, is many orders of magnitude smaller
than `dL/dh_T` — i.e. it vanishes.

---

## Part B: Hidden State Table (t = 1 … 100)

Using the recursion `z_t = W*h_(t-1) + U*x + b`, `h_t = tanh(z_t)`, and the
activation derivative

```
d(tanh(z_t))/dz_t = sech^2(z_t) = 1 - tanh^2(z_t) = 1 - h_t^2
```

we compute `h_t` and `sech^2(z_t)` for every one of the 100 time steps.
Starting from `h_0 = 0`:

* `t=1`: `z_1 = 0.9(0) + 0.5(1) + 0.1 = 0.600000`, `h_1 = tanh(0.6) = 0.537050`
* `t=2`: `z_2 = 0.9(0.537050) + 0.6 = 1.083345`, `h_2 = tanh(1.083345) = 0.794436`
* … and so on, applied recursively 100 times.

Because `W = 0.9 < 1` and the input keeps driving `z_t` upward, the sequence
`h_t` converges rapidly (by about `t ≈ 11`) to the fixed point of
`h* = tanh(0.9*h* + 0.6) ≈ 0.884494`, at which point `sech^2(z_t)` also
settles to a constant, `≈ 0.217671`. The **full 100-row table**:

| t | z_t | h_t = tanh(z_t) | sech²(z_t) = 1 − h_t² |
|---|---|---|---|
| 1 | 0.600000 | 0.537050 | 0.711578 |
| 2 | 1.083345 | 0.794436 | 0.368871 |
| 3 | 1.314992 | 0.865533 | 0.250852 |
| 4 | 1.378980 | 0.880723 | 0.224328 |
| 5 | 1.392650 | 0.883753 | 0.218981 |
| 6 | 1.395377 | 0.884348 | 0.217928 |
| 7 | 1.395914 | 0.884465 | 0.217721 |
| 8 | 1.396019 | 0.884488 | 0.217681 |
| 9 | 1.396039 | 0.884493 | 0.217673 |
| 10 | 1.396043 | 0.884493 | 0.217671 |
| 11 | 1.396044 | 0.884494 | 0.217671 |
| 12 | 1.396044 | 0.884494 | 0.217671 |
| 13 | 1.396044 | 0.884494 | 0.217671 |
| 14 | 1.396044 | 0.884494 | 0.217671 |
| 15 | 1.396044 | 0.884494 | 0.217671 |
| 16 | 1.396044 | 0.884494 | 0.217671 |
| 17 | 1.396044 | 0.884494 | 0.217671 |
| 18 | 1.396044 | 0.884494 | 0.217671 |
| 19 | 1.396044 | 0.884494 | 0.217671 |
| 20 | 1.396044 | 0.884494 | 0.217671 |
| 21 | 1.396044 | 0.884494 | 0.217671 |
| 22 | 1.396044 | 0.884494 | 0.217671 |
| 23 | 1.396044 | 0.884494 | 0.217671 |
| 24 | 1.396044 | 0.884494 | 0.217671 |
| 25 | 1.396044 | 0.884494 | 0.217671 |
| 26 | 1.396044 | 0.884494 | 0.217671 |
| 27 | 1.396044 | 0.884494 | 0.217671 |
| 28 | 1.396044 | 0.884494 | 0.217671 |
| 29 | 1.396044 | 0.884494 | 0.217671 |
| 30 | 1.396044 | 0.884494 | 0.217671 |
| 31 | 1.396044 | 0.884494 | 0.217671 |
| 32 | 1.396044 | 0.884494 | 0.217671 |
| 33 | 1.396044 | 0.884494 | 0.217671 |
| 34 | 1.396044 | 0.884494 | 0.217671 |
| 35 | 1.396044 | 0.884494 | 0.217671 |
| 36 | 1.396044 | 0.884494 | 0.217671 |
| 37 | 1.396044 | 0.884494 | 0.217671 |
| 38 | 1.396044 | 0.884494 | 0.217671 |
| 39 | 1.396044 | 0.884494 | 0.217671 |
| 40 | 1.396044 | 0.884494 | 0.217671 |
| 41 | 1.396044 | 0.884494 | 0.217671 |
| 42 | 1.396044 | 0.884494 | 0.217671 |
| 43 | 1.396044 | 0.884494 | 0.217671 |
| 44 | 1.396044 | 0.884494 | 0.217671 |
| 45 | 1.396044 | 0.884494 | 0.217671 |
| 46 | 1.396044 | 0.884494 | 0.217671 |
| 47 | 1.396044 | 0.884494 | 0.217671 |
| 48 | 1.396044 | 0.884494 | 0.217671 |
| 49 | 1.396044 | 0.884494 | 0.217671 |
| 50 | 1.396044 | 0.884494 | 0.217671 |
| 51 | 1.396044 | 0.884494 | 0.217671 |
| 52 | 1.396044 | 0.884494 | 0.217671 |
| 53 | 1.396044 | 0.884494 | 0.217671 |
| 54 | 1.396044 | 0.884494 | 0.217671 |
| 55 | 1.396044 | 0.884494 | 0.217671 |
| 56 | 1.396044 | 0.884494 | 0.217671 |
| 57 | 1.396044 | 0.884494 | 0.217671 |
| 58 | 1.396044 | 0.884494 | 0.217671 |
| 59 | 1.396044 | 0.884494 | 0.217671 |
| 60 | 1.396044 | 0.884494 | 0.217671 |
| 61 | 1.396044 | 0.884494 | 0.217671 |
| 62 | 1.396044 | 0.884494 | 0.217671 |
| 63 | 1.396044 | 0.884494 | 0.217671 |
| 64 | 1.396044 | 0.884494 | 0.217671 |
| 65 | 1.396044 | 0.884494 | 0.217671 |
| 66 | 1.396044 | 0.884494 | 0.217671 |
| 67 | 1.396044 | 0.884494 | 0.217671 |
| 68 | 1.396044 | 0.884494 | 0.217671 |
| 69 | 1.396044 | 0.884494 | 0.217671 |
| 70 | 1.396044 | 0.884494 | 0.217671 |
| 71 | 1.396044 | 0.884494 | 0.217671 |
| 72 | 1.396044 | 0.884494 | 0.217671 |
| 73 | 1.396044 | 0.884494 | 0.217671 |
| 74 | 1.396044 | 0.884494 | 0.217671 |
| 75 | 1.396044 | 0.884494 | 0.217671 |
| 76 | 1.396044 | 0.884494 | 0.217671 |
| 77 | 1.396044 | 0.884494 | 0.217671 |
| 78 | 1.396044 | 0.884494 | 0.217671 |
| 79 | 1.396044 | 0.884494 | 0.217671 |
| 80 | 1.396044 | 0.884494 | 0.217671 |
| 81 | 1.396044 | 0.884494 | 0.217671 |
| 82 | 1.396044 | 0.884494 | 0.217671 |
| 83 | 1.396044 | 0.884494 | 0.217671 |
| 84 | 1.396044 | 0.884494 | 0.217671 |
| 85 | 1.396044 | 0.884494 | 0.217671 |
| 86 | 1.396044 | 0.884494 | 0.217671 |
| 87 | 1.396044 | 0.884494 | 0.217671 |
| 88 | 1.396044 | 0.884494 | 0.217671 |
| 89 | 1.396044 | 0.884494 | 0.217671 |
| 90 | 1.396044 | 0.884494 | 0.217671 |
| 91 | 1.396044 | 0.884494 | 0.217671 |
| 92 | 1.396044 | 0.884494 | 0.217671 |
| 93 | 1.396044 | 0.884494 | 0.217671 |
| 94 | 1.396044 | 0.884494 | 0.217671 |
| 95 | 1.396044 | 0.884494 | 0.217671 |
| 96 | 1.396044 | 0.884494 | 0.217671 |
| 97 | 1.396044 | 0.884494 | 0.217671 |
| 98 | 1.396044 | 0.884494 | 0.217671 |
| 99 | 1.396044 | 0.884494 | 0.217671 |
| 100 | 1.396044 | 0.884494 | 0.217671 |

*(Rows 11–100 print identically at 6 decimal places because the system has
converged to its fixed point to within `10^-6`; the underlying values keep
approaching the fixed point asymptotically and are never exactly equal to
it. Full-precision values are produced by `compute_gradients.py`.)*

---

## Part C: Chain Rule Calculation of `dL/dh_1`

### Step 1 — set up the recursive dependency

Each hidden state depends on the previous one:
`h_k = tanh(W*h_(k-1) + U*x_k + b)`, so

```
dh_k / dh_(k-1) = tanh'(z_k) * W = sech^2(z_k) * W
```

The loss depends on `h_1` only through the entire chain
`h_1 → h_2 → h_3 → … → h_T → L`. By the multivariate chain rule this
dependency is a **product of single-step Jacobians**:

```
dL/dh_1 = dL/dh_T * dh_T/dh_(T-1) * dh_(T-1)/dh_(T-2) * … * dh_2/dh_1

        = dL/dh_T * Π_{k=2}^{T} ( dh_k/dh_(k-1) )

        = dL/dh_T * Π_{k=2}^{T} ( W * sech^2(z_k) )
```

This is the key formula: `dL/dh_1` is `dL/dh_T` multiplied by **99
factors**, each of the form `W * sech²(z_k)`.

### Step 2 — evaluate `dL/dh_T`

From Part A, `L = 1/2 (y - h_T)^2` ⇒ `dL/dh_T = h_T - y`. Using the table
(`h_100 = 0.884494`, `y = 0.8`):

```
dL/dh_100 = 0.884494 - 0.8 = 0.084494
```

### Step 3 — evaluate the product term-by-term

Using `W = 0.9` and the `sech²(z_k)` column of the Part B table:

| k | sech²(z_k) | term = W·sech²(z_k) | running product Π_{j=2}^{k} |
|---|---|---|---|
| 2 | 0.368871 | 0.331984 | 0.331984 |
| 3 | 0.250852 | 0.225767 | 0.074951 |
| 4 | 0.224328 | 0.201895 | 0.015132 |
| 5 | 0.218981 | 0.197083 | 0.002982 |
| 6 | 0.217928 | 0.196135 | 0.000585 |
| 7 | 0.217721 | 0.195949 | 0.0001146 |

Each factor is `≈ 0.196–0.332`, always **less than 1**, so the running
product shrinks geometrically. Once the system settles at its fixed point
(`t ≳ 11`, `sech²(z_k) ≈ 0.217671`), every remaining factor is essentially
identical:

```
term_steady = W * sech^2(z*) = 0.9 * 0.217671 ≈ 0.195904
```

so for large `k` the product behaves like a pure geometric sequence with
ratio `r ≈ 0.1959`:

```
Π_{k=2}^{100} (W·sech²(z_k)) ≈ term_2 · term_3 · … · term_10 · (0.195904)^90
                              ≈ 1.657 × 10^-70
```

(the exact value, computed term-by-term for all 99 factors by
`compute_gradients.py`, is `1.656899 × 10^-70`).

### Step 4 — combine

```
dL/dh_1 = dL/dh_100 * Π_{k=2}^{100}(W·sech²(z_k))
        = 0.084494 * 1.656899×10^-70
        ≈ 1.399974 × 10^-71
```

**Result:** `dL/dh_1 ≈ 1.400 × 10⁻⁷¹` — for all practical purposes zero.
Even though `dL/dh_100 ≈ 0.084` is a perfectly ordinary gradient magnitude,
by the time it is propagated 99 steps back to `h_1` it has been multiplied
by ~99 factors that are each well under 1, and it has collapsed to a number
with **71 leading zeros after the decimal point**. Gradient-based updates to
whatever parameters influenced `h_1` (e.g. an embedding used at `t=1`)
would therefore receive essentially no learning signal — this is the
vanishing gradient problem.

---

## Part D: Analysis — From Vanishing to Exploding

### Why `W = 0.9` vanishes

Every multiplicative factor in the chain, `W·sech²(z_k)`, is bounded above
by `W` (since `sech²(z) ≤ 1` for all `z`, with equality only at `z = 0`).
Because `W = 0.9 < 1`, **every single factor in the product is
guaranteed to be less than 1**, so the product is a strictly decreasing
sequence bounded by `0.9^(k-1)` — it must vanish geometrically no matter
how many steps are taken.

### Re-calculating the first 5 steps with a larger weight

To see the opposite regime we need at least one factor `W·sech²(z_k) > 1`,
which requires `W > 1` (since `sech²(z_k) ≤ 1`). We recompute the same
recursion with the recurrent weight raised to **`W' = 1.5`**. To isolate the
effect of the weight itself (rather than have the strong constant input
`x=1` immediately drive `tanh` into saturation, which would crush
`sech²(z_k)` before the weight's effect could be seen), we set `x_t = 0`,
`b = 0`, keeping a small nonzero initial state `h_0 = 0.1` — this keeps
`z_t` near 0 for the first several steps, where `sech²(z_t) ≈ 1` and the
recursion is approximately **linear**, `h_t ≈ W'·h_(t-1)`, so the product
of Jacobians is approximately `(W')^n`.

**`W = 0.9` (vanishing case), first 5 steps:**

| t | z_t | h_t | sech²(z_t) | term = W·sech²(z_t) | running product |
|---|---|---|---|---|---|
| 1 | 0.090000 | 0.089758 | 0.991944 | 0.892749 | 0.892749 |
| 2 | 0.080782 | 0.080607 | 0.993503 | 0.894152 | 0.798254 |
| 3 | 0.072546 | 0.072419 | 0.994755 | 0.895280 | 0.714661 |
| 4 | 0.065177 | 0.065085 | 0.995764 | 0.896188 | 0.640470 |
| 5 | 0.058577 | 0.058510 | 0.996577 | 0.896919 | **0.574450** |

**`W' = 1.5` (exploding case), first 5 steps:**

| t | z_t | h_t | sech²(z_t) | term = W'·sech²(z_t) | running product |
|---|---|---|---|---|---|
| 1 | 0.150000 | 0.148885 | 0.977833 | 1.466750 | 1.466750 |
| 2 | 0.223328 | 0.219687 | 0.951737 | 1.427606 | 2.093941 |
| 3 | 0.329531 | 0.318099 | 0.898813 | 1.348219 | 2.823092 |
| 4 | 0.477149 | 0.443957 | 0.802902 | 1.204353 | 3.399999 |
| 5 | 0.665936 | 0.582300 | 0.660926 | 0.991390 | **3.370723** |

### How the result changes from vanishing to exploding

* With `W = 0.9`, every term is `< 1`, so the running product falls
  monotonically at every step (`0.893 → 0.798 → 0.715 → 0.640 → 0.574`,
  → 0 as `t → ∞`). This is **vanishing**: the further back a hidden state
  is, the smaller its influence on the loss gradient, exponentially fast.
* With `W' = 1.5`, the first four terms are all `> 1` (because
  `sech²(z_t) ≈ 1` while `h_t` is still small), so the running product
  **grows** at every step (`1.467 → 2.094 → 2.823 → 3.400`) instead of
  shrinking. This is **exploding**: gradients flowing through these early
  steps are amplified rather than attenuated, which in a real network
  produces huge, unstable parameter updates.
* Note the qualitative turning point at `t = 5`: as `h_t` grows toward ±1,
  `sech²(z_t)` starts collapsing toward 0, and the 5th term (`0.991`) has
  already dropped back below 1. This illustrates a subtlety specific to a
  *bounded* activation like `tanh`: because `sech²(z) ≤ 1` always, a
  `tanh`-RNN cannot sustain true long-run exponential explosion — an
  oversized `W` can only produce a **transient** burst of growth before
  saturation forces the same geometric decay seen in the `W = 0.9` case.
  (A linear/unbounded recurrence, or a `W` acting on many co-active units
  as in a real multi-unit RNN, is what allows explosion to persist over
  long horizons — the mechanism Pascanu, Mikolov & Bengio, 2013, formalize
  with the condition `‖W‖ > 1/max sech²(z) = 1` for exploding gradients to
  even be possible.)

### Final values

* **`dL/dh_1`, main system (Parts A–C: `W = 0.9`, `x_t = 1`, `b = 0.1`,
  `h_0 = 0`, `T = 100`)**:

  ```
  dL/dh_1 ≈ 1.399974 × 10^-71   (vanishing)
  ```

* **Product of the first 5 Jacobian terms, Part D comparison
  (`x_t = 0`, `b = 0`, `h_0 = 0.1`)**:

  ```
  W  = 0.9  →  Π_{t=1}^{5} (W·sech²(z_t))  ≈ 0.574450   (shrinking, on track to vanish)
  W' = 1.5  →  Π_{t=1}^{5} (W'·sech²(z_t)) ≈ 3.370723   (growing, exploding over this window)
  ```

---

## Summary

Backpropagation through time multiplies together as many Jacobian factors
`W·sech²(z_k)` as there are time steps separating a hidden state from the
loss. Because `sech²(z) ≤ 1`, whether this product shrinks or grows is
governed almost entirely by whether `|W| < 1` or `|W| > 1`:

* `|W| < 1` ⇒ every factor is `< 1` ⇒ the gradient **vanishes**
  geometrically (`dL/dh_1 ≈ 1.4 × 10⁻⁷¹` after 99 steps in our example).
* `|W| > 1` ⇒ some early factors can exceed 1 ⇒ the gradient can
  **explode** over a window of steps, though a bounded activation such as
  `tanh` eventually pulls the product back down as the hidden state
  saturates.

This is exactly why vanilla RNNs struggle to learn long-range dependencies,
and why architectures like LSTMs/GRUs (additive, gated state updates) and
techniques like gradient clipping were introduced.
