# Exercise 3: Sequence-to-Sequence with Attention — Transformer Encoder-Decoder

In this exercise a small Transformer translates the **2-token English input**

```
x = [ "thank", "you" ]
```

into the **1-token French output**

```
y = [ "merci" ]
```

Every calculation is written out below. All numbers are rounded to 4 decimal
places. You can reproduce them with `transformer_seq2seq.py` (Python 3 +
NumPy), which is in this repository and prints every intermediate matrix.

> Rounding note: each value is computed at full precision and only rounded when
> shown, so recomputing a line by hand from the rounded inputs can differ in the
> 4th decimal (e.g. 4.0648 vs 4.0647).

---

## Part A: Transformer Calculations

### 1. Setup

| Item | Choice |
|---|---|
| Architecture | 1 encoder layer + 1 decoder layer, 1 attention head |
| Vector convention | Row vectors: each token is one row, so `Q = X·W_Q` |
| Sub-layer wrapper | Residual connection `H = X + Sublayer(X)`. LayerNorm and dropout are left out to keep the arithmetic readable. |
| Attention | Scaled dot-product: `Attention(Q,K,V) = softmax(QKᵀ / √d_k + M) · V` |
| Activation in FFN | ReLU |
| Decoding | Greedy: one decoder step starting from `<sos>`, pick the argmax token |

### 2. Dimensions

| Symbol | Meaning | Value |
|---|---|---|
| `d_model` | embedding / hidden size | 2 |
| `d_k = d_v` | query/key/value size | 2 |
| `h` | number of heads | 1 |
| `d_ff` | FFN hidden size | 2 |
| `n` | source length | 2 |
| `m` | target length | 1 |
| `|V_tgt|` | target vocabulary size | 3 → {`merci`, `bonjour`, `<eos>`} |
| `√d_k` | scaling factor | √2 = 1.4142 |

### 3. Input (source embeddings)

| Token | Position | Embedding |
|---|---|---|
| `thank` | 0 | [1, 0] |
| `you` | 1 | [0, 1] |

```
X = | 1  0 |      (2 × 2)
    | 0  1 |
```

### 4. Output (target side)

The decoder starts from the start-of-sequence token:

| Token | Position | Embedding |
|---|---|---|
| `<sos>` | 0 | [0.5, 0.5] |

Target vocabulary (column order of the output projection): `[merci, bonjour, <eos>]`.
Expected output: **`merci`**.

### 5. Weights (attention)

**Encoder self-attention**

```
W_Q^enc = | 1  0 |   W_K^enc = | 1  1 |   W_V^enc = | 1  0 |   W_O^enc = | 1  0.5 |
          | 1  1 |             | 0  1 |             | 0  1 |             | 0  1   |
```

**Decoder masked self-attention**

```
W_Q^m = W_K^m = W_V^m = I = | 1 0 |       W_O^m = | 0.5  0   |
                            | 0 1 |               | 0    0.5 |
```

**Decoder cross-attention**

```
W_Q^c = W_K^c = W_V^c = I                 W_O^c = | 0.5  0   |
                                                  | 0    0.5 |
```

### 6. Feed-forward weights

`FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂`

| | W₁ | b₁ | W₂ | b₂ |
|---|---|---|---|---|
| Encoder | `[[1, −1], [0.5, 0.5]]` | [0, 0] | `[[0.5, 0], [0, 0.5]]` | [0, 0] |
| Decoder | `[[1, 0], [0, 1]]` | [0, −1] | `[[0.5, 0], [0, 0.5]]` | [0, 0] |

**Output projection** (d_model × |V|), bias `b_vocab = [0, 0, 0]`:

```
                merci  bonjour  <eos>
W_vocab = |     0.5    0.2     -0.3  |
          |     0.3    0.2      0.1  |
```

---

## ENCODER

### 7. Positional Encoding

Sinusoidal encoding (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

With `d_model = 2` there is only `i = 0`, so the divisor is `10000^0 = 1`:

```
PE(pos) = [ sin(pos), cos(pos) ]
```

| pos | sin(pos) | cos(pos) | PE |
|---|---|---|---|
| 0 | sin 0 = 0 | cos 0 = 1 | [0, 1] |
| 1 | sin 1 = 0.8415 | cos 1 = 0.5403 | [0.8415, 0.5403] |

Encoder input = embedding + positional encoding:

```
X_in = X + PE
     = | 1 + 0       0 + 1      |  =  | 1.0000  1.0000 |   ← "thank"
       | 0 + 0.8415  1 + 0.5403 |     | 0.8415  1.5403 |   ← "you"
```

### 8. Self-Attention (encoder)

**Step 8.1 – Queries, keys, values**

```
Q = X_in · W_Q^enc
  row 1: [1·1 + 1·1,            1·0 + 1·1]      = [2.0000, 1.0000]
  row 2: [0.8415·1 + 1.5403·1,  0.8415·0 + 1.5403·1] = [2.3818, 1.5403]

K = X_in · W_K^enc
  row 1: [1·1 + 1·0,            1·1 + 1·1]      = [1.0000, 2.0000]
  row 2: [0.8415·1 + 1.5403·0,  0.8415·1 + 1.5403·1] = [0.8415, 2.3818]

V = X_in · W_V^enc = X_in · I
  = | 1.0000  1.0000 |
    | 0.8415  1.5403 |
```

**Step 8.2 – Raw scores `S = Q·Kᵀ`**

```
S11 = q1·k1 = 2(1)      + 1(2)           = 4.0000
S12 = q1·k2 = 2(0.8415) + 1(2.3818)      = 4.0647
S21 = q2·k1 = 2.3818(1) + 1.5403(2)      = 5.4624
S22 = q2·k2 = 2.3818(0.8415) + 1.5403(2.3818) = 5.6728

S = | 4.0000  4.0647 |
    | 5.4624  5.6728 |
```

**Step 8.3 – Scale by √d_k = 1.4142**

```
S / √2 = | 2.8284  2.8742 |
         | 3.8625  4.0113 |
```

**Step 8.4 – Softmax per row** (encoder attention has no mask: every token sees every token)

```
Row 1: e^2.8284 = 16.9184,  e^2.8742 = 17.7112,  sum = 34.6296
       α1 = [16.9184/34.6296, 17.7112/34.6296] = [0.4886, 0.5114]

Row 2: e^3.8625 = 47.5842,  e^4.0113 = 55.2186,  sum = 102.8028
       α2 = [47.5842/102.8028, 55.2186/102.8028] = [0.4629, 0.5371]

A_enc = | 0.4886  0.5114 |     ("thank" attends 49% to itself, 51% to "you")
        | 0.4629  0.5371 |     ("you"   attends 46% to "thank", 54% to itself)
```

**Step 8.5 – Weighted sum `Z = A·V`**

```
z1 = 0.4886·[1, 1] + 0.5114·[0.8415, 1.5403]
   = [0.4886 + 0.4303, 0.4886 + 0.7877] = [0.9189, 1.2763]

z2 = 0.4629·[1, 1] + 0.5371·[0.8415, 1.5403]
   = [0.4629 + 0.4520, 0.4629 + 0.8273] = [0.9148, 1.2902]
```

### 9. Projection and Residual (encoder)

**Output projection `Z·W_O^enc`**, with `W_O^enc = [[1, 0.5], [0, 1]]`:

```
row 1: [0.9189,  0.5·0.9189 + 1.2763] = [0.9189, 1.7358]
row 2: [0.9148,  0.5·0.9148 + 1.2902] = [0.9148, 1.7476]
```

**Residual `H₁ = X_in + Z·W_O`**

```
H₁ = | 1.0000 + 0.9189   1.0000 + 1.7358 |  =  | 1.9189  2.7358 |
     | 0.8415 + 0.9148   1.5403 + 1.7476 |     | 1.7563  3.2879 |
```

### 10. Feed-Forward Network (encoder)

**Step 10.1 – First linear layer `H₁·W₁ + b₁`**, `W₁ = [[1, −1], [0.5, 0.5]]`

```
row 1: [1.9189 + 0.5·2.7358,  −1.9189 + 0.5·2.7358] = [ 3.2868, −0.5510]
row 2: [1.7563 + 0.5·3.2879,  −1.7563 + 0.5·3.2879] = [ 3.4003, −0.1123]
```

**Step 10.2 – ReLU** (negative values become 0)

```
ReLU = | 3.2868  0 |
       | 3.4003  0 |
```

**Step 10.3 – Second linear layer `·W₂ + b₂`**, `W₂ = 0.5·I`

```
FFN = | 1.6434  0 |
      | 1.7001  0 |
```

**Step 10.4 – Residual → encoder output ("memory")**

```
Enc = H₁ + FFN = | 1.9189 + 1.6434   2.7358 + 0 |  =  | 3.5623  2.7358 |   ← "thank"
                 | 1.7563 + 1.7001   3.2879 + 0 |     | 3.4565  3.2879 |   ← "you"
```

The decoder reads this matrix in cross-attention as its keys and values.

---

## DECODER

### Decoder positional encoding

```
Y_in = E(<sos>) + PE(0) = [0.5, 0.5] + [0, 1] = [0.5, 1.5]
```

### 11. Masked Self-Attention (decoder)

**Step 11.1 – The causal mask.** When decoding, position *t* may only attend to
positions ≤ *t*. The mask adds 0 to allowed positions and −∞ to future
positions before the softmax:

```
M[i, j] = 0    if j ≤ i
M[i, j] = −∞   if j > i
```

For a 2-token target it would be

```
M = | 0   −∞ |     so e^(−∞) = 0: token 1 cannot see token 2.
    | 0    0 |
```

Here the target prefix is only `<sos>` (length 1), so `M = [0]`. There is
no future position to hide, but the mask is still applied.

**Step 11.2 – Q, K, V** (all weights are I):

```
Q = K = V = Y_in · I = [0.5, 1.5]
```

**Step 11.3 – Scores, scaling, mask, softmax**

```
Q·Kᵀ = 0.5·0.5 + 1.5·1.5 = 0.25 + 2.25 = 2.5000
/ √2 = 2.5000 / 1.4142      = 1.7678
+ M  = 1.7678 + 0           = 1.7678
softmax([1.7678]) = e^1.7678 / e^1.7678 = [1.0000]
```

**Step 11.4 – Weighted sum**

```
Z_m = 1.0000 · [0.5, 1.5] = [0.5000, 1.5000]
```

**Step 11.5 – Projection and residual**

```
Z_m · W_O^m = [0.5·0.5, 0.5·1.5] = [0.2500, 0.7500]
H₁ᵈ = Y_in + Z_m·W_O^m = [0.5 + 0.25, 1.5 + 0.75] = [0.7500, 2.2500]
```

### 12. Cross-Attention (encoder-decoder attention)

The queries come from the **decoder** (`H₁ᵈ`). The keys and values come from
the **encoder output** (`Enc`).

**Step 12.1 – Q, K, V** (all weights are I)

```
Q = H₁ᵈ = [0.7500, 2.2500]

K = V = Enc = | 3.5623  2.7358 |   ← "thank"
              | 3.4565  3.2879 |   ← "you"
```

**Step 12.2 – Scores**

```
q·k_thank = 0.75·3.5623 + 2.25·2.7358 = 2.6717 + 6.1556 = 8.8273
q·k_you   = 0.75·3.4565 + 2.25·3.2879 = 2.5924 + 7.3978 = 9.9902
```

**Step 12.3 – Scale**

```
[8.8273, 9.9902] / 1.4142 = [6.2418, 7.0642]
```

**Step 12.4 – Softmax** (no mask: the decoder may look at the whole source)

```
e^6.2418 = 513.7825,  e^7.0642 = 1169.3461,  sum = 1683.1286
α = [513.7825/1683.1286, 1169.3461/1683.1286] = [0.3053, 0.6947]
```

When generating `merci`, the decoder puts 30.5% of its attention on "thank"
and 69.5% on "you". Both source words feed into the single French word, which
is the point of attention in translation: "merci" stands for the whole phrase
"thank you".

**Step 12.5 – Context vector**

```
Z_c = 0.3053·[3.5623, 2.7358] + 0.6947·[3.4565, 3.2879]
    = [1.0876 + 2.4012, 0.8352 + 2.2841]
    = [3.4888, 3.1194]
```

**Step 12.6 – Projection and residual**

```
Z_c · W_O^c = [0.5·3.4888, 0.5·3.1194] = [1.7444, 1.5597]
H₂ᵈ = H₁ᵈ + Z_c·W_O^c = [0.75 + 1.7444, 2.25 + 1.5597] = [2.4944, 3.8097]
```

### 13. Final Output (decoder FFN → decoder output)

**Step 13.1 – First linear layer**, `W₁ = I`, `b₁ = [0, −1]`

```
H₂ᵈ·W₁ + b₁ = [2.4944 + 0, 3.8097 − 1] = [2.4944, 2.8097]
```

**Step 13.2 – ReLU** (both values are positive, so nothing changes)

```
ReLU = [2.4944, 2.8097]
```

**Step 13.3 – Second linear layer**, `W₂ = 0.5·I`

```
FFN = [1.2472, 1.4048]
```

**Step 13.4 – Residual → final decoder output**

```
D = H₂ᵈ + FFN = [2.4944 + 1.2472, 3.8097 + 1.4048] = [3.7416, 5.2145]
```

### 14. Projection (to vocabulary logits)

```
logits = D · W_vocab + b_vocab

merci   = 0.5·3.7416 + 0.3·5.2145 = 1.8708 + 1.5644 =  3.4352
bonjour = 0.2·3.7416 + 0.2·5.2145 = 0.7483 + 1.0429 =  1.7912
<eos>   = −0.3·3.7416 + 0.1·5.2145 = −1.1225 + 0.5215 = −0.6010

logits = [3.4352, 1.7912, −0.6010]
```

### 15. Softmax (probabilities)

```
e^3.4352  = 31.0362
e^1.7912  =  5.9968
e^−0.6010 =  0.5483
sum       = 37.5813

P(merci)   = 31.0362 / 37.5813 = 0.8258
P(bonjour) =  5.9968 / 37.5813 = 0.1596
P(<eos>)   =  0.5483 / 37.5813 = 0.0146
                                  ------
                         total  = 1.0000
```

---

## Final Translated Output

| Token | Probability |
|---|---|
| **merci** | **0.8258** ← argmax |
| bonjour | 0.1596 |
| <eos> | 0.0146 |

```
Input :  [ "thank", "you" ]
Output:  [ "merci" ]      (probability 82.6%)
```

---

## Summary of the pipeline

| Stage | Result |
|---|---|
| Embedding + PE (encoder) | `[[1, 1], [0.8415, 1.5403]]` |
| Encoder self-attention weights | `[[0.4886, 0.5114], [0.4629, 0.5371]]` |
| After projection + residual | `[[1.9189, 2.7358], [1.7563, 3.2879]]` |
| Encoder output (after FFN + residual) | `[[3.5623, 2.7358], [3.4565, 3.2879]]` |
| Decoder input `<sos>` + PE | `[0.5, 1.5]` |
| Masked self-attention (weight 1.0) + residual | `[0.75, 2.25]` |
| Cross-attention weights (thank, you) | `[0.3053, 0.6947]` |
| After cross-attention + residual | `[2.4944, 3.8097]` |
| Decoder output (after FFN + residual) | `[3.7416, 5.2145]` |
| Logits (merci, bonjour, <eos>) | `[3.4352, 1.7912, −0.6010]` |
| Softmax | `[0.8258, 0.1596, 0.0146]` |
| **Prediction** | **merci** |

**Main points**

1. **Positional encoding** is what distinguishes "thank" at position 0 from
   "you" at position 1. Without it, self-attention treats its input as an
   unordered set.
2. **Encoder self-attention** mixes information across the source, so each
   encoder output row carries context from both words.
3. **Masked self-attention** stops the decoder from looking at future target
   tokens, which keeps training consistent with left-to-right generation.
4. **Cross-attention** is where the translation happens: the decoder's query
   decides how much each source word contributes to the next target word.
5. **Residual connections** keep the original signal after every sub-layer.
6. The **projection + softmax** turn the final 2-D vector into a probability
   distribution over the target vocabulary, and greedy decoding picks `merci`.
