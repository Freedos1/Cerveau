"""
Exercise 3: Sequence-to-Sequence with Attention (Transformer Encoder-Decoder)

Reproduces every number in Exercise3_Transformer_Seq2Seq.md.
Task: translate the 2-token English input  ["thank", "you"]
      into the 1-token French output        ["merci"].

Conventions
  - Row-vector convention: each token is a row, so Q = X @ W_Q.
  - d_model = d_k = d_v = 2, d_ff = 2, one attention head, one encoder
    layer and one decoder layer.
  - LayerNorm and dropout are omitted (simplified Transformer); every
    sub-layer is followed only by a residual connection.

Run:  python3 transformer_seq2seq.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------- setup
d_model, d_k, d_ff = 2, 2, 2
src_tokens = ["thank", "you"]
tgt_vocab = ["merci", "bonjour", "<eos>"]

E_src = {"thank": np.array([1.0, 0.0]),
         "you":   np.array([0.0, 1.0])}
E_sos = np.array([0.5, 0.5])            # decoder start token <sos>

# encoder weights
W_Q_e = np.array([[1.0, 0.0], [1.0, 1.0]])
W_K_e = np.array([[1.0, 1.0], [0.0, 1.0]])
W_V_e = np.array([[1.0, 0.0], [0.0, 1.0]])
W_O_e = np.array([[1.0, 0.5], [0.0, 1.0]])
W1_e = np.array([[1.0, -1.0], [0.5, 0.5]]); b1_e = np.array([0.0, 0.0])
W2_e = np.array([[0.5, 0.0], [0.0, 0.5]]); b2_e = np.array([0.0, 0.0])

# decoder masked self-attention weights
W_Q_m = np.array([[1.0, 0.0], [0.0, 1.0]])
W_K_m = np.array([[1.0, 0.0], [0.0, 1.0]])
W_V_m = np.array([[1.0, 0.0], [0.0, 1.0]])
W_O_m = np.array([[0.5, 0.0], [0.0, 0.5]])

# decoder cross-attention weights
W_Q_c = np.array([[1.0, 0.0], [0.0, 1.0]])
W_K_c = np.array([[1.0, 0.0], [0.0, 1.0]])
W_V_c = np.array([[1.0, 0.0], [0.0, 1.0]])
W_O_c = np.array([[0.5, 0.0], [0.0, 0.5]])

# decoder FFN
W1_d = np.array([[1.0, 0.0], [0.0, 1.0]]); b1_d = np.array([0.0, -1.0])
W2_d = np.array([[0.5, 0.0], [0.0, 0.5]]); b2_d = np.array([0.0, 0.0])

# final linear projection to the target vocabulary (d_model x |V|)
W_vocab = np.array([[0.5, 0.2, -0.3],
                    [0.3, 0.2, 0.1]])
b_vocab = np.array([0.0, 0.0, 0.0])


def pe(pos, d=d_model):
    out = np.zeros(d)
    for i in range(0, d, 2):
        out[i] = np.sin(pos / 10000 ** (i / d))
        out[i + 1] = np.cos(pos / 10000 ** (i / d))
    return out


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Xq, Xkv, W_Q, W_K, W_V, mask=None, name=""):
    Q, K, V = Xq @ W_Q, Xkv @ W_K, Xkv @ W_V
    S = Q @ K.T
    Ss = S / np.sqrt(d_k)
    if mask is not None:
        Ss = Ss + mask
    A = softmax(Ss)
    Z = A @ V
    print(f"\n[{name}]\nQ=\n{Q}\nK=\n{K}\nV=\n{V}\nQK^T=\n{S}\n"
          f"QK^T/sqrt(d_k)=\n{Ss}\nweights=\n{A}\nZ=AV=\n{Z}")
    return Z


# --------------------------------------------------------------- encoder
X = np.stack([E_src[t] for t in src_tokens])
P = np.stack([pe(p) for p in range(len(src_tokens))])
X_in = X + P
print("Positional encodings:\n", P, "\nEncoder input X+PE:\n", X_in)

Z_e = attention(X_in, X_in, W_Q_e, W_K_e, W_V_e, name="Encoder self-attention")
O_e = Z_e @ W_O_e
H1_e = X_in + O_e
print("Z W_O =\n", O_e, "\nresidual H1 = X_in + Z W_O =\n", H1_e)

pre = H1_e @ W1_e + b1_e
act = np.maximum(0, pre)
F_e = act @ W2_e + b2_e
enc_out = H1_e + F_e
print("\n[Encoder FFN]\nH1 W1 + b1 =\n", pre, "\nReLU =\n", act,
      "\nFFN out =\n", F_e, "\nEncoder output (memory) =\n", enc_out)

# --------------------------------------------------------------- decoder
Y_in = (E_sos + pe(0)).reshape(1, -1)
print("\nDecoder input <sos> + PE(0):\n", Y_in)
mask = np.zeros((1, 1))   # upper triangle is empty for a length-1 prefix
Z_m = attention(Y_in, Y_in, W_Q_m, W_K_m, W_V_m, mask=mask,
                name="Decoder masked self-attention")
H1_d = Y_in + Z_m @ W_O_m
print("Z W_O =\n", Z_m @ W_O_m, "\nresidual =\n", H1_d)

Z_c = attention(H1_d, enc_out, W_Q_c, W_K_c, W_V_c, name="Cross-attention")
H2_d = H1_d + Z_c @ W_O_c
print("Z W_O =\n", Z_c @ W_O_c, "\nresidual =\n", H2_d)

pre = H2_d @ W1_d + b1_d
act = np.maximum(0, pre)
F_d = act @ W2_d + b2_d
dec_out = H2_d + F_d
print("\n[Decoder FFN]\nH2 W1 + b1 =\n", pre, "\nReLU =\n", act,
      "\nFFN out =\n", F_d, "\nDecoder output =\n", dec_out)

logits = dec_out @ W_vocab + b_vocab
probs = softmax(logits)
print("\nlogits =", logits, "\nexp =", np.exp(logits), "sum =", np.exp(logits).sum(),
      "\nprobs =", probs)
print("Predicted token:", tgt_vocab[int(np.argmax(probs))])
