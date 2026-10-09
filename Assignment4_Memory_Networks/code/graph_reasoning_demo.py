"""Part C demo: build a small sub-graph per answer choice and score it with a
relation-aware graph attention network (R-GCN style) fused with the question text.

The text encoder is a stand-in (a seeded random vector per string). In a real
system, replace `encode` with the [CLS] vector of a pre-trained encoder such as
BERT or RoBERTa run on "question [SEP] choice".
"""
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 32
torch.manual_seed(0)

# Background knowledge (ConceptNet-style triplets) plus triplets from Part B.
KB = [("engine", "HasSubevent", "stop"), ("engine", "Uses", "fuel"),
      ("fuel tank", "Contains", "fuel"), ("fuel tank", "HasState", "empty"),
      ("empty", "Implies", "lack of fuel"), ("lack of fuel", "Causes", "stop"),
      ("driver", "Uses", "radio"), ("radio", "PartOf", "car"),
      ("engine", "PartOf", "car"), ("tires", "PartOf", "car"),
      ("tires", "HasProperty", "new")]
RELS = sorted({r for _, r, _ in KB}) + ["Context"]
N_REL = 2 * len(RELS)                     # forward + reverse direction


def encode(text):
    """Stand-in text encoder: a fixed random vector per string."""
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    return torch.randn(D, generator=torch.Generator().manual_seed(seed))


def build_graph(q_ents, a_ents):
    """Grounded entities + their 1-hop KB neighbours + a QA context node (id 0)."""
    ents = set(q_ents) | set(a_ents)
    nodes = set(ents)
    for h, _, t in KB:
        if h in ents or t in ents:
            nodes |= {h, t}
    nodes = ["[QA]"] + sorted(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    trip = [(h, r, t) for h, r, t in KB if h in idx and t in idx]
    trip += [("[QA]", "Context", e) for e in sorted(ents)]
    src, dst, rel = [], [], []
    for h, r, t in trip:                  # add both directions of every edge
        src += [idx[h], idx[t]]
        dst += [idx[t], idx[h]]
        rel += [RELS.index(r), RELS.index(r) + len(RELS)]
    return nodes, trip, torch.tensor(src), torch.tensor(dst), torch.tensor(rel)


class RelAttnLayer(nn.Module):
    """h_v' = ReLU(W0 h_v + sum_u alpha_uv W_r h_u), alpha = softmax over in-edges."""

    def __init__(self, d, n_rel):
        super().__init__()
        self.W0 = nn.Linear(d, d)
        self.W_rel = nn.Parameter(torch.randn(n_rel, d, d) * d ** -0.5)
        self.rel_emb = nn.Embedding(n_rel, d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(2 * d, d)

    def forward(self, h, src, dst, rel):
        msg = torch.einsum("ed,edk->ek", h[src], self.W_rel[rel])      # W_r h_u
        key = self.k(torch.cat([h[src], self.rel_emb(rel)], dim=-1))
        score = (self.q(h)[dst] * key).sum(-1) / h.size(1) ** 0.5
        alpha = torch.zeros_like(score)
        for v in dst.unique():                # softmax over each node's in-edges
            m = dst == v
            alpha[m] = F.softmax(score[m], dim=0)
        agg = torch.zeros_like(h).index_add_(0, dst, alpha.unsqueeze(-1) * msg)
        return F.relu(self.W0(h) + agg), alpha


class GraphQA(nn.Module):
    def __init__(self, d=D, n_rel=N_REL, layers=3):
        super().__init__()
        self.layers = nn.ModuleList(RelAttnLayer(d, n_rel) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, z, nodes, src, dst, rel):
        h = torch.stack([z] + [encode(n) for n in nodes[1:]])  # node 0 = text
        alphas = []
        for layer in self.layers:
            h, a = layer(h, src, dst, rel)
            alphas.append(a)
        att = F.softmax(h[1:] @ z, dim=0)     # text-guided pooling of graph nodes
        g = (att.unsqueeze(-1) * h[1:]).sum(0)
        return self.out(torch.cat([z, h[0], g])).squeeze(), alphas


def best_path(nodes, src, dst, alpha, start, goal, max_len=4):
    """Walk from an answer entity to a question entity, following the
    highest-attention edges (score = sum of log attention along the path)."""
    best, stack = (float("-inf"), None), [(nodes.index(start), [start], 0.0)]
    while stack:
        u, path, s = stack.pop()
        if nodes[u] == goal:
            best = max(best, (s, path))
            continue
        if len(path) > max_len:
            continue
        for e in (src == u).nonzero().flatten().tolist():
            v = dst[e].item()
            if v != 0 and nodes[v] not in path:
                stack.append((v, path + [nodes[v]], s + alpha[e].log().item()))
    return best


if __name__ == "__main__":
    question = "Why did the engine stop?"
    q_ents = ["engine", "stop"]
    choices = {"The fuel tank was empty.": ["fuel tank", "empty"],
               "The driver turned on the radio.": ["driver", "radio"],
               "The tires were new.": ["tires", "new"]}
    model = GraphQA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits, graphs = [], []
    for text, a_ents in choices.items():
        z = encode(question + " [SEP] " + text)       # fused question + choice text
        g = build_graph(q_ents, a_ents)
        s, alphas = model(z, g[0], *g[2:])
        logits.append(s)
        graphs.append((g, alphas, a_ents))
    logits = torch.stack(logits)
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))  # gold = choice A
    opt.zero_grad(); loss.backward(); opt.step()

    print("P(choice | question):", F.softmax(logits, 0).detach().numpy().round(3))
    print("loss:", round(loss.item(), 4))
    (nodes, trip, src, dst, rel), alphas, a_ents = graphs[0]
    mean_alpha = torch.stack(alphas).mean(0).detach()
    print("nodes:", nodes)
    print("path:", best_path(nodes, src, dst, mean_alpha, "empty", "stop"))
