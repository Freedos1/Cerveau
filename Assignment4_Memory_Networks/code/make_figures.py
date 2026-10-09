import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
out = sys.argv[1]
plt.rcParams["font.family"] = "DejaVu Serif"

def box(ax, x, y, w, h, text, fc="#ffffff", ec="#222222", fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)

def arrow(ax, p, q, text=None, off=(0, 0.12), style="-|>", ls="-", color="#222222", rad=0.0, fs=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, lw=1.3, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0]+q[0])/2 + off[0], (p[1]+q[1])/2 + off[1], text, ha="center", va="center", fontsize=fs,
                style="italic", bbox=dict(fc="white", ec="none", pad=1))

# ---------- Figure 1: memory network flow ----------
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.0); ax.axis("off")
box(ax, 0.8, 3.3, 1.3, 0.8, "Input x\n(fact or\nquestion)", fs=10)
box(ax, 2.9, 3.3, 1.6, 0.8, "I\ninput feature map", fc="#e8eef7", fs=10)
box(ax, 5.2, 3.3, 1.6, 0.8, "G\ngeneralization\n(write)", fc="#e8eef7", fs=10)
box(ax, 5.2, 1.2, 3.0, 1.1, "", fc="#f5f5f5")
ax.text(5.2, 1.62, "Memory  m", ha="center", fontsize=11, weight="bold")
for i, lab in enumerate(["m₁", "m₂", "m₃", "…", "m_N"]):
    ax.add_patch(plt.Rectangle((3.95 + i*0.5, 0.82), 0.46, 0.5, fc="white", ec="#444"))
    ax.text(4.18 + i*0.5, 1.07, lab, ha="center", va="center", fontsize=9)
box(ax, 7.5, 3.3, 1.6, 0.8, "O\noutput feature map\n(read + infer)", fc="#e8eef7", fs=10)
box(ax, 9.25, 3.3, 1.1, 0.8, "R\nresponse", fc="#e8eef7", fs=10)
arrow(ax, (1.45, 3.3), (2.1, 3.3))
arrow(ax, (3.7, 3.3), (4.4, 3.3), "I(x)")
arrow(ax, (5.2, 2.9), (5.2, 1.78), "mᵢ = G(mᵢ, I(x), m)", off=(0.95, 0))
arrow(ax, (2.9, 3.72), (7.5, 3.72), "I(x) is also passed to O", off=(0, 0.62), ls="--", rad=-0.22)
ax.text(5.2, 2.78, "", fontsize=8)
arrow(ax, (6.75, 1.4), (7.4, 2.9), "score & select\nsupporting memories", off=(0.95, 0), rad=0.15)
arrow(ax, (8.3, 3.3), (8.7, 3.3))
ax.text(8.5, 3.55, "o", ha="center", fontsize=10, style="italic")
ax.text(9.25, 2.6, "r = R(o)", ha="center", fontsize=10, style="italic")
ax.text(5.2, 4.75, "Read path:  o = O(I(x), m)        Write path:  m_H(x) = I(x)", ha="center", fontsize=10)
plt.tight_layout(); plt.savefig(f"{out}/fig1_memnn_flow.png", dpi=220); plt.close()

# ---------- Figure 2: subgraph ----------
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.set_xlim(-0.2, 10.2); ax.set_ylim(0, 5.4); ax.axis("off")
N = {
  "z":      (5.0, 4.85, "z  (QA context node)\nencodes question + choice text", "#fdf1d6", 4.0),
  "engine": (1.3, 3.2, "engine", "#e8eef7", 1.5),
  "stop":   (1.3, 1.2, "stop\n(engine stops)", "#e8eef7", 1.7),
  "fuel":   (5.0, 3.2, "fuel", "#ffffff", 1.4),
  "nofuel": (5.0, 1.2, "lack of fuel", "#ffffff", 1.8),
  "tank":   (8.7, 3.2, "fuel tank", "#e6f2e6", 1.6),
  "empty":  (8.7, 1.2, "empty", "#e6f2e6", 1.4),
}
for k, (x, y, t, c, w) in N.items():
    box(ax, x, y, w if k != "z" else 3.6, 0.7 if k != "z" else 0.75, t, fc=c, fs=10)
E = [("engine","stop","HasSubevent", (-0.65,0), 0, "-"),
     ("engine","fuel","Uses", (-0.55,0.15), 0, "-"),
     ("tank","fuel","Contains", (0.55,0.15), 0, "-"),
     ("tank","empty","HasState", (0.55,0), 0, "-"),
     ("empty","nofuel","Implies", (0,0.15), 0, "-"),
     ("nofuel","stop","Causes", (0,0.15), 0, "-")]
def edge_pts(a, b):
    xa, ya = N[a][:2]; xb, yb = N[b][:2]
    wa, wb = N[a][4]/2, N[b][4]/2
    if abs(ya - yb) < 1e-6:
        s = 1 if xb > xa else -1
        return (xa + s*wa, ya), (xb - s*wb, yb)
    s = 1 if yb > ya else -1
    return (xa, ya + s*0.35), (xb, yb - s*0.35)
for a, b, lab, off, rad, ls in E:
    p, q = edge_pts(a, b)
    arrow(ax, p, q, lab, off=off)
# context links (dashed)
dash = dict(style="-", ls=(0,(3,3)), color="#999999")
arrow(ax, (3.4, 4.47), (1.3, 3.56), None, **dash)
arrow(ax, (6.6, 4.47), (8.7, 3.56), None, **dash)
arrow(ax, (4.2, 4.47), (2.0, 1.56), None, **dash)
arrow(ax, (5.8, 4.47), (8.0, 1.56), None, **dash)
ax.text(0.9, 4.1, "question entities", fontsize=9, color="#555", style="italic")
ax.text(8.0, 4.1, "answer entities", fontsize=9, color="#555", style="italic")
# reasoning path highlight
ax.annotate("", xy=(1.3, 0.62), xytext=(8.7, 0.62),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#b03a2e"))
ax.text(5.0, 0.32, "reasoning path:  empty fuel tank → lack of fuel → engine stops", ha="center",
        fontsize=10, color="#b03a2e")
plt.tight_layout(); plt.savefig(f"{out}/fig2_subgraph.png", dpi=220); plt.close()
print("done")
