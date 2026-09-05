import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "figure.dpi": 300,
})
GREY = "0.45"

# ---------------------------------------------------------------- Fig: training
ep = [1,2,3,4,5]
tr = [0.0939,0.0776,0.0656,0.0562,0.0496]
dv = [0.0867,0.0746,0.0693,0.0677,0.0685]
P  = [0.7713,0.8049,0.8203,0.8318,0.8373]
R  = [0.8308,0.8427,0.8609,0.8672,0.8696]
F  = [0.7999,0.8234,0.8401,0.8491,0.8532]

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
ax[0].plot(ep, tr, "o-", color="black", ms=4, lw=1.2, label="Training")
ax[0].plot(ep, dv, "s--", color=GREY, ms=4, lw=1.2, label="Development")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-entropy loss")
ax[0].set_xticks(ep); ax[0].legend(frameon=False)
ax[0].annotate("minimum", xy=(4, 0.0677), xytext=(3.1, 0.0605),
               fontsize=7, color=GREY,
               arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7))

ax[1].plot(ep, P, "s-",  color=GREY,  ms=4, lw=1.2, label="Precision")
ax[1].plot(ep, R, "^--", color="0.15", ms=4, lw=1.2, label="Recall")
ax[1].plot(ep, F, "o-",  color="black", ms=4, lw=1.6, label=r"$F_1$")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Score")
ax[1].set_xticks(ep); ax[1].legend(frameon=False, loc="lower right")

for a in ax:
    a.grid(True, alpha=0.22, lw=0.5); a.set_axisbelow(True)
    for s in ("top","right"): a.spines[s].set_visible(False)
fig.savefig("training_dynamics.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: F1 vs support
ent = ["PERSON","GPE","ORG","NORP","LANGUAGE","WORK_OF_ART","LOC","LAW","FAC","EVENT","PRODUCT"]
f1  = [0.9045,0.8456,0.7760,0.7603,0.7432,0.8330,0.6443,0.6466,0.5564,0.4222,0.3514]
sup = [25839,8401,8902,1860,456,284,610,59,219,76,252]

fig, a = plt.subplots(figsize=(5.0, 3.0))
a.scatter(sup, f1, s=34, facecolors="none", edgecolors="black", lw=1.0, zorder=3)
ls = np.log10(sup); m, b = np.polyfit(ls, f1, 1)
xs = np.linspace(ls.min()-0.2, ls.max()+0.2, 100)
a.plot(10**xs, m*xs+b, "--", color=GREY, lw=1.0, zorder=2)
off = {"PERSON":(-10,8),"GPE":(7,5),"ORG":(6,-12),"NORP":(8,2),"LANGUAGE":(-16,-14),
       "WORK_OF_ART":(-26,8),"LOC":(8,1),"LAW":(7,3),"FAC":(8,1),"EVENT":(8,1),"PRODUCT":(8,-3)}
for e,x,y in zip(ent,sup,f1):
    a.annotate(e,(x,y),textcoords="offset points",xytext=off[e],fontsize=7)
a.set_xscale("log"); a.set_xlim(40,60000); a.set_ylim(0.28,0.99)
a.set_xlabel("Entity support in test set (log scale)"); a.set_ylabel(r"Entity-level $F_1$")
a.grid(True, alpha=0.22, lw=0.5); a.set_axisbelow(True)
for s in ("top","right"): a.spines[s].set_visible(False)
fig.savefig("f1_vs_support.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: confusions
pairs  = ["GPE $\\rightarrow$ ORG","PERSON $\\rightarrow$ GPE","ORG $\\rightarrow$ GPE",
          "ORG $\\rightarrow$ NORP","GPE $\\rightarrow$ LOC"]
counts = [1280, 961, 628, 163, 13]
fig, a = plt.subplots(figsize=(4.6, 2.1))
y = np.arange(len(pairs))[::-1]
a.barh(y, counts, height=0.6, color="0.78", edgecolor="black", lw=0.7)
for yi, c in zip(y, counts):
    a.text(c+28, yi, f"{c:,}", va="center", fontsize=7.5)
a.set_yticks(y); a.set_yticklabels(pairs)
a.set_xlabel("Misclassified tokens"); a.set_xlim(0, 1500)
a.grid(True, axis="x", alpha=0.22, lw=0.5); a.set_axisbelow(True)
for s in ("top","right","left"): a.spines[s].set_visible(False)
a.tick_params(axis="y", length=0)
fig.savefig("confusions.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: per-entity dBLEU
e2  = ["PERSON","GPE","ORG","NORP","LOC","LANGUAGE","WORK_OF_ART","FAC","PRODUCT","EVENT","LAW"]
n2  = [13754,5041,4606,927,337,246,211,178,136,35,23]
d2  = [0.41,0.23,0.27,0.23,-0.83,-0.54,0.79,1.26,-0.33,-1.09,1.02]
order = np.argsort(n2)[::-1]
e2 = [e2[i] for i in order]; n2 = [n2[i] for i in order]; d2 = [d2[i] for i in order]

fig, a = plt.subplots(figsize=(6.0, 2.7))
x = np.arange(len(e2))
cols = ["0.35" if n >= 900 else "0.82" for n in n2]
a.bar(x, d2, width=0.62, color=cols, edgecolor="black", lw=0.7)
a.axhline(0, color="black", lw=0.8)
a.axhline(0.331, color=GREY, ls=":", lw=1.0)
a.text(len(e2)-0.4, 0.40, "corpus mean $+0.33$", fontsize=7, color=GREY, ha="right")
a.axvline(3.5, color=GREY, ls="--", lw=0.8)
a.text(1.7, -1.30, "adequate support", fontsize=7, color=GREY, ha="center")
a.text(7.4, -1.30, "sparse: differences dominated by sample size",
       fontsize=7, color=GREY, ha="center")
for xi, (d, n) in enumerate(zip(d2, n2)):
    va = "bottom" if d >= 0 else "top"
    a.text(xi, d + (0.055 if d >= 0 else -0.055), f"n={n:,}", ha="center", va=va, fontsize=6.3)
a.set_xticks(x); a.set_xticklabels(e2, rotation=30, ha="right", fontsize=7.5)
a.set_ylabel(r"$\Delta$BLEU (marked $-$ baseline)"); a.set_ylim(-1.55, 1.75)
a.grid(True, axis="y", alpha=0.22, lw=0.5); a.set_axisbelow(True)
for s in ("top","right"): a.spines[s].set_visible(False)
fig.savefig("per_entity_delta.pdf"); plt.close(fig)

print("all figures written")
