"""Draw the article's figures from results/results.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": BG, "axes.facecolor": BG})

def title(fig, t, sub):
    fig.text(0.02, 0.965, t, ha="left", va="top", fontsize=15, color=INK, weight="bold")
    fig.text(0.02, 0.905, sub, ha="left", va="top", fontsize=11, color=INK2)

# ---------- 1: who recognises what ----------
C = R["curves"]; x = np.array(C["log10_rank"])
fig, ax = plt.subplots(figsize=(11, 5.6))
e = np.array(C["hist_edges"]); mid = (e[:-1] + e[1:]) / 2; w = e[1] - e[0]
for k, c, lab in (("everyday", INK2, "everyday clinic cases"), ("nejm", ORANGE, "NEJM-style puzzle cases")):
    h = np.array(C["hist"][k], float); h = h / h.max() * 28
    ax.bar(mid, h, width=w * 0.92, color=c, alpha=0.18, lw=0)
    if k == "everyday": ax.text(0.02, 31, lab, ha="left", fontsize=10, color=c)
    else: ax.text(3.32, 31, lab, ha="right", fontsize=10, color=c)
ax.plot(x, 100 * np.array(C["generalist"]), color=BLUE, lw=2.2, ls="--", label="Generalist, no references")
ax.plot(x, 100 * np.array(C["with_refs"]), color=BLUE, lw=2.2, label="Generalist with references")
ax.plot(x, 100 * np.array(C["ai"]), color=ORANGE, lw=2.2, label="AI system")
ax.scatter(C["specialist_ranks"], 100 * np.array(C["specialist_in_field"]), s=9, color=AQUA, alpha=0.6, lw=0, label="Specialist, inside own field")
ax.set_xticks([0, 1, 2, 3], ["most common\ndisease", "#10", "#100", "#1,000"])
ax.set_xlim(-0.05, 3.35); ax.set_ylim(0, 105)
ax.set_xlabel("How common the disease is (frequency rank, log scale)"); ax.set_ylabel("Chance of recognising it (%)")
ax.legend(frameon=False, fontsize=10, loc="lower left", bbox_to_anchor=(0.27, 0.2)); ax.grid(axis="y", color=GRID)
title(fig, "Puzzle cases live exactly where a lone generalist is weakest",
      "The model's recognition curves (calibrated to the Science study), with where each type of case comes from (shaded)")
fig.subplots_adjust(top=0.84, left=0.08, right=0.97, bottom=0.17)
fig.savefig("charts/01_recognition_curves.png", dpi=200); plt.close(fig)

# ---------- 2: change one rule at a time (NEJM-style cases) ----------
S = R["scenarios"]["nejm"]; SW = R["specialist_sweep"]
rows = [("Solo generalist, no references\n(the contest rules)", S["Solo generalist, no references"], None, BLUE),
        ("Solo generalist, with references\n(the contest rules)", S["Solo generalist, with references"], None, BLUE),
        ("+ one colleague to talk it through", S["Two generalists, with references"], None, BLUE),
        ("Team of nine generalists", S["Team of nine generalists"], None, BLUE),
        ("Generalist + matching specialist\n+ one colleague", S["Generalist + matching specialist + 1 colleague"],
         (min(v["gp_spec_colleague"] for v in SW.values()), max(v["gp_spec_colleague"] for v in SW.values())), AQUA),
        ("Specialist-led team of five", S["Specialist-led team of five"],
         (min(v["team"] for v in SW.values()), max(v["team"] for v in SW.values())), AQUA)]
ai = S["AI system alone"]
fig, ax = plt.subplots(figsize=(11, 6.2))
for i, (lab, v, rg, c) in enumerate(rows):
    ax.barh(i, 100 * v, color=c, height=0.6)
    if rg:
        ax.plot([100 * rg[0], 100 * rg[1]], [i, i], color=INK, lw=1.6)
        for xx in rg: ax.plot([100 * xx] * 2, [i - .12, i + .12], color=INK, lw=1.6)
    ax.text(100 * (rg[1] if rg else v) + 1.2, i, f"{100 * v:.0f}%", va="center", fontsize=11.5, color=INK)
ax.axvline(100 * ai, color=ORANGE, lw=2.2)
ax.text(100 * ai, -0.75, f"AI alone {100 * ai:.0f}%", color=ORANGE, ha="center", fontsize=11, weight="bold")
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10.5); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_ylim(len(rows) - 0.4, -1.1)
ax.set_xlabel("Correct diagnoses on NEJM-style puzzle cases (%)"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
title(fig, "Change the rules and the 'superhuman' gap mostly closes",
      "Simulated accuracy on rare puzzle cases; whiskers show the range from 'specialists no better than generalists' to 'much better'")
fig.subplots_adjust(top=0.84, left=0.27, right=0.97, bottom=0.1)
fig.savefig("charts/02_rule_changes.png", dpi=200); plt.close(fig)

# ---------- 3: same players, two case mixes ----------
E, N = R["scenarios"]["everyday"], R["scenarios"]["nejm"]
groups = ["Everyday clinic cases", "NEJM-style puzzle cases"]
doc = [E["Solo generalist, with references"], N["Solo generalist, with references"]]
aiv = [E["AI system alone"], N["AI system alone"]]
fig, ax = plt.subplots(figsize=(11, 5.2))
xs = np.arange(2); bw = 0.34
for off, vals, c, lab in ((-bw / 2 - 0.01, doc, BLUE, "Solo generalist with references"), (bw / 2 + 0.01, aiv, ORANGE, "AI system alone")):
    ax.bar(xs + off, [100 * v for v in vals], width=bw, color=c, label=lab)
    for xi, v in zip(xs + off, vals): ax.text(xi, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=12, color=INK)
for xi, d, a in zip(xs, doc, aiv):
    ax.text(xi, 103, f"gap: {100 * (a - d):.0f} points", ha="center", fontsize=11.5, color=INK, weight="bold")
ax.set_xticks(xs, groups); ax.set_ylim(0, 124); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("Correct diagnoses (%)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10.5); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
title(fig, "Same doctor, same AI: the case mix decides how big the gap looks",
      "Simulated accuracy when cases come in proportion to how common diseases are, versus picked from the rare, hard tail")
fig.subplots_adjust(top=0.84, left=0.08, right=0.97, bottom=0.1)
fig.savefig("charts/03_case_mix.png", dpi=200); plt.close(fig)

# ---------- 4: validation against Barnett et al. ----------
B = R["barnett_check"]["accuracy_by_group_size"]
ks = sorted(int(k) for k in B)
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(ks, [100 * B[str(k)] for k in ks], color=BLUE, lw=2.2, marker="o", ms=8, mec=BG, mew=2, label="This model")
real = {1: 62.5, 2: 75.1, 9: 85.6}
ax.scatter(list(real), list(real.values()), s=120, marker="D", color=ORANGE, zorder=3, label="Barnett et al. 2019 (real doctors)")
for k, v in real.items(): ax.text(k - 0.18, v + 1.2, f"{v}%", color=ORANGE, fontsize=10.5, ha="right")
ax.set_xticks(ks); ax.set_xlabel("Number of doctors pooling their diagnoses"); ax.set_ylabel("Correct diagnoses (%)")
ax.set_ylim(50, 100); ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", color=GRID)
title(fig, "Sanity check: the model's teams behave like real teams of doctors",
      "Accuracy by group size on a broad case mix, compared with 1,572 real cases from the Human Diagnosis Project")
fig.subplots_adjust(top=0.84, left=0.08, right=0.97, bottom=0.12)
fig.savefig("charts/04_validation.png", dpi=200); plt.close(fig)
print("ok")
