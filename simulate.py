"""How much of 'AI 83% vs doctors 36%' is the contest rules?

A toy diagnosis game, calibrated to the published numbers and then used to change one rule at a time.

World
  2,000 diseases with Zipf-like frequency (common colds at the top, zebras in the tail), spread across 12 specialties.
  Each player has a probability of recognising each disease if a case of it lands on their desk.
    generalist:   p = sigmoid(a + b * log10(frequency rank))  -> high for common diseases, low for rare ones
    + references: a lookup rescues a share r of the cases the generalist would otherwise miss
    specialist:   inside their own specialty, recognition of rare diseases is much higher (sigmoid with a better curve)
    AI:           broad, almost flat recognition with a mild drop for rare diseases
  A player who does not recognise the disease names a wrong diagnosis drawn from 3 shared look-alikes for that case,
  so the errors of different doctors are correlated (they fall for the same red herrings).
  Groups decide by plurality vote, ties broken at random (as in collective-intelligence studies).

Case sets
  "NEJM-style": cases drawn from the rare tail, then kept with probability proportional to how hard they are for a
               generalist (journals pick cases that stump people).
  "Everyday":   cases drawn in proportion to how common each disease is.

Calibration targets (from reporting on the Science paper): solo generalist without references 22.3%,
with references 36.1%, MAI-DxO 83.2% on NEJM cases. Validation target (Barnett et al. 2019): individual 62.5% ->
groups of nine 85.6%.
"""
import json
import numpy as np

rng0 = np.random.default_rng(0)
ND, NSPEC = 2000, 12
rank = np.arange(1, ND + 1)
freq = 1 / rank ** 1.05; freq /= freq.sum()
spec_of = rng0.integers(0, NSPEC, ND)
L = np.log10(rank)                                    # 0 for the most common disease, ~3.3 for the rarest

def sig(x): return 1 / (1 + np.exp(-x))

def make_cases(kind, n, rng, p_gen=None):
    if kind == "everyday":
        return rng.choice(ND, n, p=freq)
    # NEJM-style: rare tail (ranks > 200), selected for difficulty
    tail = np.arange(200, ND)
    w = (1 - p_gen[tail]); w = w / w.sum()
    return rng.choice(tail, n, p=w)

SIGMA = 0.0             # spread of shared case difficulty (logit units), set by calibrate_persuasion()
PERSUADE = (0.5, 1.0)   # (c, g) set by calibrate_persuasion(): if the vote is wrong, the people who recognised the
                        # disease win the discussion with probability c * (share of group who recognised it) ** g

def play(cases, P_list, rng):
    """P_list: one recognition-probability vector per group member. Returns share of correct group diagnoses.
    Groups first vote; if the vote is wrong but someone recognised the disease, they persuade the group with
    probability PERSUADE (calibrated so pairs match Barnett et al.'s 62.5% -> 75.1%)."""
    n = len(cases); k = len(P_list)
    votes = np.empty((n, k), dtype=int)               # 0 = correct, 1..3 = shared look-alikes
    lure = rng.dirichlet([2, 1, 1], n)                # how tempting each look-alike is for each case
    u = rng.normal(0, SIGMA, n)                       # shared case-level difficulty: doctors struggle on the same cases
    for j, P in enumerate(P_list):
        p = np.clip(P[cases], 1e-6, 1 - 1e-6)
        ok = rng.random(n) < sig(np.log(p / (1 - p)) + u)
        wrong = 1 + (rng.random(n)[:, None] > np.cumsum(lure, 1)).sum(1)
        votes[:, j] = np.where(ok, 0, np.minimum(wrong, 3))
    counts = np.stack([(votes == v).sum(1) for v in range(4)], 1).astype(float)
    counts += rng.random(counts.shape) * 0.1          # random tie-break
    right = counts.argmax(1) == 0
    if k > 1:
        share = (votes == 0).mean(1)
        c, g = PERSUADE
        right = right | ((share > 0) & (rng.random(n) < c * share ** g))
    return float(right.mean())

def players(a, b, r, ai_hi, ai_slope, spec_boost):
    gen = sig(a + b * L)
    ref = 1 - (1 - gen) * (1 - r)
    ai = np.clip(ai_hi - ai_slope * L, 0, 1)
    spec = np.empty((NSPEC, ND))
    for s in range(NSPEC):
        inside = spec_of == s
        spec[s] = np.where(inside, sig(a + spec_boost + b * 0.55 * L), gen)
        spec[s] = 1 - (1 - spec[s]) * (1 - r)         # specialists also have their references
    return gen, ref, ai, spec

def calibrate():
    rng = np.random.default_rng(1)
    best = None
    for a in np.arange(3.0, 6.01, 0.25):
        for b in np.arange(-3.0, -1.49, 0.1):
            gen = sig(a + b * L)
            cases = make_cases("nejm", 20000, rng, gen)
            solo = gen[cases].mean()
            if abs(solo - 0.223) > 0.02:
                continue
            # Barnett validation on a broad case mix where individuals score ~62.5%: handled separately below
            for r in np.arange(0.05, 0.4, 0.01):
                ref = 1 - (1 - gen) * (1 - r)
                withref = ref[cases].mean()
                err = abs(solo - 0.223) + abs(withref - 0.361)
                if best is None or err < best[0]:
                    best = (err, a, b, r)
    _, a, b, r = best
    gen = sig(a + b * L)
    cases = make_cases("nejm", 20000, rng, gen)
    best_ai = None
    for hi in np.arange(0.80, 1.0, 0.005):
        ai = np.clip(hi - 0.03 * L, 0, 1)
        e = abs(ai[cases].mean() - 0.832)
        if best_ai is None or e < best_ai[0]:
            best_ai = (e, hi)
    return dict(a=float(a), b=float(b), r=float(r), ai_hi=float(best_ai[1]), ai_slope=0.03)

def scenario_table(prm, spec_boost=3.0, n=40000, seed=7):
    rng = np.random.default_rng(seed)
    gen, ref, ai, spec = players(prm["a"], prm["b"], prm["r"], prm["ai_hi"], prm["ai_slope"], spec_boost)
    out = {}
    for kind in ("nejm", "everyday"):
        cases = make_cases(kind, n, rng, gen)
        right_spec = spec[spec_of[cases], :]          # each case routed to the matching specialist
        S = {}
        S["Solo generalist, no references"] = play(cases, [gen], rng)
        S["Solo generalist, with references"] = play(cases, [ref], rng)
        S["Two generalists, with references"] = play(cases, [ref, ref], rng)
        S["Generalist + matching specialist + 1 colleague"] = float(np.mean([
            play(cases, [ref, right_spec[np.arange(n), cases], ref], rng)]))
        S["Team of nine generalists"] = play(cases, [ref] * 9, rng)
        S["Specialist-led team of five"] = play(cases, [right_spec[np.arange(n), cases]] * 2 + [ref] * 3, rng)
        S["AI system alone"] = play(cases, [ai], rng)
        out[kind] = S
    return out

def barnett_cases(prm, n, rng):
    gen = sig(prm["a"] + prm["b"] * L)
    ref = 1 - (1 - gen) * (1 - prm["r"])
    for lo in range(1, 400, 2):
        cases = rng.integers(lo, lo * 8, n).clip(0, ND - 1)
        if ref[cases].mean() <= 0.625:
            return cases, ref, lo

def calibrate_persuasion(prm):
    global PERSUADE, SIGMA
    rng = np.random.default_rng(5)
    cases, ref, _ = barnett_cases(prm, 20000, rng)
    best = None
    for sd in np.arange(0.0, 3.01, 0.25):
        for c in np.arange(0.2, 1.01, 0.05):
            SIGMA, PERSUADE = sd, (c, 1.0)
            one = play(cases, [ref], rng)
            e = abs(one - 0.625) + abs(play(cases, [ref, ref], rng) - 0.751) + abs(play(cases, [ref] * 9, rng) - 0.856)
            if best is None or e < best[0]:
                best = (e, float(sd), float(c))
    SIGMA, PERSUADE = best[1], (best[2], 1.0)
    return {"sigma": SIGMA, "persuade": PERSUADE[0]}

def barnett_check(prm, n=40000, seed=3):
    """Do our pooling rules reproduce 62.5% -> 85.6% for groups of nine on a broad case mix?"""
    rng = np.random.default_rng(seed)
    cases, ref, lo = barnett_cases(prm, n, rng)
    res = {k: play(cases, [ref] * k, rng) for k in (1, 2, 3, 5, 9)}
    return {"case_ranks": [lo, lo * 8], "accuracy_by_group_size": res}

def spec_boost_sweep(prm):
    out = {}
    for sb in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        t = scenario_table(prm, spec_boost=sb, n=20000)
        out[str(sb)] = {"team": t["nejm"]["Specialist-led team of five"],
                        "gp_spec_colleague": t["nejm"]["Generalist + matching specialist + 1 colleague"]}
    return out

if __name__ == "__main__":
    prm = calibrate()
    prm.update(calibrate_persuasion(prm))
    print("calibrated:", prm)
    T = scenario_table(prm)
    for kind, S in T.items():
        print(kind); [print(f"  {k:48s} {100*v:5.1f}%") for k, v in S.items()]
    B = barnett_check(prm); print("Barnett check:", B)
    SB = spec_boost_sweep(prm); print("specialist strength sweep:", SB)
    gen, ref, ai, spec = players(prm["a"], prm["b"], prm["r"], prm["ai_hi"], prm["ai_slope"], 3.0)
    curve = {"log10_rank": L[::10].tolist(), "generalist": gen[::10].tolist(), "with_refs": ref[::10].tolist(),
             "ai": ai[::10].tolist(), "specialist_in_field": (spec[0][spec_of == 0][::2]).tolist(),
             "specialist_ranks": (L[spec_of == 0][::2]).tolist()}
    rngh = np.random.default_rng(11)
    hist = {k: np.histogram(L[make_cases(k, 50000, rngh, gen)], bins=np.linspace(0, 3.31, 34))[0].tolist() for k in ("nejm", "everyday")}
    curve["hist_edges"] = np.linspace(0, 3.31, 34).tolist(); curve["hist"] = hist
    json.dump({"params": prm, "scenarios": T, "barnett_check": B, "specialist_sweep": SB, "curves": curve},
              open("results/results.json", "w"), indent=1)
