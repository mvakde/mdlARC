"""Keyed kaleidoscope wedges.

Parallel strips (separated by one-cell background dividers) contain one dense
two-colour template tile.  The tile anchors a reflection lattice: strip
dividers and the template's longitudinal edges act as mirror seams, so every
macro cell (strip offset dr, block offset dc) holds the template reflected
across rows when dr is odd and across columns when dc is odd.  Only macro
cells inside the cue-anchored diagonal wedge |dc| >= |dr| are painted.  Each
non-cue strip is recoloured independently: sparse key cells, read back
through the reflection lattice, say which template role receives which
colour; an unkeyed role stays the strip's fill.  Background holes persist.
"""
from __future__ import annotations

import json

import numpy as np

MAX_SIDE = 30


# --------------------------------------------------------------------------
# counts


def _resolve_counts(num_train, num_test):
    if num_train is None:
        num_train = 4
    if num_test is None:
        num_test = 1
    for value, name in ((num_train, "num_train"), (num_test, "num_test")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


# --------------------------------------------------------------------------
# shared lattice geometry (canonical orientation: horizontal strips)


def _lattice(x, x0, width, dr):
    """Block offset and template column for interior column x."""
    d = x - x0
    dc = d // width
    u = d % width
    tc = u if dc % 2 == 0 else width - 1 - u
    return dc, tc


def _template_row(a, height, dr):
    return a if dr % 2 == 0 else height - 1 - a


def _selected(dc, dr, mode="ge"):
    if mode == "ge":
        return abs(dc) >= abs(dr)
    if mode == "gt":
        return abs(dc) > abs(dr) or dr == 0
    return True  # "all": ignore the wedge


# --------------------------------------------------------------------------
# input-only parsing


class ParseError(ValueError):
    pass


def _parse(grid):
    """Recover every structural element from an input grid alone."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) < 5:
        raise ParseError("grid too small")
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg) and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ParseError("no uniform border")
    rows = [r for r in range(1, g.shape[0] - 1) if np.all(g[r] == bg)]
    cols = [c for c in range(1, g.shape[1] - 1) if np.all(g[:, c] == bg)]
    if rows and not cols:
        vertical = False
        c = g
    elif cols and not rows:
        vertical = True
        c = g.T
    else:
        raise ParseError("cannot determine strip orientation")
    R, C = c.shape
    dividers = [0] + [r for r in range(1, R - 1) if np.all(c[r] == bg)] + [R - 1]
    strips = []
    for top, bottom in zip(dividers[:-1], dividers[1:]):
        if bottom - top - 1 <= 0:
            raise ParseError("adjacent dividers")
        strips.append(top + 1)
    heights = {b - t - 1 for t, b in zip(dividers[:-1], dividers[1:])}
    if len(heights) != 1:
        raise ParseError("unequal strips")
    H = heights.pop()
    S = len(strips)
    L = C - 2
    if S < 3:
        raise ParseError("need at least three strips")
    body = np.concatenate([c[s:s + H, 1:C - 1] for s in strips], axis=0)
    vals, counts = np.unique(body[body != bg], return_counts=True)
    if len(vals) == 0:
        raise ParseError("empty strips")
    fill = int(vals[np.argmax(counts)])
    if np.sum(counts == counts.max()) > 1:
        raise ParseError("ambiguous fill")
    # template: the unique run of interior columns, inside one strip, whose
    # every cell is neither fill nor background.
    runs = []
    for si, top in enumerate(strips):
        block = c[top:top + H, 1:C - 1]
        full = np.all((block != fill) & (block != bg), axis=0)
        x = 0
        while x < L:
            if full[x]:
                start = x
                while x < L and full[x]:
                    x += 1
                runs.append((si, start, x - start))
            else:
                x += 1
    if len(runs) != 1:
        raise ParseError(f"expected one template run, found {len(runs)}")
    s0, x0, W = runs[0]
    T = c[strips[s0]:strips[s0] + H, 1 + x0:1 + x0 + W].copy()
    tcolors = sorted(int(v) for v in np.unique(T))
    if len(tcolors) != 2:
        raise ParseError("template must use exactly two colours")
    # keys: non-fill, non-background cells outside the template strip
    keys = []
    for si, top in enumerate(strips):
        dr = si - s0
        for a in range(H):
            for x in range(L):
                v = int(c[top + a, 1 + x])
                if v in (fill, bg):
                    continue
                if si == s0:
                    if x0 <= x < x0 + W:
                        continue
                    raise ParseError("stray colour in cue strip")
                dc, tc = _lattice(x, x0, W, dr)
                keys.append({"strip": si, "dr": dr, "a": a, "x": x, "dc": dc, "color": v,
                             "role": int(T[_template_row(a, H, dr), tc]),
                             "translated_role": int(T[a, (x - x0) % W])})
    holes = [(si, a, x) for si, top in enumerate(strips) for a in range(H) for x in range(L)
             if c[top + a, 1 + x] == bg]
    return {"canon": c, "vertical": vertical, "bg": bg, "fill": fill, "strips": strips, "H": H,
            "S": S, "L": L, "s0": s0, "x0": x0, "W": W, "T": T, "tcolors": tcolors,
            "keys": keys, "holes": holes}


def _mappings(info, use_reflection=True):
    maps = {}
    for k in info["keys"]:
        role = k["role"] if use_reflection else k["translated_role"]
        m = maps.setdefault(k["strip"], {})
        if m.get(role, k["color"]) != k["color"]:
            raise ParseError("contradictory keys")
        m[role] = k["color"]
    for si, m in maps.items():
        if len(m) == 2 and len(set(m.values())) == 1:
            raise ParseError("both roles keyed with one colour")
    return maps


def _render(info, *, wedge="ge", reflect=True, recolor=True, unkeyed="fill"):
    c = info["canon"].copy()
    H, W, x0, s0, T = info["H"], info["W"], info["x0"], info["s0"], info["T"]
    maps = _mappings(info, use_reflection=reflect)
    default = info["fill"] if unkeyed == "fill" else info["bg"]
    for si, top in enumerate(info["strips"]):
        dr = si - s0
        m = maps.get(si, {})
        for x in range(info["L"]):
            if reflect:
                dc, tc = _lattice(x, x0, W, dr)
            else:
                dc, tc = (x - x0) // W, (x - x0) % W
            if not _selected(dc, dr, wedge):
                continue
            for a in range(H):
                if c[top + a, 1 + x] == info["bg"]:
                    continue  # holes persist
                tr = _template_row(a, H, dr) if reflect else a
                role = int(T[tr, tc])
                if dr == 0 or not recolor:
                    c[top + a, 1 + x] = role
                else:
                    c[top + a, 1 + x] = m.get(role, default)
    return c.T if info["vertical"] else c


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _render(info).astype(int)


# --------------------------------------------------------------------------
# generation


def _make_template(rng, H, W):
    for _ in range(400):
        style = rng.choice(["random", "lshape", "cross", "diag", "blocky"])
        T = np.zeros((H, W), dtype=int)
        if style == "random":
            p = rng.uniform(0.25, 0.7)
            T = (rng.random((H, W)) < p).astype(int)
        elif style == "lshape":
            r, col = int(rng.integers(H)), int(rng.integers(W))
            T[r, :] = 1
            T[:, col] = 1
            T[int(rng.integers(H)), int(rng.integers(W))] ^= 1
        elif style == "cross":
            r = int(rng.integers(H))
            T[r, : int(rng.integers(1, W + 1))] = 1
            T[: int(rng.integers(1, H + 1)), int(rng.integers(W))] = 1
        elif style == "diag":
            off = int(rng.integers(-1, 2))
            for i in range(H):
                for j in range(W):
                    if j - i == off or j - i == off + 1 and rng.random() < 0.5:
                        T[i, j] = 1
        else:
            h, w = int(rng.integers(1, H)), int(rng.integers(1, W))
            r, col = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
            T[r:r + h, col:col + w] = 1
        if rng.random() < 0.5:
            T = 1 - T
        n1 = int(T.sum())
        if n1 == 0 or n1 == H * W:
            continue
        if np.array_equal(T, T[::-1, :]) or np.array_equal(T, T[:, ::-1]):
            continue
        return T
    raise RuntimeError("template sampling failed")


def _geometry(rng, tier):
    H = int(rng.choice([2, 2, 3, 3, 4]))
    W = int(rng.choice([2, 3, 3, 4, 4, 5]))
    max_s = (MAX_SIDE - 1) // (H + 1)
    if tier == "expanded":
        S = int(rng.integers(min(5, max_s), max_s + 1))
        L = int(rng.integers(max(16, 3 * W + 2), 29))
    else:
        S = int(rng.integers(3, min(6, max_s) + 1))
        L = int(rng.integers(max(9, 3 * W + 1), min(22, 28) + 1))
    s0 = int(rng.integers(S))
    if rng.random() < 0.5 and S >= 3:
        s0 = int(rng.integers(1, S - 1))  # interior cue more often: two vertical wings
    x0 = int(rng.integers(0, L - W + 1))
    return H, W, S, L, s0, x0


def _palette(rng, fixed=None):
    if fixed is not None:
        return fixed
    cols = [int(v) for v in rng.permutation(10)]
    return {"bg": cols[0], "fill": cols[1], "c0": cols[2], "c1": cols[3]}


def _attempt_pair(rng, *, tier, vertical, pal, holes, want):
    H, W, S, L, s0, x0 = _geometry(rng, tier)
    bg, fill = pal["bg"], pal["fill"]
    roles = _make_template(rng, H, W)
    T = np.where(roles == 1, pal["c1"], pal["c0"])
    tcol = [pal["c0"], pal["c1"]]
    R, C = 1 + S * (H + 1), L + 2
    if R > MAX_SIDE or C > MAX_SIDE:
        return None
    inp = np.full((R, C), bg, dtype=int)
    strips = [1 + s * (H + 1) for s in range(S)]
    for top in strips:
        inp[top:top + H, 1:C - 1] = fill
    inp[strips[s0]:strips[s0] + H, 1 + x0:1 + x0 + W] = T
    pool = [v for v in range(10) if v not in (bg, fill)]
    out = inp.copy()
    keyed_strips = 0
    strip_records = []
    for si, top in enumerate(strips):
        dr = si - s0
        cells = []
        for x in range(L):
            dc, tc = _lattice(x, x0, W, dr)
            if not _selected(dc, dr):
                continue
            for a in range(H):
                tr = _template_row(a, H, dr)
                cells.append((a, x, dc, int(T[tr, tc]), int(T[a, (x - x0) % W])))
        if dr == 0:
            for a, x, dc, role, _ in cells:
                out[top + a, 1 + x] = role
            continue
        if not cells:
            continue
        keyed_strips += 1
        present = sorted({cell[3] for cell in cells})
        if len(present) == 2 and rng.random() < (0.55 if want.get("one") else 0.5):
            kind = "one"
        elif len(present) == 2:
            kind = "both"
        else:
            kind = "one"
        if kind == "both":
            m0, m1 = (int(v) for v in rng.choice(pool, size=2, replace=False))
            if rng.random() < 0.15:
                m0, m1 = tcol[1], tcol[0]  # role swap
            mapping = {tcol[0]: m0, tcol[1]: m1}
            keyed_roles = present
        else:
            role = int(rng.choice(present))
            mapping = {role: int(rng.choice(pool))}
            keyed_roles = [role]
        wings = {"left": [c for c in cells if c[2] < 0], "right": [c for c in cells if c[2] > 0]}
        one_wing = None
        if wings["left"] and wings["right"] and rng.random() < (0.5 if want.get("onewing") else 0.3):
            one_wing = str(rng.choice(["left", "right"]))
        keys = []
        for role in keyed_roles:
            choices = [c for c in (wings[one_wing] if one_wing else cells) if c[3] == role]
            if not choices:
                return None
            sensitive = [c for c in choices if c[4] != c[3]]
            n = int(rng.choice([1, 1, 1, 2, 2, 3]))
            picks = []
            for _ in range(n):
                src = sensitive if sensitive and rng.random() < 0.6 else choices
                picks.append(src[int(rng.integers(len(src)))])
            for p in picks:
                if (p[0], p[1]) not in [(k[0], k[1]) for k in keys]:
                    keys.append((p[0], p[1], mapping[role]))
        # never let keys fill a full cross-section (it would mimic a template)
        colcount = {}
        for a, x, _ in keys:
            colcount[x] = colcount.get(x, 0) + 1
        if any(n >= H for n in colcount.values()):
            return None
        for a, x, v in keys:
            inp[top + a, 1 + x] = v
        for a, x, dc, role, _ in cells:
            out[top + a, 1 + x] = mapping.get(role, fill)
        strip_records.append((dr, kind, one_wing))
    if keyed_strips == 0:
        return None
    if holes:
        free = [(top + a, 1 + x) for si, top in enumerate(strips) for a in range(H) for x in range(L)
                if inp[top + a, 1 + x] == fill]
        frac = rng.uniform(0.03, 0.08)
        n = max(3, int(round(frac * len(free))))
        idx = rng.choice(len(free), size=min(n, len(free)), replace=False)
        for i in idx:
            r, cc = free[int(i)]
            inp[r, cc] = bg
            out[r, cc] = bg
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return inp, out


def _pair_facts(inp, out):
    """Facts derived only from the returned grids (used by validation)."""
    info = _parse(inp)
    maps = _mappings(info)
    pred = _render(info)
    if not np.array_equal(pred, out):
        raise AssertionError("solve disagrees with output")
    H, W, x0, s0 = info["H"], info["W"], info["x0"], info["s0"]
    T = info["T"]
    if np.array_equal(T, T[::-1, :]) or np.array_equal(T, T[:, ::-1]):
        raise AssertionError("template must be asymmetric under both reflections")
    hole_set = set(info["holes"])
    facts = {"vertical": info["vertical"], "one": False, "both": False, "onewing": False,
             "deep": False, "sides": set(), "sensitive": False, "nonidentity": False,
             "holes": len(info["holes"]) > 0, "hole_selected": False, "keyed_dr": set(),
             "H": H, "W": W, "S": info["S"], "L": info["L"], "max_dr": 0}
    for si, top in enumerate(info["strips"]):
        dr = si - s0
        sel_roles, wings = set(), set()
        for x in range(info["L"]):
            dc, tc = _lattice(x, x0, W, dr)
            if not _selected(dc, dr):
                continue
            for a in range(H):
                if (si, a, x) in hole_set:
                    facts["hole_selected"] = True
                    continue
                sel_roles.add(int(T[_template_row(a, H, dr), tc]))
                if dr != 0:
                    wings.add("L" if dc < 0 else "R")
        if dr == 0 or not sel_roles:
            if dr != 0 and si in maps:
                raise AssertionError("key in a strip without wedge cells")
            continue
        if si not in maps:
            raise AssertionError("wedge strip without keys")
        m = maps[si]
        if any(r not in sel_roles for r in m):
            raise AssertionError("keyed role not visible in wedge")
        keys = [k for k in info["keys"] if k["strip"] == si]
        if any(not _selected(k["dc"], dr) for k in keys):
            raise AssertionError("key outside wedge")
        facts["keyed_dr"].add(dr)
        facts["max_dr"] = max(facts["max_dr"], abs(dr))
        facts["sides"].add(np.sign(dr))
        if len(sel_roles) == 2 and len(m) == 1:
            facts["one"] = True
        if len(m) == 2:
            facts["both"] = True
        if any(m.get(r, info["fill"]) != r for r in sel_roles):
            facts["nonidentity"] = True
        key_wings = {"L" if k["dc"] < 0 else "R" for k in keys}
        if len(wings) == 2 and len(key_wings) == 1:
            facts["onewing"] = True
        if abs(dr) >= 2:
            facts["deep"] = True
        if any(k["role"] != k["translated_role"] for k in keys):
            facts["sensitive"] = True
    # shortcut divergence: each simpler rule must change this pair's answer
    for kwargs in ({"wedge": "gt"}, {"wedge": "all"}, {"reflect": False}, {"recolor": False}):
        try:
            alt = _render(info, **kwargs)
        except ParseError:
            continue
        if np.array_equal(alt, out):
            raise AssertionError(f"shortcut {kwargs} reproduces the output")
    if facts["one"]:
        facts["bgdefault_differs"] = not np.array_equal(_render(info, unkeyed="bg"), out)
    else:
        facts["bgdefault_differs"] = False
    return facts


def _train_requirements(train_facts, test_facts):
    problems = []
    if not any(f["one"] for f in train_facts):
        problems.append("no one-key strip in training")
    if not any(f["both"] for f in train_facts):
        problems.append("no two-key strip in training")
    if not any(f["onewing"] for f in train_facts):
        problems.append("no single-wing keyed strip in training")
    if not any(f["deep"] for f in train_facts):
        problems.append("no |dr|>=2 strip in training")
    if not any(len(f["sides"]) == 2 for f in train_facts):
        problems.append("no two-sided strip wedge in training")
    if not any(f["bgdefault_differs"] for f in train_facts):
        problems.append("unkeyed-fill convention not distinguished from background")
    orients = {f["vertical"] for f in train_facts}
    if any(f["vertical"] not in orients for f in test_facts):
        problems.append("test orientation unseen in training")
    if any(f["holes"] for f in test_facts) and not any(f["hole_selected"] for f in train_facts):
        problems.append("hole preservation not demonstrated")
    for f in train_facts + test_facts:
        if not f["sensitive"] or not f["nonidentity"]:
            problems.append("pair lacks reflection-sensitive key or recolouring")
            break
    return problems


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    orient_mode = str(rng.choice(["h", "v", "mixed", "mixed"]))
    holes = bool(rng.random() < 0.45)
    fixed = _palette(rng) if rng.random() < 0.4 else None
    for _ in range(300):
        pairs, facts, seen = [], [], set()
        ok = True
        for i in range(num_train + num_test):
            is_test = i >= num_train
            if orient_mode == "mixed":
                vertical = bool(rng.random() < 0.5) if is_test or i >= 2 else (i == 1) ^ bool(seed % 2)
            else:
                vertical = orient_mode == "v"
            tier = ("expanded" if rng.random() < 0.6 else "familiar") if is_test else \
                ("familiar" if rng.random() < 0.8 else "expanded")
            want = {"one": i % 2 == 0, "onewing": i % 3 == 1}
            made = None
            for _ in range(60):
                made = _attempt_pair(rng, tier=tier, vertical=vertical, pal=_palette(rng, fixed),
                                     holes=holes, want=want)
                if made is None:
                    continue
                key = json.dumps(made[0].tolist())
                if key in seen:
                    made = None
                    continue
                try:
                    f = _pair_facts(*made)
                except (AssertionError, ParseError):
                    made = None
                    continue
                if not (f["sensitive"] and f["nonidentity"]):
                    made = None
                    continue
                break
            if made is None:
                ok = False
                break
            seen.add(json.dumps(made[0].tolist()))
            pairs.append({"input": made[0], "output": made[1]})
            facts.append(f)
        if not ok:
            continue
        if _train_requirements(facts[:num_train], facts[num_train:]):
            continue
        return {"train": pairs[:num_train], "test": pairs[num_train:]}
    raise RuntimeError("could not build a valid puzzle")


# --------------------------------------------------------------------------
# validation


def validate_generated_puzzle(puzzle: dict) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have train/test")
    train, test = puzzle["train"], puzzle["test"]
    _resolve_counts(len(train), len(test))
    seen = set()
    facts = []
    for pair in train + test:
        if set(pair) != {"input", "output"}:
            raise AssertionError("pair keys")
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > MAX_SIDE:
            raise AssertionError("bad shape")
        if inp.min() < 0 or inp.max() > 9 or out.min() < 0 or out.max() > 9:
            raise AssertionError("bad colours")
        key = json.dumps(inp.tolist())
        if key in seen:
            raise AssertionError("duplicate input")
        seen.add(key)
        if np.array_equal(inp, out):
            raise AssertionError("identity pair")
        facts.append(_pair_facts(inp, out))
    problems = _train_requirements(facts[:len(train)], facts[len(train):])
    if problems:
        raise AssertionError("; ".join(problems))


def self_test() -> None:
    stats = {"vertical": set(), "H": set(), "W": set(), "S": set(), "holes": set(),
             "test_deeper": 0, "test_onewing": 0, "test_one": 0}
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(json.dumps({k: [{r: np.asarray(v).tolist() for r, v in q.items()} for q in p[k]]
                                   for k in p}))
        tr = [_pair_facts(np.asarray(q["input"]), np.asarray(q["output"])) for q in p["train"]]
        te = [_pair_facts(np.asarray(q["input"]), np.asarray(q["output"])) for q in p["test"]]
        for f in tr + te:
            stats["vertical"].add(f["vertical"])
            stats["H"].add(f["H"])
            stats["W"].add(f["W"])
            stats["S"].add(f["S"])
            stats["holes"].add(f["holes"])
        if te[0]["max_dr"] > max(f["max_dr"] for f in tr):
            stats["test_deeper"] += 1
        stats["test_onewing"] += te[0]["onewing"]
        stats["test_one"] += te[0]["one"]
    assert len(signatures) == 42
    assert stats["vertical"] == {True, False}
    assert stats["H"] >= {2, 3, 4} and stats["W"] >= {2, 3, 4, 5}
    assert len(stats["S"]) >= 4 and stats["holes"] == {True, False}
    assert stats["test_deeper"] >= 3 and stats["test_onewing"] >= 3 and stats["test_one"] >= 5
    # determinism and count handling
    a, b = generate_puzzle(7, 3, 2), generate_puzzle(7, 3, 2)
    assert all(np.array_equal(x[r], y[r]) for k in a for x, y in zip(a[k], b[k]) for r in x)
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted illegal counts {bad}")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
