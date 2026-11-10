"""Marker-registered assembly, then plus-count selection.

Every input holds several colour "sets".  Each set is scattered as fragments;
every fragment is one marker cell (shared marker colour) plus payload cells of
the set's colour, and is a separate 8-connected island.  Overlaying all
fragments of one colour so that their markers coincide (parent A) rebuilds a
picture made of complete plus signs, single cells and dominoes.  The answer is
the rebuilt picture (marker included, cropped to its bounding box) of the set
whose assembly contains the unique largest number of complete plus signs
(parent B).  Pluses are frequently split between fragments, so they can only be
counted after registration.
"""
from __future__ import annotations

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))
PLUS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
NOISE_SHAPES = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 0)))
PLUS_SET = frozenset(PLUS)


# --------------------------------------------------------------------------
# shared helpers (used by solve and by the validator)
# --------------------------------------------------------------------------

def _components(cells, nbrs):
    cells = set(cells)
    out = []
    while cells:
        start = cells.pop()
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in nbrs:
                q = (r + dr, c + dc)
                if q in cells:
                    cells.remove(q)
                    comp.append(q)
                    stack.append(q)
        out.append(comp)
    return out


def _is_plus(comp):
    if len(comp) != 5:
        return False
    rs = sorted(r for r, _ in comp)
    cs = sorted(c for _, c in comp)
    cr, cc = rs[2], cs[2]
    return frozenset((r - cr, c - cc) for r, c in comp) == PLUS_SET


def _plus_count(cells):
    return sum(_is_plus(comp) for comp in _components(cells, N4))


def _background(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _parse(grid):
    """Return background, marker colour and the list of fragments.

    A fragment is (colour, frozenset of payload offsets relative to its marker).
    """
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    fg = [(int(r), int(c)) for r, c in zip(*np.nonzero(grid != bg))]
    comps = _components(fg, N8)
    if len(comps) < 2:
        raise ValueError("need several fragments")
    common = None
    for comp in comps:
        colours = {int(grid[p]) for p in comp}
        common = colours if common is None else common & colours
    if not common or len(common) != 1:
        raise ValueError("no unique marker colour")
    marker = common.pop()
    frags = []
    for comp in comps:
        marks = [p for p in comp if grid[p] == marker]
        if len(marks) != 1:
            raise ValueError("fragment without a unique marker")
        mr, mc = marks[0]
        pay = [p for p in comp if grid[p] != marker]
        colours = {int(grid[p]) for p in pay}
        if len(colours) != 1:
            raise ValueError("fragment payload must be one colour")
        frags.append((colours.pop(), frozenset((r - mr, c - mc) for r, c in pay)))
    return bg, marker, frags


def _assemble(frags):
    sets = {}
    for colour, offs in frags:
        cells = sets.setdefault(colour, set())
        if cells & offs or (0, 0) in offs:
            raise ValueError("registered fragments overlap")
        cells |= offs
    return sets


def _render(bg, marker, colour, cells):
    pts = list(cells) + [(0, 0)]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    h = max(r for r, _ in pts) - r0 + 1
    w = max(c for _, c in pts) - c0 + 1
    out = np.full((h, w), bg, dtype=int)
    out[-r0, -c0] = marker
    for r, c in cells:
        out[r - r0, c - c0] = colour
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    bg, marker, frags = _parse(input_grid)
    sets = _assemble(frags)
    scores = {colour: _plus_count(cells) for colour, cells in sets.items()}
    best = max(scores.values())
    winners = [c for c, s in scores.items() if s == best]
    if len(winners) != 1:
        raise ValueError("plus-count maximum is not unique")
    return _render(bg, marker, winners[0], sets[winners[0]])


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} outside {low}..{high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _composite(rng, n_plus, n_noise, window, min_roots):
    """Grow a marker-connected picture of pluses/singles/dominoes."""
    (r_lo, r_hi), (c_lo, c_hi) = window
    cells_all = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    for _ in range(25):
        occ = {(0, 0)}
        payload = set()
        objs = ["plus"] * n_plus + ["noise"] * n_noise
        rng.shuffle(objs)
        ok = True
        for obj in objs:
            if obj == "plus":
                shape = PLUS
            else:
                shape = NOISE_SHAPES[int(rng.choice(3, p=[.5, .25, .25]))]
            blocked = set(occ)
            for r, c in payload:
                for dr, dc in N4:
                    blocked.add((r + dr, c + dc))
            touch = set()
            for r, c in occ:
                for dr, dc in N8:
                    touch.add((r + dr, c + dc))
            free = {p for p in cells_all if p not in blocked}
            cands = []
            for r, c in cells_all:
                cells = [(r + dr, c + dc) for dr, dc in shape]
                if all(p in free for p in cells) and any(p in touch for p in cells):
                    cands.append(cells)
            if not cands:
                ok = False
                break
            pick = cands[int(rng.integers(len(cands)))]
            payload.update(pick)
            occ.update(pick)
        if not ok:
            continue
        roots = [p for p in payload if max(abs(p[0]), abs(p[1])) == 1]
        if len(roots) < min_roots:
            continue
        if _plus_count(payload) != n_plus:
            continue
        if all(p in payload for p in N4):
            continue  # avoid a mixed-colour "plus" around the marker
        return payload
    return None


def _split(rng, payload, k, pref=None):
    """Partition payload into k fragments, each 8-connected with the marker.

    pref='split' biases growth so plus signs receive several labels,
    pref='intact' biases growth so each plus keeps one label.
    """
    roots = [p for p in sorted(payload) if max(abs(p[0]), abs(p[1])) == 1]
    if len(roots) < k:
        return None
    owner = {}
    for idx, comp in enumerate(_components(payload, N4)):
        for p in comp:
            owner[p] = idx
    order = [roots[i] for i in rng.permutation(len(roots))]
    label = {}
    for i, p in enumerate(order):
        label[p] = i if i < k else int(rng.integers(k))
    rest = set(payload) - set(label)
    while rest:
        frontier = sorted(p for p in rest
                          if any((p[0] + dr, p[1] + dc) in label for dr, dc in N8))
        p = frontier[int(rng.integers(len(frontier)))]
        nb = sorted({label[(p[0] + dr, p[1] + dc)] for dr, dc in N8 if (p[0] + dr, p[1] + dc) in label})
        if pref is not None and len(nb) > 1:
            mine = {label[q] for q in label if owner[q] == owner[p]}
            if pref == "split":
                alt = [x for x in nb if x not in mine]
            else:
                alt = [x for x in nb if x in mine]
            if alt and rng.random() < .85:
                nb = alt
        label[p] = nb[int(rng.integers(len(nb)))]
        rest.remove(p)
    return [frozenset(p for p in payload if label[p] == i) for i in range(k)]


def _raw_plus(frags):
    return sum(_plus_count(f) for f in frags)


def _make_set(rng, n_plus, n_noise, window, k, raw_hi=None, raw_lo=0, pref=None):
    """Composite plus fragment partition whose intact-plus count lies in range."""
    for _ in range(10):
        payload = _composite(rng, n_plus, n_noise, window, max(k, 1))
        if payload is None:
            continue
        for _ in range(20):
            frags = _split(rng, payload, k, pref)
            if frags is None:
                break
            raw = _raw_plus(frags)
            if raw < raw_lo or (raw_hi is not None and raw > raw_hi):
                continue
            return payload, frags
    return None


def _window(rng, n_plus, n_noise, big):
    top = 5 if big else 4
    spans = [[-int(rng.integers(1, top)), int(rng.integers(1, top))] for _ in range(2)]
    need = 12 * n_plus + 3 * n_noise + 4
    while (spans[0][1] - spans[0][0] + 1) * (spans[1][1] - spans[1][0] + 1) < need:
        axis = int(rng.integers(2))
        if spans[axis][1] - spans[axis][0] >= 8:
            axis = 1 - axis
        if int(rng.integers(2)):
            spans[axis][1] += 1
        else:
            spans[axis][0] -= 1
    return tuple(tuple(s) for s in spans)


def _place(rng, frags_with_colour, marker, bg):
    """Scatter fragments (marker + payload) with a blank 8-neighbourhood gap."""
    pieces = []
    for colour, offs in frags_with_colour:
        pts = list(offs) + [(0, 0)]
        r0 = min(r for r, _ in pts)
        c0 = min(c for _, c in pts)
        cells = [((r - r0, c - c0), colour) for r, c in offs] + [((-r0, -c0), marker)]
        h = max(r for r, _ in pts) - r0 + 1
        w = max(c for _, c in pts) - c0 + 1
        pieces.append((h, w, cells))
    area = sum((h + 1) * (w + 1) for h, w, _ in pieces)
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i][0] * pieces[i][1])
    for attempt in range(40):
        side = np.sqrt(area * float(rng.uniform(1.6, 3.2)) + 4 * attempt * attempt)
        aspect = float(np.exp(rng.uniform(-0.6, 0.6)))
        H = int(np.clip(round(side * aspect), max(h for h, _, _ in pieces) + 1, 30))
        W = int(np.clip(round(side / aspect), max(w for _, w, _ in pieces) + 1, 30))
        grid = np.full((H, W), bg, dtype=int)
        blocked = np.zeros((H + 2, W + 2), dtype=bool)
        ok = True
        for i in order:
            h, w, cells = pieces[i]
            spots = [(y, x) for y in range(H - h + 1) for x in range(W - w + 1)
                     if not any(blocked[y + r + 1, x + c + 1] for (r, c), _ in cells)]
            if not spots:
                ok = False
                break
            y, x = spots[int(rng.integers(len(spots)))]
            for (r, c), col in cells:
                grid[y + r, x + c] = col
                blocked[y + r:y + r + 3, x + c:x + c + 3] = True
        if ok:
            return grid
    return None


def _make_pair(rng, cfg, marker, bg, n_sets, force):
    """force: subset of {'raw', 'pixels', 'allsplit'} that must hold in this pair."""
    for _ in range(60):
        palette = [c for c in range(10) if c not in (marker, bg)]
        colours = [int(c) for c in rng.permutation(palette)[:n_sets]]
        top = int(rng.integers(cfg["win_lo"], cfg["win_hi"] + 1))
        if "raw" in force:
            top = max(top, 2)
        plus = [top] + [int(rng.integers(0, top)) for _ in range(n_sets - 1)]
        noise = [int(rng.integers(0, 5)) for _ in range(n_sets)]
        if "raw" in force:
            plus[1] = int(rng.integers(1, top))
        if "pixels" in force:
            # a loser with nearly as many pluses but many more loose marks
            j = n_sets - 1
            plus[j] = top - 1
            noise[0] = int(rng.integers(0, 3))
            noise[j] = noise[0] + int(rng.integers(4, 7))
        frags_n = [int(rng.integers(2, 5))]
        for i in range(1, n_sets):
            lo = 1 if rng.random() < .15 else 2
            frags_n.append(int(rng.integers(lo, 5)))
        for i in range(n_sets):
            if plus[i] == 0:
                noise[i] = max(noise[i], frags_n[i], 1)
        sets = [None] * n_sets
        good = True
        for i in list(range(1, n_sets)) + [0]:
            window = _window(rng, plus[i], noise[i], cfg.get("big", False) and i == 0)
            if i == 0:
                distract_raw = max(_raw_plus(s[1]) for s in sets[1:])
                hi = None
                if "raw" in force:
                    hi = distract_raw - 1
                if "allsplit" in force:
                    hi = 0
                pref = "split" if hi is not None or rng.random() < .5 else None
                made = _make_set(rng, plus[0], noise[0], window, frags_n[0], raw_hi=hi, pref=pref)
            else:
                want_intact = "raw" in force and i == 1
                made = _make_set(rng, plus[i], noise[i], window, frags_n[i],
                                 raw_lo=1 if want_intact else 0,
                                 pref="intact" if want_intact or rng.random() < .3 else None)
            if made is None:
                good = False
                break
            sets[i] = made
        if not good:
            continue
        info = _pair_info(colours, sets)
        if "raw" in force and not info["raw_fails"]:
            continue
        if "pixels" in force and not info["pixels_fails"]:
            continue
        frag_list = [(colours[i], f) for i in range(n_sets) for f in sets[i][1]]
        if len(frag_list) > 16:
            continue
        order = rng.permutation(len(frag_list))
        grid = _place(rng, [frag_list[int(i)] for i in order], marker, bg)
        if grid is None:
            continue
        out = _render(bg, marker, colours[0], sets[0][0])
        return {"input": grid, "output": out}, info
    return None, None


def _pair_info(colours, sets):
    plus = [_plus_count(p) for p, _ in sets]
    raw = [_raw_plus(f) for _, f in sets]
    pix = [len(p) for p, _ in sets]
    nfr = [len(f) for _, f in sets]

    def fails(vals):
        m = max(vals)
        return vals.count(m) > 1 or vals[0] != m

    return {"plus": plus, "raw": raw, "raw_fails": fails(raw) and max(raw) > raw[0],
            "pixels_fails": max(pix[1:]) > pix[0], "frags_fails": max(nfr[1:]) >= nfr[0],
            "winner_split": raw[0] < plus[0], "n_sets": len(sets)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(30):
        puzzle = _try_puzzle(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("generation failed")


def _try_puzzle(rng, num_train, num_test):
    marker = int(rng.integers(10))
    fixed_bg = rng.random() < .55
    bg0 = int(rng.choice([c for c in range(10) if c != marker]))
    shift = str(rng.choice(["more_sets", "all_split", "bigger", "plain"], p=[.3, .3, .25, .15]))
    train_sets_max = int(rng.choice([2, 3], p=[.45, .55]))
    train_cfg = {"win_lo": 1, "win_hi": 3, "big": False}
    forced = {}
    idx = [int(i) for i in rng.permutation(num_train)]
    forced.setdefault(idx[0], set()).add("raw")
    forced.setdefault(idx[1], set()).add("pixels")
    pairs, infos, seen = [], [], set()

    def add(cfg, n_sets, force, bg):
        for _ in range(6):
            pair, info = _make_pair(rng, cfg, marker, bg, n_sets, force)
            if pair is None:
                return False
            key = pair["input"].tobytes() + bytes(pair["input"].shape)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
            infos.append(info)
            return True
        return False

    def pick_bg():
        if fixed_bg:
            return bg0
        return int(rng.choice([c for c in range(10) if c != marker]))

    for i in range(num_train):
        n_sets = 2 if train_sets_max == 2 else int(rng.choice([2, 3], p=[.4, .6]))
        if not add(train_cfg, n_sets, forced.get(i, set()), pick_bg()):
            return None
    train_n = len(pairs)
    for _ in range(num_test):
        cfg = dict(train_cfg)
        n_sets = int(rng.integers(2, train_sets_max + 1))
        force = set()
        if shift == "more_sets":
            n_sets = min(4, train_sets_max + 1)
        elif shift == "all_split":
            force = {"allsplit", "raw"}
        elif shift == "bigger":
            cfg = {"win_lo": 3, "win_hi": 4, "big": True}
        if rng.random() < .5:
            force.add("raw")
        if rng.random() < .3:
            force.add("pixels")
        if not add(cfg, n_sets, force, pick_bg()):
            return None
    return {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in pairs[:train_n]],
            "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in pairs[train_n:]]}


# --------------------------------------------------------------------------
# validation and self test
# --------------------------------------------------------------------------

def _analyse(grid):
    bg, marker, frags = _parse(grid)
    sets = _assemble(frags)
    info = {}
    for colour, cells in sets.items():
        for comp in _components(cells, N4):
            if not (_is_plus(comp) or len(comp) == 1 or
                    (len(comp) == 2 and abs(comp[0][0] - comp[1][0]) + abs(comp[0][1] - comp[1][1]) == 1)):
                raise AssertionError("assembled component is not a plus, single or domino")
        mine = [f for c, f in frags if c == colour]
        info[colour] = {"plus": _plus_count(cells), "raw": _raw_plus(mine),
                        "pix": len(cells), "frags": len(mine)}
    return bg, marker, sets, info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    seen = set()
    witnesses = {"raw": False, "pixels": False}
    markers = set()
    for split in ("train", "test"):
        assert puzzle[split], "empty split"
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            gi = np.asarray(pair["input"], dtype=int)
            go = np.asarray(pair["output"], dtype=int)
            for g in (gi, go):
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                assert g.min() >= 0 and g.max() <= 9
            key = gi.tobytes() + bytes(gi.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            bg, marker, sets, info = _analyse(gi)
            markers.add(marker)
            assert 2 <= len(sets) <= 4
            best = max(v["plus"] for v in info.values())
            win = [c for c, v in info.items() if v["plus"] == best]
            assert best >= 1 and len(win) == 1, "plus maximum must be unique and positive"
            w = win[0]
            assert info[w]["frags"] >= 2, "winner must be assembled from several fragments"
            assert np.array_equal(_render(bg, marker, w, sets[w]), go), "output is not the winning assembly"
            assert np.array_equal(solve(gi), go)
            if split == "train":
                raws = {c: v["raw"] for c, v in info.items()}
                if max(raws.values()) > raws[w]:
                    witnesses["raw"] = True
                if max(v["pix"] for v in info.values()) > info[w]["pix"]:
                    witnesses["pixels"] = True
    assert len(markers) == 1, "marker colour is shared across the puzzle"
    assert all(witnesses.values()), f"training lacks shortcut witnesses: {witnesses}"


def self_test() -> None:
    shapes, set_counts, frag_counts, plus_counts = set(), set(), set(), set()
    raw_fail_tests = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                go = np.asarray(pair["output"])
                shapes.add(go.shape)
                _, _, sets, info = _analyse(np.asarray(pair["input"]))
                set_counts.add(len(sets))
                for v in info.values():
                    frag_counts.add(v["frags"])
                    plus_counts.add(v["plus"])
                if split == "test":
                    w = max(info, key=lambda c: info[c]["plus"])
                    if max(v["raw"] for v in info.values()) > info[w]["raw"]:
                        raw_fail_tests += 1
    assert len(shapes) >= 15, shapes
    assert set_counts >= {2, 3, 4}, set_counts
    assert len(frag_counts) >= 4 and len(plus_counts) >= 4
    assert raw_fail_tests >= 10, raw_fail_tests
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {bad}")
    assert generate_puzzle(5) == generate_puzzle(5)
