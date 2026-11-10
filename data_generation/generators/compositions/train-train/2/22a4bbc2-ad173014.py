"""Gate-anchored every-third relay around a panel track.

Composition of parent A (22a4bbc2: number a top-anchored sequence of bands and
select bands 1, 4, 7, ...) with parent B (ad173014: framed motif panels mounted
on a rectangular track, ordered clockwise, each receiving the accent colour of
its clockwise successor).

New rule: a single gate cell on the track anchors the clockwise count.  Panels
are numbered 1, 2, 3, ... clockwise starting with the first panel after the
gate.  Panels numbered 1, 4, 7, 10 are "stations".  Each station's accent mask
is recoloured with the accent of the NEXT STATION clockwise (wrapping from the
last station back to station 1).  Non-station panels, frames, track, gate and
background are unchanged.
"""
from __future__ import annotations

import numpy as np

PERIOD = 3
MIN_PANELS = 4
MAX_PANELS = 10


# ----------------------------------------------------------------------------
# Geometry helpers shared by generation and solving
# ----------------------------------------------------------------------------

def _perimeter(r0, r1, c0, c1):
    """Clockwise list of track cells starting at the top-left corner."""
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r, c1) for r in range(r0 + 1, r1 + 1)]
    cells += [(r1, c) for c in range(c1 - 1, c0 - 1, -1)]
    cells += [(r, c0) for r in range(r1 - 1, r0, -1)]
    return cells


def _station_indices(n):
    return list(range(0, n, PERIOD))


def _relay(accents, stations):
    """New accent list: each station takes the accent of the next station."""
    new = list(accents)
    m = len(stations)
    for k, s in enumerate(stations):
        new[s] = accents[stations[(k + 1) % m]]
    return new


# ----------------------------------------------------------------------------
# Input-only parser and solver
# ----------------------------------------------------------------------------

def _parse(grid):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    colors, counts = np.unique(g, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    bg = int(colors[order[0]])
    if len(counts) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("ambiguous background")
    singles = [int(c) for c, k in zip(colors, counts) if k == 1 and c != bg]
    if len(singles) != 1:
        raise ValueError("expected exactly one gate cell")
    gate = singles[0]
    gr, gc = map(int, np.argwhere(g == gate)[0])
    neigh = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = gr + dr, gc + dc
        if 0 <= rr < H and 0 <= cc < W and g[rr, cc] != bg:
            neigh.add(int(g[rr, cc]))
    if len(neigh) != 1:
        raise ValueError("gate must sit on a plain track")
    track = neigh.pop()
    pts = np.argwhere((g == track) | (g == gate))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
    per = _perimeter(r0, r1, c0, c1)
    index = {p: i for i, p in enumerate(per)}
    if (gr, gc) not in index:
        raise ValueError("gate is not on the track rectangle")
    frame = None
    for p in per:
        v = int(g[p])
        if v not in (track, gate):
            frame = v
            break
    if frame is None or frame == bg:
        raise ValueError("no panels on the track")
    # 4-connected components of frame colour
    seen = np.zeros_like(g, dtype=bool)
    panels = []
    for sr, sc in np.argwhere(g == frame):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and not seen[rr, cc] and g[rr, cc] == frame:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        pr0, pr1, pc0, pc1 = min(rs), max(rs), min(cs), max(cs)
        inner = g[pr0 + 1:pr1, pc0 + 1:pc1]
        vals = set(int(v) for v in np.unique(inner)) - {bg}
        if len(vals) != 1:
            raise ValueError("each panel needs exactly one accent")
        accent = vals.pop()
        mask = np.zeros_like(g, dtype=bool)
        mask[pr0 + 1:pr1, pc0 + 1:pc1] = inner == accent
        gidx = index[(gr, gc)]
        L = len(per)
        keys = [(index[(r, c)] - gidx) % L for r in range(pr0, pr1 + 1)
                for c in range(pc0, pc1 + 1) if (r, c) in index]
        if not keys:
            raise ValueError("panel is not mounted on the track")
        panels.append({"box": (pr0, pr1, pc0, pc1), "accent": accent,
                       "mask": mask, "key": min(keys)})
    panels.sort(key=lambda p: p["key"])
    return {"bg": bg, "gate": gate, "gate_pos": (gr, gc), "track": track,
            "frame": frame, "rect": (r0, r1, c0, c1), "panels": panels}


def _render(grid, info, new_accents):
    out = np.array(grid, dtype=int).copy()
    for p, a in zip(info["panels"], new_accents):
        out[p["mask"]] = a
    return out


def solve(input_grid):
    g = np.asarray(input_grid, dtype=int)
    info = _parse(g)
    accents = [p["accent"] for p in info["panels"]]
    stations = _station_indices(len(accents))
    return _render(g, info, _relay(accents, stations))


# ----------------------------------------------------------------------------
# Alternative (shortcut) rules used to prove the demonstrations discriminate
# ----------------------------------------------------------------------------

def _alternatives(grid):
    g = np.asarray(grid, dtype=int)
    info = _parse(g)
    acc = [p["accent"] for p in info["panels"]]
    n = len(acc)
    st = _station_indices(n)
    alts = {}
    # B only: every panel takes its clockwise successor's accent.
    alts["all_panels_successor"] = _relay(acc, list(range(n)))
    # Stations relay counter-clockwise instead of clockwise.
    new = list(acc)
    for k, s in enumerate(st):
        new[s] = acc[st[(k - 1) % len(st)]]
    alts["stations_reverse_direction"] = new
    # Station takes accent of the panel three steps ahead (no sub-cycle wrap).
    new = list(acc)
    for s in st:
        new[s] = acc[(s + PERIOD) % n]
    alts["plus_three_mod_n"] = new
    # Station takes accent of its immediate clockwise neighbour.
    new = list(acc)
    for s in st:
        new[s] = acc[(s + 1) % n]
    alts["station_takes_neighbour"] = new
    # Period 2 instead of 3.
    alts["every_other"] = _relay(acc, list(range(0, n, 2)))
    # Count anchored at top-left corner instead of the gate.
    r0, r1, c0, c1 = info["rect"]
    per = _perimeter(r0, r1, c0, c1)
    index = {p: i for i, p in enumerate(per)}
    gidx = index[info["gate_pos"]]
    abs_keys = [(p["key"] + gidx) % len(per) for p in info["panels"]]
    corner_order = sorted(range(n), key=lambda i: abs_keys[i])
    cst = [corner_order[k] for k in range(0, n, PERIOD)]
    alts["corner_anchor"] = _relay(acc, cst)
    # Count counter-clockwise from the gate (panel just before gate is #1).
    ccw = list(range(n - 1, -1, -1))
    cst = [ccw[k] for k in range(0, n, PERIOD)]
    alts["counterclockwise_count"] = _relay(acc, cst)
    # A only: stations untouched / identity.
    alts["identity"] = list(acc)
    return {k: _render(g, info, v) for k, v in alts.items()}


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _motif(rng, ih, iw):
    """Binary motif mask with at least two accent cells and one background cell."""
    for _ in range(100):
        kind = rng.integers(0, 5)
        m = np.zeros((ih, iw), dtype=bool)
        if kind == 0:
            m = rng.random((ih, iw)) < rng.uniform(0.3, 0.7)
        elif kind == 1:  # mirror-symmetric
            half = rng.random((ih, (iw + 1) // 2)) < rng.uniform(0.35, 0.65)
            m = np.concatenate([half, half[:, ::-1][:, iw % 2:]], axis=1)
        elif kind == 2:  # plus / cross
            r = rng.integers(0, ih)
            c = rng.integers(0, iw)
            m[r, :] = True
            m[:, c] = True
        elif kind == 3:  # diagonal staircase
            for r in range(ih):
                c = (r + rng.integers(0, 2)) % iw if rng.random() < 0.3 else r % iw
                m[r, c] = True
                if c + 1 < iw and rng.random() < 0.5:
                    m[r, c + 1] = True
        else:  # sub-rectangle block
            a, b = sorted(rng.integers(0, ih, size=2))
            c, d = sorted(rng.integers(0, iw, size=2))
            m[a:b + 1, c:d + 1] = True
        if 2 <= m.sum() < ih * iw:
            return m
    m = np.zeros((ih, iw), dtype=bool)
    m[0, 0] = m[ih - 1, iw - 1] = True
    return m


def _side_cells(side, r0, r1, c0, c1):
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _box_ok(box, side, panels, per, own, H, W):
    pr0, pr1, pc0, pc1 = box
    if pr0 < 0 or pc0 < 0 or pr1 >= H or pc1 >= W:
        return False
    e = (pr0 - 1, pr1 + 1, pc0 - 1, pc1 + 1)
    for q in panels:
        qr0, qr1, qc0, qc1 = q["box"]
        if not (qr1 < e[0] or qr0 > e[1] or qc1 < e[2] or qc0 > e[3]):
            return False
    for (r, c) in per:
        if e[0] <= r <= e[1] and e[2] <= c <= e[3] and (r, c) not in own[side]:
            return False
    return True


def _layout(rng, n, H, W, uniform):
    """Try to place n panels around a track inside an H x W canvas."""
    mmax = int(rng.choice([3, 4, 6]))
    r0 = int(rng.integers(0, mmax))
    c0 = int(rng.integers(0, mmax))
    r1 = H - 1 - int(rng.integers(0, mmax))
    c1 = W - 1 - int(rng.integers(0, mmax))
    if r1 - r0 < 9 or c1 - c0 < 9:
        return None
    sides = ("top", "right", "bottom", "left")
    along_len = {"top": c1 - c0 - 3, "bottom": c1 - c0 - 3,
                 "left": r1 - r0 - 3, "right": r1 - r0 - 3}
    usize = (int(rng.integers(4, 7)), int(rng.integers(4, 7)))
    # distribute panels over sides
    weights = rng.dirichlet(np.ones(4) * rng.choice([0.6, 1.5, 4.0]))
    counts = dict.fromkeys(sides, 0)
    for _ in range(n):
        cap = [s for s in sides if (counts[s] + 1) * 5 - 1 <= along_len[s]]
        if not cap:
            return None
        w = np.array([weights[sides.index(s)] for s in cap])
        s = cap[int(rng.choice(len(cap), p=w / w.sum()))]
        counts[s] += 1
    per = _perimeter(r0, r1, c0, c1)
    own = {s: set(_side_cells(s, r0, r1, c0, c1)) for s in sides}
    panels = []
    for s in sides:
        k = counts[s]
        if k == 0:
            continue
        A = along_len[s]
        for _ in range(30):
            if uniform:
                alongs = [usize[1] if s in ("top", "bottom") else usize[0]] * k
            else:
                alongs = [int(rng.integers(4, 8)) for _ in range(k)]
            if sum(alongs) + (k - 1) <= A:
                break
        else:
            return None
        if sum(alongs) + (k - 1) > A:
            return None
        slack = A - sum(alongs) - (k - 1)
        cuts = np.sort(rng.integers(0, slack + 1, size=k))
        start = (c0 + 2) if s in ("top", "bottom") else (r0 + 2)
        pos = []
        prev = 0
        cur = start
        for i in range(k):
            cur += int(cuts[i] - prev)
            prev = int(cuts[i])
            pos.append(cur)
            cur += alongs[i] + 1
        for a, al in zip(pos, alongs):
            room = {"top": r0, "bottom": H - 1 - r1, "left": c0, "right": W - 1 - c1}[s]
            placed = None
            for _try in range(25):
                if uniform and _try < 5:
                    perp = usize[0] if s in ("top", "bottom") else usize[1]
                else:
                    perp = int(rng.integers(4, 8))
                off = int(rng.integers(0, min(perp - 1, room) + 1))
                if s == "top":
                    box = (r0 - off, r0 - off + perp - 1, a, a + al - 1)
                elif s == "bottom":
                    box = (r1 - (perp - 1 - off), r1 + off, a, a + al - 1)
                elif s == "left":
                    box = (a, a + al - 1, c0 - off, c0 - off + perp - 1)
                else:
                    box = (a, a + al - 1, c1 - (perp - 1 - off), c1 + off)
                if _box_ok(box, s, panels, per, own, H, W):
                    placed = box
                    break
            if placed is None:
                return None
            panels.append({"side": s, "box": placed})
    # gate candidates: non-corner track cells outside every expanded box
    cand = []
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    for (r, c) in per:
        if (r, c) in corners:
            continue
        ok = True
        for p in panels:
            pr0, pr1, pc0, pc1 = p["box"]
            if pr0 - 1 <= r <= pr1 + 1 and pc0 - 1 <= c <= pc1 + 1:
                ok = False
                break
        if ok:
            cand.append((r, c))
    if not cand:
        return None
    gate = cand[int(rng.integers(0, len(cand)))]
    return {"rect": (r0, r1, c0, c1), "panels": panels, "gate": gate}


def _canvas_range(n):
    if n <= 5:
        return 13, 24
    if n <= 7:
        return 17, 28
    return 21, 30


def _make_pair(rng, n, roles, accent_pool, distinct_pref):
    bg, track, frame, gate_c = roles
    lo, hi = _canvas_range(n)
    for _ in range(400):
        H = int(rng.integers(lo, hi + 1))
        W = int(rng.integers(lo, hi + 1))
        lay = _layout(rng, n, H, W, uniform=rng.random() < 0.3)
        if lay is None:
            continue
        r0, r1, c0, c1 = lay["rect"]
        per = _perimeter(r0, r1, c0, c1)
        index = {p: i for i, p in enumerate(per)}
        gidx = index[lay["gate"]]
        L = len(per)

        def key(p):
            pr0, pr1, pc0, pc1 = p["box"]
            return min((index[(r, c)] - gidx) % L for r in range(pr0, pr1 + 1)
                       for c in range(pc0, pc1 + 1) if (r, c) in index)

        panels = sorted(lay["panels"], key=key)
        stations = _station_indices(n)
        pool = list(accent_pool)
        if distinct_pref and n <= len(pool):
            acc = [int(x) for x in rng.permutation(pool)[:n]]
        else:
            st_cols = [int(x) for x in rng.permutation(pool)[:len(stations)]]
            acc = [int(rng.choice(pool)) for _ in range(n)]
            for s, cval in zip(stations, st_cols):
                acc[s] = cval
        g = np.full((H, W), bg, dtype=int)
        for (r, c) in per:
            g[r, c] = track
        for p, a in zip(panels, acc):
            pr0, pr1, pc0, pc1 = p["box"]
            g[pr0:pr1 + 1, pc0:pc1 + 1] = frame
            ih, iw = pr1 - pr0 - 1, pc1 - pc0 - 1
            inner = np.full((ih, iw), bg, dtype=int)
            inner[_motif(rng, ih, iw)] = a
            g[pr0 + 1:pr1, pc0 + 1:pc1] = inner
        g[lay["gate"]] = gate_c
        try:
            info = _parse(g)
        except ValueError:
            continue
        if [p["accent"] for p in info["panels"]] != acc:
            continue
        out = solve(g)
        if np.array_equal(out, g):
            continue
        return g, out, n
    raise RuntimeError("could not build a pair")


def _choose_roles(rng):
    bg = 0 if rng.random() < 0.7 else int(rng.integers(0, 10))
    rest = [c for c in range(10) if c != bg]
    rng.shuffle(rest)
    return (bg, int(rest[0]), int(rest[1]), int(rest[2])), [int(x) for x in rest[3:]]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    if isinstance(num_train, bool) or isinstance(num_test, bool):
        raise ValueError("counts must be integers")
    if not (isinstance(num_train, (int, np.integer)) and isinstance(num_test, (int, np.integer))):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    rng = np.random.default_rng(seed)
    for attempt in range(60):
        fixed_roles = rng.random() < 0.6
        roles, pool = _choose_roles(rng)
        distinct_pref = rng.random() < 0.7
        # training panel counts: 4..8, with at least one 7 or 8 (>= 3 stations)
        ns = [int(rng.integers(4, 9)) for _ in range(num_train)]
        ns[int(rng.integers(0, num_train))] = int(rng.choice([7, 8]))
        if all(n % 3 == 0 for n in ns):
            ns[int(rng.integers(0, num_train))] = int(rng.choice([4, 5, 7, 8]))
        tlo = max(7, max(ns))
        test_ns = [int(rng.integers(tlo, MAX_PANELS + 1)) for _ in range(num_test)]
        pairs = []
        seen = set()
        ok = True
        for i, n in enumerate(ns + test_ns):
            if fixed_roles:
                r, p = roles, pool
            else:
                r, p = _choose_roles(rng)
            for _ in range(5):
                gi, go, _n = _make_pair(rng, n, r, p, distinct_pref)
                k = gi.tobytes() + bytes(gi.shape)
                if k not in seen:
                    seen.add(k)
                    break
            else:
                ok = False
                break
            pairs.append({"input": gi, "output": go})
        if not ok:
            continue
        puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
        if _discriminates(puzzle):
            return puzzle
    raise RuntimeError("generation failed")


def _discriminates(puzzle):
    """Every shortcut must be contradicted by at least one training pair, and
    the main shortcuts (B-only, corner anchor) also on every test pair."""
    names = None
    refuted = {}
    for pair in puzzle["train"]:
        alts = _alternatives(pair["input"])
        names = list(alts)
        for k, v in alts.items():
            if not np.array_equal(v, pair["output"]):
                refuted[k] = True
    if names is None or not all(refuted.get(k) for k in names):
        return False
    for pair in puzzle["test"]:
        alts = _alternatives(pair["input"])
        for k in ("all_panels_successor", "corner_anchor", "identity"):
            if np.array_equal(alts[k], pair["output"]):
                return False
    return True


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    assert set(puzzle) == {"train", "test"}
    seen = set()
    max_train_n = 0
    max_train_stations = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            gi = np.asarray(pair["input"], dtype=int)
            go = np.asarray(pair["output"], dtype=int)
            assert gi.ndim == 2 and gi.shape == go.shape
            assert 1 <= gi.shape[0] <= 30 and 1 <= gi.shape[1] <= 30
            assert gi.min() >= 0 and gi.max() <= 9
            key = gi.tobytes() + bytes(gi.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(gi)
            roles = {info["bg"], info["track"], info["frame"], info["gate"]}
            assert len(roles) == 4, "structural roles must be distinct"
            n = len(info["panels"])
            assert MIN_PANELS <= n <= MAX_PANELS
            acc = [p["accent"] for p in info["panels"]]
            assert not (set(acc) & roles), "accent colours must differ from structural roles"
            st = _station_indices(n)
            assert len({acc[s] for s in st}) == len(st), "station accents must be distinct"
            for p in info["panels"]:
                pr0, pr1, pc0, pc1 = p["box"]
                assert pr1 - pr0 >= 3 and pc1 - pc0 >= 3
                border = np.ones((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
                border[1:-1, 1:-1] = False
                assert np.all(gi[pr0:pr1 + 1, pc0:pc1 + 1][border] == info["frame"])
                assert p["mask"].sum() >= 2
                inner = gi[pr0 + 1:pr1, pc0 + 1:pc1]
                assert np.any(inner == info["bg"])
            # every non-panel cell is track, gate or background
            covered = np.zeros(gi.shape, dtype=bool)
            for p in info["panels"]:
                pr0, pr1, pc0, pc1 = p["box"]
                covered[pr0:pr1 + 1, pc0:pc1 + 1] = True
            r0, r1, c0, c1 = info["rect"]
            trackmask = np.zeros(gi.shape, dtype=bool)
            for cell in _perimeter(r0, r1, c0, c1):
                trackmask[cell] = True
            rest = gi[~covered]
            assert set(np.unique(rest).tolist()) <= {info["bg"], info["track"], info["gate"]}
            assert np.all(gi[~covered & ~trackmask] == info["bg"])
            assert np.array_equal(solve(gi), go), "output disagrees with rule"
            diff = gi != go
            allowed = np.zeros(gi.shape, dtype=bool)
            for s in st:
                allowed |= info["panels"][s]["mask"]
            assert not np.any(diff & ~allowed), "only station accent cells may change"
            assert np.any(diff), "pair must change"
            if split == "train":
                max_train_n = max(max_train_n, n)
                max_train_stations = max(max_train_stations, len(st))
            else:
                assert n >= max(7, max_train_n), "test must extend the sequence"
    assert max_train_stations >= 3, "direction needs a training pair with >=3 stations"
    assert _discriminates(puzzle), "demonstrations do not rule out shortcuts"
    return True


def self_test():
    seeds = list(range(42))
    ns, shapes, gate_sides, multi_side = set(), set(), set(), 0
    bgs = set()
    for seed in seeds:
        pz = generate_puzzle(seed, num_train=3 + seed % 4, num_test=1 + seed % 2)
        validate_generated_puzzle(pz)
        for pair in pz["train"] + pz["test"]:
            gi = np.asarray(pair["input"])
            info = _parse(gi)
            ns.add(len(info["panels"]))
            shapes.add(gi.shape)
            bgs.add(info["bg"])
            r0, r1, c0, c1 = info["rect"]
            gr, gc = info["gate_pos"]
            gate_sides.add("top" if gr == r0 else "bottom" if gr == r1 else
                           "left" if gc == c0 else "right")
    assert ns >= set(range(4, 11)), ns
    assert len(shapes) >= 40
    assert gate_sides == {"top", "bottom", "left", "right"}
    assert len(bgs) >= 2
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (4, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p1 = generate_puzzle(7)
    p2 = generate_puzzle(7)
    assert all(np.array_equal(x["input"], y["input"]) for x, y in zip(p1["train"], p2["train"]))
    return True


if __name__ == "__main__":
    self_test()
    print("ok")
