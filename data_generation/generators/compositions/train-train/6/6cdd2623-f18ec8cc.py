"""Relinked bands with seam-activated border lines.

Composition of 6cdd2623 (opposite-edge marker pairs become complete lines,
everything else of that color is erased) and f18ec8cc (full-span color bands
carrying cross-color labels are permuted intact, the original first band ends
last).

Rule (canonical vertical-band orientation; some puzzles use the transpose):
  * The grid is tiled by K >= 3 full-height vertical bands, each with its own
    dominant color.  Every band holds exactly one "link" cell painted in the
    dominant color of another band: the band it points to must stand
    immediately to its right in the output.  The links form a single cycle;
    the original leftmost band is cut out of the cycle and placed last, so the
    band named by the leftmost band's link leads the output.
  * Bands move intact (link cells included).
  * One extra "beacon" color, which is no band's dominant color, sits on band
    edge columns and on the top/bottom rows.  After the bands have been
    rearranged, every row whose beacon cells lie on BOTH the new left edge and
    the new right edge becomes a complete beacon line; every column with beacon
    cells on both the top and bottom rows becomes a complete beacon line.
    All other beacon cells are erased to the dominant color of their band.
    Lines are drawn last and paint through everything.

Because the outer left/right edges only exist after the permutation, the
pairs that become lines are determined by the decoded band order: beacons on
interior seams in the input can become active, and pairs that are aligned on
the input's own border are decoys.
"""

from __future__ import annotations

import numpy as np

MIN_BANDS = 3
MAX_BANDS = 5
MIN_WIDTH = 3
MAX_WIDTH = 8
MIN_SPAN = 8          # length of the band axis (height for vertical bands)
MAX_SPAN = 20
PURITY = 0.6          # every band line must be at least this pure for parsing
GEN_PURITY = 0.65     # generator keeps a safety margin above PURITY
MAX_WIDTH_SHARE = 0.58


# --------------------------------------------------------------------------
# Solver (input only)
# --------------------------------------------------------------------------

def _parse_vertical(grid):
    """Parse grid as vertical bands.  Returns a dict or None if invalid."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return None
    h, w = g.shape
    if h < 3 or w < 3 * MIN_WIDTH:
        return None
    dom = []
    for c in range(w):
        counts = np.bincount(g[:, c], minlength=10)
        d = int(np.argmax(counts))
        if counts[d] < PURITY * h:
            return None
        dom.append(d)
    bands = []  # (start, end_exclusive, color)
    start = 0
    for c in range(1, w + 1):
        if c == w or dom[c] != dom[start]:
            bands.append((start, c, dom[start]))
            start = c
    k = len(bands)
    if k < MIN_BANDS:
        return None
    colors = [b[2] for b in bands]
    if len(set(colors)) != k:
        return None
    if any(e - s < MIN_WIDTH for s, e, _ in bands):
        return None
    band_set = set(colors)
    others = sorted(set(np.unique(g).tolist()) - band_set)
    if len(others) != 1:
        return None
    beacon = others[0]
    color_index = {c: i for i, c in enumerate(colors)}
    succ = []
    for i, (s, e, col) in enumerate(bands):
        sub = g[:, s:e]
        links = [int(v) for v in sub.ravel() if v in band_set and v != col]
        if len(links) != 1:
            return None
        succ.append(color_index[links[0]])
    # links must form a single K-cycle
    seen, cur = set(), 0
    for _ in range(k):
        seen.add(cur)
        cur = succ[cur]
    if cur != 0 or len(seen) != k:
        return None
    order = []
    cur = succ[0]
    while cur != 0:
        order.append(cur)
        cur = succ[cur]
    order.append(0)
    return {"bands": bands, "colors": colors, "beacon": beacon, "succ": succ,
            "order": order}


def _apply(grid, parse):
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    beacon = parse["beacon"]
    pieces = []
    for idx in parse["order"]:
        s, e, col = parse["bands"][idx]
        pieces.append(g[:, s:e].copy())
    moved = np.concatenate(pieces, axis=1)
    out = moved.copy()
    # erase every beacon cell to its band's dominant color
    pos = 0
    for idx in parse["order"]:
        s, e, col = parse["bands"][idx]
        block = out[:, pos:pos + (e - s)]
        block[block == beacon] = col
        pos += e - s
    w = moved.shape[1]
    rows = [r for r in range(h) if moved[r, 0] == beacon and moved[r, w - 1] == beacon]
    cols = [c for c in range(w) if moved[0, c] == beacon and moved[h - 1, c] == beacon]
    for r in rows:
        out[r, :] = beacon
    for c in cols:
        out[:, c] = beacon
    return out


def _parse_any(grid):
    g = np.asarray(grid, dtype=int)
    pv = _parse_vertical(g)
    ph = _parse_vertical(g.T)
    return pv, ph


def solve(input_grid):
    g = np.asarray(input_grid, dtype=int)
    pv, ph = _parse_any(g)
    if pv is not None:
        return _apply(g, pv)
    if ph is not None:
        return _apply(g.T, ph).T.copy()
    raise ValueError("input does not parse as a relinked-band grid")


# --------------------------------------------------------------------------
# Shortcut rules used only to prove that demonstrations discriminate
# --------------------------------------------------------------------------

def _lines_on(g, beacon, fill_of_col):
    h, w = g.shape
    out = g.copy()
    for c in range(w):
        col = out[:, c]
        col[col == beacon] = fill_of_col[c]
    rows = [r for r in range(h) if g[r, 0] == beacon and g[r, w - 1] == beacon]
    cols = [c for c in range(w) if g[0, c] == beacon and g[h - 1, c] == beacon]
    for r in rows:
        out[r, :] = beacon
    for c in cols:
        out[:, c] = beacon
    return out


def _shortcuts(g_canon, parse):
    """Alternative outputs (canonical orientation) of plausible wrong rules."""
    g = np.asarray(g_canon, dtype=int)
    k = len(parse["bands"])
    beacon = parse["beacon"]
    fill = []
    for s, e, col in parse["bands"]:
        fill.extend([col] * (e - s))
    res = {}
    # A only: complete lines on the input's own border, no band move.
    res["no_permute"] = _lines_on(g, beacon, fill)
    # B only: move bands, keep beacons untouched, no lines.
    res["no_lines"] = np.concatenate(
        [g[:, parse["bands"][i][0]:parse["bands"][i][1]] for i in parse["order"]], axis=1)
    # A before B: lines from the input border, then move bands.
    pre = res["no_permute"]
    res["lines_then_move"] = np.concatenate(
        [pre[:, parse["bands"][i][0]:parse["bands"][i][1]] for i in parse["order"]], axis=1)

    def with_order(order):
        p = dict(parse)
        p["order"] = list(order)
        return _apply(g, p)

    # fixed rotation (B-parent cycle) ignoring the links
    res["rotate"] = with_order(list(range(1, k)) + [0])
    # links read as predecessors instead of successors
    pred = {parse["succ"][i]: i for i in range(k)}
    order, cur = [], pred[0]
    while cur != 0:
        order.append(cur)
        cur = pred[cur]
    res["predecessor"] = with_order(order + [0])
    # leftmost band first instead of last
    res["leftmost_first"] = with_order([0] + parse["order"][:-1])
    return res


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, v in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _sample_widths(rng, k):
    for _ in range(200):
        style = rng.random()
        if style < 0.3:
            w0 = int(rng.integers(MIN_WIDTH, 7))
            widths = [w0] * k
        else:
            widths = [int(rng.integers(MIN_WIDTH, MAX_WIDTH + 1)) for _ in range(k)]
        total = sum(widths)
        if total <= 30 and max(widths) / total < MAX_WIDTH_SHARE:
            return widths
    return [MIN_WIDTH] * k


def _gen_pair(rng, k, need):
    """Build one pair in canonical vertical orientation.

    need: dict of flags
      order: 'rot' | 'nonrot' | None
      edge_decoy, single_edge, vline, vdecoy, alt_head: bool
    Returns (input, output, info) or None on failure.
    """
    widths = _sample_widths(rng, k)
    h = int(rng.integers(MIN_SPAN, MAX_SPAN + 1))
    w = sum(widths)
    starts = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(int).tolist()
    palette = [int(x) for x in rng.permutation(10)[:k + 1]]
    colors, beacon = palette[:k], palette[k]

    # link cycle
    for _ in range(50):
        cyc = [0] + [int(x) + 1 for x in rng.permutation(k - 1)]
        is_rot = cyc == list(range(k))
        if need.get("order") == "rot" and not is_rot:
            continue
        if need.get("order") == "nonrot" and is_rot:
            continue
        break
    else:
        return None
    succ = [0] * k
    for i in range(k):
        succ[cyc[i]] = cyc[(i + 1) % k]
    order = cyc[1:] + [0]
    head = order[0]

    left = [starts[i] for i in range(k)]
    right = [starts[i] + widths[i] - 1 for i in range(k)]
    head_l, tail_r = left[head], right[0]
    interior_cols = [c for i in range(k) for c in range(left[i] + 1, right[i])]

    beacons = set()
    inner_rows = list(range(1, h - 1))
    n_h = int(rng.choice([1, 1, 2, 2, 3]))
    n_h = min(n_h, max(1, (h - 2) // 4))
    line_rows = sorted(int(r) for r in rng.choice(inner_rows, n_h, replace=False))
    for r in line_rows:
        beacons.add((r, head_l))
        beacons.add((r, tail_r))
    free_rows = [r for r in inner_rows if r not in line_rows]

    # vertical lines
    if need.get("vline"):
        n_v = int(rng.choice([1, 1, 2]))
    else:
        n_v = int(rng.choice([0, 0, 1, 1, 2]))
    vcols = sorted(int(c) for c in rng.choice(interior_cols, n_v, replace=False)) if n_v else []
    for c in vcols:
        beacons.add((0, c))
        beacons.add((h - 1, c))

    used_rows = set()

    def take_row():
        cand = [r for r in free_rows if r not in used_rows]
        if not cand:
            return None
        r = int(rng.choice(cand))
        used_rows.add(r)
        return r

    flags = {"edge_decoy": False, "single_edge": False, "alt_head": False,
             "vdecoy": False, "seam": 0}
    # decoy: pair aligned on the input's own left/right border
    if need.get("edge_decoy") or rng.random() < 0.35:
        r = take_row()
        if r is not None:
            beacons.add((r, 0))
            beacons.add((r, w - 1))
            flags["edge_decoy"] = True
    # decoy: a beacon on exactly one future outer edge
    if need.get("single_edge") or rng.random() < 0.4:
        r = take_row()
        if r is not None:
            beacons.add((r, head_l) if rng.random() < 0.5 else (r, tail_r))
            flags["single_edge"] = True
    # decoy: the pair that a wrong head band would complete
    alts = [b for b in range(1, k) if b != head]
    if alts and (need.get("alt_head") or rng.random() < 0.45):
        r = take_row()
        if r is not None:
            b = int(rng.choice(alts))
            beacons.add((r, left[b]))
            beacons.add((r, tail_r))
            flags["alt_head"] = True
    # seam clutter on other band edges
    seam_cols = sorted(set(left + right) - {head_l, tail_r, 0, w - 1})
    for _ in range(int(rng.integers(0, 3))):
        r = take_row()
        if r is None or not seam_cols:
            break
        beacons.add((r, int(rng.choice(seam_cols))))
        flags["seam"] += 1
    # vertical decoys: one end only
    if need.get("vdecoy") or rng.random() < 0.4:
        cand = [c for c in interior_cols if c not in vcols]
        if cand:
            c = int(rng.choice(cand))
            beacons.add((0, c) if rng.random() < 0.5 else (h - 1, c))
            flags["vdecoy"] = True

    for key in ("edge_decoy", "single_edge", "alt_head", "vdecoy"):
        if need.get(key) and not flags[key]:
            return None

    grid = np.zeros((h, w), dtype=int)
    for i in range(k):
        grid[:, left[i]:right[i] + 1] = colors[i]
    # link cells: interior column of each band, interior row, not a beacon
    for i in range(k):
        cand = [(r, c) for r in inner_rows for c in range(left[i] + 1, right[i])
                if (r, c) not in beacons]
        r, c = cand[int(rng.integers(len(cand)))]
        grid[r, c] = colors[succ[i]]
    clean = grid.copy()
    for (r, c) in beacons:
        grid[r, c] = beacon

    # purity margin
    for c in range(w):
        counts = np.bincount(grid[:, c], minlength=10)
        if counts.max() < GEN_PURITY * h:
            return None

    # expected output from construction parameters
    out = np.concatenate([clean[:, left[i]:right[i] + 1] for i in order], axis=1)
    new_pos = {}
    pos = 0
    for i in order:
        for c in range(left[i], right[i] + 1):
            new_pos[c] = pos + c - left[i]
        pos += widths[i]
    for r in line_rows:
        out[r, :] = beacon
    for c in vcols:
        out[:, new_pos[c]] = beacon

    info = {"k": k, "order": order, "is_rot": order == list(range(1, k)) + [0],
            "line_rows": line_rows, "vcols": vcols, "flags": flags,
            "widths": widths, "h": h}
    return grid, out, info


def _pair_needs(rng, num_train, num_test):
    train = [dict() for _ in range(num_train)]
    idx = [int(i) for i in rng.permutation(num_train)]
    # band counts: a 3-band and a 4-band demonstration always exist
    ks = [3, 4] + [int(rng.choice([3, 4, 4, 5])) for _ in range(num_train - 2)]
    for slot, kk in zip(idx, ks):
        train[slot]["k"] = kk
    # one non-rotation order is required; a rotation where room allows
    train[idx[0]]["order"] = "nonrot"
    if num_train >= 3:
        train[idx[1]]["order"] = "rot"
    # feature coverage spread over shuffled slots
    feats = ["edge_decoy", "single_edge", "alt_head", "vline", "vdecoy"]
    idx2 = [int(i) for i in rng.permutation(num_train)]
    for j, f in enumerate(feats):
        train[idx2[j % num_train]][f] = True
    tests = []
    for t in range(num_test):
        d = {"k": int(rng.choice([3, 4, 4, 5, 5])), "edge_decoy": True}
        if t == 0:
            d["order"] = "nonrot"
            d["alt_head"] = True
        tests.append(d)
    return train, tests


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    horizontal = bool(rng.random() < 0.35)
    for _attempt in range(50):
        train_needs, test_needs = _pair_needs(rng, num_train, num_test)
        pairs, infos, seen = [], [], set()
        ok = True
        for need in train_needs + test_needs:
            res = None
            for _ in range(200):
                res = _gen_pair(rng, need["k"], need)
                if res is None:
                    continue
                gi, go, info = res
                if horizontal:
                    gi, go = gi.T.copy(), go.T.copy()
                key = gi.tobytes() + bytes(gi.shape)
                if key in seen:
                    res = None
                    continue
                seen.add(key)
                break
            if res is None:
                ok = False
                break
            pairs.append({"input": gi, "output": go})
            infos.append(info)
        if not ok:
            continue
        puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not generate a valid puzzle")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _orient(grid):
    g = np.asarray(grid, dtype=int)
    pv, ph = _parse_any(g)
    assert (pv is None) != (ph is None), "exactly one band orientation must parse"
    if pv is not None:
        return "v", g, pv
    return "h", g.T, ph


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    orients = set()
    stats = {"train_k": set(), "train_nonrot": 0, "test_nonrot": 0}
    shortcut_beaten = {name: 0 for name in ("rotate", "predecessor", "leftmost_first")}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            gi = np.asarray(pair["input"], dtype=int)
            go = np.asarray(pair["output"], dtype=int)
            for g in (gi, go):
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                assert g.min() >= 0 and g.max() <= 9
            assert gi.shape == go.shape
            key = gi.tobytes() + bytes(gi.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            o, canon, parse = _orient(gi)
            orients.add(o)
            pred = solve(gi)
            assert np.array_equal(pred, go), "solve disagrees with output"
            k = len(parse["bands"])
            assert MIN_BANDS <= k <= MAX_BANDS
            assert parse["order"][-1] == 0
            # bands move intact apart from beacons and lines
            out_c = go if o == "v" else go.T
            beacon = parse["beacon"]
            h, w = canon.shape
            # at least one completed horizontal line from the new outer edges
            full_rows = [r for r in range(h) if np.all(out_c[r] == beacon)]
            assert full_rows, "each pair must activate a permutation-dependent line"
            # input-border pairs never coincide with a completed line
            for r in range(h):
                if canon[r, 0] == beacon and canon[r, w - 1] == beacon:
                    assert r not in full_rows
            # no beacon survives outside a completed line
            full_cols = [c for c in range(w) if np.all(out_c[:, c] == beacon)]
            mask = np.zeros_like(out_c, dtype=bool)
            mask[full_rows, :] = True
            mask[:, full_cols] = True
            assert not np.any((out_c == beacon) & ~mask)
            alts = _shortcuts(canon, parse)
            for name in ("no_permute", "no_lines", "lines_then_move"):
                assert not np.array_equal(alts[name], out_c), f"{name} shortcut matches"
            for name in shortcut_beaten:
                if not np.array_equal(alts[name], out_c):
                    shortcut_beaten[name] += split == "train"
            is_rot = parse["order"] == list(range(1, k)) + [0]
            if split == "train":
                stats["train_k"].add(k)
                stats["train_nonrot"] += not is_rot
            else:
                stats["test_nonrot"] += not is_rot
    assert len(orients) == 1, "orientation must be shared within a puzzle"
    assert {3, 4} <= stats["train_k"]
    assert stats["train_nonrot"] >= 1 and stats["test_nonrot"] >= 1
    for name, n in shortcut_beaten.items():
        assert n >= 1, f"training never refutes the {name} shortcut"
    # the first test must refute the fixed rotation shortcut
    o, canon, parse = _orient(test[0]["input"])
    out_c = np.asarray(test[0]["output"]) if o == "v" else np.asarray(test[0]["output"]).T
    assert not np.array_equal(_shortcuts(canon, parse)["rotate"], out_c)
    return True


def self_test():
    seeds = list(range(42))
    shapes, ks, orients, orders, sigs = set(), set(), set(), set(), set()
    vline_pairs = 0
    for seed in seeds:
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        for a, b in zip(p["train"] + p["test"], q["train"] + q["test"]):
            assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        validate_generated_puzzle(p)
        sigs.add(b"".join(np.asarray(x["input"]).tobytes() for x in p["train"] + p["test"]))
        for pair in p["train"] + p["test"]:
            gi = np.asarray(pair["input"])
            shapes.add(gi.shape)
            o, canon, parse = _orient(gi)
            orients.add(o)
            ks.add(len(parse["bands"]))
            orders.add(tuple(parse["order"]))
            go = np.asarray(pair["output"])
            out_c = go if o == "v" else go.T
            if any(np.all(out_c[:, c] == parse["beacon"]) for c in range(out_c.shape[1])):
                vline_pairs += 1
    assert len(sigs) == len(seeds)
    assert orients == {"v", "h"}
    assert ks == {3, 4, 5}
    assert len(shapes) >= 40
    assert len(orders) >= 15
    assert vline_pairs >= 20
    for a, b in ((2, 1), (10, 4), (5, 4), (3, 2)):
        p = generate_puzzle(7, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
