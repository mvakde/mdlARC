"""Cursor-hopping cross: an ordered frame-to-frame program relocates a crossing.

A straight divider separates a two-column instruction table from a payload
region.  The payload holds several hollow one-cell rectangular frames, each in
its own colour, and one pair of perpendicular full-span lines whose crossing
lies strictly inside exactly one frame (the *start* frame, parent A).  The
table rows are ordered ``source -> destination`` frame-colour pairs (parent B).
A single cursor starts at the start frame's colour; the rows are executed once,
top to bottom, each replacing the cursor when it currently equals the row's
source.  The crossing is then moved to the upper-right corner of the frame
named by the final cursor: the horizontal line follows that frame's top edge,
the vertical line its right edge, both spanning the payload region, and every
frame is drawn above every line (frames are opaque).  Old lines are erased.
Divider, table and frames are unchanged.
"""
from __future__ import annotations

import numpy as np

COLORS = tuple(range(10))


# ----------------------------------------------------------------------------
# Input-only parsing and solving
# ----------------------------------------------------------------------------

def _background(g):
    vals, counts = np.unique(g, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(order) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("ambiguous background")
    return int(vals[order[0]])


def _parse(grid):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _background(g)
    cands = []
    for c in range(W):
        col = g[:, c]
        if col[0] != bg and np.all(col == col[0]):
            cands.append(("v", c, int(col[0])))
    for r in range(H):
        row = g[r, :]
        if row[0] != bg and np.all(row == row[0]):
            cands.append(("h", r, int(row[0])))
    if len(cands) != 1:
        raise ValueError(f"expected one divider, found {len(cands)}")
    axis, idx, div = cands[0]
    if axis == "v":
        sides = [(0, H, 0, idx), (0, H, idx + 1, W)]
    else:
        sides = [(0, idx, 0, W), (idx + 1, H, 0, W)]
    if any(r1 <= r0 or c1 <= c0 for r0, r1, c0, c1 in sides):
        raise ValueError("divider on grid edge")

    def table_of(box):
        r0, r1, c0, c1 = box
        sub = g[r0:r1, c0:c1]
        mask = sub != bg
        if not mask.any():
            return None
        rr, cc = np.nonzero(mask)
        a, b, l, rt = rr.min(), rr.max(), cc.min(), cc.max()
        if rt - l != 1 or not mask[a:b + 1, l:rt + 1].all() or mask.sum() != 2 * (b - a + 1):
            return None
        return [(int(sub[i, l]), int(sub[i, l + 1])) for i in range(a, b + 1)]

    tables = [table_of(s) for s in sides]
    if sum(t is not None for t in tables) != 1:
        raise ValueError("expected one instruction table")
    ti = 0 if tables[0] is not None else 1
    program = tables[ti]
    pr0, pr1, pc0, pc1 = sides[1 - ti]
    P = g[pr0:pr1, pc0:pc1]

    frames, others = {}, []
    for col in sorted(set(np.unique(P).tolist()) - {bg}):
        rr, cc = np.nonzero(P == col)
        t, b, l, r = rr.min(), rr.max(), cc.min(), cc.max()
        per = np.zeros((b - t + 1, r - l + 1), dtype=bool)
        per[0, :] = per[-1, :] = per[:, 0] = per[:, -1] = True
        box = P[t:b + 1, l:r + 1] == col
        if b - t >= 2 and r - l >= 2 and np.array_equal(box, per):
            frames[col] = (int(t), int(l), int(b), int(r))
        else:
            others.append(col)
    if len(others) != 1:
        raise ValueError("expected exactly one line colour")
    line = others[0]
    lm = P == line
    rc, cc = lm.sum(axis=1), lm.sum(axis=0)
    row = int(np.argmax(rc))
    colm = int(np.argmax(cc))
    if rc[row] < 2 or cc[colm] < 2 or np.sum(rc == rc[row]) != 1 or np.sum(cc == cc[colm]) != 1:
        raise ValueError("cannot locate crossing")
    # every line cell must lie on the crossing row or column
    if np.any(lm & ~(np.arange(P.shape[0])[:, None] == row) & ~(np.arange(P.shape[1])[None, :] == colm)):
        raise ValueError("stray line cells")
    if P[row, colm] != line:
        raise ValueError("crossing hidden")
    starts = [f for f, (t, l, b, r) in frames.items() if t < row < b and l < colm < r]
    if len(starts) != 1:
        raise ValueError("crossing must lie inside exactly one frame")
    return {
        "bg": bg, "axis": axis, "divider_index": idx, "divider": div,
        "table_side": ti, "program": program, "payload": (pr0, pr1, pc0, pc1),
        "frames": frames, "line": line, "cross": (row, colm), "start": starts[0],
    }


def _run(program, start):
    token = start
    for s, d in program:
        if token == s:
            token = d
    return token


def _render(grid, info, final):
    out = np.array(grid, dtype=int).copy()
    pr0, pr1, pc0, pc1 = info["payload"]
    P = out[pr0:pr1, pc0:pc1]
    bg, line = info["bg"], info["line"]
    P[P == line] = bg
    t, l, b, r = info["frames"][final]
    rowv = P[t, :]
    rowv[rowv == bg] = line
    colv = P[:, r]
    colv[colv == bg] = line
    return out


def solve(input_grid):
    g = np.asarray(input_grid, dtype=int)
    info = _parse(g)
    final = _run(info["program"], info["start"])
    if final not in info["frames"]:
        raise ValueError("program names a missing frame")
    return _render(g, info, final)


# ----------------------------------------------------------------------------
# Shortcut predictors (cursor level) used to certify the demonstrations
# ----------------------------------------------------------------------------

def _sc_ignore_program(prog, start):
    return start


def _sc_first_match(prog, start):
    for s, d in prog:
        if s == start:
            return d
    return start


def _sc_fixed_point(prog, start):
    token = start
    for _ in range(12):
        new = _run(prog, token)
        if new == token:
            return token
        token = new
    return None  # no convergence: certainly differs from a one-pass answer


def _sc_graph_walk(prog, start):
    token, seen = start, {start}
    while True:
        nxt = next((d for s, d in prog if s == token), None)
        if nxt is None or nxt in seen:
            return token if nxt is None else None
        seen.add(nxt)
        token = nxt


def _sc_last_dest(prog, start):
    return prog[-1][1]


def _sc_first_source(prog, start):
    return _run(prog, prog[0][0])


def _sc_reverse(prog, start):
    return _run(prog[::-1], start)


SHORTCUTS = {
    "ignore_program": _sc_ignore_program,
    "first_match_only": _sc_first_match,
    "fixed_point": _sc_fixed_point,
    "graph_walk": _sc_graph_walk,
    "last_destination": _sc_last_dest,
    "start_at_first_source": _sc_first_source,
    "bottom_to_top": _sc_reverse,
}


def _refuted(prog, start):
    final = _run(prog, start)
    return {name for name, f in SHORTCUTS.items() if f(prog, start) != final}


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _make_program(rng, colors, start, length, hits):
    positions = set(int(x) for x in rng.choice(length, size=hits, replace=False))
    token, rows = start, []
    for i in range(length):
        if i in positions:
            d = int(rng.choice([c for c in colors if c != token]))
            rows.append((token, d))
            token = d
        else:
            s = int(rng.choice([c for c in colors if c != token]))
            d = int(rng.choice([c for c in colors if c != s]))
            rows.append((s, d))
    return rows, token


def _place_frames(rng, PH, PW, n, big):
    rects = []
    for i in range(n):
        for _ in range(200):
            lo = 4 if i == 0 else 3
            hi = 9 if big else 8
            h = int(rng.integers(lo, min(hi, PH - 1) + 1))
            w = int(rng.integers(lo, min(hi, PW - 2) + 1))
            t = int(rng.integers(0, PH - h + 1))
            l = int(rng.integers(0, PW - w + 1))
            b, r = t + h - 1, l + w - 1
            if all(b + 1 < t2 or b2 + 1 < t or r + 1 < l2 or r2 + 1 < l
                   for (t2, l2, b2, r2) in rects):
                rects.append((t, l, b, r))
                break
        else:
            return None
    return rects


def _choose_program(rng, kind):
    """Abstract program over frame indices 0..n-1; the crossing frame is index 0."""
    for _ in range(4000):
        if kind == "chain":
            n = int(rng.integers(3, 5)); L = int(rng.integers(2, 5)); hits = int(rng.integers(2, L + 1))
            need = {"ignore_program", "first_match_only", "bottom_to_top"}
        elif kind == "order":
            n = int(rng.integers(3, 5)); L = int(rng.integers(3, 6)); hits = int(rng.integers(1, L))
            need = {"ignore_program", "fixed_point", "graph_walk", "last_destination",
                    "start_at_first_source"}
        elif kind == "free":
            n = int(rng.integers(2, 5)); L = int(rng.integers(1, 5))
            hits = int(rng.integers(0 if rng.random() < 0.25 else 1, min(3, L) + 1))
            need = set()
        else:  # test
            n = int(rng.integers(3, 6)); L = int(rng.integers(3, 8)); hits = int(rng.integers(2, min(4, L) + 1))
            need = {"ignore_program"}
        prog, final = _make_program(rng, list(range(n)), 0, L, hits)
        ref = _refuted(prog, 0)
        if not need <= ref:
            continue
        if kind == "test" and len(ref) < 3:
            continue
        if kind == "free" and final == 0 and hits == 0 and n == 2:
            continue
        return n, prog
    return None


def _make_pair(rng, orient, table_side, n_frames, prog_idx, colors_spec, big):
    bg, div, line, fcols = colors_spec
    length = len(prog_idx)
    for _ in range(60):
        band = int(rng.integers(2, 6)) if orient == "v" else int(rng.integers(length, length + 4))
        if orient == "v":
            PH = int(rng.integers(max(9, length + 2), 27))
            PW = int(rng.integers(9, min(27, 30 - band - 1) + 1))
        else:
            PH = int(rng.integers(9, min(27, 30 - band - 1) + 1))
            PW = int(rng.integers(9, 27))
        if PH * PW < 30 * n_frames + 40:
            continue
        rects = _place_frames(rng, PH, PW, n_frames, big)
        if rects is None:
            continue
        t, l, b, r = rects[0]
        edge_rows = {x for (t2, _, b2, _) in rects[1:] for x in (t2, b2)}
        edge_cols = {x for (_, l2, _, r2) in rects[1:] for x in (l2, r2)}
        rows_ok = [x for x in range(t + 1, b) if x not in edge_rows]
        cols_ok = [x for x in range(l + 1, r) if x not in edge_cols]
        if not rows_ok or not cols_ok:
            continue
        # prefer crossings whose lines pierce another frame (a distractor)
        if rng.random() < 0.6:
            pr = [x for x in rows_ok if any(t2 < x < b2 for (t2, _, b2, _) in rects[1:])]
            pc = [x for x in cols_ok if any(l2 < x < r2 for (_, l2, _, r2) in rects[1:])]
            rows_ok = pr or rows_ok
            cols_ok = pc or cols_ok
        cr = int(rng.choice(rows_ok))
        cc = int(rng.choice(cols_ok))
        fc = [int(c) for c in fcols[:n_frames]]
        prog = [(fc[s], fc[d]) for s, d in prog_idx]
        P = np.full((PH, PW), bg, dtype=int)
        P[cr, :] = line
        P[:, cc] = line
        for i, (t2, l2, b2, r2) in enumerate(rects):
            col = fc[i]
            P[t2, l2:r2 + 1] = col
            P[b2, l2:r2 + 1] = col
            P[t2:b2 + 1, l2] = col
            P[t2:b2 + 1, r2] = col
        if orient == "v":
            T = np.full((PH, band), bg, dtype=int)
            r0 = int(rng.integers(0, PH - length + 1))
            c0 = int(rng.integers(0, band - 1))
            for i, (s, d) in enumerate(prog):
                T[r0 + i, c0] = s
                T[r0 + i, c0 + 1] = d
            D = np.full((PH, 1), div, dtype=int)
            parts = [T, D, P] if table_side == 0 else [P, D, T]
            g = np.concatenate(parts, axis=1)
        else:
            T = np.full((band, PW), bg, dtype=int)
            r0 = int(rng.integers(0, band - length + 1))
            c0 = int(rng.integers(0, PW - 1))
            for i, (s, d) in enumerate(prog):
                T[r0 + i, c0] = s
                T[r0 + i, c0 + 1] = d
            D = np.full((1, PW), div, dtype=int)
            parts = [T, D, P] if table_side == 0 else [P, D, T]
            g = np.concatenate(parts, axis=0)
        try:
            info = _parse(g)
        except ValueError:
            continue
        if info["start"] != fc[0] or info["program"] != prog or info["line"] != line:
            continue
        return {"input": g, "output": solve(g)}
    return None


def _check_counts(num_train, num_test):
    if num_test is None:
        num_test = 1
    if num_train is None:
        num_train = max(4, num_test + 1)
    for v in (num_train, num_test):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    return int(num_train), int(num_test)


def _attempt(rng, num_train, num_test):
    regime = ["stable", "rotate", "mixed"][int(rng.choice(3, p=[0.35, 0.3, 0.35]))]
    fixed_bg = rng.random() < 0.4
    base_orient = "v" if rng.random() < 0.55 else "h"
    base_side = int(rng.integers(0, 2))
    bg0 = int(rng.integers(0, 10))
    kinds = ["chain", "order"] + ["free"] * (num_train - 2) + ["test"] * num_test
    programs = [_choose_program(rng, k) for k in kinds]
    if any(p is None for p in programs):
        return None
    refuted = set().union(*(_refuted(p, 0) for _, p in programs[:num_train]))
    if refuted != set(SHORTCUTS):
        return None
    pairs = []
    for k, (n_frames, prog) in enumerate(programs):
        is_test = k >= num_train
        if regime == "stable":
            orient, side = base_orient, base_side
        elif regime == "rotate":
            orient = base_orient if not is_test else ("h" if base_orient == "v" else "v")
            side = int(rng.integers(0, 2))
        else:
            orient = "v" if rng.random() < 0.5 else "h"
            side = int(rng.integers(0, 2))
        bg = bg0 if fixed_bg else int(rng.integers(0, 10))
        rest = [c for c in COLORS if c != bg]
        rng.shuffle(rest)
        spec = (bg, int(rest[0]), int(rest[1]), [int(c) for c in rest[2:2 + n_frames]])
        pair = _make_pair(rng, orient, side, n_frames, prog, spec, is_test)
        if pair is None:
            return None
        pairs.append(pair)
    train = pairs[:num_train]
    order = rng.permutation(num_train)
    train = [train[i] for i in order]
    return {"train": train, "test": pairs[num_train:]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is None:
            continue
        keys = [p["input"].tobytes() + bytes(p["input"].shape) for s in ("train", "test") for p in puzzle[s]]
        if len(set(keys)) != len(keys):
            continue
        try:
            validate_generated_puzzle(puzzle)
        except (AssertionError, ValueError):
            continue
        return {s: [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in puzzle[s]]
                for s in ("train", "test")}
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------

def _pair_record(pair):
    gi = np.asarray(pair["input"], dtype=int)
    go = np.asarray(pair["output"], dtype=int)
    assert gi.ndim == 2 and gi.shape == go.shape, "shape mismatch"
    assert 1 <= gi.shape[0] <= 30 and 1 <= gi.shape[1] <= 30
    info = _parse(gi)
    frames, prog = info["frames"], info["program"]
    assert len(frames) >= 2, "need at least two frames"
    assert prog, "empty program"
    for s, d in prog:
        assert s != d, "row with equal source and destination"
        assert s in frames and d in frames, "program colour without a frame"
    roles = {info["bg"], info["divider"], info["line"]} | set(frames)
    assert len(roles) == 3 + len(frames), "colour roles must be distinct within a pair"
    # lines are not collinear with another frame's edge
    row, col = info["cross"]
    for f, (t, l, b, r) in frames.items():
        if f != info["start"]:
            assert row not in (t, b) and col not in (l, r), "line collinear with a frame edge"
    final = _run(prog, info["start"])
    assert np.array_equal(go, solve(gi)), "output disagrees with rule"
    assert not np.array_equal(gi, go), "identity pair"
    # structural invariants on the output: table, divider and frames unchanged
    pr0, pr1, pc0, pc1 = info["payload"]
    mask = np.ones_like(gi, dtype=bool)
    mask[pr0:pr1, pc0:pc1] = False
    assert np.array_equal(gi[mask], go[mask]), "non-payload cells changed"
    for f in frames:
        assert np.array_equal(gi == f, go == f), "frame changed"
    pierced = False
    for f, (t, l, b, r) in frames.items():
        if f != info["start"] and (t < row < b or l < col < r):
            pierced = True
    return {"info": info, "final": final, "refuted": _refuted(prog, info["start"]), "pierced": pierced}


def validate_generated_puzzle(puzzle):
    assert set(puzzle) == {"train", "test"}
    train = [_pair_record(p) for p in puzzle["train"]]
    test = [_pair_record(p) for p in puzzle["test"]]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    keys = [np.asarray(p["input"]).tobytes() + bytes(np.asarray(p["input"]).shape)
            for s in ("train", "test") for p in puzzle[s]]
    assert len(set(keys)) == len(keys), "duplicate inputs"
    refuted = set().union(*(r["refuted"] for r in train))
    missing = set(SHORTCUTS) - refuted
    assert not missing, f"training does not refute {sorted(missing)}"
    for r in test:
        assert r["final"] != r["info"]["start"], "test must need the program"
        assert len(r["refuted"]) >= 3, "test too easy for shortcuts"
    return True


def self_test():
    stats = {"orient": set(), "side": set(), "frames": set(), "length": set(),
             "cyclic": 0, "repeat_src": 0, "inert": 0, "pierced": 0, "bgs": set(),
             "shapes": set(), "rotate_shift": 0, "return_to_start": 0}
    for seed in range(42):
        p = generate_puzzle(seed, 4, 1)
        assert p == generate_puzzle(seed, 4, 1), "nondeterministic"
        validate_generated_puzzle(p)
        orients = []
        for split in ("train", "test"):
            for pair in p[split]:
                info = _parse(np.asarray(pair["input"]))
                prog = info["program"]
                orients.append(info["axis"])
                stats["orient"].add(info["axis"])
                stats["side"].add((info["axis"], info["table_side"]))
                stats["frames"].add(len(info["frames"]))
                stats["length"].add(len(prog))
                stats["bgs"].add(info["bg"])
                stats["shapes"].add(np.asarray(pair["input"]).shape)
                srcs = [s for s, _ in prog]
                stats["repeat_src"] += len(set(srcs)) < len(srcs)
                edges = {}
                for s, d in prog:
                    edges.setdefault(s, set()).add(d)
                stats["cyclic"] += any(s in edges.get(d, ()) for s, d in prog) or _sc_fixed_point(prog, info["start"]) is None
                token = info["start"]
                inert = False
                for s, d in prog:
                    if s != token:
                        inert = True
                    else:
                        token = d
                stats["inert"] += inert
                stats["return_to_start"] += _run(prog, info["start"]) == info["start"]
                row, col = info["cross"]
                stats["pierced"] += any(f != info["start"] and (t < row < b or l < col < r)
                                        for f, (t, l, b, r) in info["frames"].items())
        stats["rotate_shift"] += len(set(orients[:-1])) == 1 and orients[-1] != orients[0]
    assert stats["orient"] == {"v", "h"}
    assert len(stats["side"]) == 4
    assert len(stats["frames"]) >= 4 and len(stats["length"]) >= 6
    assert stats["cyclic"] >= 10 and stats["repeat_src"] >= 20 and stats["inert"] >= 60
    assert stats["pierced"] >= 40 and len(stats["bgs"]) == 10 and len(stats["shapes"]) >= 60
    assert stats["rotate_shift"] >= 5 and stats["return_to_start"] >= 3
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (3, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    stats["shapes"] = len(stats["shapes"])
    return {k: (sorted(v, key=str) if isinstance(v, set) else v) for k, v in stats.items()}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), default=str))
