"""Marker shafts: scattered markers select columns, runs of selected columns become
shafts, and each two-thick piece is rotated, size-matched and dropped into the
shaft of matching width, after which the shaft is completed as a beam.

Parent A (67c52801): size-matched slot filling, quarter-turn normalisation to a
two-cell-thick piece, bottom alignment, erasing the original piece.
Parent B (f5b8619d): occupancy-based column selection and completion of every
background cell of the selected columns.
Connecting concepts: run-grouping of selected columns into shafts whose width
is the slot size, and a gravity floor (one solid border line) that fixes which
direction counts as "down" (the whole scene may be rotated).
"""
from __future__ import annotations

import numpy as np

MAX_SIDE = 30


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def _components(grid: np.ndarray, background: int):
    """4-connected monochrome components of non-background cells."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == background:
                continue
            color = grid[r, c]
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((int(color), cells))
    return comps


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _floor_turns(grid: np.ndarray, background: int):
    """Quarter turns k such that np.rot90(grid, k) has the floor on the bottom."""
    turns = []
    for k in range(4):
        g = np.rot90(grid, k)
        row = g[-1]
        if g.shape[0] >= 3 and np.all(row == row[0]) and row[0] != background:
            turns.append(k)
    return turns


def _runs(active: np.ndarray):
    runs, start = [], None
    for c, flag in enumerate(list(active) + [False]):
        if flag and start is None:
            start = c
        elif not flag and start is not None:
            runs.append((start, c - start))
            start = None
    return runs


def _piece_size(height: int, width: int) -> int:
    """Slot width of a piece: the dimension that is not its two-cell thickness."""
    if height == 2:
        return width
    if width == 2:
        return height
    raise ValueError("piece is not two cells thick")


def _analyse_canonical(grid: np.ndarray, background: int):
    """Return markers, pieces and shafts of a floor-at-bottom grid."""
    h, w = grid.shape
    upper = grid[: h - 1]
    markers, pieces = [], []
    for color, cells in _components(upper, background):
        if len(cells) == 1:
            markers.append((color, cells[0]))
        else:
            rows = [y for y, _ in cells]
            cols = [x for _, x in cells]
            top, left = min(rows), min(cols)
            ph, pw = max(rows) - top + 1, max(cols) - left + 1
            if ph * pw != len(cells):
                raise ValueError("non-rectangular piece")
            pieces.append({"color": color, "top": top, "left": left, "h": ph, "w": pw,
                           "size": _piece_size(ph, pw)})
    active = np.zeros(w, dtype=bool)
    for _, (_, x) in markers:
        active[x] = True
    shafts = []
    for start, width in _runs(active):
        colors = [col for col, (_, x) in markers if start <= x < start + width]
        beam = max(set(colors), key=colors.count)
        shafts.append({"start": start, "width": width, "beam": beam})
    return markers, pieces, shafts


def _solve_canonical(grid: np.ndarray, background: int) -> np.ndarray:
    h, w = grid.shape
    markers, pieces, shafts = _analyse_canonical(grid, background)
    by_width = {}
    for s in shafts:
        if s["width"] in by_width:
            raise ValueError("repeated shaft width")
        by_width[s["width"]] = s
    if sorted(p["size"] for p in pieces) != sorted(by_width):
        raise ValueError("pieces do not match shafts")
    out = grid.copy()
    for p in pieces:  # erase every original piece location
        out[p["top"]:p["top"] + p["h"], p["left"]:p["left"] + p["w"]] = background
    for s in shafts:  # complete each selected column as a beam
        block = out[: h - 1, s["start"]:s["start"] + s["width"]]
        block[block == background] = s["beam"]
    for p in pieces:  # drop the normalised piece onto the floor of its shaft
        s = by_width[p["size"]]
        out[h - 3:h - 1, s["start"]:s["start"] + s["width"]] = p["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    background = _background(grid)
    turns = _floor_turns(grid, background)
    if len(turns) != 1:
        raise ValueError("floor is not uniquely identifiable")
    k = turns[0]
    return np.rot90(_solve_canonical(np.rot90(grid, k), background), -k).copy()


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        out[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
    return out


def _place_markers(rng, height, shafts, dense, blocked):
    """Isolated single-cell markers; each shaft column gets at least one."""
    limit = height - 4  # markers live in rows 0..height-4 (above the landing zone)
    if limit < 0:
        return None
    cells = []
    for start, width in shafts:
        prev_rows = []
        for c in range(start, start + width):
            options = [r for r in range(limit + 1)
                       if not blocked[r, c] and all(abs(r - p) >= 2 for p in prev_rows)]
            if not options:
                return None
            rows = [int(rng.choice(options))]
            if dense and rng.random() < 0.45:
                more = [r for r in options if abs(r - rows[0]) >= 2]
                if more:
                    rows.append(int(rng.choice(more)))
            cells.extend((r, c) for r in rows)
            prev_rows = rows
    return cells


def _make_pair(rng, sizes, order, colors, height, turns, opts):
    """Build one pair in canonical (floor-at-bottom) frame, then rotate."""
    bg, floor, marker = colors["bg"], colors["floor"], colors["marker"]
    n = len(sizes)
    shaft_widths = [sizes[i] for i in order]
    gaps = [int(rng.integers(1, opts["max_gap"] + 1)) for _ in range(n - 1)]
    left = int(rng.integers(0, opts["max_margin"] + 1))
    right = int(rng.integers(0, opts["max_margin"] + 1))
    width = left + right + sum(shaft_widths) + sum(gaps)
    min_w = max(max(sizes) + 1, 3)
    if width < min_w:
        right += min_w - width
        width = min_w
    if width > MAX_SIDE or height > MAX_SIDE:
        return None
    shafts, x = [], left
    for i, sw in enumerate(shaft_widths):
        shafts.append((x, sw))
        x += sw + (gaps[i] if i < n - 1 else 0)
    grid = np.full((height, width), bg, dtype=int)
    grid[height - 1, :] = floor
    taken = np.zeros((height, width), dtype=bool)
    taken[height - 1, :] = True
    blocked = _dilate(taken)
    pieces = []
    piece_colors = colors["pieces"]
    for idx in rng.permutation(n):
        k = sizes[idx]
        if k == 2:
            ph, pw = 2, 2
        else:
            upright = rng.random() < opts["upright_prob"]
            ph, pw = (2, k) if upright else (k, 2)
        if ph > height - 2 or pw > width:
            return None
        spots = [(t, l) for t in range(0, height - 2 - ph + 1) for l in range(0, width - pw + 1)
                 if not blocked[t:t + ph, l:l + pw].any()]
        if not spots:
            return None
        t, l = spots[int(rng.integers(len(spots)))]
        grid[t:t + ph, l:l + pw] = piece_colors[idx]
        taken[t:t + ph, l:l + pw] = True
        blocked = _dilate(taken)
        pieces.append((int(idx), t, l, ph, pw))
    marks = _place_markers(rng, height, shafts, opts["dense_markers"], blocked)
    if marks is None:
        return None
    for r, c in marks:
        grid[r, c] = marker
    canonical_out = _solve_canonical(grid, bg)
    inp = np.rot90(grid, -turns).copy()
    out = np.rot90(canonical_out, -turns).copy()
    meta = {"sizes": list(sizes), "shaft_widths": shaft_widths, "turns": turns,
            "pieces": pieces, "shafts": shafts, "markers": len(marks),
            "rotated": any(ph != 2 for _, _, _, ph, _ in pieces)}
    return inp, out, meta


def _pair_colors(rng, n, fixed_bg):
    bg = 0 if (fixed_bg or rng.random() < 0.7) else int(rng.integers(1, 10))
    rest = [c for c in range(10) if c != bg]
    rng.shuffle(rest)
    floor, marker = rest[0], rest[1]
    pool = rest[2:]
    if rng.random() < 0.25 and n >= 2:
        # some pieces share a colour; sizes still decide their shafts
        base = [int(rng.choice(pool)) for _ in range(max(1, n - 1))]
        piece_colors = [base[i % len(base)] for i in range(n)]
        rng.shuffle(piece_colors)
    else:
        piece_colors = [int(c) for c in rng.choice(pool, size=n, replace=n > len(pool))]
    return {"bg": bg, "floor": floor, "marker": marker, "pieces": [int(c) for c in piece_colors]}


def _shortcuts(inp: np.ndarray):
    """Outputs of plausible simpler rules (canonical frame), for validation."""
    grid = np.asarray(inp, dtype=int)
    bg = _background(grid)
    k = _floor_turns(grid, bg)[0]
    g = np.rot90(grid, k)
    h, w = g.shape
    markers, pieces, shafts = _analyse_canonical(g, bg)
    results = {}
    # (1) beams only, pieces untouched (ignores parent A)
    beams = g.copy()
    for s in shafts:
        block = beams[: h - 1, s["start"]:s["start"] + s["width"]]
        block[block == bg] = s["beam"]
    results["beams_only"] = beams
    # (2) slot filling with no beams (ignores parent B's completion)
    full = _solve_canonical(g, bg)
    nobeam = full.copy()
    for s in shafts:
        block = nobeam[: h - 3, s["start"]:s["start"] + s["width"]]
        block[block == s["beam"]] = bg
        for _, (y, x) in markers:
            nobeam[y, x] = g[y, x]
    results["no_beams"] = nobeam
    # (3) drop each piece straight down, unrotated, beams on marker columns
    straight = g.copy()
    for p in pieces:
        straight[p["top"]:p["top"] + p["h"], p["left"]:p["left"] + p["w"]] = bg
    for s in shafts:
        block = straight[: h - 1, s["start"]:s["start"] + s["width"]]
        block[block == bg] = s["beam"]
    for p in pieces:
        straight[h - 1 - p["h"]:h - 1, p["left"]:p["left"] + p["w"]] = p["color"]
    results["drop_straight"] = straight
    # (4) slot order by left-to-right piece order instead of size
    ordered = g.copy()
    for p in pieces:
        ordered[p["top"]:p["top"] + p["h"], p["left"]:p["left"] + p["w"]] = bg
    for s in shafts:
        block = ordered[: h - 1, s["start"]:s["start"] + s["width"]]
        block[block == bg] = s["beam"]
    for p, s in zip(sorted(pieces, key=lambda q: (q["left"], q["top"])), shafts):
        span = min(s["width"], p["size"])
        ordered[h - 3:h - 1, s["start"]:s["start"] + span] = p["color"]
    results["order_match"] = ordered
    # (5) match by the piece's current horizontal width, never turning it
    norot = g.copy()
    for p in pieces:
        norot[p["top"]:p["top"] + p["h"], p["left"]:p["left"] + p["w"]] = bg
    for s in shafts:
        block = norot[: h - 1, s["start"]:s["start"] + s["width"]]
        block[block == bg] = s["beam"]
    for p in pieces:
        s = next((s for s in shafts if s["width"] == p["w"]), None)
        if s is not None:
            norot[h - 1 - p["h"]:h - 1, s["start"]:s["start"] + p["w"]] = p["color"]
    results["match_by_width_no_rotation"] = norot
    return {name: np.rot90(v, -k) for name, v in results.items()}


def _training_covers(train_meta, train_pairs):
    if not any(len(m["sizes"]) >= 2 for m in train_meta):
        return False
    if not any(max(m["sizes"]) >= 2 for m in train_meta):
        return False
    if not any(m["rotated"] for m in train_meta):
        return False
    if not any(any(p[3] == 2 and p[4] != 2 for p in m["pieces"]) for m in train_meta):
        return False  # also show an already-upright non-square piece
    for name in ("drop_straight", "order_match", "match_by_width_no_rotation"):
        if not any(not np.array_equal(_shortcuts(p["input"])[name], p["output"])
                   for p in train_pairs):
            return False
    return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # puzzle-level choices
    orient_mode = rng.choice(["fixed", "fixed", "mixed"])
    if orient_mode == "fixed":
        base_turn = int(rng.choice([0, 0, 0, 1, 2, 3]))
        train_turns = [base_turn] * num_train
    else:
        pool = [int(t) for t in rng.permutation(4)[: int(rng.integers(2, 4))]]
        train_turns = [pool[i % len(pool)] for i in range(num_train)]
        train_turns = [int(t) for t in rng.permutation(train_turns)]
    shift = str(rng.choice(["wider", "more", "familiar"]))
    fixed_bg = rng.random() < 0.6
    train_max_width = int(rng.integers(3, 5))  # 3 or 4
    opts_base = {
        "max_gap": int(rng.integers(1, 4)),
        "max_margin": int(rng.integers(0, 4)),
        "dense_markers": bool(rng.random() < 0.5),
        "upright_prob": float(rng.uniform(0.3, 0.7)),
    }
    compact = rng.random() < 0.35

    def sizes_for(count, max_w, must_include=None):
        widths = list(range(1, max_w + 1))
        chosen = [int(v) for v in rng.choice(widths, size=min(count, len(widths)), replace=False)]
        if must_include is not None and must_include not in chosen:
            chosen[int(rng.integers(len(chosen)))] = must_include
        return chosen

    def order_for(n):
        roll = rng.random()
        if roll < 0.3:
            return None  # ascending by size (source-like rack)
        if roll < 0.4:
            return "desc"
        return [int(i) for i in rng.permutation(n)]

    def build(count, max_w, turns, must_include=None, tall=False):
        for attempt in range(120):
            bump = attempt // 12
            sizes = sizes_for(count, max_w, must_include)
            n = len(sizes)
            o = order_for(n)
            if o is None:
                order = sorted(range(n), key=lambda i: sizes[i])
            elif o == "desc":
                order = sorted(range(n), key=lambda i: -sizes[i])
            else:
                order = o
            colors = _pair_colors(rng, n, fixed_bg)
            lo = max(max(sizes) + 3, 5)
            if compact and not tall:
                height = int(rng.integers(lo, lo + 3)) + bump
            else:
                height = int(rng.integers(lo + 1, min(lo + 8, 16) + 1)) + bump
            height = min(height, 20)
            res = _make_pair(rng, sizes, order, colors, height, turns, opts_base)
            if res is None:
                continue
            inp, out, meta = res
            if _background(inp) != colors["bg"] or len(_floor_turns(inp, colors["bg"])) != 1:
                continue
            if np.array_equal(inp, out):
                continue
            return inp, out, meta
        raise RuntimeError("could not build pair")

    for _attempt in range(80):
        train, train_meta, seen = [], [], set()
        max_count = 1
        for i in range(num_train):
            count = int(rng.choice([1, 2, 2, 3, 3, 3, 4]))
            count = min(count, train_max_width)
            for _ in range(20):
                inp, out, meta = build(count, train_max_width, train_turns[i])
                key = inp.tobytes() + bytes(inp.shape)
                if key not in seen:
                    break
            seen.add(key)
            max_count = max(max_count, len(meta["sizes"]))
            train.append({"input": inp, "output": out})
            train_meta.append(meta)
        if not _training_covers(train_meta, train):
            continue
        test = []
        test_turn_pool = sorted(set(train_turns))
        ok = True
        for j in range(num_test):
            turn = int(rng.choice(test_turn_pool))
            if shift == "wider":
                new_w = train_max_width + int(rng.integers(1, 3))
                count = int(rng.integers(2, min(4, new_w) + 1))
                args = (count, new_w, turn, new_w)
            elif shift == "more":
                count = min(max_count + 1, 5)
                args = (count, max(count, train_max_width), turn, None)
            else:
                count = int(rng.integers(1, train_max_width + 1))
                args = (count, train_max_width, turn, None)
            for _ in range(20):
                inp, out, meta = build(*args, tall=True)
                key = inp.tobytes() + bytes(inp.shape)
                if key not in seen:
                    break
            else:
                ok = False
                break
            seen.add(key)
            test.append({"input": inp, "output": out})
        if not ok:
            continue
        puzzle = {"train": train, "test": test}
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("could not generate puzzle")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def _validate_pair(pair):
    inp = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    assert inp.ndim == 2 and inp.shape == out.shape
    assert all(1 <= n <= MAX_SIDE for n in inp.shape)
    bg = _background(inp)
    turns = _floor_turns(inp, bg)
    assert len(turns) == 1, "floor must be a unique solid border line"
    g = np.rot90(inp, turns[0])
    o = np.rot90(out, turns[0])
    h, w = g.shape
    floor = int(g[-1, 0])
    markers, pieces, shafts = _analyse_canonical(g, bg)
    assert markers and pieces and shafts
    mcolors = {c for c, _ in markers}
    assert len(mcolors) == 1, "one marker colour per pair"
    marker = mcolors.pop()
    assert marker not in (bg, floor)
    assert all(p["color"] not in (bg, floor, marker) for p in pieces)
    # markers are isolated singletons above the landing zone
    for _, (y, x) in markers:
        assert y <= h - 4
        nb = g[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        assert int(np.sum(nb != bg)) == 1, "marker must be isolated"
    # pieces are solid, two-thick, detached from everything else
    mask = g != bg
    mask[-1, :] = False
    for p in pieces:
        assert 2 in (p["h"], p["w"])
        assert p["top"] + p["h"] - 1 <= h - 3, "piece touches the floor"
        t, l = p["top"], p["left"]
        ring = mask[max(0, t - 1):t + p["h"] + 1, max(0, l - 1):l + p["w"] + 1]
        assert int(ring.sum()) == p["h"] * p["w"], "piece must be detached"
    widths = [s["width"] for s in shafts]
    assert len(set(widths)) == len(widths), "shaft widths must be distinct"
    assert sorted(widths) == sorted(p["size"] for p in pieces)
    assert np.array_equal(solve(inp), out)
    assert not np.array_equal(inp, out)
    assert np.all(o[-1] == floor)
    return {"turn": turns[0], "widths": widths, "count": len(pieces),
            "rotated": any(p["h"] != 2 for p in pieces), "bg": bg,
            "shape": inp.shape}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    infos = []
    for pair in train + test:
        assert set(pair) == {"input", "output"}
        key = np.asarray(pair["input"]).tobytes() + bytes(np.asarray(pair["input"]).shape)
        assert key not in seen, "duplicate input"
        seen.add(key)
        infos.append(_validate_pair(pair))
    tr = infos[: len(train)]
    te = infos[len(train):]
    assert any(i["count"] >= 2 for i in tr), "training needs a multi-piece pair"
    assert any(max(i["widths"]) >= 2 for i in tr), "training needs a multi-column shaft"
    assert any(i["rotated"] for i in tr), "training needs a piece that must turn"
    train_turns = {i["turn"] for i in tr}
    assert all(i["turn"] in train_turns for i in te), "test floor side must be demonstrated"
    tr_pairs = [{"input": np.asarray(p["input"]), "output": np.asarray(p["output"])} for p in train]
    for name in ("beams_only", "no_beams", "drop_straight", "order_match",
                 "match_by_width_no_rotation"):
        assert any(not np.array_equal(_shortcuts(p["input"])[name], p["output"])
                   for p in tr_pairs), f"training does not refute shortcut {name}"


def self_test() -> None:
    stats = {"turn_sets": set(), "counts": set(), "widths": set(), "shapes": set(),
             "bgs": set(), "test_max": set()}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed)
        assert all(np.array_equal(np.asarray(a[r]), np.asarray(b[r]))
                   for s in ("train", "test") for a, b in zip(p[s], q[s]) for r in ("input", "output"))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _validate_pair(pair)
                stats["counts"].add(info["count"])
                stats["widths"].update(info["widths"])
                stats["shapes"].add(info["shape"])
                stats["bgs"].add(info["bg"])
                stats["turn_sets"].add(info["turn"])
    assert len(stats["turn_sets"]) == 4, stats["turn_sets"]
    assert len(stats["counts"]) >= 4, stats["counts"]
    assert len(stats["widths"]) >= 5, stats["widths"]
    assert len(stats["shapes"]) >= 40
    assert len(stats["bgs"]) >= 3
    for bad in ((1, 1), (11, 1), (4, 0), (2, 2), (3, 4), (6, 5)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
