"""Normalize a stretched pixel board, look up exact glyphs, and report a line.

Only standard-library modules and NumPy are used. No files, example answers,
parent generators, global RNG state, or construction metadata are read by solve.
All optional diagnostic JSON must use json.dumps(..., separators=(",", ":")).
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


HOST_SEEDS = (0, 776645947, 489994690, 807829634, 73056570, 688642035,
              668512330, 335264676, 846092744, 417999162, 742714564,
              525285040, 550971920, 302103474, 466229193, 878392316,
              372969394, 986363351, 779108433, 752856677, 663135351,
              768761613, 379326074, 934016766, 65109631, 806032786,
              619197601, 885000753, 527540213, 904651136, 121271097,
              227285227, 404888182, 892691100, 563276673, 249910224,
              528589442, 454824444, 255438060, 508698276, 526148942,
              449615783)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer)
             and bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.int8).tobytes()).hexdigest()


def _starts(a, axis):
    """Compare complete pixel vectors, never logical slot vectors."""
    lines = a if axis == 0 else a.T
    return [0] + [i for i in range(1, len(lines))
                  if not np.array_equal(lines[i], lines[i - 1])]


def _compress(a):
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _runs(a, axis):
    edges = _starts(a, axis) + [a.shape[axis]]
    return tuple(b - x for x, b in zip(edges, edges[1:]))


def _line_axes(selected, nr, nc):
    s = set(selected)
    axes = set()
    if any(all(r * nc + c in s for c in range(nc)) for r in range(nr)):
        axes.add("row")
    if any(all(r * nc + c in s for r in range(nr)) for c in range(nc)):
        axes.add("col")
    return axes


def _diagonal(s, nr, nc):
    return nr == nc and nr >= 3 and set(s) in (
        {i * nc + i for i in range(nr)},
        {i * nc + nc - i - 1 for i in range(nr)})


def _parse(input_grid):
    a = _grid(input_grid)
    _require(a.shape[1] >= 7 and a.shape[0] >= 14, "scene too small")
    bg, frame, divider, success, failure = map(int, a[0, :5])
    _require(len({bg, frame, divider, success, failure}) == 5, "ambiguous legend")
    _require(np.all(a[0, 5:] == bg), "unexpected legend tail")
    bars = [i for i in range(1, len(a)) if np.all(a[i] == frame)]
    _require(len(bars) == 3 and bars[0] == 1 and bars[-1] == len(a) - 2,
             "expected three compartment bars")
    gh = bars[1] - 2
    _require(gh in (2, 3), "invalid query height")
    tray = a[2:bars[1]]
    raw = a[bars[1] + 1:bars[2]]
    _require(np.all(a[-1] == bg), "input status must be blank")
    board = _compress(raw)
    row_divs = [i for i in range(len(board)) if np.all(board[i] == divider)]
    col_divs = [i for i in range(board.shape[1]) if np.all(board[:, i] == divider)]
    nr, nc = len(row_divs) - 1, len(col_divs) - 1
    _require(2 <= nr <= nc <= 4, "invalid board dimensions")
    _require(row_divs == list(range(0, board.shape[0], gh + 1))
             and row_divs[-1] == len(board) - 1, "invalid recovered row lattice")
    gw = col_divs[1] - col_divs[0] - 1
    _require(gw in (2, 3) and (gh == 3 or gw == 3), "invalid glyph width")
    _require(col_divs == list(range(0, board.shape[1], gw + 1))
             and col_divs[0] == 0 and col_divs[-1] == board.shape[1] - 1,
             "invalid recovered column lattice")
    _require(np.all(tray[:, col_divs] == divider)
             and np.all(tray[:, board.shape[1]:] == bg), "invalid query separators")
    tokens, lookup = [], {}
    for r, c in itertools.product(range(nr), range(nc)):
        token = board[r * (gh + 1) + 1:(r + 1) * (gh + 1),
                      c * (gw + 1) + 1:(c + 1) * (gw + 1)].copy()
        colors = set(map(int, np.unique(token))) - {bg}
        _require(len(colors) == 1 and not colors & {frame, divider, success, failure},
                 "glyph must have one nonstructural ink")
        mask = token != bg
        _require(all(np.any(mask[x]) for x in (0, -1))
                 and all(np.any(mask[:, x]) for x in (0, -1)), "glyph must span its canvas")
        _require(_key(token) not in lookup, "duplicate colored glyph")
        lookup[_key(token)] = r * nc + c
        tokens.append(token)
    query, slots = [], []
    for c in range(nc):
        token = tray[:, c * (gw + 1) + 1:(c + 1) * (gw + 1)]
        if np.all(token == bg):
            slots.append(None)
        else:
            _require(_key(token) in lookup, "query has no exact recovered match")
            p = lookup[_key(token)]
            query.append(p)
            slots.append(p)
    _require(1 <= len(query) <= nc and len(set(query)) == len(query), "invalid query set")
    return dict(grid=a, bg=bg, frame=frame, divider=divider, success=success,
                failure=failure, gh=gh, gw=gw, nr=nr, nc=nc, raw=raw,
                board=board, tray=tray, tokens=tokens, query=query, slots=slots,
                win=bool(_line_axes(query, nr, nc)), axes=_line_axes(query, nr, nc),
                board0=bars[1] + 1, rows=_runs(raw, 0), cols=_runs(raw, 1))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer regions, normalize the board, exact-match, paint, then decide status."""
    info = _parse(input_grid)
    board = info["board"].copy()
    gh, gw, nc = (info[k] for k in ("gh", "gw", "nc"))
    bg, success = info["bg"], info["success"]
    for p in info["query"]:
        r, c = divmod(p, nc)
        slot = board[r * (gh + 1) + 1:(r + 1) * (gh + 1),
                     c * (gw + 1) + 1:(c + 1) * (gw + 1)]
        slot[slot == bg] = success
    w = board.shape[1]
    prefix = info["grid"][:info["board0"], :w].copy()
    if info["win"]:
        tray = prefix[2:gh + 2]
        tray[tray == bg] = success
    status = success if info["win"] else info["failure"]
    return np.vstack((prefix, board, np.full((1, w), info["frame"], np.int8),
                      np.full((1, w), status, np.int8)))


def _count(value, name, lo, hi):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)) and lo <= int(value) <= hi,
             name + " must be an integer in its permitted range")
    return int(value)


def _selection(rng, nr, nc, kind, axis, count=None):
    lines = ([[r * nc + c for c in range(nc)] for r in range(nr)]
             if axis == "row" else [[r * nc + c for r in range(nr)] for c in range(nc)])
    if kind == "win":
        return lines[int(rng.integers(len(lines)))]
    if kind == "near":
        choices = []
        for line in lines:
            for missing in line:
                for extra in range(nr * nc):
                    s = sorted((set(line) - {missing}) | {extra})
                    if extra not in line and not _line_axes(s, nr, nc) and not _diagonal(s, nr, nc):
                        choices.append(s)
        return choices[int(rng.integers(len(choices)))]
    count = int(rng.integers(1, nr)) if kind == "partial" else count
    # At most C(16,4)=1820 candidates, with no unbounded rejection loop.
    choices = [s for s in itertools.combinations(range(nr * nc), count)
               if not _line_axes(s, nr, nc) and not _diagonal(s, nr, nc)]
    return list(choices[int(rng.integers(len(choices)))])


def _mask_bank(gh, gw):
    bank = []
    for code in range(1, 1 << (gh * gw)):
        m = np.array([(code >> i) & 1 for i in range(gh * gw)], bool).reshape(gh, gw)
        if (all(np.any(m[x]) for x in (0, -1))
                and all(np.any(m[:, x]) for x in (0, -1))):
            bank.append(m)
    return bank


def _normal_board(rng, nr, nc, gh, gw, roles, selected, ink_count):
    bg, _, divider, _, _ = roles
    inks = list(map(int, rng.choice([x for x in range(10) if x not in roles],
                                    ink_count, replace=False)))
    bank = _mask_bank(gh, gw)
    anchors = [m for m in bank if _compress(m).shape == m.shape]
    # The selected stripe contains real repeated local rows or columns. A
    # different tile witnesses their boundaries in the complete board vectors.
    stripe = np.ones((gh, gw), bool)
    if gw == 3:
        stripe[:, 1] = False
    else:
        stripe[1, :] = False
    outside = [p for p in range(nr * nc) if p not in selected]
    p = int(rng.choice(selected))
    q_options = [x for x in outside if (x // nc != p // nc if gw == 3 else x % nc != p % nc)]
    q = int(rng.choice(q_options))
    t = int(rng.choice([x for x in outside if x != q]))
    for attempt in range(17):
        pool = anchors if attempt == 16 or rng.random() < .45 else bank
        forced = {p: (stripe, inks[0]), q: (stripe, inks[1]),
                  t: (anchors[int(rng.integers(len(anchors)))], inks[0])}
        used = {(_key(m), ink) for m, ink in forced.values()}
        catalogue = [(m, ink) for m in pool for ink in inks if (_key(m), ink) not in used]
        order = rng.permutation(len(catalogue))
        tokens, j = [], 0
        for pos in range(nr * nc):
            if pos in forced:
                m, ink = forced[pos]
            else:
                m, ink = catalogue[int(order[j])]
                j += 1
            tokens.append(np.where(m, ink, bg).astype(np.int8))
        board = np.full((nr * (gh + 1) + 1, nc * (gw + 1) + 1), divider, np.int8)
        for pos, token in enumerate(tokens):
            r, c = divmod(pos, nc)
            board[r * (gh + 1) + 1:(r + 1) * (gh + 1),
                  c * (gw + 1) + 1:(c + 1) * (gw + 1)] = token
        if _compress(board).shape == board.shape:
            return board, tokens
    raise AssertionError("anchor construction must have observable full-line boundaries")


def _schedule_runs(rng, length, budget, testing):
    runs = np.ones(length, dtype=int)
    order = rng.permutation(length)
    # Train uses at most 2; test has a concentrated band of 3 or 4. In either
    # case at least one 1 survives, so a single uniform resize factor fails.
    runs[int(order[0])] = int(rng.integers(3, 5)) if testing else 2
    remaining = budget - int(runs.sum())
    extras = int(rng.integers(0, min(remaining, max(1, length // 2)) + 1))
    for i in order[1:1 + extras]:
        runs[int(i)] += 1
    return tuple(map(int, runs))


def _render(rng, nr, nc, gh, gw, roles, selected, testing):
    board, tokens = _normal_board(rng, nr, nc, gh, gw, roles, selected,
                                  int(rng.integers(2, 6)))
    bg, frame, divider, success, failure = roles
    rr = _schedule_runs(rng, len(board), 30 - gh - 5, testing)
    cc = _schedule_runs(rng, board.shape[1], 30, testing)
    raw = np.repeat(np.repeat(board, rr, axis=0), cc, axis=1)
    inp = np.full((gh + 5 + len(raw), raw.shape[1]), bg, np.int8)
    inp[0, :5] = roles
    inp[1] = inp[gh + 2] = inp[-2] = frame
    tray = inp[2:gh + 2]
    tray[:, :board.shape[1]:gw + 1] = divider
    order = list(selected)
    if rng.random() < .8:
        rng.shuffle(order)
    slots = sorted(map(int, rng.choice(nc, len(order), replace=False)))
    for slot, pos in zip(slots, order):
        tray[:, slot * (gw + 1) + 1:(slot + 1) * (gw + 1)] = tokens[pos]
    inp[gh + 3:-2] = raw
    # Construction-time labels use the latent selection, not solve().
    marked = board.copy()
    for pos in selected:
        r, c = divmod(pos, nc)
        part = marked[r * (gh + 1) + 1:(r + 1) * (gh + 1),
                      c * (gw + 1) + 1:(c + 1) * (gw + 1)]
        part[part == bg] = success
    win = bool(_line_axes(selected, nr, nc))
    out = np.full((gh + 5 + len(board), board.shape[1]), bg, np.int8)
    out[0, :5] = roles
    out[1] = out[gh + 2] = out[-2] = frame
    out[2:gh + 2] = tray[:, :board.shape[1]]
    if win:
        part = out[2:gh + 2]
        part[part == bg] = success
    out[gh + 3:-2] = marked
    out[-1] = success if win else failure
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    nr, nc = ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4))[int(rng.integers(6))]
    gh, gw = ((2, 3), (3, 2), (3, 3))[int(rng.integers(3))]
    roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    first_axis = "row" if rng.integers(2) == 0 else "col"
    other_axis = "col" if first_axis == "row" else "row"
    lesson_count = nc if first_axis == "row" else nr
    schedule = [("win", first_axis, None), ("near", first_axis, lesson_count)]
    if nt >= 3:
        schedule.append(("win", other_axis, None))
    if nt >= 4:
        schedule.append(("partial", first_axis, None))
    while len(schedule) < nt:
        kind = str(rng.choice(["win", "near", "scatter", "partial"]))
        schedule.append((kind, str(rng.choice(["row", "col"])), lesson_count))
    taught_axes = [first_axis] if nt == 2 else ["row", "col"]
    for _ in range(nq):
        failures = ["near", "scatter"] + (["partial"] if nt >= 4 else [])
        kind = "win" if rng.random() < .5 else str(rng.choice(failures))
        axis = str(rng.choice(taught_axes)) if kind == "win" else first_axis
        schedule.append((kind, axis, lesson_count))
    pairs, seen = [], set()
    for index, (kind, axis, count) in enumerate(schedule):
        selected = _selection(rng, nr, nc, kind, axis, count)
        for _ in range(8):
            pair = _render(rng, nr, nc, gh, gw, roles, selected, index >= nt)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                pairs.append(pair)
                break
        else:
            raise RuntimeError("distinct input construction exhausted")
    puzzle = {"train": [pairs[int(i)] for i in rng.permutation(nt)],
              "test": [pairs[nt + int(i)] for i in rng.permutation(nq)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    masks = [_key(t != info["bg"]) for t in info["tokens"]]
    inks = [int(t[t != info["bg"]][0]) for t in info["tokens"]]
    selected = set(info["query"])
    outside = set(range(len(masks))) - selected
    return dict(color_needed=any(masks[p] == masks[q] and inks[p] != inks[q]
                                 for p in selected for q in outside),
                shape_needed=any(inks[p] == inks[q] and masks[p] != masks[q]
                                 for p in selected for q in outside),
                local_repeats=any(_compress(info["tokens"][p]).shape != info["tokens"][p].shape
                                  for p in selected))


def validate_generated_puzzle(puzzle):
    """Check semantic pixels, observable normalization, teaching, and the shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(nq, "num_test", 1, 4)
    _require(nt > nq, "invalid split counts")
    infos, seen = [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            _require(np.array_equal(solve(inp), out), "output violates the family rule")
            # Full-line run recovery must be lossless before any highlighting.
            restored = np.repeat(np.repeat(info["board"], info["rows"], axis=0), info["cols"], axis=1)
            _require(np.array_equal(restored, info["raw"]), "normalization not lossless")
            _require(all(min(info[k]) == 1 and len(set(info[k])) > 1 for k in ("rows", "cols")),
                     "normalization lacks unequal bands")
            expected_max = (3, 4) if split == "test" else (2,)
            _require(all(max(info[k]) in expected_max for k in ("rows", "cols")), "wrong band shift")
            _require(all(_features(info).values()), "missing exact-color/shape/global-compression witnesses")
            _require(not _diagonal(info["query"], info["nr"], info["nc"]), "untaught full diagonal")
            _require(out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1], "compression omitted")
            infos.append(info)
    train, tests = infos[:nt], infos[nt:]
    conventions = {(i["bg"], i["frame"], i["divider"], i["success"], i["failure"],
                    i["nr"], i["nc"], i["gh"], i["gw"]) for i in infos}
    _require(len(conventions) == 1, "within-puzzle conventions changed")
    _require(any(i["win"] for i in train) and any(not i["win"] for i in train), "both statuses must be taught")
    _require(any(len(a["query"]) == len(b["query"]) and a["win"] != b["win"]
                 for a in train for b in train), "query-count shortcut not refuted")
    taught_axes = set().union(*(i["axes"] for i in train))
    _require(all(i["axes"] <= taught_axes for i in tests), "unwitnessed winning axis")
    if any(None in i["slots"] for i in tests):
        _require(any(None in i["slots"] for i in train), "empty slots not taught")
    if any(not i["win"] and None not in i["slots"] for i in tests):
        _require(any(not i["win"] and None not in i["slots"] for i in train), "full-tray failure not taught")


def self_test():
    """42 host seeds; bounded streaming metrics, ablations, and damaged labels."""
    layouts, shapes, palettes, schedules, selections, boards = (set() for _ in range(6))
    outcomes, puzzles = set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        _require(all(np.array_equal(x[k], y[k]) for s in puzzle
                     for x, y in zip(puzzle[s], again[s]) for k in x), "nondeterminism")
        puzzles.add(tuple(_digest(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                i = _parse(pair["input"])
                layouts.add((i["nr"], i["nc"], i["gh"], i["gw"]))
                shapes.add(pair["input"].shape)
                palettes.add(tuple(map(int, pair["input"][0, :5])))
                schedules.add((i["rows"], i["cols"]))
                selections.add((i["nr"], i["nc"], tuple(sorted(i["query"]))))
                boards.add(_digest(i["board"]))
                outcomes.add("+".join(sorted(i["axes"])) if i["win"] else "fail")
                # Independent expression for B, plus axis-order invariance.
                raw = i["raw"]
                keep_r = np.r_[True, np.any(raw[1:] != raw[:-1], axis=1)]
                keep_c = np.r_[True, np.any(raw[:, 1:] != raw[:, :-1], axis=0)]
                _require(np.array_equal(raw[keep_r][:, keep_c], i["board"]), "compression mismatch")
                _require(np.array_equal(_compress(raw.T).T, i["board"]), "axis asymmetry")
                # Remove A: plain compression leaves status blank and slots unpainted.
                plain = np.vstack((pair["input"][:i["board0"], :i["board"].shape[1]],
                                   i["board"], np.full((1, i["board"].shape[1]), i["frame"], np.int8),
                                   np.full((1, i["board"].shape[1]), i["bg"], np.int8)))
                _require(not np.array_equal(plain, pair["output"]), "A is unnecessary")
                # Remove B: retaining raw pixel coordinates has wrong dimensions;
                # local normalization of the selected striped glyph also loses strokes.
                _require(raw.shape != i["board"].shape and _features(i)["local_repeats"], "B is unnecessary")
                # Separated full separator rows and columns must all survive.
                _require(sum(np.all(i["board"] == i["divider"], axis=1)) == i["nr"] + 1,
                         "separated row repeats lost")
                _require(sum(np.all(i["board"] == i["divider"], axis=0)) == i["nc"] + 1,
                         "separated column repeats lost")
    _require(len(puzzles) == 42 and len(boards) >= 200, "insufficient fresh boards")
    _require(len(layouts) >= 12 and len(shapes) >= 40 and len(palettes) >= 35,
             "insufficient geometry or palette variation")
    _require(len(schedules) >= 200 and len(selections) >= 45, "insufficient structural variation")
    _require(outcomes >= {"row", "col", "fail"}, "missing outcomes")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(8675309, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.2), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    base = generate_puzzle(0)
    # The solver's family is broader than the sampler's train/test schedules:
    # no enlargement, one-axis repetition, and uniform column repetition work.
    pair = base["train"][0]
    i = _parse(pair["input"])
    bh, bw = i["board"].shape
    for first_row_run, column_run in ((1, 1), (2, 1), (1, 2), (2, 2)):
        rows = [first_row_run] + [1] * (bh - 1)
        raw = np.repeat(np.repeat(i["board"], rows, axis=0), column_run, axis=1)
        prefix = np.full((i["board0"], raw.shape[1]), i["bg"], np.int8)
        prefix[:, :bw] = pair["input"][:i["board0"], :bw]
        prefix[1] = prefix[i["gh"] + 2] = i["frame"]
        variant = np.vstack((prefix, raw,
                             np.full((1, raw.shape[1]), i["frame"], np.int8),
                             np.full((1, raw.shape[1]), i["bg"], np.int8)))
        _require(np.array_equal(solve(variant), pair["output"]), "solver assumed sampled runs")
    for defect in ("status", "ink", "duplicate"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        if defect == "duplicate":
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        elif defect == "status":
            bad["test"][0]["output"][-1, 0] = (int(bad["test"][0]["output"][-1, 0]) + 1) % 10
        else:
            pair = bad["test"][0]
            i = _parse(pair["input"])
            p = i["query"][0]
            r, c = divmod(p, i["nc"])
            tr, tc = map(int, np.argwhere(i["tokens"][p] != i["bg"])[0])
            pair["output"][i["board0"] + r * (i["gh"] + 1) + 1 + tr,
                           c * (i["gw"] + 1) + 1 + tc] = i["bg"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("validator accepted " + defect)


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and all legal explicit counts")
