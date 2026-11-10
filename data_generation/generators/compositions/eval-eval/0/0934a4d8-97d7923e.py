"""Mirror shutter: unary ranks of rod spans inside an erased rectangle.

Only the returned input pixels are used by solve.  Centered axial reflections
recover the scene; the erased rectangle then defines a new, local competition.
See task_spec.md for the semantic domain, rather than sampler preferences.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _mask(a):
    candidates = []
    for color in np.unique(a):
        rr, cc = np.nonzero(a == color)
        r, s, c, d = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        if len(rr) >= 2 and len(rr) == (s - r) * (d - c):
            candidates.append((int(color), (r, s, c, d)))
    _require(len(candidates) == 1, "shutter must be the unique rectangular color")
    return candidates[0]


def _recover(a):
    """Find exact centered reflections and fill their cell orbits.

    A reflection needs visible, nonconstant paired evidence as well as zero
    visible conflicts.  Its scope is the entire scene, never an unrelated patch.
    """
    color, box = _mask(a)
    axes = []
    for axis in (0, 1):
        mate = np.flip(a, axis=axis)
        coord = np.indices(a.shape)[axis]
        visible = (a != color) & (mate != color) & (coord < (a.shape[axis] - 1) / 2)
        if (visible.sum() >= max(6, int(a.size * .08))
                and len(np.unique(a[visible])) >= 2
                and np.array_equal(a[visible], mate[visible])):
            axes.append(axis)
    _require(axes, "no supported reflection")
    full = a.copy()
    visited = set()
    witnesses = []
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            if (row, col) in visited:
                continue
            rows = {row, a.shape[0] - 1 - row} if 0 in axes else {row}
            cols = {col, a.shape[1] - 1 - col} if 1 in axes else {col}
            orbit = [(r, c) for r in rows for c in cols]
            visited.update(orbit)
            values = {int(a[r, c]) for r, c in orbit if a[r, c] != color}
            _require(len(values) == 1, "erased orbit or conflicting reflection witnesses")
            value = next(iter(values))
            for r, c in orbit:
                full[r, c] = value
                if a[r, c] == color:
                    witnesses.append(sum(a[x, y] != color for x, y in orbit))
    _require(not np.any(full == color), "shutter color cannot be scene content")
    return full, {"mask": color, "box": box, "axes": tuple(axes),
                  "minimum_witnesses": int(min(witnesses))}


def _runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(indices) else []


def _parse_axis(full, axis, bg):
    """Segment tracks, allowing adjacent rods and multiple rods on a track.

    Uniform runs are mirrored unary keys. Nonuniform runs must have the exact
    cap/payload/cap grammar. End-cap and payload palettes are disjoint locally.
    """
    work = full if axis == 0 else full.T
    keys, bars = [], []
    for col in range(work.shape[1]):
        for run in _runs(np.flatnonzero(work[:, col] != bg)):
            start, stop = int(run[0]), int(run[-1]) + 1
            values = work[start:stop, col]
            cap = int(values[0])
            cells = tuple((r, col) if axis == 0 else (col, r) for r in range(start, stop))
            if np.all(values == cap):
                _require(stop - start <= 4, "key rank exceeds four")
                keys.append({"cap": cap, "rank": stop - start, "cells": cells})
            else:
                _require(stop - start >= 3 and int(values[-1]) == cap, "invalid rod caps")
                payload = int(values[1])
                _require(payload != cap and np.all(values[1:-1] == payload), "invalid rod interior")
                bars.append({"cap": cap, "payload": payload, "length": stop - start,
                             "cells": cells, "track": col, "start": start})
    _require(keys and bars, "missing key or rods")
    ranks = {}
    for key in keys:
        _require(key["cap"] not in ranks or ranks[key["cap"]] == key["rank"], "key copies disagree")
        ranks[key["cap"]] = key["rank"]
    _require(set(ranks) == {b["cap"] for b in bars}, "keys and rod families do not match")
    _require(not set(ranks) & {b["payload"] for b in bars}, "payload cannot stand for a family")
    return {"axis": axis, "background": bg, "ranks": ranks, "keys": keys, "bars": bars}


def _decode(full):
    counts = np.bincount(full.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg]) == 1, "background must be uniquely modal")
    candidates = []
    for axis in (0, 1):
        try:
            candidates.append(_parse_axis(full, axis, bg))
        except AssertionError:
            pass
    _require(len(candidates) == 1, "rod axis must be unambiguous")
    return candidates[0]


def _chords(metadata, box):
    r, s, c, d = box
    chords = []
    for bar in metadata["bars"]:
        cells = tuple((x, y) for x, y in bar["cells"] if r <= x < s and c <= y < d)
        if cells:
            chords.append(dict(bar, hidden_cells=cells, span=len(cells),
                               fully_hidden=len(cells) == bar["length"]))
    return chords


def _render(metadata, box, mode="local"):
    r, s, c, d = box
    out = np.full((s - r, d - c), metadata["background"], dtype=np.int8)
    chords = _chords(metadata, box)
    selected = []
    for cap, rank in metadata["ranks"].items():
        family = [b for b in chords if b["cap"] == cap]
        _require(family, "a keyed family has no rod behind the shutter")
        levels = sorted({b["span"] for b in family}, reverse=True)
        _require(rank <= len(levels), "requested hidden-span level does not exist")
        if mode in ("full_length", "global_full_length"):
            competitors = family if mode == "full_length" else [b for b in metadata["bars"] if b["cap"] == cap]
            scores = sorted({b["length"] for b in competitors}, reverse=True)
            winners = [b for b in family if b["length"] == scores[rank - 1]] if rank <= len(scores) else []
        elif mode in ("left", "right"):
            ordered = sorted(family, key=lambda b: (b["track"], b["start"]), reverse=mode == "right")
            winners = [ordered[rank - 1]]
        elif mode == "global_span":
            scores = sorted({b["span"] for b in chords}, reverse=True)
            winners = [b for b in family if b["span"] == scores[rank - 1]]
        else:
            score = (levels[rank - 1] if mode in ("local", "visible_only") else
                     levels[-rank] if mode == "ascending" else
                     levels[0] if mode == "longest" else levels[-1])
            winners = [b for b in family if b["span"] == score]
            if mode == "visible_only":
                winners = [b for b in winners if not b["fully_hidden"]]
        for bar in winners:
            for x, y in bar["hidden_cells"]:
                out[x - r, y - c] = cap
            selected.append(bar)
    return out, {"chords": chords, "selected": selected,
                 "levels": {cap: tuple(sorted({b["span"] for b in chords if b["cap"] == cap}, reverse=True))
                            for cap in metadata["ranks"]}}


def _analyze_input(a):
    full, reflection = _recover(a)
    metadata = _decode(full)
    output, selection = _render(metadata, reflection["box"])
    return output, dict(reflection, full=full, metadata=metadata, **selection)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the selected hidden spans, using only this input."""
    output, _ = _analyze_input(_grid(input_grid))
    return output


def _reflect(base, axes, odd_rows, odd_cols):
    full = base.copy()
    if 1 in axes:
        full = np.concatenate((full, full[:, -2::-1] if odd_cols else full[:, ::-1]), axis=1)
    if 0 in axes:
        full = np.concatenate((full, full[-2::-1] if odd_rows else full[::-1]), axis=0)
    return full


def _scene(rng, parameters, ranks, anchor, axes):
    f = len(ranks)
    palette = list(parameters["palette"])
    if parameters["vary_colors"]:
        palette = [int(v) for v in rng.permutation(10) if v != parameters["anchor_color"]]
        bg, mask = palette[:2]
        controls = [parameters["anchor_color"]] + palette[2:f + 1]
        if not anchor:
            controls = palette[2:f + 2]
        payloads = [v for v in palette if v not in controls and v not in (bg, mask)]
    else:
        bg, mask = palette[:2]
        controls = palette[2:2 + f]
        payloads = palette[2 + f:]
    classes = 1 if parameters.get("teach_shared", False) else int(rng.integers(1, min(3, len(payloads)) + 1))
    payloads = payloads[:classes]
    # Population is not determined by the unary value. The teaching rank-two
    # anchor has an extra level to demonstrate a non-extreme choice.
    counts = [int(rng.integers(max(2, n + (anchor and i == 0 and n == 2)), 6))
              for i, n in enumerate(ranks)]
    transverse_limit = 15 if 1 in axes else 29
    while sum(counts) > transverse_limit - 2:
        choices = [i for i, n in enumerate(counts) if n > max(2, ranks[i])]
        _require(choices, "too many rods for the canvas")
        counts[int(rng.choice(choices))] -= 1
    side_key = bool(rng.integers(2)) and sum(counts) + 2 * f + 2 <= transverse_limit
    axial_limit = 15 if 0 in axes else 29
    height = int(rng.integers(max(11, max(counts) + 7), axial_limit + 1))
    key_row = int(rng.integers(0, 3)) if side_key else 0
    key_columns = [2 * i for i in range(f)]
    reserved = 2 * f + 1 if side_key else 0
    minimum_width = reserved + sum(counts) + 1
    width = int(rng.integers(minimum_width, transverse_limit + 1))
    gap_budget = width - minimum_width
    gaps = [0] * max(0, sum(counts) - 1)
    for _ in range(int(rng.integers(gap_budget + 1))):
        slots = [i for i, gap in enumerate(gaps) if gap < 2]
        if slots:
            gaps[int(rng.choice(slots))] += 1
    order = [i for i in range(f) for _ in range(counts[i])]
    if parameters["mixing"] == "shuffle" or rng.random() < .35:
        rng.shuffle(order)
    else:
        family_order = [int(i) for i in rng.permutation(f)]
        order = [i for i in family_order for _ in range(counts[i])]
    columns, col = [], reserved
    for i in range(len(order)):
        columns.append(col)
        if i < len(gaps):
            col += 1 + gaps[i]
    rod_top = 0 if side_key else max(ranks) + 1
    maximum_length = height - 1 - rod_top
    lengths = {i: [int(v) for v in rng.choice(np.arange(3, maximum_length + 1), counts[i], replace=False)]
               for i in range(f)}
    base = np.full((height, width), bg, dtype=np.int8)
    for x, i in zip(key_columns, rng.permutation(f)):
        base[key_row:key_row + ranks[i], x] = controls[i]
    cursors = [0] * f
    for x, i in zip(columns, order):
        length = lengths[i][cursors[i]]
        cursors[i] += 1
        start = int(rng.integers(rod_top, height - length))
        base[start, x] = base[start + length - 1, x] = controls[i]
        base[start + 1:start + length - 1, x] = int(rng.choice(payloads))
    full = _reflect(base, axes, bool(rng.integers(2)), bool(rng.integers(2)))
    metadata = _decode(full)
    return full, metadata, mask


def _window(rng, full, metadata, axes, seam, quadrant=False):
    h, w = full.shape
    # A one-axis window stays to one side. With two axes it can cross one seam
    # while the other reflection supplies witnesses for every covered cell.
    protected = int(rng.choice(axes)) if len(axes) == 2 else axes[0]
    if seam and len(axes) == 2:
        protected = 1
    limits = [h, w]
    limits[protected] //= 2
    if quadrant:
        limits = [h // 2, w // 2]
    height = int(rng.integers(3, limits[0] + 1))
    width = int(rng.integers(max(2, limits[1] // 2), limits[1] + 1))
    row = int(rng.integers(0, limits[0] - height + 1))
    col = int(rng.integers(0, limits[1] - width + 1))
    if seam and len(axes) == 2:
        height = int(rng.integers(4, min(h - 1, 2 * (h // 2 - 1)) + 1))
        row = int(np.clip(h // 2 - int(rng.integers(1, height)), 0, h - height))
        if not row <= (h - 1) / 2 < row + height - 1:
            return None
    if bool(rng.integers(2)):
        if protected == 0:
            row = h - row - height
        else:
            col = w - col - width
    if quadrant and bool(rng.integers(2)):
        if protected == 0:
            col = w - col - width
        else:
            row = h - row - height
    if height * width > full.size * .42:
        return None
    return row, row + height, col, col + width


def _make_pair(rng, parameters, ranks, anchor, axes, seam=False,
               need_full_contrast=False, need_hidden_winner=False, need_tie=False,
               need_group_contrast=False, key_visibility=None, quadrant=False, seen=None):
    seen = set() if seen is None else seen
    for attempt in range(240):
        try:
            full, metadata, mask = _scene(rng, parameters, ranks, anchor, axes)
        except AssertionError:
            continue
        for _ in range(70):
            box = _window(rng, full, metadata, axes, seam, quadrant)
            if box is None:
                continue
            r, s, c, d = box
            key_overlaps = [sum(r <= x < s and c <= y < d for x, y in k["cells"]) for k in metadata["keys"]]
            if key_visibility == "visible" and any(key_overlaps):
                continue
            if key_visibility == "hidden" and not any(n == k["rank"] for n, k in zip(key_overlaps, metadata["keys"])):
                continue
            try:
                output, selection = _render(metadata, box)
            except AssertionError:
                continue
            # Losers and multiple scores expose selection in every example.
            if any(len(v) < 2 for v in selection["levels"].values()):
                continue
            if anchor and ranks[0] == 2 and len(selection["levels"][parameters["anchor_color"]]) < 3:
                continue
            if need_hidden_winner and not any(b["fully_hidden"] for b in selection["selected"]):
                continue
            if need_tie and not any(len({b["length"] for b in selection["selected"] if b["cap"] == cap}) >= 2
                                    for cap in metadata["ranks"]):
                continue
            if need_full_contrast:
                alternative, _ = _render(metadata, box, "full_length")
                if np.array_equal(alternative, output):
                    continue
            if need_group_contrast and np.array_equal(_render(metadata, box, "global_span")[0], output):
                continue
            inp = full.copy()
            inp[r:s, c:d] = mask
            try:
                predicted, record = _analyze_input(inp)
            except AssertionError:
                continue
            if not np.array_equal(record["full"], full) or not np.array_equal(predicted, output):
                continue
            rotation = parameters["rotation"] if not parameters["mixed_orientation"] else int(rng.integers(4))
            if parameters.get("test_rotation") is not None:
                rotation = parameters["test_rotation"]
            inp, output = (np.ascontiguousarray(np.rot90(a, rotation)) for a in (inp, output))
            if _signature(inp) in seen:
                continue
            return {"input": inp, "output": output}
    raise AssertionError("bounded scene/window sampling exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(v) for v in rng.permutation(10))
    shift = str(rng.choice(["seam", "orientation", "families", "hidden_key"]))
    parameters = {"palette": palette, "anchor_color": palette[2], "vary_colors": bool(rng.integers(2)),
                  "rotation": int(rng.integers(4)), "mixed_orientation": bool(rng.integers(2)),
                  "mixing": str(rng.choice(["shuffle", "blocks"]))}
    if shift == "orientation":
        parameters["mixed_orientation"] = False
    max_families = int(rng.choice([2, 3]))
    max_rank = 3 if nt >= 3 and rng.random() < .55 else 2
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            anchor = split == "train" and index < 2
            f = 2 if anchor else int(rng.integers(1, (2 if shift == "families" and split == "train" else max_families) + 1))
            if shift == "families" and split == "test":
                f = 3
            ranks = [int(rng.integers(1, max_rank + 1)) for _ in range(f)]
            if anchor:
                ranks[0] = index + 1
                ranks[1] = 1 if index == 0 else 2
            if split == "train" and index == 2 and max_rank == 3:
                ranks[0] = 3
            axes = (0, 1) if anchor or shift == "seam" else [(0,), (1,), (0, 1)][int(rng.integers(3))]
            seam = split == "test" and shift == "seam"
            if split == "train" and index >= 2 and shift != "seam":
                seam = axes == (0, 1) and rng.random() < .30
            pair_parameters = dict(parameters)
            pair_parameters["teach_shared"] = anchor
            if split == "test" and shift == "orientation":
                pair_parameters["test_rotation"] = (parameters["rotation"] + 1) % 4
            pair = _make_pair(rng, pair_parameters, ranks, anchor, axes, seam=seam,
                              need_full_contrast=anchor,
                              need_hidden_winner=anchor or split == "test",
                              need_tie=anchor and index == 0,
                              need_group_contrast=anchor and index == 1,
                              key_visibility=("visible" if split == "train" else "hidden") if shift == "hidden_key" else None,
                              quadrant=split == "train" and shift == "seam", seen=seen)
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    # A spatial nth-object shortcut must fail on at least one demonstration.
    analyses = [_analyze_input(p["input"])[1] for p in puzzle["train"]]
    surviving = [mode for mode in ("left", "right") if all(
        np.array_equal(_render(a["metadata"], a["box"], mode)[0], p["output"])
        for a, p in zip(analyses, puzzle["train"]))]
    if surviving:
        old = puzzle["train"][0]
        seen.remove(_signature(old["input"]))
        for _ in range(30):
            pair = _make_pair(rng, parameters, [1, 1], True, (0, 1),
                              need_full_contrast=True, need_hidden_winner=True, need_tie=True,
                              key_visibility="visible" if shift == "hidden_key" else None,
                              quadrant=shift == "seam", seen=seen)
            a = _analyze_input(pair["input"])[1]
            if all(not np.array_equal(_render(a["metadata"], a["box"], mode)[0], pair["output"])
                   for mode in surviving):
                puzzle["train"][0] = pair
                break
        else:
            raise AssertionError("spatial teaching contrast exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_signature(a) not in seen, "duplicate input")
            seen.add(_signature(a))
            expected, record = _analyze_input(a)
            _require(np.array_equal(expected, b), "incorrect hidden-span silhouette")
            _require(record["mask"] not in b, "shutter color survived")
            _require(set(np.unique(b)) <= {record["metadata"]["background"], *record["metadata"]["ranks"]},
                     "payload or key entered the silhouette")
            _require(all(len(v) >= 2 for v in record["levels"].values()), "selection needs a losing span")
            _require(any(chord not in record["selected"] for chord in record["chords"]), "no losing rod")
            records[split].append(record)
    train = records["train"]
    ranks = {n for record in train for n in record["metadata"]["ranks"].values()}
    _require({1, 2} <= ranks, "training must expose unary ranks one and two")
    _require(any(n == 2 and len(record["levels"][cap]) >= 3
                 for record in train for cap, n in record["metadata"]["ranks"].items()),
             "rank two needs a non-extreme teaching case")
    cap_history = {}
    for record in train:
        for cap, n in record["metadata"]["ranks"].items():
            cap_history.setdefault(cap, set()).add(n)
    _require(any({1, 2} <= ns for ns in cap_history.values()), "same cap must change key rank")
    _require(any(len({b["length"] for b in record["selected"] if b["cap"] == cap}) >= 2
                 for record in train for cap in record["metadata"]["ranks"]), "training must show clipped ties")
    for mode in ("full_length", "global_full_length", "global_span", "ascending", "longest", "shortest", "left", "right", "visible_only"):
        _require(any(not np.array_equal(_render(record["metadata"], record["box"], mode)[0], pair["output"])
                     for record, pair in zip(train, puzzle["train"])), f"training admits {mode} shortcut")
    _require(any(not np.array_equal(record["full"][record["box"][0]:record["box"][1],
                                                   record["box"][2]:record["box"][3]], pair["output"])
                 for record, pair in zip(train, puzzle["train"])), "mere restoration suffices")
    for record in records["test"]:
        _require(any(b["fully_hidden"] for b in record["selected"]), "test must require erased-object recovery")
        _require(set(record["metadata"]["ranks"].values()) <= ranks, "test rank was not demonstrated")


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(np.asarray(p["input"])), _signature(np.asarray(p["output"])))
                       for p in puzzle[split]) for split in ("train", "test"))


def _hand_checked_fixtures():
    """Small literal answers exercise semantics beyond sampler templates."""
    def field(height, rank, rods, odd=False):
        base = np.zeros((height, 8), dtype=np.int8)
        base[:rank, 0] = 2
        for col, start, length in rods:
            base[start, col] = base[start + length - 1, col] = 2
            base[start + 1:start + length - 1, col] = 5
        return _reflect(base, (1,), False, odd)

    def erased(full, box):
        a = full.copy()
        r, s, c, d = box
        a[r:s, c:d] = 8
        return a

    # Whole lengths 8/6/4 become local lengths 2/5/4. Unary two selects
    # the wholly erased four-cell rod, independently of full-length order.
    full = field(13, 2, [(2, 1, 8), (3, 6, 6), (4, 8, 4)])
    a = erased(full, (7, 12, 2, 5))
    expected = np.zeros((5, 3), dtype=np.int8)
    expected[1:, 2] = 2
    _require(np.array_equal(solve(a), expected), "literal clipped-rank fixture")
    _require(np.array_equal(a[7:12, 2:5], np.full((5, 3), 8)), "solve mutated its input")
    # The same scene hides the entire unary key. The shorter positive chord
    # wins at rank two; erased keys and the losing rod disappear from output.
    a = erased(full, (0, 9, 0, 4))
    expected = np.zeros((9, 4), dtype=np.int8)
    expected[6:9, 3] = 2
    _require(np.array_equal(solve(a), expected), "literal hidden-key fixture")
    # Different whole lengths tie after clipping. Both must survive.
    full = field(11, 1, [(2, 1, 8), (3, 3, 6), (4, 5, 3)])
    a = erased(full, (4, 8, 2, 5))
    expected = np.zeros((4, 3), dtype=np.int8)
    expected[:, :2] = 2
    _require(np.array_equal(solve(a), expected), "literal all-tied-winners fixture")
    # Rank four and a through-cell column axis are legal without a generator
    # preference or a stored training packet. Output positions are literal.
    full = field(13, 4, [(2, 4, 8), (3, 5, 6), (4, 6, 5), (5, 8, 4), (6, 4, 3)], odd=True)
    a = erased(full, (4, 12, 2, 7))
    expected = np.zeros((8, 5), dtype=np.int8)
    expected[4:8, 3] = 2
    _require(np.array_equal(solve(a), expected), "literal rank-four fixture")
    # A thin shutter and one positive length level are semantic members even
    # though the sampler favors multiple levels and broader windows.
    full = field(13, 1, [(2, 1, 8), (3, 6, 6), (4, 8, 4)])
    a = erased(full, (8, 9, 2, 5))
    _require(np.array_equal(solve(a), np.full((1, 3), 2)), "literal thin-window fixture")
    a[0, -1] = 5  # Contradict the visible reflected unary key.
    try:
        solve(a)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting visible reflection accepted")
    a = erased(full, (8, 9, 2, 14))  # Both members of several column-reflection orbits.
    try:
        solve(a)
    except AssertionError:
        pass
    else:
        raise AssertionError("completely erased reflection orbit accepted")


def self_test() -> dict:
    """Bounded semantic, metamorphic and variation checks over 42 seeds."""
    _hand_checked_fixtures()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, output_shapes, populations, family_counts, span_levels = set(), set(), set(), set(), set()
    axes, orientations, backgrounds, masks, payload_classes = set(), set(), set(), set(), set()
    signatures, normalized, ties, fully_hidden, reversed_orders = set(), set(), 0, 0, 0
    for puzzle in puzzles:
        signatures.add(_puzzle_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                expected, record = _analyze_input(a)
                _require(np.array_equal(expected, b), "solve check")
                shapes.add(a.shape)
                output_shapes.add(b.shape)
                roles = {}
                normalized.add((a.shape, tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)))
                m = record["metadata"]
                orientations.add(m["axis"])
                axes.add(record["axes"])
                backgrounds.add(m["background"])
                masks.add(record["mask"])
                family_counts.add(len(m["ranks"]))
                payload_classes.add(len({bar["payload"] for bar in m["bars"]}))
                for cap, n in m["ranks"].items():
                    levels = record["levels"][cap]
                    span_levels.add((len(levels), n))
                    populations.add(len([bar for bar in record["chords"] if bar["cap"] == cap]))
                    ties += sum(bar["cap"] == cap for bar in record["selected"]) > 1
                fully_hidden += any(bar["fully_hidden"] for bar in record["selected"])
                reversed_orders += not np.array_equal(_render(m, record["box"], "full_length")[0], b)
    _require(len(signatures) == 42 and len(normalized) >= 160, "structural variation")
    _require(len(shapes) >= 35 and len(output_shapes) >= 45, "geometry variation")
    _require(family_counts == {1, 2, 3}, "family count variation")
    _require(axes == {(0,), (1,), (0, 1)} and orientations == {0, 1}, "reflection/rod axes")
    _require(len(backgrounds) >= 8 and len(masks) >= 8 and len(payload_classes) >= 2, "palette roles")
    _require(ties >= 15 and fully_hidden >= 100 and reversed_orders >= 80, "interaction coverage")
    _require(len(populations) >= 5 and {1, 2, 3} <= {n for _, n in span_levels}, "ranking variation")
    for seed in (0, 7, 41):
        _require(_puzzle_signature(puzzles[seed]) == _puzzle_signature(generate_puzzle(seed)), "determinism")
    # Global rotations/reflections and independent pair-specific color bijections
    # preserve the rule, including original target coordinate order.
    base = puzzles[7]
    for turn in range(4):
        changed = {s: [] for s in base}
        for split in base:
            for i, pair in enumerate(base[split]):
                perm = np.roll(np.arange(10, dtype=np.int8), turn + 1)
                changed[split].append({k: perm[np.rot90(np.fliplr(v), turn)].copy() for k, v in pair.items()})
                individual = np.roll(np.arange(10, dtype=np.int8), i + 1)
                _require(np.array_equal(solve(individual[pair["input"]]), individual[pair["output"]]),
                         "pair-specific palette invariance")
        validate_generated_puzzle(changed)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "puzzles": len(signatures), "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "normalized_inputs": len(normalized),
            "tied_family_selections": ties, "hidden_winner_pairs": fully_hidden,
            "full_length_disagreements": reversed_orders, "literal_semantic_fixtures": 5}
