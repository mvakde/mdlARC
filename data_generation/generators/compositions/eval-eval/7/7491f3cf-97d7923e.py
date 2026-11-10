"""Rank through a window: regional payload length selects capped-bar fragments.

Two congruent framed panels carry a spatial/unary instruction and a bar rack.
The solver reads only pixels. Sampling policies never change the family rule.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    h, w = mask.shape
    todo = set(map(tuple, np.argwhere(mask)))
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, queue = [start], deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    cells.append(p)
                    queue.append(p)
        result.append(cells)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _near(mask):
    h, w = mask.shape
    result = mask.copy()
    for r, c in np.argwhere(mask):
        result[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
    return result


def _partition(cut):
    """Accept an unbranched digital path separating two four-connected regions."""
    cells = set(map(tuple, np.argwhere(cut)))
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if not (dr or dc) or q not in cells:
                    continue
                # An orthogonal elbow is not a triangular branch.
                if dr and dc and ((r+dr, c) in cells or (r, c+dc) in cells):
                    continue
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p in graph if len(graph[p]) == 1]
    _require(len(cells) >= 3 and len(ends) == 2
             and all(1 <= len(v) <= 2 for v in graph.values()), "cut is not a simple path")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    _require(all(border(p) for p in ends)
             and all(not border(p) for p in cells if p not in ends),
             "only the two cut endpoints may meet the panel boundary")
    _require(len(_components(cut, True)) == 1, "cut is disconnected")
    pieces = _components(~cut)
    _require(len(pieces) == 2, "cut must separate exactly two regions")
    regions = [_mask(cut.shape, p) for p in pieces]
    sites = [list(map(tuple, np.argwhere(region & ~_near(cut)))) for region in regions]
    _require(all(sites), "both sides need a separated marker site")
    return regions, sites


def _pack(control, rack, frame):
    h, w = control.shape
    result = np.full((h+2, 2*w+3), frame, dtype=np.int8)
    result[1:-1, 1:w+1] = control
    result[1:-1, w+2:-1] = rack
    return result


def _runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1) if len(indices) else []


def _read_bars(rack, keys, bg, forbidden, axis):
    oriented = rack if axis == 0 else rack.T
    bars = []
    for c in range(oriented.shape[1]):
        segments = _runs(np.flatnonzero(oriented[:, c] != bg))
        _require(len(segments) <= 1, "one bar per occupied axis line")
        if not segments:
            continue
        run = segments[0]
        a, b = int(run[0]), int(run[-1])
        values = oriented[a:b+1, c]
        cap = int(values[0])
        _require(b-a >= 2 and cap in keys and int(values[-1]) == cap,
                 "each bar needs equal keyed caps and a payload")
        payload = int(values[1])
        _require(payload not in forbidden and np.all(values[1:-1] == payload),
                 "bar payload must be uniform and distinct from structural colors/caps")
        to_original = lambda r: (r, c) if axis == 0 else (c, r)
        bars.append({"cap": cap, "payload": payload, "length": b-a+1,
                     "position": c, "ends": (to_original(a), to_original(b)),
                     "cells": tuple(to_original(r) for r in range(a+1, b))})
    _require(bars and set(b["cap"] for b in bars) == set(keys), "missing keyed family")
    return bars


def _decode(input_grid):
    inp = _grid(input_grid)
    h, width = inp.shape
    _require(h >= 5 and width >= 9 and (width-3) % 2 == 0, "invalid two-panel frame")
    w = (width-3)//2
    frame = int(inp[0, 0])
    _require(np.all(inp[[0, -1], :] == frame)
             and np.all(inp[:, [0, w+1, width-1]] == frame), "broken frame/divider")
    control, rack = inp[1:-1, 1:w+1], inp[1:-1, w+2:-1]
    _require(frame not in control and frame not in rack, "frame color in an interior")
    frequencies = np.bincount(control.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(np.sum(frequencies == frequencies[bg]) == 1, "instruction background must be modal")
    candidates = []
    for color in set(map(int, np.unique(control))) - {bg}:
        pieces = sorted(_components(control == color, True), key=len)
        if len(pieces) != 2 or len(pieces[0]) != 1 or len(pieces[1]) < 3:
            continue
        cut = _mask(control.shape, pieces[1])
        try:
            regions, sites = _partition(cut)
            marker = pieces[0][0]
            side = next(i for i, region in enumerate(regions) if region[marker])
            _require(marker in sites[side], "marker touches the cut")
        except (AssertionError, StopIteration):
            continue
        candidates.append((color, cut, marker, regions, sites, side))
    _require(len(candidates) == 1, "cut and isolated marker must be unique")
    color, cut, marker, regions, sites, side = candidates[0]
    caps = set(map(int, np.unique(control))) - {bg, color}
    _require(caps, "missing unary keys")
    keys = {}
    for cap in caps:
        cells = list(map(tuple, np.argwhere(control == cap)))
        rows, cols = zip(*cells)
        _require((len(set(rows)) == 1 and max(cols)-min(cols)+1 == len(cells))
                 or (len(set(cols)) == 1 and max(rows)-min(rows)+1 == len(cells)),
                 "each key must be one straight contiguous unary run")
        _require(not any(_near(cut)[p] for p in cells), "key touches the cut")
        keys[cap] = len(cells)
    forbidden = caps | {frame, bg, color}
    axes = []
    for axis in (0, 1):
        try:
            bars = _read_bars(rack, keys, bg, forbidden, axis)
        except AssertionError:
            continue
        axes.append((axis, bars))
    _require(len(axes) == 1, "rack must have one unambiguous common axis")
    axis, bars = axes[0]
    marked, other = regions[side], regions[1-side]
    groups = {cap: [] for cap in keys}
    for bar in bars:
        bar["scores"] = tuple(sum(bool(region[p]) for p in bar["cells"]) for region in regions)
        bar["score"] = bar["scores"][side]
        groups[bar["cap"]].append(bar)
    for cap, family in groups.items():
        _require(len({b["length"] for b in family}) == len(family), "whole lengths tie within a family")
        eligible = [b for b in family if b["score"] > 0]
        _require(len(eligible) >= 2 and keys[cap] <= len(eligible), "insufficient exposed competitors")
        _require(len({b["score"] for b in eligible}) == len(eligible), "positive regional lengths tie")
    return {"input": inp, "control": control, "rack": rack, "frame": frame, "bg": bg,
            "cut_color": color, "cut": cut, "marker": marker, "regions": regions,
            "sites": sites, "side": side, "marked": marked, "other": other,
            "keys": keys, "bars": bars, "groups": groups, "axis": axis, "width": w}


def _choose(d, rule="regional"):
    chosen = []
    for cap, family in d["groups"].items():
        eligible = [b for b in family if b["score"] > 0]
        rank = d["keys"][cap]
        if rule == "pooled":
            order = sorted([b for b in d["bars"] if b["score"] > 0],
                           key=lambda b: b["score"], reverse=True)
        elif rule == "whole_all":
            order = sorted(family, key=lambda b: b["length"], reverse=True)
        elif rule == "whole_eligible":
            order = sorted(eligible, key=lambda b: b["length"], reverse=True)
        elif rule in ("first", "last"):
            order = sorted(eligible, key=lambda b: b["position"], reverse=rule == "last")
        else:
            order = sorted(eligible, key=lambda b: b["score"], reverse=rule != "ascending")
        if rule == "longest":
            rank = 1
        if rule == "shortest":
            rank = len(order)
        chosen.append(order[rank-1])
    return chosen


def _paint_answer(d, chosen, full=False, seam=True):
    out = d["input"].copy()
    target = out[1:-1, d["width"]+2:-1]
    for bar in chosen:
        for p in bar["cells"]:
            if full or d["marked"][p] or (seam and d["cut"][p]):
                target[p] = bar["cap"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select by strictly regional payload length; fill region plus seam only."""
    d = _decode(input_grid)
    return _paint_answer(d, _choose(d))


def _signature(grid):
    return grid.shape, grid.tobytes()


def _cut_kind(cut):
    cells = np.argwhere(cut)
    if len(set(cells[:, 0])) == 1 or len(set(cells[:, 1])) == 1:
        return "straight"
    # A digital diagonal has no multiple cut pixels in either axis line.
    if np.all(cut.sum(axis=0) <= 1) or np.all(cut.sum(axis=1) <= 1):
        return "diagonal"
    h, w = cut.shape
    endpoints = [(r, c) for r, c in cells if r in (0, h-1) or c in (0, w-1)]
    if len(endpoints) == 2:
        a, b = endpoints
        if (a[0] in (0, h-1)) != (b[0] in (0, h-1)):
            return "elbow"
    return "stepped"


def _sample_cut(rng, h, w, kind):
    for _ in range(80):
        cut = np.zeros((h, w), dtype=bool)
        center = int(rng.integers(max(3, h//3), min(h-3, 2*h//3)+1))
        if kind == "straight":
            cut[center, :] = True
        elif kind == "diagonal":
            a, b = sorted(int(v) for v in rng.choice(np.arange(2, h-2), 2, replace=False))
            # One pixel per column, consecutive pixels at most one row apart.
            b = min(b, a+w-1)
            rows = np.rint(np.linspace(a, b, w)).astype(int)
            if bool(rng.integers(2)):
                rows = rows[::-1]
            cut[rows, np.arange(w)] = True
        elif kind == "stepped":
            columns = list(range(2, w-2, 3))
            count = int(rng.integers(1, min(4, len(columns))+1))
            changes = sorted(int(v) for v in rng.choice(columns, count, replace=False))
            row, previous = center, 0
            for col in changes:
                cut[row, previous:col+1] = True
                choices = [r for r in range(2, h-2) if 2 <= abs(r-row) <= max(2, h//3)]
                next_row = int(rng.choice(choices))
                cut[min(row, next_row):max(row, next_row)+1, col] = True
                row, previous = next_row, col
            cut[row, previous:] = True
        elif kind == "elbow":
            col = int(rng.integers(2, max(3, w//2)))
            cut[:center+1, col] = True
            cut[center, col:] = True
        else:
            raise ValueError("unknown cut kind")
        try:
            regions, sites = _partition(cut)
            return cut, regions, sites
        except AssertionError:
            pass
    raise RuntimeError("could not sample a readable cut")


def _sample_rack(rng, cut, regions, side, counts, hidden, both):
    """Place intact bars; choose from interval geometries, never edit their pixels."""
    h, w = cut.shape
    total = sum(counts) + sum(hidden)
    if total > w:
        return None
    columns = sorted(int(v) for v in rng.choice(w, total, replace=False))
    sequence = [(f, False) for f, n in enumerate(counts) for _ in range(n)]
    sequence += [(f, True) for f, n in enumerate(hidden) for _ in range(n)]
    if rng.random() < .25:
        family_order = list(map(int, rng.permutation(len(counts))))
        sequence.sort(key=lambda item: family_order.index(item[0]))
    else:
        rng.shuffle(sequence)
    used_lengths = [set() for _ in counts]
    used_scores = [[set(), set()] for _ in counts]
    records = []
    for column, (family, invisible) in zip(columns, sequence):
        options = []
        prefix = [np.concatenate(([0], np.cumsum(region[:, column]))) for region in regions]
        for top in range(h-2):
            for bottom in range(top+2, h):
                length = bottom-top+1
                scores = tuple(int(p[bottom]-p[top+1]) for p in prefix)
                if length in used_lengths[family]:
                    continue
                if invisible:
                    if scores[side] != 0:
                        continue
                elif scores[side] == 0 or scores[side] in used_scores[family][side]:
                    continue
                if both and any(s == 0 or s in used_scores[family][i] for i, s in enumerate(scores)):
                    continue
                options.append((top, bottom, scores))
        if not options:
            return None
        top, bottom, scores = options[int(rng.integers(len(options)))]
        used_lengths[family].add(bottom-top+1)
        for i, score in enumerate(scores):
            if score:
                used_scores[family][i].add(score)
        records.append((family, column, top, bottom))
    return records


def _key_slots(rng, cut, markers, lengths):
    blocked = _near(cut)
    for r, c in markers:
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    h, w = cut.shape
    slots = []
    for n in lengths:
        options = []
        for r in range(h):
            for c in range(w):
                for dr, dc in ((0, 1), (1, 0)):
                    cells = [(r+j*dr, c+j*dc) for j in range(n)]
                    if cells[-1][0] >= h or cells[-1][1] >= w:
                        continue
                    if all(not blocked[p] for p in cells):
                        options.append(cells)
        if not options:
            return None
        cells = options[int(rng.integers(len(options)))]
        slots.append(cells)
        for r, c in cells:
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return slots


def _render(cut, marker, slots, ranks, records, palette, payload_ids, transpose, flips):
    frame, bg, control = palette[:3]
    caps = palette[3:3+len(ranks)]
    control_panel = np.full(cut.shape, bg, dtype=np.int8)
    control_panel[cut] = control
    control_panel[marker] = control
    for cap, cells, rank in zip(caps, slots, ranks):
        for p in cells[:rank]:
            control_panel[p] = cap
    rack = np.full(cut.shape, bg, dtype=np.int8)
    for record, payload in zip(records, payload_ids):
        family, c, a, b = record
        rack[a, c] = rack[b, c] = caps[family]
        rack[a+1:b, c] = payload
    if transpose:
        control_panel, rack = control_panel.T, rack.T
    for axis in flips:
        control_panel, rack = np.flip(control_panel, axis), np.flip(rack, axis)
    inp = _pack(control_panel, rack, frame)
    return {"input": inp, "output": solve(inp)}


def _new_scene(rng, params, *, anchor=False, ranks=None, axis=None, kind=None,
               test=False, higher=False):
    """Bounded geometry search; all information required by solve is rendered."""
    axis = params["axis"] if axis is None else axis
    for _ in range(120):
        f = 2 if anchor else int(rng.integers(1, params["max_families"]+1))
        if test and params["shift"] == "population":
            f = int(rng.integers(1, 3))
        chosen_ranks = [1, 2] if anchor else list(ranks or [])
        if not chosen_ranks:
            limit = 3 if higher else 2
            chosen_ranks = [int(rng.integers(1, limit+1)) for _ in range(f)]
        if higher and not test:
            chosen_ranks[0] = 3
        f = len(chosen_ranks)
        if axis == 0:
            h = int(rng.integers(15 if anchor else 10, 29))
            w = int(rng.integers(11 if anchor else 8, 14))
        else:
            h = int(rng.integers(11 if anchor else 9, 14))
            w = int(rng.integers(12 if anchor else 8, 29))
        if test and params["shift"] == "population":
            h = int(rng.integers(22, 29)) if axis == 0 else 13
            w = 13 if axis == 0 else int(rng.integers(16, 29))
        k = kind or str(rng.choice(params["train_kinds"] if not test else params["test_kinds"]))
        cut, regions, sites = _sample_cut(rng, h, w, k)
        side = int(rng.integers(2))
        markers = [sites[side][int(rng.integers(len(sites[side])))],
                   sites[1-side][int(rng.integers(len(sites[1-side])))]]
        if anchor:
            counts = [3, 3] if params["shift"] == "population" else [int(rng.integers(3, 5)) for _ in range(f)]
        elif test and params["shift"] == "population":
            counts = [int(rng.integers(4, 6)) for _ in range(f)]
        else:
            upper = 3 if (params["shift"] == "population" and not test) or f == 3 else 5
            counts = [int(rng.integers(max(2, r), upper+1)) for r in chosen_ranks]
        hidden = [0 if anchor else int(rng.random() < .3) for _ in range(f)]
        if sum(counts) + sum(hidden) > w:
            continue
        records = _sample_rack(rng, cut, regions, side, counts, hidden, anchor)
        if records is None:
            continue
        slot_lengths = [2, 2] if anchor else chosen_ranks
        slots = _key_slots(rng, cut, markers, slot_lengths)
        if slots is None:
            continue
        palette = params["palette"] if anchor or params["fixed_palette"] else list(map(int, rng.permutation(10)))
        payload_pool = palette[3+f:]
        classes = 1 if anchor or params["simple_payload"] else int(rng.integers(1, min(3, len(payload_pool))+1))
        payload_ids = [int(rng.choice(payload_pool[:classes])) for _ in records]
        flips = tuple(i for i in (0, 1) if bool(rng.integers(2)))
        try:
            pair = _render(cut, markers[0], slots, chosen_ranks, records, palette, payload_ids, axis == 1, flips)
            if anchor:
                # With only two demonstrations change key and marker together.
                second_marker = markers[1] if params["num_train"] == 2 else markers[0]
                second = _render(cut, second_marker, slots, [2, 1], records, palette, payload_ids, axis == 1, flips)
                third = _render(cut, markers[1], slots, chosen_ranks, records, palette, payload_ids, axis == 1, flips)
                return [pair, second, third]
            return pair
        except AssertionError:
            pass
    raise RuntimeError("could not construct a valid scene")


def _training_shortcuts(analyses, pairs):
    survivors = []
    for rule in ("whole_all", "whole_eligible", "first", "last", "ascending", "longest", "shortest", "pooled"):
        if all(np.array_equal(_paint_answer(d, _choose(d, rule)), pair["output"])
               for d, pair in zip(analyses, pairs)):
            survivors.append(rule)
    if all(np.array_equal(_paint_answer(d, _choose(d), full=True), pair["output"])
           for d, pair in zip(analyses, pairs)):
        survivors.append("fill_whole_bar")
    if all(np.array_equal(_paint_answer(d, d["bars"]), pair["output"])
           for d, pair in zip(analyses, pairs)):
        survivors.append("fill_all_fragments")
    if all(np.array_equal(_paint_answer(d, _choose(d), seam=False), pair["output"])
           for d, pair in zip(analyses, pairs)):
        survivors.append("omit_seam")
    # Ignore the marker and always use the first flood-filled region.
    fixed_outputs = []
    for d in analyses:
        fixed = dict(d, marked=d["regions"][0], side=0)
        groups = {cap: [dict(b, score=b["scores"][0]) for b in family]
                  for cap, family in d["groups"].items()}
        fixed["groups"] = groups
        if any(len([b for b in family if b["score"] > 0]) < fixed["keys"][cap]
               for cap, family in groups.items()):
            fixed_outputs.append(None)
        else:
            fixed_outputs.append(_paint_answer(fixed, _choose(fixed)))
    if all(x is not None and np.array_equal(x, p["output"]) for x, p in zip(fixed_outputs, pairs)):
        survivors.append("ignore_marker")
    return survivors


def _metric_prediction(d, include_caps, include_cut, tie_mode):
    """Nearby measurement alternatives used only to audit the curriculum.

    A short curriculum need not distinguish observationally identical metrics,
    but a test may not require a distinction absent from its demonstrations.
    """
    window = d["marked"] | d["cut"] if include_cut else d["marked"]
    chosen = []
    for cap, family in d["groups"].items():
        scores = {}
        for i, bar in enumerate(family):
            cells = bar["cells"] + bar["ends"] if include_caps else bar["cells"]
            scores[i] = sum(bool(window[p]) for p in cells)
        eligible = [i for i in scores if scores[i] > 0]
        rank = d["keys"][cap]
        if len(eligible) < rank:
            return None
        if tie_mode == "whole":
            secondary = lambda i: family[i]["length"]
        else:
            secondary = lambda i: family[i]["position"] * (1 if tie_mode == "last" else -1)
        order = sorted(eligible, key=lambda i: (scores[i], secondary(i)), reverse=True)
        chosen.append(family[order[rank-1]])
    return _paint_answer(d, chosen)


_METRICS = tuple((caps, seam, tie) for caps, seam in ((True, False), (False, True), (True, True))
                 for tie in ("whole", "first", "last"))


def _surviving_metrics(analyses, pairs):
    return [model for model in _METRICS
            if all((answer := _metric_prediction(d, *model)) is not None
                   and np.array_equal(answer, pair["output"])
                   for d, pair in zip(analyses, pairs))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["cut", "population", "axis"]))
    anchor_kind = str(rng.choice(["straight", "diagonal"]))
    params = {"shift": shift, "axis": int(rng.integers(2)), "num_train": int(num_train),
              "palette": list(map(int, rng.permutation(10))), "fixed_palette": bool(rng.random() < .3),
              "simple_payload": bool(rng.random() < .4), "max_families": int(rng.integers(2, 4)),
              "train_kinds": ["straight", "diagonal"] if shift == "cut" else ["straight", "diagonal", "stepped", "elbow"],
              "test_kinds": ["stepped", "elbow"] if shift == "cut" else [anchor_kind]}
    for _ in range(80):
        anchors = _new_scene(rng, params, anchor=True, kind=anchor_kind)
        train = anchors[:min(num_train, 3)]
        for i in range(3, num_train):
            train.append(_new_scene(rng, params, higher=i == 3))
        analyses = [_decode(p["input"]) for p in train]
        if _training_shortcuts(analyses, train):
            continue
        # Test ranks never exceed a rank already visible in training.
        taught = {rank for d in analyses for rank in d["keys"].values()}
        surviving_metrics = _surviving_metrics(analyses, train)
        tests, seen = [], {_signature(p["input"]) for p in train}
        for _index in range(num_test):
            for _try in range(40):
                test_axis = 1-params["axis"] if shift == "axis" else params["axis"]
                pair = _new_scene(rng, params, test=True, axis=test_axis, higher=3 in taught)
                d = _decode(pair["input"])
                teachable = all((answer := _metric_prediction(d, *model)) is not None
                                and np.array_equal(answer, pair["output"])
                                for model in surviving_metrics)
                if _signature(pair["input"]) not in seen and teachable:
                    tests.append(pair)
                    seen.add(_signature(pair["input"]))
                    break
            else:
                raise RuntimeError("could not separate test inputs")
        puzzle = {"train": train, "test": tests}
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("could not make contrastive demonstrations")


def validate_generated_puzzle(puzzle):
    """Check semantics from pixels and compare ordinal counts, without solve()."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            _require(_signature(inp) not in seen, "duplicate input")
            seen.add(_signature(inp))
            d = _decode(inp)
            expected = inp.copy()
            dest = expected[1:-1, d["width"]+2:-1]
            for cap, family in d["groups"].items():
                for bar in family:
                    ordinal = 1 + sum(other["score"] > bar["score"] for other in family)
                    if bar["score"] and ordinal == d["keys"][cap]:
                        for p in bar["cells"]:
                            if d["marked"][p] or d["cut"][p]:
                                dest[p] = cap
            _require(np.array_equal(out, expected), "incorrect regional ranking/fill or changed instruction/frame")
            _require(np.any(inp != out), "identity example")
            changed = out[1:-1, d["width"]+2:-1] != d["rack"]
            _require(not np.any(changed & d["other"]), "hidden-side cell changed")
            for b in d["bars"]:
                for p in b["ends"]:
                    _require(out[p[0]+1, p[1]+d["width"]+2] == b["cap"], "cap changed")
            analyses[split].append(d)
    train = analyses["train"]
    _require({d["side"] for d in train} == {0, 1}, "training must expose both marker sides")
    _require(not _training_shortcuts(train, puzzle["train"]), "training admits a listed shortcut")
    taught = {r for d in train for r in d["keys"].values()}
    _require({1, 2} <= taught, "training must show unary ranks one and two")
    history = {}
    for d in train:
        for cap, rank in d["keys"].items():
            history.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= ranks for ranks in history.values()), "same-cap key contrast missing")
    _require(any(1 < d["keys"][cap] < sum(b["score"] > 0 for b in family)
                 for d in train for cap, family in d["groups"].items()), "no intermediate ordinal demonstration")
    _require(all(r in taught for d in analyses["test"] for r in d["keys"].values()),
             "test ordinal was not demonstrated")
    for model in _surviving_metrics(train, puzzle["train"]):
        for d, pair in zip(analyses["test"], puzzle["test"]):
            answer = _metric_prediction(d, *model)
            _require(answer is not None and np.array_equal(answer, pair["output"]),
                     "test requires an undemonstrated measurement distinction")


def self_test():
    """Bounded 42-seed semantic, diversity, and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, kinds, axes, palettes, populations, ranks = (set() for _ in range(7))
    hidden = partial = whole_mismatch = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagrees")
                shapes.add(pair["input"].shape)
                mapping = {}
                structures.add(tuple(mapping.setdefault(int(v), len(mapping)) for v in pair["input"].flat))
                kinds.add(_cut_kind(d["cut"]))
                axes.add(d["axis"])
                palettes.add((d["frame"], d["bg"], d["cut_color"], tuple(sorted(d["keys"]))))
                ranks.update(d["keys"].values())
                for family in d["groups"].values():
                    populations.add(sum(b["score"] > 0 for b in family))
                    hidden += sum(b["score"] == 0 for b in family)
                partial += any(any(d["other"][p] for p in b["cells"]) for b in _choose(d))
                whole_mismatch += not np.array_equal(_paint_answer(d, _choose(d, "whole_eligible")), pair["output"])
        # Consistent palette permutation preserves the answer.
        pair = puzzle["train"][0]
        mapping = np.array([7, 2, 9, 4, 0, 8, 1, 6, 3, 5], dtype=np.int8)
        _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "palette covariance")
        d = _decode(pair["input"])
        for axis in (0, 1):
            inp = _pack(np.flip(d["control"], axis), np.flip(d["rack"], axis), d["frame"])
            answer_rack = pair["output"][1:-1, d["width"]+2:-1]
            out = _pack(np.flip(d["control"], axis), np.flip(answer_rack, axis), d["frame"])
            _require(np.array_equal(solve(inp), out), "reflection covariance")
    _require(len(shapes) >= 25 and len(structures) >= 120 and len(palettes) >= 35,
             "size/structure/palette diversity collapsed")
    _require(kinds == {"straight", "diagonal", "stepped", "elbow"} and axes == {0, 1},
             "cut/axis diversity collapsed")
    _require({2, 3, 4, 5} <= populations and {1, 2, 3} <= ranks, "population/ordinal coverage")
    _require(hidden >= 10 and partial >= 84 and whole_mismatch >= 42,
             "occlusion or regional-order evidence collapsed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(0, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(2930432, 2, 1), generate_puzzle(2930432, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x),
             "determinism failed")
    # Corrupt a changed payload and a protected instruction cell.
    for instruction in (False, True):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        p = bad["test"][0]
        if instruction:
            p["output"][0, 0] = (int(p["output"][0, 0])+1) % 10
        else:
            r, c = np.argwhere(p["input"] != p["output"])[0]
            p["output"][r, c] = p["input"][r, c]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "shapes": len(shapes), "structures": len(structures), "palettes": len(palettes),
            "cut_kinds": sorted(kinds), "axes": sorted(axes), "eligible_populations": sorted(populations),
            "ranks": sorted(ranks), "hidden_bars": hidden, "partial_fills": partial,
            "regional_vs_whole_disagreements": whole_mismatch}


if __name__ == "__main__":
    print(self_test())
