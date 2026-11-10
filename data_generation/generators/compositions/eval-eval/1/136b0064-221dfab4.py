"""Feedback scanner: periodic cross-sections conditionally execute arrow strokes.

Only the standard library and NumPy are used. All solving conventions are
geometric or visible in the input; no puzzle-level dictionary is hidden.
"""

import numpy as np


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(mask):
    """Monochrome eight-connected components, including diagonal glyph ink."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        pending, cells = [root], []
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = r + dy, c + dx
                    if p in remaining:
                        remaining.remove(p)
                        pending.append(p)
        result.append(cells)
    return result


def _arrow_mask(direction, width):
    # The central tip points toward motion. Width minus one is displacement.
    mask = np.zeros((3, width), dtype=bool)
    mask[0, :-1] = mask[2, :-1] = True
    mask[1, -1] = True
    return mask if direction == 1 else mask[:, ::-1]


def _decode_component(cells):
    rr, cc = np.array(cells).T
    top, left = int(rr.min()), int(cc.min())
    h, w = int(rr.max() - top + 1), int(cc.max() - left + 1)
    if h != 3 or w not in (2, 3, 4):
        return None
    local = np.zeros((h, w), dtype=bool)
    local[rr - top, cc - left] = True
    for direction in (-1, 1):
        if np.array_equal(local, _arrow_mask(direction, w)):
            return {"row": top + 1, "tip": left + (w - 1 if direction == 1 else 0),
                    "delta": direction * (w - 1), "width": w, "cells": cells}
    return None


def _analyze(input_grid):
    original = _grid(input_grid)
    values, counts = np.unique(original, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if counts.max() <= original.size / 2:
        raise ValueError("background must occupy more than half the input")
    candidates = []
    for rotation in range(4):
        g = np.rot90(original, rotation)
        if len(g) < 2:
            continue
        for marker in values:
            marker = int(marker)
            if marker == bg:
                continue
            points = np.argwhere(g == marker)
            if not np.all(points[:, 0] == 0):
                continue
            cols = points[:, 1]
            s, w = int(cols.min()), len(cols)
            if not np.array_equal(np.sort(cols), np.arange(s, s + w)):
                continue
            pilot = int(g[1, s])
            if pilot in (bg, marker) or np.count_nonzero(g == pilot) != 1:
                continue
            # Apart from the bar, the seed edge is blank.
            if np.count_nonzero(g[0] != bg) != w:
                continue
            candidates.append((rotation, g.copy(), marker, pilot, s, w))
    if len(candidates) != 1:
        raise ValueError("need one boundary bar with a unique inward pilot pixel")
    rotation, g, marker, pilot, s, w = candidates[0]
    arrows = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, marker, pilot):
            continue
        for cells in _components(g == color):
            arrow = _decode_component(cells)
            if arrow is not None:
                arrow["color"] = color
                arrows.append(arrow)
    arrows.sort(key=lambda a: (a["row"], a["tip"]))
    return {"grid": g, "rotation": rotation, "background": bg, "marker": marker,
            "pilot": pilot, "start": s, "length": w, "arrows": arrows}


def _sweep(info, policy="normal"):
    """One interleaved scan/execute procedure; policies are ablation checks only."""
    original = info["grid"]
    bg, marker, pilot = (info[k] for k in ("background", "marker", "pilot"))
    s, w = info["start"], info["length"]
    result = original.copy()
    foreground = ~np.isin(original, (bg, marker, pilot))
    strips = np.zeros_like(original, dtype=bool)
    connectors = np.zeros_like(original, dtype=bool)
    events, spans, pending = [], [], None
    for d in range(len(original)):
        spans.append(s)
        strips[d, s:s + w] = True
        result[d, s:s + w] = bg
        special = d % 6 == 4
        if d % 2 == 0:
            result[d, s:s + w] = pilot if special else marker
        if special and policy != "no_remote_highlight":
            result[d, foreground[d]] = pilot
        if pending is not None:
            old, new, color = pending
            a, b = sorted((old, new))
            result[d, a:b + 1] = color
            connectors[d, a:b + 1] = True
            s, pending = (old if policy == "freeze_span" else new), None
        # Connector rows cannot also run an instruction: their phase is five.
        eligible = [a for a in info["arrows"] if a["row"] == d
                    and (policy == "ignore_span" or s <= a["tip"] < s + w)]
        gate = special or (policy == "ignore_period" and d % 6 != 5)
        if not gate or policy == "no_commands" or not eligible or d + 1 == len(original):
            continue
        if len(eligible) != 1:
            raise ValueError("more than one eligible arrow on a command cross-section")
        arrow = eligible[0]
        new = s + arrow["delta"]
        if not 0 <= new <= original.shape[1] - w:
            raise ValueError("activated command moves the full bar outside the canvas")
        events.append({**arrow, "old": s, "new": new})
        pending = s, new, arrow["color"]
    return {"output": result, "strips": strips, "connectors": connectors,
            "events": events, "spans": spans, "foreground": foreground}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the result solely from the given input, including its pilot hue."""
    info = _analyze(input_grid)
    return np.rot90(_sweep(info)["output"], -info["rotation"]).copy()


def _arrow_cells(row, tip, delta):
    direction, width = int(np.sign(delta)), abs(delta) + 1
    left = tip - width + 1 if direction == 1 else tip
    rr, cc = np.nonzero(_arrow_mask(direction, width))
    return [(row - 1 + int(r), left + int(c)) for r, c in zip(rr, cc)]


def _put_arrow(rng, grid, row, tips, deltas, palette, bg):
    """Finite candidate search, preserving recognizable monochrome components."""
    options = [(int(t), int(delta)) for t in tips for delta in deltas]
    for index in rng.permutation(len(options)):
        tip, delta = options[int(index)]
        cells = _arrow_cells(row, tip, delta)
        if any(not (0 <= r < len(grid) and 0 <= c < grid.shape[1]) or grid[r, c] != bg
               for r, c in cells):
            continue
        for color in rng.permutation(palette):
            if any(grid[rr, cc] == color for r, c in cells
                   for rr in range(max(0, r - 1), min(len(grid), r + 2))
                   for cc in range(max(0, c - 1), min(grid.shape[1], c + 2))):
                continue
            for r, c in cells:
                grid[r, c] = color
            return tip, delta
    return None


def _plans(rng, span, w, rows, commands, feedback):
    """Find anchor/tip geometry for a planned sequence without an unbounded retry."""
    offsets = np.cumsum([0] + list(commands))
    low, high = 2 - int(offsets.min()), span - w - 2 - int(offsets.max())
    if high < low:
        return None
    for initial in rng.permutation(np.arange(low, high + 1)):
        initial = int(initial)
        anchors, tips = [], []
        for j, (row, delta) in enumerate(zip(rows, commands)):
            s = initial + int(offsets[j])
            possible = [t for t in range(s, s + w)
                        if all(0 <= c < span for _, c in _arrow_cells(row, t, delta))
                        and (not feedback or j != 1 or not initial <= t < initial + w)]
            if not possible:
                break
            anchors.append(s)
            tips.append(int(rng.choice(possible)))
        else:
            return initial, anchors, tips
    return None


def _objects(rng, grid, bg, marker, pilot, palette, density):
    """Natural rectangles, rings, and connected growth around protected glyph ink."""
    h, w = grid.shape
    occupied = grid != bg
    protected = occupied.copy()
    # Objects do not touch instruction glyphs; unlike a repeated witness stamp,
    # their seeds, dimensions, topology and relation to scan phases are random.
    for r, c in np.argwhere(occupied):
        protected[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
    protected[0] = True
    target = int(h * w * density)
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(18):
        if int(mask.sum()) >= target:
            break
        color = int(rng.choice(palette))
        candidate = np.zeros_like(mask)
        kind = int(rng.integers(3))
        if kind < 2:
            rh, cw = int(rng.integers(2, min(8, h) + 1)), int(rng.integers(2, min(8, w) + 1))
            r, c = int(rng.integers(1, h - rh + 1)), int(rng.integers(w - cw + 1))
            candidate[r:r + rh, c:c + cw] = True
            if kind == 1 and rh >= 4 and cw >= 4:
                candidate[r + 1:r + rh - 1, c + 1:c + cw - 1] = False
        else:
            available = np.argwhere(~protected & ~mask)
            if not len(available):
                break
            root = tuple(available[int(rng.integers(len(available)))])
            pending, queued = [root], {root}
            budget = int(rng.integers(4, max(5, min(70, target - int(mask.sum()) + 1))))
            for _ in range(budget):
                if not pending:
                    break
                r, c = pending.pop(int(rng.integers(len(pending))))
                candidate[r, c] = True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    p = r + dy, c + dx
                    if (0 <= p[0] < h and 0 <= p[1] < w and not protected[p]
                            and not mask[p] and p not in queued):
                        queued.add(p)
                        pending.append(p)
        candidate &= ~protected & ~mask
        if candidate.sum() < 4 or np.count_nonzero(grid != bg) + candidate.sum() >= h * w * .47:
            continue
        proposed = (grid == color) | candidate
        if any(_decode_component(cells) is not None for cells in _components(proposed)):
            continue  # Passive objects must not accidentally spell an instruction.
        grid[candidate] = color
        mask |= candidate
    return grid


def _make_pair(rng, rotation, depth, span, w, commands, roles, feedback=False,
               teach_decoys=False, density=.16, active_rows=None):
    bg, marker, pilot = roles[:3]
    arrow_palette, object_palette = roles[3:6], roles[6:8]
    rows = list(range(4, depth - 1, 6))
    active_rows = rows if active_rows is None else active_rows
    plan = _plans(rng, span, w, active_rows, commands, feedback)
    if plan is None:
        return None
    initial, _, tips = plan
    anchors = [initial + sum(delta for r, delta in zip(active_rows, commands) if r < row)
               for row in rows]
    g = np.full((depth, span), bg, dtype=np.int8)
    g[0, initial:initial + w] = marker
    g[1, initial] = pilot
    for row, tip, delta in zip(active_rows, tips, commands):
        # Three distinct demonstrated meanings use at most two hues, so color
        # alone cannot select the motion even in a two-training-pair request.
        command_palette = arrow_palette[:2] if teach_decoys else arrow_palette
        if _put_arrow(rng, g, row, [tip], [delta], command_palette, bg) is None:
            return None
    # A highlighted but remote glyph is visible yet inactive. The same row's
    # foreground is still recolored, independently of instruction activation.
    for j, row in enumerate(rows):
        if teach_decoys or rng.random() < .6:
            outside = [t for t in range(span) if not anchors[j] <= t < anchors[j] + w]
            _put_arrow(rng, g, row, outside, [-3, -2, -1, 1, 2, 3], arrow_palette, bg)
    # At phase one of the next cycle, an in-span arrow must NOT execute.
    for j in range(len(rows) - 1):
        if teach_decoys or rng.random() < .65:
            s = anchors[j + 1]
            placed = _put_arrow(rng, g, rows[j] + 3, range(s, s + w),
                                [-3, -2, -1, 1, 2, 3], arrow_palette, bg)
            if teach_decoys and placed is None:
                return None
    # Optional arrows on other phases or remote cross-sections. They never
    # create a second eligible tip on a special row.
    for _ in range(int(rng.integers(0, 4))):
        row = int(rng.integers(2, depth - 1))
        if row % 6 == 4:
            continue
        _put_arrow(rng, g, row, range(span), [-3, -2, -1, 1, 2, 3], arrow_palette, bg)
    _objects(rng, g, bg, marker, pilot, object_palette, density)
    a = np.rot90(g, rotation).copy()
    try:
        info = _analyze(a)
        sweep = _sweep(info)
    except ValueError:
        return None
    if len(sweep["events"]) != len(commands):
        return None
    pair = {"input": a, "output": np.rot90(sweep["output"], -info["rotation"]).copy()}
    return pair


def _evidence(info, sweep):
    fg, strips = sweep["foreground"], sweep["strips"]
    high = np.broadcast_to(np.arange(len(fg))[:, None] % 6 == 4, fg.shape)
    ordinary = np.broadcast_to(np.isin(np.arange(len(fg)) % 6, (0, 2))[:, None], fg.shape)
    odd = np.broadcast_to(np.arange(len(fg))[:, None] % 2 == 1, fg.shape)
    arrows = info["arrows"]
    spans, w = sweep["spans"], info["length"]
    initial = info["start"]
    return {
        "feedback": any(not initial <= e["tip"] < initial + w for e in sweep["events"][1:]),
        "off_phase": any(a["row"] % 6 != 4 and spans[a["row"]] <= a["tip"] < spans[a["row"]] + w
                         for a in arrows),
        "remote_arrow": any(a["row"] % 6 == 4 and not spans[a["row"]] <= a["tip"] < spans[a["row"]] + w
                            for a in arrows),
        "remote_highlight": bool(np.any(fg & high & ~strips)),
        "outside_preserved": bool(np.any(fg & ~high & ~strips & ~sweep["connectors"])),
        "outside_background": bool(np.any((info["grid"] == info["background"]) & high & ~strips)),
        "clearing": bool(np.any(fg & strips & odd & ~sweep["connectors"])),
        "ordinary_overwrite": bool(np.any(fg & strips & ordinary)),
        "connector": bool(sweep["events"]),
    }


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    vocabulary = [int(x) for x in rng.permutation([-3, -2, -1, 1, 2, 3])]
    base = int(rng.integers(4))
    puzzle, seen, evidence = {"train": [], "test": []}, set(), []
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            mandatory = split == "train" and index < 2
            rotation = ((base + 2 * index) % 4 if split == "train" and index < 2
                        else (base + 1 + 2 * index) % 4 if split == "test"
                        else int(rng.integers(4)))
            accepted = None
            # Bounded natural proposals; final proposals use generous geometry,
            # rather than adding a repeated foreground evidence stamp.
            for attempt in range(80):
                depth = int(rng.integers(18, 23) if mandatory else
                            rng.integers(25, 31) if split == "test" else rng.integers(12, 23))
                span = int(rng.integers(18 if mandatory else 16, 31))
                w = int(rng.integers(1, 7))
                rows = list(range(4, depth - 1, 6))
                active_rows = rows if mandatory else [r for r in rows if rng.random() < .76]
                commands = vocabulary[3 * index:3 * index + 3] if mandatory else [
                    int(rng.choice([-3, -2, -1, 1, 2, 3])) for _ in active_rows]
                if attempt >= 64:
                    span, w = 30, 3
                roles = [int(v) for v in rng.permutation(10)]
                pair = _make_pair(rng, rotation, depth, span, w, commands, roles,
                                  feedback=mandatory and index == 0,
                                  teach_decoys=mandatory,
                                  density=float(rng.uniform(.07, .25)), active_rows=active_rows)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                info = _analyze(pair["input"])
                observed = _evidence(info, _sweep(info))
                if mandatory and not all(observed[k] for k in (
                        "off_phase", "remote_arrow", "remote_highlight", "outside_preserved",
                        "outside_background", "clearing", "ordinary_overwrite", "connector")):
                    continue
                if mandatory and index == 0 and not observed["feedback"]:
                    continue
                accepted = pair
                if split == "train":
                    evidence.append(observed)
                break
            if accepted is None:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_key(accepted["input"]))
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, evidence, train_commands, meanings_by_color = set(), [], set(), {}
    train_cycles, test_cycles = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            sweep = _sweep(info)
            out = np.rot90(b, info["rotation"])
            assert np.array_equal(out, sweep["output"]), "incorrect scanner output"
            # Semantic conservation: untouched pixels include remote foreground
            # and blank cells; only special rows, current strips and strokes change.
            high = np.broadcast_to(np.arange(len(out))[:, None] % 6 == 4, out.shape)
            untouched = ~sweep["strips"] & ~sweep["connectors"] & ~(high & sweep["foreground"])
            assert np.array_equal(out[untouched], info["grid"][untouched])
            for event in sweep["events"]:
                assert event["new"] - event["old"] == event["delta"]
                lo, hi = sorted((event["old"], event["new"]))
                assert np.all(out[event["row"] + 1, lo:hi + 1] == event["color"])
                assert sweep["spans"][event["row"] + 1] == event["old"]
                if event["row"] + 2 < len(out):
                    assert sweep["spans"][event["row"] + 2] == event["new"]
            cycles = len(range(4, len(out), 6))
            if split == "train":
                evidence.append(_evidence(info, sweep))
                train_commands.update(e["delta"] for e in sweep["events"])
                for e in sweep["events"]:
                    meanings_by_color.setdefault(e["color"], set()).add(e["delta"])
                train_cycles.append(cycles)
            else:
                test_cycles.append(cycles)
    assert train_commands == {-3, -2, -1, 1, 2, 3}, "all arrow meanings must be demonstrated"
    assert any(len(v) > 1 for v in meanings_by_color.values()), "color-only command shortcut"
    assert min(test_cycles) > max(train_cycles), "tests must extend the demonstrated clock"
    for key in ("feedback", "off_phase", "remote_arrow", "remote_highlight", "outside_preserved",
                "outside_background", "clearing", "ordinary_overwrite", "connector"):
        assert any(e[key] for e in evidence), f"missing training witness: {key}"
    # Omitting either parent yields a concrete changed demonstration.
    first = _analyze(puzzle["train"][0]["input"])
    correct = _sweep(first)["output"]
    assert not np.array_equal(correct, _sweep(first, "no_commands")["output"])
    assert not np.array_equal(correct, _sweep(first, "no_remote_highlight")["output"])
    # Ablations can be geometrically invalid as well as produce wrong pixels.
    for policy in ("freeze_span", "ignore_period", "ignore_span"):
        try:
            alternate = _sweep(first, policy)["output"]
        except ValueError:
            pass
        else:
            assert not np.array_equal(correct, alternate), f"unexposed shortcut: {policy}"


def _normalized_key(grid):
    roles = {}
    return grid.shape, tuple(roles.setdefault(int(v), len(roles)) for v in grid.flat)


def self_test():
    """Bounded variation, semantic, orientation, boundary, and rejection checks."""
    shapes, structures, rotations, lengths, objects, paths, palettes = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                sweep = _sweep(info)
                shapes.add(pair["input"].shape)
                structures.add(_normalized_key(pair["input"]))
                rotations.add(info["rotation"])
                lengths.add(info["length"])
                objects.add(int(sweep["foreground"].sum()))
                paths.add(tuple(e["delta"] for e in sweep["events"]))
                palettes.add(tuple(info[k] for k in ("background", "marker", "pilot")))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                # Rotating the entire input must rotate the entire answer.
                if seed < 4:
                    for k in range(1, 4):
                        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
    assert len(shapes) >= 50 and len(structures) == 210
    assert rotations == {0, 1, 2, 3} and lengths == set(range(1, 7))
    assert len(objects) >= 60 and len(paths) >= 100 and len(palettes) >= 100
    a, b = generate_puzzle(443812052), generate_puzzle(443812052)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 5 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Hand-built terminal, empty-program, full-span and boundary cases. The pilot
    # makes even a corner singleton's inward direction explicit.
    for depth, span, start, width in ((5, 8, 0, 1), (8, 8, 0, 8), (11, 12, 3, 2)):
        g = np.zeros((depth, span), dtype=np.int8)
        g[0, start:start + width] = 8
        g[1, start] = 7
        g[4, -1] = 2
        expected = g.copy()
        for d in range(depth):
            expected[d, start:start + width] = 7 if d % 6 == 4 else 8 if d % 2 == 0 else 0
        expected[4, -1] = 7
        for rotation in range(4):
            assert np.array_equal(solve(np.rot90(g, rotation)), np.rot90(expected, rotation))
    # Snapshot and exact-boundary regression: glyph wings are cleared before
    # reading their tips, and the second stroke returns the span to the boundary.
    g = np.zeros((14, 14), dtype=np.int8)
    g[0, :2], g[1, 0] = 8, 7
    for r, c in _arrow_cells(4, 1, 1):
        g[r, c] = 3
    for r, c in _arrow_cells(10, 1, -1):
        g[r, c] = 5
    g[4:6, -1] = 2
    out = solve(g)
    assert out[3, 0] == 0 and np.all(out[5, :2] == 3)
    assert np.all(out[6, 1:3] == 8)
    assert np.all(out[11, :2] == 5) and out[11, 2] == 0
    assert np.all(out[12, :2] == 8) and out[4, -1] == 7 and out[5, -1] == 2
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(g, rotation)), np.rot90(out, rotation))
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus count, boundary, and rotation checks)")
