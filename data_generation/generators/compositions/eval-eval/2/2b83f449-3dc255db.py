"""Rewired counterflow balances: topology determines which unary counts cancel.

No parent imports, files, example lookup, or hidden conventions are used by solve.
Generation and checking are in memory; this module never writes artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 884837088, 433561138, 951779592, 205843289, 697784463,
              285228168, 463934218, 782965271, 683442977, 969932737, 932726578,
              147474463, 533489296, 801064679, 114376800, 405426653, 372808554,
              253034880, 929256359, 93794529, 98137046, 500682157, 80891836,
              197263676, 834059932, 350578392, 852633076, 265333702, 274934068,
              483664611, 890474279, 667961564, 899070028, 517291900, 852575311,
              708200774, 100412159, 787246551, 854490010, 346821378, 201684266)
_FOUR = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=False):
    left = set(cells)
    steps = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]
             if diagonal else _FOUR)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        todo, component = [p], {p}
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _graph(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    adjacent = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _FOUR
                    if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    assert cells and all(1 <= len(v) <= 2 for v in adjacent.values()), "branched or isolated track"
    paths, cycles = [], []
    for component in _components(cells):
        ends = sorted(p for p in component if len(adjacent[p]) == 1)
        if not ends:
            cycles.append(component)
            continue
        assert len(ends) == 2
        path, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            choices = [p for p in adjacent[current] if p != previous]
            assert len(choices) == 1
            previous, current = current, choices[0]
            path.append(current)
        assert len(path) == len(component)
        paths.append(path)
    return paths, cycles


def _turn(grid, centers, track, divider):
    """Agreeing simultaneous writes allow corner overlaps between switch tiles."""
    writes = {}
    for r, c in centers:
        for y in range(r - 1, r + 2):
            for x in range(c - 1, c + 2):
                value = divider if x == c else track
                assert (y, x) not in writes or writes[y, x] == value, "conflicting switches"
                writes[y, x] = value
    result = grid.copy()
    for p, value in writes.items():
        result[p] = value
    return result


def _read_right(a):
    """Recognize right-pointing reservoirs, then the network they feed.

    Frame recognition uses color components and their three cross sections,
    not the alternating placement slots used by the constructor.
    """
    bg = int(a[0, 0])
    assert np.all(a[[0, -1], :] == bg) and np.all(a[:, [0, -1]] == bg)
    candidates = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        frames = []
        for comp in _components(map(tuple, np.argwhere(a == color)), diagonal=True):
            r0 = min(p[0] for p in comp)
            r1 = max(p[0] for p in comp)
            if r1 - r0 != 2:
                break
            top = sorted(c for r, c in comp if r == r0)
            bottom = sorted(c for r, c in comp if r == r1)
            middle = sorted(c for r, c in comp if r == r0 + 1)
            if (len(top) < 2 or top != bottom or top != list(range(top[0], top[-1] + 1))
                    or middle != [top[-1] + 1]):
                break
            y, tip = r0 + 1, middle[0]
            cavity = [(y, c) for c in top]
            values = {int(a[p]) for p in cavity} - {bg}
            if len(values) != 1 or color in values or tip + 1 >= a.shape[1]:
                break
            payload_color = next(iter(values))
            payload = {p for p in cavity if a[p] == payload_color}
            frames.append(dict(tip=(y, tip), start=(y, tip + 1), frame=comp,
                               cavity=cavity, payload=payload, color=payload_color,
                               q=len(payload), depth=len(top)))
        else:
            if len(frames) >= 2:
                candidates.append((color, sorted(frames, key=lambda f: f["tip"])))
    assert len(candidates) == 1, "one geometrically identifiable reservoir bank required"
    frame_color, frames = candidates[0]
    rows = [f["tip"][0] for f in frames]
    assert rows == list(range(rows[0], rows[-1] + 1, 2)), "one reservoir per lane"
    tracks = {int(a[f["start"]]) for f in frames}
    assert len(tracks) == 1
    track = next(iter(tracks))
    payload_colors = {f["color"] for f in frames}
    assert track not in {bg, frame_color} | payload_colors
    end = int(np.argwhere(a == track)[:, 1].max())
    max_tip = max(f["tip"][1] for f in frames)
    separators = list(range(rows[0] + 1, rows[-1], 2))
    separator_colors = {int(a[r, c]) for r in separators
                        for c in range(max_tip + 1, end + 1)} - {bg}
    assert len(separator_colors) == 1
    divider = next(iter(separator_colors))
    assert divider not in {bg, track, frame_color} | payload_colors
    expected = np.full(a.shape, bg, dtype=np.int8)
    breaks = []
    for f in frames:
        for p in f["frame"]:
            expected[p] = frame_color
        for p in f["payload"]:
            expected[p] = f["color"]
        r, start = f["start"]
        assert start < end and a[r, end] == track
        for c in range(start, end + 1):
            assert a[r, c] in (bg, track)
            expected[r, c] = a[r, c]
            if a[r, c] == bg:
                breaks.append((r, c))
    centers = []
    for r in separators:
        xs = [c for c in range(max_tip + 1, end + 1) if a[r, c] == divider]
        starts = [c for c in xs if c - 1 not in xs]
        for start in starts:
            assert all(c in xs for c in range(start, start + 3)) and start + 3 not in xs
            c = start + 1
            assert c - 1 > max_tip and c + 1 < end
            assert np.all(a[r - 1, c - 1:c + 2] == track)
            assert np.all(a[r + 1, c - 1:c + 2] == track)
            expected[r, c - 1:c + 2] = divider
            centers.append((r, c))
    assert centers and np.array_equal(a, expected), "unaccounted cells or broken switch neighborhood"
    cleared = a.copy()
    for f in frames:
        for p in f["payload"]:
            cleared[p] = bg
    structural = _turn(cleared, centers, track, divider)
    paths, cycles = _graph(structural == track)
    assert not cycles, "closed routes are outside the family"
    sources = {f["start"]: f for f in frames}
    ends = {p for path in paths for p in (path[0], path[-1])}
    assert sources.keys() <= ends, "every reservoir must feed a path endpoint"
    return dict(a=a, bg=bg, track=track, frame_color=frame_color, divider=divider,
                frames=frames, sources=sources, centers=centers, breaks=breaks,
                paths=paths, structural=structural, cleared=cleared, rows=rows)


def _read(value):
    a = _grid(value)
    matches = []
    for rotation in range(4):
        canonical = np.rot90(a, rotation).copy()
        try:
            info = _read_right(canonical)
        except (AssertionError, IndexError, ValueError):
            continue
        matches.append((rotation, info))
    if len(matches) != 1:
        raise ValueError("input has no unique valid counterflow interpretation")
    return matches[0]


def _emit(info, paths=None, unit_counts=False, cancel=True):
    result = info["structural"].copy()
    records = []
    for path in info["paths"] if paths is None else paths:
        sources = [info["sources"][p] for p in (path[0], path[-1]) if p in info["sources"]]
        if not sources:
            continue
        counts = [1 if unit_counts else f["q"] for f in sources]
        targets = []
        if len(sources) == 1:
            targets = [(sources[0], counts[0])]
        elif not cancel:
            targets = list(zip(sources, counts))
        elif counts[0] != counts[1]:
            winner = int(counts[1] > counts[0])
            targets = [(sources[winner], abs(counts[0] - counts[1]))]
        painted = []
        for frame, quantity in targets:
            ordered = path if path[0] == frame["start"] else path[::-1]
            cells = ordered[:quantity]
            for p in cells:
                # For the no-rewiring diagnostic, avoid painting a new divider.
                if result[p] == info["track"] or result[p] in {f["color"] for f in sources}:
                    result[p] = frame["color"]
            painted.extend(cells)
        records.append(dict(sources=sources, counts=counts, painted=painted,
                            residual=sum(q for _, q in targets), capacity=len(path)))
    return result, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every role, direction, route, quantity and winner from input only."""
    rotation, info = _read(input_grid)
    answer, _ = _emit(info)
    return np.rot90(answer, -rotation).copy()


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError("counts must be integers: train 2..10 and test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _base(layout, palette):
    bg, track, frame_color, divider = palette[:4]
    a = np.full(layout["shape"], bg, dtype=np.int8)
    frames = []
    for i, (tip, depth) in enumerate(zip(layout["tips"], layout["depths"])):
        r = layout["top"] + 1 + 2 * i
        cells = {(r, tip)} | {(y, x) for y in (r - 1, r + 1)
                                for x in range(tip - depth, tip)}
        for p in cells:
            a[p] = frame_color
        a[r, tip + 1:layout["end"] + 1] = track
        frames.append(dict(tip=(r, tip), start=(r, tip + 1), frame=cells,
                           cavity=[(r, x) for x in range(tip - depth, tip)], depth=depth))
    for r, c in layout["centers"]:
        a[r, c - 1:c + 2] = divider
    for p in layout["breaks"]:
        a[p] = bg
    return a, frames


def _layout(rng, anchor=False, busy=False):
    lanes = int(rng.integers(5, 8) if anchor else rng.integers(2, 8))
    max_depth = 6 if anchor else int(rng.integers(2, 7))
    left, top = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    low = left + max_depth
    tips = [low + (max_depth + 2) * (i % 2) for i in range(lanes)]
    depths = [max_depth if anchor else int(rng.integers(2, max_depth + 1)) for _ in tips]
    first = max(tips) + 3
    end = int(rng.integers(first + (7 if busy else 5), 29))
    width = int(rng.integers(end + 2, 31))
    height = top + 2 * lanes + 1 + int(rng.integers(1, 4))
    centers = []
    block = None
    if anchor:
        block = int(rng.integers(lanes - 3))
        centers = [(top + 2 + 2 * (block + j), first + (2 if j != 1 else 0)) for j in range(3)]
    else:
        forced = int(rng.integers(lanes - 1)) if busy else None
        previous = []
        for i in range(lanes - 1):
            if i == forced:
                # An arbitrary busy separator, not a distinguished final row.
                pairs = [(x, y) for x in range(first, end - 1)
                         for y in range(x + 4, end - 1)]
                chosen = list(pairs[int(rng.integers(len(pairs)))])
                centers = [(r, c) for r, c in centers
                           if r != top + 2 * i or all(abs(c - x) >= 2 for x in chosen)]
            else:
                chosen = []
                wanted = int(rng.integers(0, 3)) if busy else int(rng.random() < .8)
                for value in rng.permutation(np.arange(first, end - 1)):
                    x = int(value)
                    if len(chosen) < wanted and all(abs(x - y) >= 4 for y in chosen) and all(abs(x - y) >= 2 for y in previous):
                        chosen.append(x)
            centers.extend((top + 2 + 2 * i, c) for c in sorted(chosen))
            previous = chosen
        if not centers:
            centers = [(top + 2 + 2 * int(rng.integers(lanes - 1)), first)]
    layout = dict(shape=(height, width), top=top, tips=tips, depths=depths,
                  centers=centers, end=end, breaks=[], block=block)
    # No arbitrary rejection loop: open each cycle at an ordinary horizontal cell.
    base, frames = _base(layout, (0, 1, 2, 3))
    if not anchor:
        for _ in range(len(centers) + 1):
            structural = _turn(base, centers, 1, 3)
            _, cycles = _graph(structural == 1)
            if not cycles:
                break
            comp = cycles[0]
            options = [p for p in sorted(comp) if p[0] in [f["tip"][0] for f in frames]
                       and base[p] == 1 and base[p[0], p[1] - 1] == base[p[0], p[1] + 1] == 1
                       and all(max(abs(p[0] - r), abs(p[1] - c)) > 1 for r, c in centers)]
            assert options, "cycle has no legal cut"
            p = options[int(rng.integers(len(options)))]
            layout["breaks"].append(p)
            base[p] = 0
        else:
            raise AssertionError("cycle opening bound exceeded")
        for _ in range(int(rng.integers(0, 4))):
            options = [(f["tip"][0], c) for f in frames for c in range(f["start"][1] + 2, end - 1)
                       if base[f["tip"][0], c - 1] == base[f["tip"][0], c] == base[f["tip"][0], c + 1] == 1
                       and all(r != f["tip"][0] or abs(c - x) >= 3 for r, x in layout["breaks"])
                       and all(max(abs(f["tip"][0] - r), abs(c - x)) > 1 for r, x in centers)]
            if not options:
                break
            p = options[int(rng.integers(len(options)))]
            layout["breaks"].append(p)
            base[p] = 0
    return layout


def _positions(cavity, quantity, kind, rng):
    if kind == "compact":
        start = int(rng.integers(len(cavity) - quantity + 1))
        return set(cavity[start:start + quantity])
    if kind == "spread":
        indices = np.rint(np.linspace(0, len(cavity) - 1, quantity)).astype(int)
        return {cavity[int(i)] for i in indices}
    return {cavity[int(i)] for i in rng.choice(len(cavity), quantity, replace=False)}


def _scene(rng, layout, palette, rotation, anchor_version=None):
    a, frames = _base(layout, palette)
    bg, track, frame_color, divider = palette[:4]
    structural = _turn(a, layout["centers"], track, divider)
    paths, cycles = _graph(structural == track)
    assert not cycles
    by_start = {f["start"]: i for i, f in enumerate(frames)}
    quantities = [int(rng.integers(1, f["depth"] + 1)) for f in frames]
    kinds = [str(rng.choice(["compact", "spread", "random"])) for _ in frames]
    colors = [int(palette[4 + i % (len(palette) - 4)]) for i in range(len(frames))]
    if anchor_version is not None:
        b = layout["block"]
        winner = b if b % 2 else b + 3  # the nearer of the two nonadjacent sources
        loser = b + 3 if winner == b else b
        quantities[winner], quantities[loser] = (6, 1) if anchor_version == 0 else (1, 4)
        # Both winners and both colors are demonstrated, including train=2.
        colors[loser] = next(int(c) for c in palette[4:] if int(c) != colors[winner])
        quantities[b + 1] = quantities[b + 2] = 2
        kinds[b + 1], kinds[b + 2] = "compact", "spread"
        # All other lanes are single sources, with a fully visible count of 3.
        for i in range(len(frames)):
            if not b <= i <= b + 3:
                quantities[i] = 3
                kinds[i] = "compact" if anchor_version == 0 else "spread"
    else:
        paired = [[by_start[p] for p in (path[0], path[-1]) if p in by_start] for path in paths]
        paired = [p for p in paired if len(p) == 2]
        if not paired:
            return None
        i, j = paired[int(rng.integers(len(paired)))]
        if rng.random() < .5:
            i, j = j, i
        quantities[i], quantities[j] = frames[i]["depth"], 1
    for i, frame in enumerate(frames):
        payload = _positions(frame["cavity"], quantities[i], kinds[i], rng)
        frame.update(q=quantities[i], color=colors[i], payload=payload)
        for p in payload:
            a[p] = colors[i]
    # Construct the label from the sampled geometry. solve later has to read it.
    info = dict(structural=structural, paths=paths, track=track,
                sources={f["start"]: f for f in frames})
    out, _ = _emit(info)
    return {"input": np.rot90(a, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.choice(10, size=int(rng.integers(6, 9)), replace=False)))
    layout = _layout(rng, anchor=True)
    rotation = int(rng.integers(4))
    anchors = [_scene(rng, layout, palette, rotation, version) for version in (0, 1)]
    if rng.random() < .5:
        anchors.reverse()
    puzzle = {"train": anchors, "test": []}
    seen = {_key(p["input"]) for p in anchors}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for _ in range(count):
            for attempt in range(48):
                local_palette = list(map(int, rng.choice(10, size=int(rng.integers(6, 9)), replace=False)))
                plan = _layout(rng, busy=split == "test")
                pair = _scene(rng, plan, local_palette, int(rng.integers(4)))
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction could not find a distinct connected balance")
    # Move the matched witnesses through the training sequence.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _ablations(info):
    raw_paths, cycles = _graph(info["cleared"] == info["track"])
    assert not cycles
    return (_emit(info, paths=raw_paths)[0], _emit(info, unit_counts=True)[0],
            _emit(info, cancel=False)[0])


def _evidence(infos):
    ties = bends = nonnearest = single = component_example = False
    residuals, repeats, matched, winners = set(), {}, {}, {}
    for info in infos:
        out, records = _emit(info)
        for frame in info["frames"]:
            repeats.setdefault(frame["q"], set()).add(len(_components(frame["payload"])))
        for rec in records:
            fs = rec["sources"]
            if rec["residual"] and rec["residual"] < rec["capacity"]:
                residuals.add(rec["residual"])
            if len(fs) == 1 and rec["residual"] < rec["capacity"]:
                single = True
            if len(fs) == 2:
                ties |= fs[0]["q"] == fs[1]["q"]
                nonnearest |= abs(fs[0]["tip"][0] - fs[1]["tip"][0]) > 2
                component_example |= (fs[0]["q"] == fs[1]["q"]
                                      and len(_components(fs[0]["payload"])) != len(_components(fs[1]["payload"])))
            bends |= any(p[0] not in info["rows"] for p in rec["painted"])
        # Same complete scene geometry/colors, different observed surplus lengths.
        skeleton = info["cleared"]
        signature = (_key(skeleton), tuple(f["color"] for f in info["frames"]))
        matched.setdefault(signature, set()).add(tuple((r["residual"], r["capacity"]) for r in records))
        for rec in records:
            fs = rec["sources"]
            if len(fs) == 2 and fs[0]["q"] != fs[1]["q"]:
                winner = max(fs, key=lambda f: f["q"])
                key = signature, tuple(f["start"] for f in fs)
                winners.setdefault(key, set()).add(winner["start"])
    assert ties and bends and nonnearest and single and component_example, "missing conceptual training witness"
    assert len(residuals) >= 2 and any(len(v) >= 2 for v in repeats.values())
    assert any(len(v) >= 2 for v in matched.values()), "no matched count intervention"
    assert any(len(v) == 2 for v in winners.values()), "no matched reversal of the winning endpoint"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            rotation, info = _read(a)
            expected, records = _emit(info)
            assert np.array_equal(b, np.rot90(expected, -rotation))
            assert not np.array_equal(a, b)
            # Frames/background/untouched track are protected by exact comparison.
            for f in info["frames"]:
                assert all(expected[p] == info["frame_color"] for p in f["frame"])
                assert all(expected[p] == info["bg"] for p in f["payload"])
            for rec in records:
                assert len(rec["painted"]) == min(rec["residual"], rec["capacity"])
            assert all(not np.array_equal(cf, expected) for cf in _ablations(info)), "parent or cancellation is dispensable"
            counts = [sum(r == rr for r, _ in info["centers"]) for rr in range(info["rows"][0] + 1, info["rows"][-1], 2)]
            if split == "train":
                assert max(counts) <= 1
                training.append(info)
            else:
                assert max(counts) >= 2, "missing test topology shift"
    _evidence(training)


def self_test():
    """Bounded streaming audit: 42 host seeds, no saved puzzles or execution logs."""
    shapes, structures, palettes, directions, depths = set(), set(), set(), set(), set()
    quantities, residuals, partners, breaks, bar_counts = set(), set(), set(), set(), set()
    fingerprints, pairs, ties, bends, clipped, nonmonotone, overlaps = set(), 0, 0, 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        repeated = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                assert all(np.array_equal(pair[k], repeated[split][index][k]) for k in ("input", "output"))
                inp = pair["input"]
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"]) and np.array_equal(inp, before)
                rotation, info = _read(inp)
                directions.add(rotation)
                normalized, roles = np.empty(inp.shape, dtype=np.int8), {}
                for p in np.ndindex(inp.shape):
                    normalized[p] = roles.setdefault(int(inp[p]), len(roles))
                structures.add((inp.shape, hashlib.sha256(normalized.tobytes()).hexdigest()))
                shapes.add(inp.shape)
                palettes.add((info["bg"], info["track"], info["frame_color"], info["divider"]))
                depths.update(f["depth"] for f in info["frames"])
                quantities.update(f["q"] for f in info["frames"])
                breaks.add(len(info["breaks"]))
                bar_counts.add(len(info["centers"]))
                _, records = _emit(info)
                for rec in records:
                    residuals.add(rec["residual"])
                    ties += len(rec["sources"]) == 2 and rec["residual"] == 0
                    clipped += rec["residual"] > rec["capacity"]
                    bends += any(p[0] not in info["rows"] for p in rec["painted"])
                    if len(rec["sources"]) == 2:
                        partners.add(abs(rec["sources"][0]["tip"][0] - rec["sources"][1]["tip"][0]) // 2)
                for path in info["paths"]:
                    signs = {int(np.sign(q[0] - p[0])) for p, q in zip(path, path[1:]) if q[0] != p[0]}
                    nonmonotone += len(signs) > 1
                neighborhoods = [{(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} for r, c in info["centers"]]
                overlaps += any(x & y for i, x in enumerate(neighborhoods) for y in neighborhoods[i + 1:])
                fingerprint.update(str(inp.shape).encode())
                fingerprint.update(inp.tobytes())
                fingerprint.update(pair["output"].tobytes())
                pairs += 1
        fingerprints.add(fingerprint.hexdigest())
    assert len(fingerprints) == 42 and len(structures) >= 150 and len(shapes) >= 25
    assert directions == {0, 1, 2, 3} and depths == set(range(2, 7))
    assert quantities == set(range(1, 7)) and {0, 1, 2, 3, 4, 5} <= residuals
    assert max(partners) >= 3 and len(breaks) >= 4 and len(bar_counts) >= 5
    assert ties and bends and overlaps and nonmonotone
    for n, m in ((2, 1), (10, 1), (10, 4), (5, 4)):
        generate_puzzle(41, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Hand-constructed local two-frame check, independently specified edits.
    layout = dict(shape=(7, 17), top=1, tips=[3, 7], depths=[2, 2], end=14,
                  centers=[(3, 10)], breaks=[], block=None)
    inp, frames = _base(layout, (0, 8, 5, 4))
    inp[2, 1:3], inp[4, 6] = 2, 3
    expected = inp.copy()
    expected[2, 1:3], expected[4, 6] = 0, 0
    expected[2:5, 9:12] = 8
    expected[2:5, 10] = 4
    expected[2, 4] = 2  # 2 minus 1, emitted from the larger reservoir.
    assert np.array_equal(solve(inp), expected)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation))
    corrupt = generate_puzzle(0, 2, 1)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(fingerprints),
                shapes=len(shapes), normalized_structures=len(structures), palettes=len(palettes),
                directions=sorted(directions), frame_depths=sorted(depths), quantities=sorted(quantities),
                residuals=sorted(residuals), partner_lane_distances=sorted(map(int, partners)),
                break_counts=sorted(breaks), divider_counts=sorted(bar_counts),
                tie_paths=ties, bent_runs=bends, clipped_runs=clipped,
                nonmonotone_paths=nonmonotone, overlapping_switch_pairs=overlaps,
                semantic_verification="generator-owned checks; no independent solver or blind prediction")
