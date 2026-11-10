"""Equal-beat combs: normalize superimposed pulse streams by event count.

Only NumPy and the standard library are used. No files are read or written.
Source runs are read nearest-first; all periods share phase zero. The ordinal
middle lane's existing number of occupied pulse cells is the target for all.
"""
from __future__ import annotations

import hashlib
import itertools
from functools import lru_cache

import numpy as np


_OPTIONS = tuple(ls for k in (1, 2, 3)
                 for ls in itertools.product(range(2, 7), repeat=k)
                 if sum(ls) <= 11)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _events(lengths, width=30):
    return [t for t in range(width) if any(t % p == 0 for p in lengths)]


@lru_cache(maxsize=4096)
def _extent(lengths, count):
    events = _events(lengths)
    return events[count - 1] + 1 if len(events) >= count else 100


def _parse(input_grid):
    """Recover orientation, packed source runs, and valid finite pulse prefixes.

    Bare separator endpoints and the first inter-pulse hole make roles unique.
    The parser does not require sampler-specific period, lane, or count caps.
    """
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    if counts.max() <= grid.size / 2:
        raise ValueError("background must occupy more than half the input")
    background = int(values[counts.argmax()])
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        height, width = a.shape
        for col in range(2, width - 1):
            separator = int(a[0, col])
            if separator == background or not np.all(a[:, col] == separator):
                continue
            if np.count_nonzero(a == separator) != height:
                continue
            if (np.any(a[0, :col] != background) or np.any(a[0, col + 1:] != background)
                    or np.any(a[-1, :col] != background)
                    or np.any(a[-1, col + 1:] != background)):
                continue
            lanes = []
            valid = True
            for row in range(1, height - 1):
                source = np.flatnonzero(a[row, :col] != background)
                drawing = np.flatnonzero(a[row, col + 1:] != background)
                if not len(source) and not len(drawing):
                    continue
                if (not len(source) or not len(drawing) or drawing[0] != 0
                        or source[-1] != col - 1
                        or not np.array_equal(source, np.arange(source[0], col))):
                    valid = False
                    break
                blocks, cursor = [], col - 1
                while cursor >= int(source[0]):
                    color = int(a[row, cursor])
                    end = cursor
                    while cursor >= int(source[0]) and a[row, cursor] == color:
                        cursor -= 1
                    blocks.append((end - cursor, color))
                if any(length < 2 for length, _ in blocks):
                    valid = False
                    break
                # Verify every cell through the last observed pulse. Internal
                # omissions, wrong collision colors and shifted phases fail.
                for t in range(int(drawing[-1]) + 1):
                    firing = [color for period, color in blocks if t % period == 0]
                    expected = firing[0] if firing else background
                    if a[row, col + 1 + t] != expected:
                        valid = False
                        break
                if not valid:
                    break
                lanes.append({"row": row, "blocks": tuple(blocks),
                              "count": int(len(drawing)), "end": int(drawing[-1]) + 1})
            if not valid or len(lanes) < 3 or len(lanes) % 2 == 0:
                continue
            if any(b["row"] - a_lane["row"] < 2 for a_lane, b in zip(lanes, lanes[1:])):
                continue
            target = lanes[len(lanes) // 2]["count"]
            if any(len(_events([p for p, _ in lane["blocks"]], width - col - 1)) < target
                   for lane in lanes):
                continue
            candidates.append({"canonical": a, "rotation": rotation,
                               "background": background, "separator": separator,
                               "col": col, "lanes": lanes, "target": target})
    if len(candidates) != 1:
        raise ValueError("input needs one unambiguous comb interpretation")
    return candidates[0]


def _predict(info, target=None, priority="near", dense=False, endpoint=False,
             multiplicity=False):
    """Cell-wise input-only implementation, with explicit rival-rule probes."""
    a, col, bg = info["canonical"], info["col"], info["background"]
    target = info["target"] if target is None else int(target)
    out = a.copy()
    center_end = info["lanes"][len(info["lanes"]) // 2]["end"]
    for lane in info["lanes"]:
        row, blocks = lane["row"], lane["blocks"]
        out[row, col + 1:] = bg
        used = 0
        for t in range(a.shape[1] - col - 1):
            if endpoint and t >= center_end:
                break
            firing = [(p, color) for p, color in blocks if t % p == 0]
            if dense:
                firing = [blocks[0]]
            if not firing:
                continue
            if not endpoint and used >= target:
                break
            winner = (min(firing, key=lambda x: x[0]) if priority == "short"
                      else max(firing, key=lambda x: x[0]) if priority == "long"
                      else firing[-1] if priority == "far" else firing[0])
            out[row, col + 1 + t] = winner[1]
            used += len(firing) if multiplicity else 1
    return np.rot90(out, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire output from visible input geometry and colors."""
    return _predict(_parse(input_grid))


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _paint_stream(grid, row, col, blocks, count, background):
    # Construction uses far-to-near slice overwrites, then counts occupied
    # cells. It does not call solve, _parse, or _predict.
    stream = np.full(grid.shape[1] - col - 1, background, dtype=np.int8)
    for period, color in reversed(blocks):
        stream[::period] = color
    occupied = np.flatnonzero(stream != background)
    if len(occupied) < count:
        raise AssertionError("insufficient pulse capacity")
    stream[occupied[count - 1] + 1:] = background
    grid[row, col + 1:] = stream


def _rows(rng, count, witness):
    gaps = [2] * (count - 1)
    if witness:
        # Unequal outer gap separates ordinal and geometric middle.
        gaps[0 if rng.integers(2) else -1] += int(rng.choice((5, 7)))
    else:
        budget = int(rng.integers(0, min(9, 27 - sum(gaps)) + 1))
        for _ in range(budget):
            gaps[int(rng.integers(len(gaps)))] += 1
    span = sum(gaps)
    height = int(rng.integers(span + 3, min(30, span + 7) + 1))
    top = int(rng.integers(1, height - span - 1))
    return height, [top] + [top + int(v) for v in np.cumsum(gaps)]


def _make_pair(rng, count, target, palette, rotation, witness=False, collision=None):
    bg, sep, color_a, color_b = palette
    middle = count // 2
    others = [i for i in range(count) if i != middle]
    rng.shuffle(others)
    if witness:
        incoming = [max(1, target - int(rng.integers(1, 3)))] * count
        incoming[middle] = target
        incoming[others[0]] = target + 1
    else:
        incoming = [int(rng.integers(1, target + 3)) for _ in range(count)]
        incoming[middle] = target
        incoming[others[0]] = max(1, target - 1)
        incoming[others[1]] = target + 1
    specs = [(2,)] * count
    colors = [None] * count
    locked = set()
    if witness:
        a, b = collision
        for index, ls, cs in ((others[0], (a, b), (color_a, color_b)),
                               (others[1], (b, a), (color_b, color_a))):
            specs[index], colors[index] = ls, cs
            locked.add(index)
        specs[middle] = (3,)
        locked.add(middle)
        # One visible beat cannot reveal its future cadence. This lane makes
        # the source ruler necessary, alongside the period-3 reference lane.
        probe = others[2]
        specs[probe], incoming[probe] = (2,), 1
        locked.add(probe)
    # Grow compositions while respecting the *combined* canvas budget.
    for index in rng.permutation(count):
        index = int(index)
        if index in locked:
            continue
        choices = []
        for ls in _OPTIONS:
            trial = specs.copy()
            trial[index] = ls
            source = max(map(sum, trial))
            draw = max(_extent(ps, max(n, target)) for ps, n in zip(trial, incoming))
            if source + 1 + draw <= 30:
                choices.append(ls)
        desired = int(rng.choice((1, 2, 3), p=(.45, .40, .15)))
        preferred = [ls for ls in choices if len(ls) == desired] or choices
        specs[index] = preferred[int(rng.integers(len(preferred)))]
    source = max(map(sum, specs))
    draw = max(_extent(ls, max(n, target)) for ls, n in zip(specs, incoming))
    slack = 29 - source - draw
    assert slack >= 0
    extra_source = int(rng.integers(min(4, slack) + 1))
    source += extra_source
    slack -= extra_source
    draw += int(rng.integers(slack + 1))
    height, rows = _rows(rng, count, witness)
    inp = np.full((height, source + 1 + draw), bg, dtype=np.int8)
    inp[:, source] = sep
    payload = [c for c in range(10) if c not in (bg, sep)]
    size = int(rng.integers(2, min(6, len(payload)) + 1))
    pool = [int(c) for c in rng.choice(payload, size=size, replace=False)]
    for index, (row, ls) in enumerate(zip(rows, specs)):
        if colors[index] is None:
            cs = []
            for _ in ls:
                cs.append(int(rng.choice([c for c in pool if not cs or c != cs[-1]])))
            colors[index] = tuple(cs)
        cursor = source
        for length, color in zip(ls, colors[index]):
            inp[row, cursor - length:cursor] = color
            cursor -= length
    out = inp.copy()
    for row, ls, cs, n in zip(rows, specs, colors, incoming):
        blocks = tuple(zip(ls, cs))
        _paint_stream(inp, row, source, blocks, n, bg)
        _paint_stream(out, row, source, blocks, target, bg)
    return {"input": np.rot90(inp, rotation).copy(),
            "output": np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice(10, size=4, replace=False))
    base = int(rng.integers(4))
    shift = str(rng.choice(("lanes", "beats", "turn")))
    collision = ((2, 3), (2, 5), (3, 4))[int(rng.integers(3))]
    lcm = int(np.lcm(*collision))
    needed = _events(collision).index(lcm) + 1
    first_target = needed + int(rng.integers(0, 2)) if needed < 7 else needed
    train_targets = [first_target, int(rng.choice([n for n in range(2, 7) if n != first_target]))]
    train_targets += [int(rng.integers(2, 8)) for _ in range(num_train - 2)]
    train_counts = [5 if shift == "lanes" else int(rng.choice((5, 7))), 3]
    train_counts += [int(rng.choice((3, 5) if shift == "lanes" else (3, 5, 7)))
                     for _ in range(num_train - 2)]
    seen, puzzle = set(), {"train": [], "test": []}
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            witness = split == "train" and index == 0
            if split == "train":
                target, count = train_targets[index], train_counts[index]
                rotation = base if shift == "turn" else (base + index % 2) % 4
            else:
                target = (max(train_targets) + int(rng.integers(1, 3)) if shift == "beats"
                          else int(rng.integers(2, 8)))
                count = int(rng.choice((7, 9) if shift == "lanes" else (3, 5, 7, 9)))
                rotation = ((base + int(rng.choice((1, 2, 3)))) % 4 if shift == "turn"
                            else int(rng.integers(4)))
            for _ in range(32):
                pair = _make_pair(rng, count, target, palette, rotation, witness, collision)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input attempts exhausted")
    # Witness location is not a split-position convention.
    rng.shuffle(puzzle["train"])
    return puzzle


def _selection_targets(info):
    lanes = info["lanes"]
    counts = [lane["count"] for lane in lanes]
    rows = [lane["row"] for lane in lanes]
    distance = [abs(2 * row - rows[0] - rows[-1]) for row in rows]
    geometric = counts[distance.index(min(distance))]
    return {"median": sorted(counts)[len(counts) // 2], "first": counts[0],
            "last": counts[-1], "minimum": min(counts), "maximum": max(counts),
            "geometric": geometric}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            canonical = np.rot90(out, info["rotation"])
            a, bg, col = info["canonical"], info["background"], info["col"]
            assert np.array_equal(canonical[:, :col + 1], a[:, :col + 1]), "source/separator changed"
            active = {lane["row"] for lane in info["lanes"]}
            for row in range(a.shape[0]):
                if row not in active:
                    assert np.array_equal(canonical[row], a[row]), "inactive lane changed"
            ends = []
            for lane in info["lanes"]:
                observed = np.flatnonzero(canonical[lane["row"], col + 1:] != bg)
                assert len(observed) == info["target"], "wrong visible beat count"
                events = _events([p for p, _ in lane["blocks"]], a.shape[1] - col - 1)
                assert observed.tolist() == events[:info["target"]], "wrong period, phase, or stop"
                for t in observed:
                    color = next(c for p, c in lane["blocks"] if int(t) % p == 0)
                    assert canonical[lane["row"], col + 1 + t] == color, "wrong collision winner"
                ends.append(int(observed[-1]) + 1)
            middle_row = info["lanes"][len(info["lanes"]) // 2]["row"]
            assert np.array_equal(canonical[middle_row], a[middle_row]), "reference changed"
            assert np.array_equal(out, solve(inp))
            assert not np.array_equal(inp, out), "no correction witness"
            if split == "train":
                train.append((pair, info, ends))
    # These are teaching checks on returned labels, not only sampler metadata.
    assert len({info["target"] for _, info, _ in train}) >= 2
    assert len({len(info["lanes"]) for _, info, _ in train}) >= 2
    assert any(min(lane["count"] for lane in info["lanes"]) < info["target"]
               < max(lane["count"] for lane in info["lanes"]) for _, info, _ in train)
    assert any(len(set(ends)) > 1 for _, _, ends in train), "need unequal equal-count extents"
    assert any(info["target"] >= 5 and any(lane["count"] == 1 and lane["blocks"][0][0] == 2
               and len(lane["blocks"]) == 1 for lane in info["lanes"])
               for _, info, _ in train), "need a source-dependent one-beat prefix"
    for selection in ("median", "first", "last", "minimum", "maximum", "geometric"):
        assert any(_selection_targets(info)[selection] != info["target"] for _, info, _ in train), selection
    for priority in ("short", "long", "far"):
        assert any(not np.array_equal(pair["output"], _predict(info, priority=priority))
                   for pair, info, _ in train), "priority shortcut: " + priority
    for kwargs in ({"dense": True}, {"endpoint": True}, {"multiplicity": True}):
        assert any(not np.array_equal(pair["output"], _predict(info, **kwargs))
                   for pair, info, _ in train), "beat-space shortcut"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """Bounded, streaming 42-seed campaign; no artifacts or retained grids."""
    signatures, shapes, layouts, directions = set(), set(), set(), set()
    references, lane_counts, periods, run_counts, backgrounds, palettes = (set() for _ in range(6))
    repeated, equal, unequal, max_pair_bytes = False, False, False, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                directions.add(info["rotation"])
                references.add(info["target"])
                backgrounds.add(info["background"])
                lane_counts.add(len(info["lanes"]))
                palettes.add((info["background"], info["separator"]))
                max_pair_bytes = max(max_pair_bytes, sum(a.nbytes for a in pair.values()))
                gaps = tuple(b["row"] - a["row"] for a, b in zip(info["lanes"], info["lanes"][1:]))
                equal |= len(set(gaps)) == 1
                unequal |= len(set(gaps)) > 1
                lane_layout = tuple((tuple(p for p, _ in lane["blocks"]), lane["count"])
                                    for lane in info["lanes"])
                layouts.add((gaps, lane_layout, info["target"]))
                for lane in info["lanes"]:
                    run_counts.add(len(lane["blocks"]))
                    periods.update(p for p, _ in lane["blocks"])
                    cs = [c for _, c in lane["blocks"]]
                    repeated |= len(set(cs)) < len(cs)
        # Palette and dihedral transformations must commute with the rule.
        pair = puzzle["test"][0]
        mapping = np.random.default_rng(seed + 991).permutation(10)
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
        for turns in range(4):
            for mirror in (False, True):
                a, b = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                if mirror:
                    a, b = np.fliplr(a), np.fliplr(b)
                assert np.array_equal(solve(a), b)
    assert len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 180
    assert directions == {0, 1, 2, 3} and lane_counts == {3, 5, 7, 9}
    assert run_counts == {1, 2, 3} and set(range(2, 7)) <= periods
    assert len(references) >= 6 and len(backgrounds) >= 8 and len(palettes) >= 25
    assert repeated and equal and unequal
    # Hold an incoming one-beat stream fixed while changing only its ruler.
    # Future pulse positions must respond to that visible instruction change.
    causal_checked = False
    for pair in generate_puzzle(0)["train"]:
        info = _parse(pair["input"])
        for lane in info["lanes"]:
            if (info["target"] >= 5 and lane["count"] == 1
                    and len(lane["blocks"]) == 1 and lane["blocks"][0][0] == 2):
                altered = info["canonical"].copy()
                row, col = lane["row"], info["col"]
                altered[row, :col] = info["background"]
                altered[row, col - 3:col] = lane["blocks"][0][1]
                changed = np.rot90(solve(np.rot90(altered, -info["rotation"])), info["rotation"])
                expected = np.rot90(pair["output"], info["rotation"])
                assert not np.array_equal(changed[row, col + 1:], expected[row, col + 1:])
                causal_checked = True
                break
        if causal_checked:
            break
    assert causal_checked
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1801, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Fixed, manually calculated pulse arithmetic: near 2/blue, far 3/red.
    # Five visible beats occur at 0,2,3,4,6: collision counts as one beat.
    a = np.zeros((7, 17), dtype=np.int8)
    a[:, 5] = 8
    a[1, :5] = [2, 2, 2, 1, 1]
    a[3, 3:5] = 3
    a[5, :5] = [1, 1, 2, 2, 2]
    a[1, [6, 8, 9]] = [1, 1, 2]
    a[3, [6, 8, 10, 12, 14]] = 3
    a[5, [6, 8, 9, 10, 12, 14]] = [2, 1, 2, 1, 2, 1]
    expected = a.copy()
    expected[1, 6:] = [1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0]
    expected[5, 6:] = [2, 0, 1, 2, 1, 0, 2, 0, 0, 0, 0]
    assert np.array_equal(solve(a), expected)
    # Changing only the visible middle count changes all target counts.
    altered = a.copy()
    altered[3, 14] = 0
    changed = solve(altered)
    assert not np.array_equal(changed[1], expected[1])
    assert np.count_nonzero(changed[1, 6:]) == 4
    # Detect a damaged preserved cell and a damaged pulse, without file writes.
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "layouts": len(layouts),
            "targets": sorted(references), "lane_counts": sorted(lane_counts),
            "periods": sorted(periods), "max_pair_array_bytes": max_pair_bytes,
            "independent_solver": False}
