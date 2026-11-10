"""Hidden-edge prefix triangles. Standard library and NumPy only.

The renderer uses scene coordinates; solve reconstructs them from input pixels.
No parent module, saved sample, seed, or generation metadata enters solve.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


CHECK_SEEDS = (0, 357192180, 489529113, 960156111, 435452339, 715143578,
               679596584, 412134032, 988334812, 167981236, 808674897,
               183438224, 562473881, 984272505, 388039591, 43182225,
               829720721, 97767881, 67682035, 887527994, 490943479,
               237038501, 108899676, 335605251, 484688539, 494958361,
               528868619, 389330262, 420523924, 706081089, 170409144,
               671020315, 926260786, 544142664, 706726938, 830770123,
               672698504, 659391785, 340942493, 542114593, 569702086,
               193741818)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _need(isinstance(value, (int, np.integer))
          and not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    return a, b


def _grid(value):
    array = np.asarray(value)
    _need(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
          "grid dimensions must be 1..30")
    _need(np.issubdtype(array.dtype, np.integer)
          and np.all((0 <= array) & (array <= 9)), "integer colors must be 0..9")
    return array.astype(np.int8, copy=True)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _inside(point, box):
    y, x = point
    t, b, l, r = box
    return t <= y <= b and l <= x <= r


def _paint_outline(array, box, rail, word):
    t, b, l, r = box
    array[t:b + 1, l] = rail
    array[t:b + 1, r] = rail
    array[t, l + 1:r] = word
    array[b, l + 1:r] = word


def _decode(input_grid):
    """Read the frame and every split pair of vertical rails from pixels alone."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _need(all(int(grid[y, x]) == bg for y, x in
              ((0, 0), (0, -1), (-1, 0), (-1, -1))), "corners must show background")
    locations = {int(c): {tuple(map(int, p)) for p in np.argwhere(grid == c)}
                 for c in np.unique(grid) if int(c) != bg}
    frames = []
    for color, points in locations.items():
        box = t, b, l, r = _bbox(points)
        if b - t < 4 or r - l < 4 or points != _perimeter(box):
            continue
        interior = grid[t + 1:b, l + 1:r]
        if np.all(interior == interior[0, 0]) and int(interior[0, 0]) != color:
            frames.append((box, color, int(interior[0, 0])))
    _need(len(frames) == 1, "expected one complete opaque frame")
    main, frame, fill = frames[0]
    mt, mb, ml, mr = main
    _need(0 < mt < mb < grid.shape[0] - 1 and 0 < ml < mr < grid.shape[1] - 1,
          "frame requires exterior evidence")
    outlines = []
    for color, points in locations.items():
        if color in (frame, fill):
            continue
        box = t, b, l, r = _bbox(points)
        if b - t < 2 or r - l < 4:
            continue
        sides = {(y, x) for y in range(t, b + 1) for x in (l, r)}
        visible_sides = {p for p in sides if not _inside(p, main)}
        if points != visible_sides:
            continue
        hidden_rows = [y for y in (t, b) if mt < y < mb]
        outer_rows = [y for y in (t, b) if y < mt or y > mb]
        if len(hidden_rows) != 1 or len(outer_rows) != 1:
            continue
        if not (l < ml or r > mr):
            continue
        start, stop = max(l + 1, ml + 1), min(r, mr)
        if stop - start < 3:
            continue
        word = grid[outer_rows[0], l + 1:r].copy()
        outlines.append({"box": box, "rail": color, "word": word,
                         "row": hidden_rows[0], "start": start, "stop": stop})
    _need(1 <= len(outlines) <= 3, "expected one to three recoverable rectangles")
    rails = {o["rail"] for o in outlines}
    signals = tuple(sorted(set(locations) - {frame, fill} - rails))
    _need(len(signals) == 2 and bg not in signals, "two visible signal roles required")
    under = np.full(grid.shape, bg, dtype=np.int8)
    occupied = set()
    for outline in outlines:
        full = _perimeter(outline["box"])
        _need(not (occupied & full), "rectangular perimeters may not intersect")
        occupied.update(full)
        word = outline["word"]
        _need(set(map(int, word)) <= {bg, *signals}, "horizontal edge contains a rail color")
        _paint_outline(under, outline["box"], outline["rail"], word)
    rendered = under.copy()
    rendered[mt:mb + 1, ml:mr + 1] = frame
    rendered[mt + 1:mb, ml + 1:mr] = fill
    _need(np.array_equal(rendered, grid), "visible paired edges or rail geometry disagree")
    outlines.sort(key=lambda o: o["row"])
    _need(len({o["row"] for o in outlines}) == len(outlines), "hidden-side row tie")
    active = outlines[0]
    row, start, stop = active["row"], active["start"], active["stop"]
    seed = under[row, start:stop].copy()
    _need(len(seed) >= 3 and len(seed) % 2 == 1, "active side span must be odd")
    _need(int(seed[0]) in signals and int(seed[-1]) in signals, "seed endpoints must be signal")
    _need(row + len(seed) // 2 < mb, "triangle tip exceeds window")
    base = under[mt + 1:mb, ml + 1:mr].copy()
    base[base == bg] = fill
    return base, {"grid": grid, "background": bg, "fill": fill, "frame": frame,
                  "main": main, "signals": signals, "outlines": outlines,
                  "row": row - mt - 1, "start": start - ml - 1,
                  "sequence": base[row - mt - 1, start - ml - 1:stop - ml - 1].copy()}


def _swap(values, signals):
    a, b = signals
    return np.where(values == a, b, np.where(values == b, a, values)).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore all outlines, crop the window, then grow the selected prefix triangle."""
    output, scene = _decode(input_grid)
    current = scene["sequence"].copy()
    row, col = scene["row"], scene["start"]
    while len(current) > 1:
        current = _swap(current[:-2], scene["signals"])
        row, col = row + 1, col + 1
        output[row, col:col + len(current)] = current
    return output


def _direct_triangle(base, sequence, row, start, signals):
    """Closed-form renderer/check: independent of solve's recursive row state."""
    output = base.copy()
    for depth in range((len(sequence) + 1) // 2):
        prefix = sequence[:len(sequence) - 2 * depth]
        values = _swap(prefix, signals) if depth % 2 else prefix
        output[row + depth, start + depth:start + depth + len(prefix)] = values
    return output


def _sequence(rng, length, style):
    """Return role codes 0=gap, 1/2=the two phases."""
    phase = int(rng.integers(1, 3))
    if style == "solid":
        seq = np.full(length, phase, dtype=np.int8)
    elif style == "alternating":
        seq = np.array([1 + (i + phase) % 2 for i in range(length)], dtype=np.int8)
    elif style == "runs":
        seq = np.empty(length, dtype=np.int8)
        at = 0
        while at < length:
            run = int(rng.integers(1, 5))
            seq[at:at + run] = phase
            phase = 3 - phase
            at += run
    elif style == "periodic":
        period = int(rng.integers(2, 6))
        word = rng.integers(0, 3, size=period, dtype=np.int8)
        seq = np.resize(word, length).copy()
    else:
        prob = (0.48, 0.26, 0.26) if style == "sparse" else (0.25, 0.375, 0.375)
        seq = rng.choice(np.array([0, 1, 2], dtype=np.int8), size=length, p=prob)
    seq[0], seq[-1] = phase, int(rng.integers(1, 3))
    if style == "solid":
        seq[-1] = phase
    if style == "diagnostic":
        seq[1], seq[2] = 0, 3 - phase
    return seq.astype(np.int8)


def _make_pair(rng, length, style, layout, approach, onset, matching_fill):
    """Construct in crop-relative coordinates, translate once, render one scene."""
    count = 1 if layout == "single" else 2 if layout == "straddle" else int(layout[-1])
    depth = length // 2
    oh = max(8, onset + depth + 1 + int(rng.integers(0, 4)), onset + 2 * (count - 1) + 1)
    if layout == "straddle":
        ow = length + int(rng.choice([2, 4]))
        boxes = [(-int(rng.integers(2, 5)), onset, -int(rng.integers(2, 5)), length),
                 (onset + 2, oh + int(rng.integers(1, 4)), -3, ow + 2)]
    elif layout.startswith("north"):
        width_floor = length + 2 * (count - 1)
        ow = width_floor + int(rng.integers(1, min(3, 26 - count - width_floor) + 1))
        boxes = [(-2 - j, onset + 2 * j, -2 - j, length + 2 * j)
                 for j in range(count)]
    elif layout.startswith("south"):
        ow = length + int(rng.integers(1, 6))
        boxes = [(onset + 2 * j, oh + count + 1 - j, -count - 1 + j, length - 2 * j)
                 for j in range(count)]
    else:
        ow = length if approach == "bridge" else length + int(rng.integers(1, 6))
        l, r = (-3, ow + 2) if approach == "bridge" else (-int(rng.integers(2, 5)), length)
        if rng.integers(2):
            boxes = [(-int(rng.integers(2, 5)), onset, l, r)]
        else:
            boxes = [(onset, oh + int(rng.integers(1, 4)), l, r)]
    if approach == "right":
        boxes = [(t, b, ow - 1 - r, ow - 1 - l) for t, b, l, r in boxes]
    palette = [int(c) for c in rng.permutation(10)]
    bg, frame = palette[:2]
    fill = bg if matching_fill else palette[2]
    signals, rails = tuple(palette[3:5]), palette[5:5 + count]
    sequence = _sequence(rng, length, style)
    words = []
    intervals = []
    for j, (t, b, l, r) in enumerate(boxes):
        lo, hi = max(0, l + 1), min(ow, r)
        codes = rng.integers(0, 3, size=r - l - 1, dtype=np.int8)
        hidden_codes = sequence if j == 0 else _sequence(rng, hi - lo, str(rng.choice(
            ["irregular", "periodic", "runs", "sparse", "solid"])))
        codes[lo - l - 1:hi - l - 1] = hidden_codes
        exterior = [c for c in range(l + 1, r) if not 0 <= c < ow]
        # Even a solid cropped seed has its complementary phase visibly encoded.
        codes[exterior[0] - l - 1] = 3 - int(hidden_codes[0])
        words.append(np.array([bg, *signals], dtype=np.int8)[codes])
        intervals.append((lo, hi))
    start = intervals[0][0]
    if style == "diagnostic" and layout == "straddle":
        # The retained gap at seed index 1 erases a nonactive edge at depth 2.
        passive_l = boxes[1][2]
        words[1][start + 3 - passive_l - 1] = signals[int(rng.integers(2))]
    main = (-1, oh, -1, ow)
    min_y = min(-1, *(b[0] for b in boxes))
    max_y = max(oh, *(b[1] for b in boxes))
    min_x = min(-1, *(b[2] for b in boxes))
    max_x = max(ow, *(b[3] for b in boxes))
    span_h, span_w = max_y - min_y + 3, max_x - min_x + 3
    _need(span_h <= 30 and span_w <= 30, "sampled scene exceeds ARC canvas")
    extra_h, extra_w = int(rng.integers(0, min(2, 30 - span_h) + 1)), int(rng.integers(0, min(2, 30 - span_w) + 1))
    top_pad, left_pad = int(rng.integers(0, extra_h + 1)), int(rng.integers(0, extra_w + 1))
    dy, dx = 1 - min_y + top_pad, 1 - min_x + left_pad
    shifted = [(t + dy, b + dy, l + dx, r + dx) for t, b, l, r in boxes]
    mt, mb, ml, mr = (main[0] + dy, main[1] + dy, main[2] + dx, main[3] + dx)
    under = np.full((span_h + extra_h, span_w + extra_w), bg, dtype=np.int8)
    for box, rail, word in zip(shifted, rails, words):
        _paint_outline(under, box, rail, word)
    base = under[mt + 1:mb, ml + 1:mr].copy()
    base[base == bg] = fill
    seed = np.array([fill, *signals], dtype=np.int8)[sequence]
    output = _direct_triangle(base, seed, onset, start, signals)
    inp = under.copy()
    inp[mt:mb + 1, ml:mr + 1] = frame
    inp[mt + 1:mb, ml + 1:mr] = fill
    return {"input": inp, "output": output}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _motif(scene):
    a, b = scene["signals"]
    seq = tuple(0 if int(v) == scene["fill"] else 1 if int(v) == a else 2
                for v in scene["sequence"])
    # Quotient out exchanging phase names, as well as literal palette changes.
    swapped = tuple(0 if v == 0 else 3 - v for v in seq)
    return min(seq, swapped)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed") % (1 << 64)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    diagnostic_length = int(rng.choice([7, 9, 11, 13]))
    onset = int(rng.integers(0, 3))
    approach = str(rng.choice(["left", "right"]))
    jobs = [(diagnostic_length, "diagnostic", "straddle", approach, onset, False),
            (diagnostic_length - 2, "solid", "south1", "right" if approach == "left" else "left",
             onset + 1 + int(rng.integers(2)), True)]
    styles = ["solid", "alternating", "runs", "irregular", "sparse", "periodic"]
    layouts = ["single", "straddle", "north2", "north3", "south2", "south3"]
    for _ in range(num_train - 2):
        extra_length = int(rng.choice(np.arange(5, diagnostic_length + 1, 2)))
        compatible_layouts = [x for x in layouts if extra_length >= 7 or not x.endswith("3")]
        jobs.append((extra_length,
                     str(rng.choice(styles[1:])), str(rng.choice(compatible_layouts)),
                     str(rng.choice(["left", "right"])), int(rng.integers(0, 5)),
                     bool(rng.integers(2))))
    rng.shuffle(jobs)
    test_styles = list(rng.permutation(styles))
    for j in range(num_test):
        length = diagnostic_length + 2 * int(rng.integers(1, min(2, (17 - diagnostic_length) // 2) + 1))
        layout = str(rng.choice(layouts))
        side = str(rng.choice(["left", "right", "bridge"])) if layout == "single" else str(rng.choice(["left", "right"]))
        jobs.append((length, str(test_styles[j]), layout, side, int(rng.integers(0, 6)), bool(rng.integers(2))))
    puzzle = {"train": [], "test": []}
    seen, motifs = set(), set()
    for i, job in enumerate(jobs):
        split = "train" if i < num_train else "test"
        for _ in range(80):
            pair = _make_pair(rng, *job)
            _, scene = _decode(pair["input"])
            key, motif = _key(pair["input"]), _motif(scene)
            reserved_solid = (split == "train" and job[1] != "solid"
                              and len(scene["sequence"]) == diagnostic_length - 2
                              and len(set(map(int, scene["sequence"]))) == 1)
            if key not in seen and motif not in motifs and not reserved_solid:
                seen.add(key)
                motifs.add(motif)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded uniqueness attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, output = _grid(pair["input"]), _grid(pair["output"])
    base, scene = _decode(inp)
    seq, row, start = scene["sequence"], scene["row"], scene["start"]
    _need(output.shape == base.shape, "output must be the strict window interior")
    expected = _direct_triangle(base, seq, row, start, scene["signals"])
    _need(np.array_equal(output, expected), "prefix triangle or reconstruction is incorrect")
    _need(np.array_equal(solve(inp), output), "input-only solve disagrees")
    triangle = np.zeros(base.shape, dtype=bool)
    for depth in range((len(seq) + 1) // 2):
        triangle[row + depth, start + depth:start + len(seq) - depth] = True
    _need(np.array_equal(output[~triangle], base[~triangle]), "outside triangle not preserved")
    _need(not np.array_equal(base, output), "omitting B did not change this answer")
    _need(int(output[row, start]) != scene["fill"], "omitting A did not change this answer")
    _need(int(output[row + len(seq) // 2, start + len(seq) // 2])
          == int(_swap(seq[:1], scene["signals"])[0] if len(seq) // 2 % 2 else seq[0]),
          "one-cell tip incorrect")
    scene["base"], scene["output"], scene["triangle"] = base, output, triangle
    return scene


def _diagnostic(scene):
    seq, fill = scene["sequence"], scene["fill"]
    if len(seq) < 7 or seq[1] != fill or np.array_equal(seq[:-2], seq[1:-1]):
        return False
    row, start = scene["row"], scene["start"]
    if not (scene["base"][row + 2, start + 3] in scene["signals"]
            and scene["output"][row + 2, start + 3] == fill):
        return False
    # This must also distinguish choosing the longer, lower reconstructed side.
    main = scene["main"]
    lower = scene["outlines"][1:]
    return any(o["stop"] - o["start"] > len(seq) and o["row"] - main[0] - 1 > row
               for o in lower)


def validate_generated_puzzle(puzzle):
    """Check the visible family contract, both ablations, and training evidence."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, motifs = set(), set()
    scenes = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            scene = _inspect_pair(pair)
            key, motif = _key(pair["input"]), _motif(scene)
            _need(key not in seen and motif not in motifs, "duplicate input or phase-normalized motif")
            seen.add(key)
            motifs.add(motif)
            scenes[split].append(scene)
    train = scenes["train"]
    _need(any(_diagnostic(s) for s in train), "missing retained-gap/collision/prefix diagnostic")
    _need(any(len(set(map(int, s["sequence"]))) == 1 for s in train), "missing solid phase-swap demo")
    _need(len({s["row"] for s in train}) >= 2, "hidden onset must vary in training")
    _need({s["fill"] == s["background"] for s in train} == {False, True}, "fill conventions not exposed")
    max_train = max(len(s["sequence"]) for s in train)
    _need(all(len(s["sequence"]) > max_train for s in scenes["test"]), "missing length extrapolation")


def _digest_puzzle(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded: 42 fixed seeds, every legal count pair once, and mutation checks.

    No files or execution traces are written. Only small hashes/features persist
    between puzzles; at most two generated puzzles are retained at once.
    """
    signatures, shapes, output_shapes, motifs, palettes, layouts = set(), set(), set(), set(), set(), set()
    lengths, onsets, fills, counts = set(), set(), set(), set()
    approaches, source_sides = set(), set()
    pairs = gap_tests = multi_tests = 0
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest_puzzle(puzzle)
        _need(signature == _digest_puzzle(generate_puzzle(seed)), "seed did not reproduce puzzle")
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _inspect_pair(pair)
                pairs += 1
                shapes.add(tuple(np.asarray(pair["input"]).shape))
                output_shapes.add(tuple(np.asarray(pair["output"]).shape))
                motifs.add(_motif(s))
                palettes.add((s["background"], s["frame"], s["fill"], s["signals"]))
                lengths.add(len(s["sequence"]))
                onsets.add(s["row"])
                fills.add(s["fill"] == s["background"])
                counts.add(len(s["outlines"]))
                t, b, l, r = s["outlines"][0]["box"]
                mt, mb, ml, mr = s["main"]
                source_sides.add("bottom" if t < mt else "top")
                approaches.add("bridge" if l < ml and r > mr else "left" if l < ml else "right")
                layouts.add((s["main"], tuple(o["box"] for o in s["outlines"])))
                if split == "test":
                    gap_tests += s["fill"] in s["sequence"][1:-2]
                    multi_tests += len(s["outlines"]) > 1
    _need(len(signatures) == 42 and len(shapes) >= 40 and len(output_shapes) >= 40,
          "insufficient scene/shape variation")
    _need(len(motifs) >= 100 and len(palettes) >= 100 and len(layouts) >= 150,
          "insufficient motif/palette/geometry variation")
    _need(counts == {1, 2, 3} and fills == {False, True} and len(lengths) >= 7,
          "missing structural range")
    _need(approaches == {"left", "right", "bridge"} and source_sides == {"top", "bottom"},
          "missing approach or hidden-side variation")
    _need(gap_tests >= 10 and multi_tests >= 20 and len(onsets) >= 6, "test variation too narrow")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(123456 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupted = generate_puzzle(0, 2, 1)
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": len(CHECK_SEEDS), "default_pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "normalized_motifs": len(motifs), "palettes": len(palettes), "layouts": len(layouts),
            "seed_lengths": sorted(lengths), "onsets": sorted(onsets), "outline_counts": sorted(counts),
            "approaches": sorted(approaches), "hidden_sides": sorted(source_sides),
            "retained_gap_tests": int(gap_tests), "multiple_outline_tests": int(multi_tests),
            "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
