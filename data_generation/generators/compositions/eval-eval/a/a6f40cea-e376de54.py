"""Occluded ribbon circuits: equalize path lengths to the middle nested track.

Only NumPy and the standard library are used. Scene descriptors are used to
draw new examples; the reader reconstructs everything from the supplied pixels.
"""

from collections import Counter
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _inside(p, box, strict=False):
    y, x = p
    t, b, l, r = box
    return t < y < b and l < x < r if strict else t <= y <= b and l <= x <= r


def _bounds(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _contains(outer, inner):
    t, b, l, r = outer
    a, z, c, d = inner
    return t < a < z < b and l < c < d < r


def _walk(box, start=None, clockwise=True):
    """Clockwise cell order, counting each corner exactly once."""
    t, b, l, r = box
    path = ([(t, c) for c in range(l, r + 1)]
            + [(y, r) for y in range(t + 1, b + 1)]
            + [(b, c) for c in range(r - 1, l - 1, -1)]
            + [(y, l) for y in range(b - 1, t, -1)])
    if start is None:
        start = (t, l)
    k = path.index(start)
    path = path[k:] + path[:k]
    return path if clockwise else [path[0]] + path[:0:-1]


def _components(points):
    remaining, groups = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], {p}
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _recover_boxes(points, mask):
    """A rectangular screen leaves at most two pieces of each rectangular loop."""
    groups = _components(points)
    assert 1 <= len(groups) <= 10
    candidates = []
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            cells = groups[i] | groups[j]
            box = _bounds(cells)
            t, b, l, r = box
            if b - t < 2 or r - l < 2:
                continue
            full = _perimeter(box)
            if {p for p in full if not _inside(p, mask)} == cells:
                candidates.append((frozenset((i, j)), box))
    answers = []

    def search(left, boxes):
        if len(answers) > 1:
            return
        if not left:
            if len(boxes) in (3, 5):
                answers.append(tuple(sorted(boxes, key=lambda b: -len(_perimeter(b)))))
            return
        first = min(left)
        for members, box in candidates:
            if first not in members or not members <= left:
                continue
            if all(_contains(box, other) or _contains(other, box) for other in boxes):
                search(left - members, boxes + [box])

    search(set(range(len(groups))), [])
    assert len(answers) == 1, "ambiguous nested rectangular tracks"
    boxes = answers[0]
    for outer, inner in zip(boxes, boxes[1:]):
        assert _contains(outer, inner)
        assert min(inner[0] - outer[0], outer[1] - inner[1],
                   inner[2] - outer[2], outer[3] - inner[3]) >= 2
    return boxes


def _read(input_grid):
    g = _grid(input_grid)
    frames = []
    for color in map(int, np.unique(g)):
        cells = {tuple(map(int, p)) for p in np.argwhere(g == color)}
        box = _bounds(cells)
        t, b, l, r = box
        if b - t < 2 or r - l < 2 or cells != _perimeter(box):
            continue
        inner = g[t + 1:b, l + 1:r]
        fill = int(inner[0, 0])
        if fill != color and np.all(inner == fill):
            frames.append((box, color, fill))
    assert len(frames) == 1, "one complete opaque screen is required"
    mask, frame, fill = frames[0]
    outside = [(y, x) for y in range(g.shape[0]) for x in range(g.shape[1])
               if not _inside((y, x), mask)]
    frequencies = Counter(int(g[p]) for p in outside)
    most = frequencies.most_common()
    assert len(most) >= 2 and most[0][1] > most[1][1]
    bg = most[0][0]
    assert frame != bg
    points = {p for p in outside if int(g[p]) != bg}
    boxes = _recover_boxes(points, mask)
    visible = [{p for p in _perimeter(b) if not _inside(p, mask)} for b in boxes]
    assert all(any(_inside(p, mask, strict=True) for p in _perimeter(b)) for b in boxes)
    counts = [Counter(int(g[p]) for p in cells) for cells in visible]

    # The anchor is the one common singleton color at corresponding corners.
    anchor_options = []
    for corner in range(4):
        corners = [[(t, l), (t, r), (b, r), (b, l)][corner] for t, b, l, r in boxes]
        if not all(p in cells for p, cells in zip(corners, visible)):
            continue
        colors = {int(g[p]) for p in corners}
        if len(colors) == 1:
            color = next(iter(colors))
            if all(c[color] == 1 for c in counts):
                anchor_options.append((color, corners))
    assert len(anchor_options) == 1, "anchor corners must be uniquely indicated"
    anchor, starts = anchor_options[0]
    singletons = set.intersection(*[{c for c, n in count.items() if n == 1} for count in counts])
    singletons.discard(anchor)
    assert len(singletons) == 1, "one visible end marker per track is required"
    end = next(iter(singletons))
    paths_cw = [_walk(box, start) for box, start in zip(boxes, starts)]
    assert all(path[1] in cells and path[-1] in cells for path, cells in zip(paths_cw, visible))
    neighbor_colors = [{int(g[path[1]]), int(g[path[-1]])} for path in paths_cw]
    common = set.intersection(*neighbor_colors)
    assert len(common) == 1, "the common empty-track color must be identifiable"
    track = next(iter(common))
    roles = {bg, frame, track, anchor, end}
    assert len(roles) == 5 and fill not in {frame, track, anchor, end}
    if fill != bg:
        roles.add(fill)
    circuits = []
    directions = set()
    for box, start, cw, cells in zip(boxes, starts, paths_cw, visible):
        clockwise = int(g[cw[-1]]) == track and int(g[cw[1]]) != track
        counterclockwise = int(g[cw[1]]) == track and int(g[cw[-1]]) != track
        assert clockwise != counterclockwise
        directions.add(clockwise)
        path = cw if clockwise else [cw[0]] + cw[:0:-1]
        finish = [p for p in cells if int(g[p]) == end]
        assert len(finish) == 1
        length = path.index(finish[0]) + 1
        assert 3 <= length <= len(path) - 2
        assert finish[0] not in {(box[0], box[2]), (box[0], box[3]),
                                  (box[1], box[2]), (box[1], box[3])}
        evidence = [(k, int(g[p])) for k, p in enumerate(path)
                    if 0 < k < length - 1 and p in cells]
        assert all(color not in roles for k, color in evidence)
        assert all(int(g[p]) == track for k, p in enumerate(path) if k >= length and p in cells)
        explanations = []
        predictions = set()
        for period in range(1, 4):
            word, witnesses = {}, [0] * period
            for k, color in evidence:
                residue = (k - 1) % period
                if residue in word and word[residue] != color:
                    break
                word[residue] = color
                witnesses[residue] += 1
            else:
                if min(witnesses) >= 2:
                    w = tuple(word[i] for i in range(period))
                    predictions.add(tuple(w[(k - 1) % period] for k in range(1, len(path) - 1)))
                    explanations.append(w)
        assert len(predictions) == 1, "the ribbon's periodic continuation is ambiguous"
        word = min(explanations, key=len)
        circuits.append({"box": box, "path": path, "length": length, "word": word,
                         "visible": cells, "start": start})
    assert len(directions) == 1, "ribbons share one winding direction within a grid"
    target = circuits[len(circuits) // 2]["length"]
    assert target <= min(len(c["path"]) - 2 for c in circuits), "normalization must not wrap"
    assert any(c["length"] != target for c in circuits)
    return {"shape": g.shape, "mask": mask, "background": bg, "frame": frame,
            "fill": fill, "track": track, "anchor": anchor, "end": end,
            "circuits": circuits, "target": target, "clockwise": next(iter(directions))}


def _reveal(scene, lengths):
    t, b, l, r = scene["mask"]
    result = np.full((b - t - 1, r - l - 1), scene["fill"], dtype=np.int8)
    for circuit, length in zip(scene["circuits"], lengths):
        word = circuit["word"]
        for k, p in enumerate(circuit["path"]):
            if not _inside(p, scene["mask"], strict=True):
                continue
            if k == 0:
                color = scene["anchor"]
            elif k == length - 1:
                color = scene["end"]
            elif k < length - 1:
                color = word[(k - 1) % len(word)]
            else:
                color = scene["track"]
            result[p[0] - t - 1, p[1] - l - 1] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _read(input_grid)
    return _reveal(scene, [scene["target"]] * len(scene["circuits"]))


def _geometry(rng, count, kind):
    # Nonconcentric nesting: the four insets at each level are independent.
    for _ in range(100):
        ih, iw = map(int, rng.integers(6 if count == 5 else 5, 11, size=2))
        boxes = [(0, ih - 1, 0, iw - 1)]
        for _ in range(count - 1):
            t, b, l, r = boxes[-1]
            a, z, c, d = map(int, rng.choice((2, 3), 4, p=(.85, .15)))
            boxes.append((t - a, b + z, l - c, r + d))
        boxes.reverse()
        t, b, l, r = boxes[0]
        oh, ow = b - t + 1, r - l + 1
        if max(oh, ow) <= 28:
            break
    else:
        raise AssertionError("nested tracks did not fit")
    h, w = int(rng.integers(oh + 2, 31)), int(rng.integers(ow + 2, 31))
    dy = int(rng.integers(1, h - oh)) - t
    dx = int(rng.integers(1, w - ow)) - l
    boxes = [(a + dy, z + dy, c + dx, d + dx) for a, z, c, d in boxes]
    t, b, l, r = boxes[0]
    a, z, c, d = boxes[-1]
    if kind == "stripe":
        width = int(rng.integers(3, min(7, d - c - 1) + 1))
        left = int(rng.integers(c + 1, d - width + 1))
        mask = (t - 1, b + 1, left, left + width - 1)
    else:
        # The screen conceals actual turns, rather than just straight sections.
        top = int(rng.integers(a, z))
        left = int(rng.integers(c, d))
        mask = (top, b + 1, left, r + 1)
    return (h, w), boxes, mask


def _draw_input(scene):
    g = np.full(scene["shape"], scene["background"], dtype=np.int8)
    for c in scene["circuits"]:
        for k, p in enumerate(c["path"]):
            if k == 0:
                color = scene["anchor"]
            elif k == c["length"] - 1:
                color = scene["end"]
            elif k < c["length"] - 1:
                color = c["word"][(k - 1) % len(c["word"])]
            else:
                color = scene["track"]
            g[p] = color
    t, b, l, r = scene["mask"]
    g[t:b + 1, l:r + 1] = scene["fill"]
    for p in _perimeter(scene["mask"]):
        g[p] = scene["frame"]
    return g


def _features(scene):
    cs, target, mask = scene["circuits"], scene["target"], scene["mask"]
    lengths = tuple(c["length"] for c in cs)
    middle = cs[len(cs) // 2]
    hidden = lambda p: _inside(p, mask, strict=True)
    extend = any(c["length"] < target and any(hidden(p) for p in c["path"][c["length"]:target]) for c in cs)
    trim = any(c["length"] > target and any(hidden(p) for p in c["path"][target:c["length"]]) for c in cs)
    hidden_ref = sum(_inside(p, mask) for p in middle["path"][:target])
    first_edge = next(k for k, p in enumerate(middle["path"][1:], 1)
                      if p[0] != middle["start"][0] and p[1] != middle["start"][1])
    outer = cs[0]["box"]
    corner_order = [(outer[0], outer[2]), (outer[0], outer[3]),
                    (outer[1], outer[3]), (outer[1], outer[2])]
    start_corner = corner_order.index(cs[0]["start"])
    hidden_periodic_bend, phase_restart_witness = False, False
    for c in cs:
        turn_step = next(k for k, p in enumerate(c["path"][1:], 1)
                         if p[0] != c["start"][0] and p[1] != c["start"][1])
        if len(c["word"]) > 1 and any(hidden(p) for p in c["path"][turn_step:target - 1]):
            hidden_periodic_bend = True
        run_start, was_hidden = 0, False
        word = c["word"]
        for k, p in enumerate(c["path"][:target - 1]):
            covered_body = k > 0 and _inside(p, mask)
            if covered_body:
                if not was_hidden:
                    run_start = k
                if hidden(p) and word[(k - 1) % len(word)] != word[(k - run_start) % len(word)]:
                    phase_restart_witness = True
            was_hidden = covered_body
    return {"count": len(cs), "target": target, "lengths": lengths,
            "periods": tuple(len(c["word"]) for c in cs),
            "hidden_extend": extend, "hidden_trim": trim,
            "hidden_reference_cells": hidden_ref,
            "nonmedian": sorted(lengths)[len(cs) // 2] != target,
            "bent_reference": target > first_edge,
            "hidden_periodic_bend": hidden_periodic_bend,
            "phase_restart_witness": phase_restart_witness,
            "fill_matches": scene["fill"] == scene["background"],
            "boxes": tuple(c["box"] for c in cs), "mask": mask,
            "shape": scene["shape"], "clockwise": scene["clockwise"],
            "start_corner": start_corner}


def _make_pair(rng, count, kind, clockwise, background, distinct_fill,
               demo=None, avoid_target=None):
    for _ in range(700):
        shape, boxes, mask = _geometry(rng, count, kind)
        palette = [int(x) for x in rng.permutation([x for x in range(10) if x != background])]
        frame, fill_color, track, anchor, end = palette[:5]
        body_pool = palette[5:]
        paths = [_walk(box, clockwise=clockwise) for box in boxes]
        corners = [{(b[0], b[2]), (b[0], b[3]), (b[1], b[2]), (b[1], b[3])} for b in boxes]
        options = [[length for length in range(4, len(path) - 1)
                    if not _inside(path[length - 1], mask) and path[length - 1] not in co]
                   for path, co in zip(paths, corners)]
        if not all(options):
            continue
        m = count // 2
        targets = [length for length in options[m] if 8 <= length <= min(map(len, paths)) - 2
                   and length != avoid_target]
        if demo == 1:
            first_side = boxes[m][1] - boxes[m][0] + 1 if not clockwise else boxes[m][3] - boxes[m][2] + 1
            targets = [length for length in targets if length > first_side + 1]
        if not targets:
            continue
        target = int(rng.choice(targets))
        lengths = [target] * count
        others = [i for i in rng.permutation(count) if i != m]
        majority_short = bool(rng.integers(2))
        failed = False
        for rank, i in enumerate(others):
            if demo == 0:
                short = majority_short if rank else not majority_short
            elif rank < 2:
                short = rank == 0
            else:
                short = bool(rng.integers(2))
                if rng.random() < .2 and target in options[i]:
                    lengths[i] = target
                    continue
            choices = [n for n in options[i] if (n < target if short else n > target)]
            if demo == 1 and i == count - 1:
                choices = [n for n in choices if n >= 8]
            if not choices:
                failed = True
                break
            lengths[i] = int(rng.choice(choices))
        if failed:
            continue
        words = []
        for i, length in enumerate(lengths):
            period = (2 if i == m and demo == 0 else 3 if demo == 1 and i in (m, count - 1)
                      else int(rng.integers(1, 4)))
            if length - 2 < 2 * period:
                period = 1
            first = body_pool[i % len(body_pool)]
            rest = [c for c in body_pool if c != first]
            word = [first]
            if period >= 2:
                word.append(int(rng.choice(rest)))
            if period == 3:
                word.append(first if rng.random() < .5 else int(rng.choice([c for c in rest if c != word[1]])))
            words.append(tuple(word))
        scene = {"shape": shape, "mask": mask, "background": background,
                 "frame": frame, "fill": fill_color if distinct_fill else background,
                 "track": track, "anchor": anchor, "end": end, "target": target,
                 "clockwise": clockwise,
                 "circuits": [{"box": box, "path": path, "length": length, "word": word,
                               "start": path[0]} for box, path, length, word in zip(boxes, paths, lengths, words)]}
        inp = _draw_input(scene)
        out = _reveal(scene, [target] * count)
        try:
            parsed = _read(inp)
            assert np.array_equal(_reveal(parsed, [parsed["target"]] * count), out)
            f = _features(parsed)
            assert f["hidden_extend"] and f["hidden_trim"]
            assert not np.array_equal(_reveal(parsed, [c["length"] for c in parsed["circuits"]]), out)
            if demo == 0:
                assert f["nonmedian"] and f["hidden_reference_cells"] > 0
                median_length = sorted(f["lengths"])[count // 2]
                if median_length <= min(len(c["path"]) - 2 for c in parsed["circuits"]):
                    assert not np.array_equal(_reveal(parsed, [median_length] * count), out)
                visible_target = target - f["hidden_reference_cells"]
                assert visible_target >= 3
                assert not np.array_equal(_reveal(parsed, [visible_target] * count), out)
                assert 2 in f["periods"]
            if demo == 1:
                assert f["bent_reference"] and 3 in f["periods"] and f["hidden_periodic_bend"]
                assert f["phase_restart_witness"]
                ref = parsed["circuits"][count // 2]
                start, finish = ref["path"][0], ref["path"][target - 1]
                straight_span = max(abs(a - b) for a, b in zip(start, finish)) + 1
                assert not np.array_equal(_reveal(parsed, [straight_span] * count), out)
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    raise AssertionError(f"could not construct readable {kind}/{count} ribbon circuits")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed % (1 << 64))
    base_rotation = int(rng.integers(4))
    backgrounds = [int(x) for x in rng.permutation(10)]
    puzzle, seen, first_target = {"train": [], "test": []}, set(), None
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            demo = index if split == "train" and index < 2 else None
            count = 5 if demo == 0 else 3 if demo == 1 else int(rng.choice((3, 5)))
            kind = "stripe" if demo == 0 else "corner" if demo == 1 else str(rng.choice(("stripe", "corner")))
            clockwise = demo == 0 if demo is not None else bool(rng.integers(2))
            distinct_fill = demo == 1 if demo is not None else bool(rng.integers(2))
            rotation = (base_rotation + (2 * (index % 2) if split == "train" else 1 + 2 * (index % 2))) % 4
            for _ in range(10):
                pair = _make_pair(rng, count, kind, clockwise,
                                  backgrounds[len(seen) % 10], distinct_fill, demo,
                                  first_target if demo == 1 else None)
                pair = {k: np.rot90(g, rotation).copy() for k, g in pair.items()}
                fingerprint = _key(pair["input"])
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    puzzle[split].append(pair)
                    if demo == 0:
                        first_target = _read(pair["input"])["target"]
                    break
            else:
                raise AssertionError("duplicate input after bounded attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            scene = _read(inp)
            cs, length = scene["circuits"], scene["target"]
            expected = _reveal(scene, [length] * len(cs))
            assert np.array_equal(expected, out), "incorrect normalized hidden ribbons"
            assert out.shape != inp.shape
            # Restore-only is a concrete ablation of the middle-length mechanism.
            assert not np.array_equal(_reveal(scene, [c["length"] for c in cs]), out)
            assert np.count_nonzero(out != scene["fill"]) > 0
            f = _features(scene)
            assert f["hidden_extend"] and f["hidden_trim"], "both changes must affect the answer"
            features[split].append(f)
    a, b = features["train"][:2]
    assert a["count"] == 5 and b["count"] == 3
    assert a["nonmedian"] and a["hidden_reference_cells"] > 0 and 2 in a["periods"]
    assert b["bent_reference"] and 3 in b["periods"] and b["hidden_periodic_bend"]
    assert b["phase_restart_witness"]
    assert a["target"] != b["target"]
    assert a["fill_matches"] and not b["fill_matches"]
    first = _read(puzzle["train"][0]["input"])
    visible_length = first["target"] - a["hidden_reference_cells"]
    assert not np.array_equal(_reveal(first, [visible_length] * a["count"]), puzzle["train"][0]["output"])
    median_length = sorted(a["lengths"])[a["count"] // 2]
    if median_length <= min(len(c["path"]) - 2 for c in first["circuits"]):
        assert not np.array_equal(_reveal(first, [median_length] * a["count"]), puzzle["train"][0]["output"])
    second = _read(puzzle["train"][1]["input"])
    ref = second["circuits"][b["count"] // 2]
    start, finish = ref["path"][0], ref["path"][b["target"] - 1]
    straight_span = max(abs(a - z) for a, z in zip(start, finish)) + 1
    assert not np.array_equal(_reveal(second, [straight_span] * b["count"]), puzzle["train"][1]["output"])


def self_test():
    """Bounded 42-seed audit, equivariance, ablations and contract regressions."""
    signatures, shapes, outputs, layouts = set(), set(), set(), set()
    counts, targets, periods, directions, backgrounds, masks = set(), set(), set(), set(), set(), set()
    rng = np.random.default_rng(91247)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in ("train", "test")
                       for p, q in zip(puzzle[s], again[s]) for k in ("input", "output"))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _read(pair["input"])
                f = _features(scene)
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                layouts.add((f["boxes"], f["mask"]))
                counts.add(f["count"])
                targets.add(f["target"])
                periods.update(f["periods"])
                directions.add((f["start_corner"], f["clockwise"]))
                backgrounds.add(scene["background"])
                masks.add(sum(len(_components(c["visible"])) for c in scene["circuits"]))
        # Geometry/color equivariance is stronger than checking generated labels.
        pair = puzzle["test"][0]
        permutation = rng.permutation(10).astype(np.int8)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        for turns in range(4):
            for mirror in (False, True):
                inp, out = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                if mirror:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                assert np.array_equal(solve(inp), out)
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(outputs) >= 35 and len(layouts) >= 170
    assert counts == {3, 5} and periods == {1, 2, 3}
    assert len(targets) >= 12 and len(directions) == 8 and len(masks) >= 3
    assert backgrounds == set(range(10))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(81357 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(991, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, D4/color checks)")
