"""Turnable mirrors: color-specific reflected completion of one erased window.

Only NumPy and the standard library are used. Generation labels are crops of
complete fields; solve reconstructs from visible pixels and bars alone.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 572747009, 903671649, 805422966, 627761369, 839258702,
              381657106, 664683100, 629614549, 690910295, 680759178,
              798951829, 406031071, 477811296, 662812918, 99323456,
              347375399, 791777309, 968718902, 902077281, 458006867,
              572564665, 850071797, 847233962, 255590589, 603154593,
              573723497, 969147154, 556736270, 591498096, 803817600,
              337027088, 593540213, 633029914, 893315953, 188506762,
              315423984, 383770592, 531862410, 507447544, 65203286,
              681988264)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b):
        raise ValueError("Example counts must be integers")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    return int(a), int(b)


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _turn(r, c, n, k):
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return r, c


def _mate(r, c, n, k):
    # F first, then the same clockwise R used by parent B. R^k F is an
    # involution, with four different axes rather than only two conjugate axes.
    return _turn(r, n - 1 - c, n, k)


def _inside(p, box):
    r, c = p
    y, z, x, w = box
    return y <= r < z and x <= c < w


def _bbox(mask):
    rr, cc = np.nonzero(mask)
    assert len(rr)
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _read_input(input_grid):
    a = _grid(input_grid)
    frames = []
    for color in np.unique(a):
        y, z, x, w = _bbox(a == color)
        if z - y != w - x or z - y < 9:
            continue
        ring = np.zeros(a.shape, dtype=bool)
        ring[y, x:w] = ring[z - 1, x:w] = True
        ring[y:z, x] = ring[y:z, w - 1] = True
        if np.array_equal(a == color, ring):
            frames.append((int(color), (y, z, x, w)))
    assert len(frames) == 1, "Need one unique square frame"
    frame, outer = frames[0]
    y, z, x, w = outer
    n = z - y - 2
    assert 7 <= n <= 18
    exterior = np.ones(a.shape, dtype=bool)
    exterior[y:z, x:w] = False
    values, sizes = np.unique(a[exterior], return_counts=True)
    assert len(values) and np.sum(sizes == sizes.max()) == 1
    bg = int(values[np.argmax(sizes)])
    assert bg != frame
    board = a[y + 1:z - 1, x + 1:w - 1].copy()
    possible = []
    for color in np.unique(board):
        color = int(color)
        if color in (bg, frame) or np.any(a[exterior] == color):
            continue
        box = _bbox(board == color)
        r, s, c, d = box
        if (s - r) * (d - c) >= 2 and np.all(board[r:s, c:d] == color):
            possible.append((color, box))
    assert len(possible) == 1, "Need one unique solid rectangular erasure"
    mask, box = possible[0]
    colors = tuple(int(v) for v in np.unique(board) if v not in (bg, mask))
    assert frame not in colors and 2 <= len(colors) <= 4
    bars, bar_cells = {}, []
    for color in np.unique(a[exterior]):
        color = int(color)
        if color == bg:
            continue
        assert color in colors, "Counter must match a visible layer"
        coords = np.argwhere(exterior & (a == color))
        rr, cc = coords.T
        horizontal = np.ptp(rr) == 0 and np.ptp(cc) + 1 == len(rr)
        vertical = np.ptp(cc) == 0 and np.ptp(rr) + 1 == len(rr)
        assert 1 <= len(rr) <= 9 and (horizontal or vertical)
        assert all(not (y - 1 <= r <= z and x - 1 <= c <= w) for r, c in coords)
        orientation = "single" if len(rr) == 1 else ("horizontal" if horizontal else "vertical")
        bars[color] = {"length": len(rr), "orientation": orientation}
        bar_cells.extend((color, int(r), int(c)) for r, c in coords)
    assert bars, "Need at least one counter"
    for i, (color, r, c) in enumerate(bar_cells):
        assert all(other == color or max(abs(r - u), abs(c - v)) > 1
                   for other, u, v in bar_cells[i + 1:])
    assert set(int(v) for v in np.unique(a)) == {bg, frame, mask} | set(colors)
    return {"board": board, "background": bg, "frame": frame, "mask": mask,
            "box": box, "outer": outer, "colors": colors, "bars": bars, "size": n}


def _complete(info):
    board, bg, m, n = info["board"], info["background"], info["mask"], info["size"]
    full = np.where(board == m, bg, board).astype(np.int8)
    witnessed = np.zeros(board.shape, dtype=bool)
    for color in info["colors"]:
        k = info["bars"].get(color, {}).get("length", 0)
        for r, c in np.argwhere(board == color):
            u, v = _mate(int(r), int(c), n, k)
            assert board[u, v] in (color, m), "Visible layer violates its controlled mirror"
            assert full[u, v] in (bg, color), "Conflicting reflected witnesses"
            full[u, v] = color
            if board[u, v] == m:
                witnessed[u, v] = True
    # No unsupported content is invented: the least mirror-completion is fixed.
    for color in info["colors"]:
        k = info["bars"].get(color, {}).get("length", 0)
        assert np.array_equal(full == color, np.rot90(np.fliplr(full == color), -k % 4))
    assert np.all(witnessed[board == m] == (full[board == m] != bg))
    return full


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only this input's frame, erasure, colored pixels, and unary bars."""
    info = _read_input(input_grid)
    full = _complete(info)
    r, s, c, d = info["box"]
    return full[r:s, c:d].copy()


def _alternative(info, *, reflection=True, direction=1, common=None):
    """Concrete shortcut predictions; -1 denotes conflicting shortcut claims."""
    board, bg, n = info["board"], info["background"], info["size"]
    out = np.full(board.shape, bg, dtype=np.int8)
    for color in info["colors"]:
        k = info["bars"].get(color, {}).get("length", 0) if common is None else common
        for r, c in np.argwhere(board == color):
            if reflection:
                u, v = _mate(int(r), int(c), n, direction * k)
            else:
                u, v = _turn(int(r), int(c), n, direction * k)
            if _inside((u, v), info["box"]):
                out[u, v] = color if out[u, v] in (bg, color) else -1
    r, s, c, d = info["box"]
    return out[r:s, c:d]


def _evidence(info, answer):
    full = _complete(info)
    bg, n = info["background"], info["size"]
    affected, axes, lengths = set(), set(), set()
    for color in info["colors"]:
        k = info["bars"].get(color, {}).get("length", 0)
        if np.any(answer == color):
            affected.add(color)
            axes.add(k % 4)
            lengths.add(k)
    counter_needed = False
    board = info["board"]
    r0, r1, c0, c1 = info["box"]
    # Can another mirror axis explain all visible pixels of this layer, yet
    # give a different, nonconflicting hole completion? Counters must resolve
    # a real ambiguity, not merely restate the axis already evident in pixels.
    for color in info["colors"]:
        for k in range(4):
            predicted = np.zeros(board.shape, dtype=bool)
            compatible = True
            for r, c in np.argwhere(board == color):
                u, v = _mate(int(r), int(c), n, k)
                if board[u, v] not in (color, info["mask"]):
                    compatible = False
                    break
                if board[u, v] == info["mask"]:
                    predicted[u, v] = True
            hidden = predicted[r0:r1, c0:c1]
            if compatible and not np.array_equal(hidden, answer == color) and not np.any(
                    hidden & (answer != color) & (answer != bg)):
                counter_needed = True
    return {"counter_needed": counter_needed,
            "no_A": not np.array_equal(answer, _alternative(info, reflection=False)),
            "no_B": not np.array_equal(answer, _alternative(info, common=0)),
            "direction": not np.array_equal(answer, _alternative(info, direction=-1)),
            "no_common": all(not np.array_equal(answer, _alternative(info, common=k)) for k in range(4)),
            "affected": affected, "axes": axes, "lengths": lengths,
            "unique_axes": {color: tuple(k for k in range(4) if np.array_equal(
                full == color, np.rot90(np.fliplr(full == color), -k))) for color in info["colors"]}}


def _window(rng, n, shift, test):
    if shift == 0:
        short = int(rng.integers(2, 5))
        long = int(rng.integers(short + 1, min(n - 1, short + 5) + 1))
        h, w = (short, long) if test else (long, short)
    elif shift == 2 and test:
        h, w = 1, int(rng.integers(6, n + 1))
        if rng.integers(2):
            h, w = w, h
    else:
        h, w = int(rng.integers(2, min(7, n - 2))), int(rng.integers(2, min(7, n - 2)))
        if h * w < 8:
            h = 4
    low = 1 if shift == 1 and not test else 0
    r = int(rng.integers(low, n - h + 1 - low))
    c = int(rng.integers(low, n - w + 1 - low))
    if shift == 1 and test:
        side = int(rng.integers(4))
        if side == 0:
            r = 0
        elif side == 1:
            c = n - w
        elif side == 2:
            r = n - h
        else:
            c = 0
    return r, r + h, c, c + w


def _orbits(n, k):
    result, seen = [], set()
    for r in range(n):
        for c in range(n):
            if (r, c) not in seen:
                orbit = tuple(dict.fromkeys(((r, c), _mate(r, c, n, k))))
                seen.update(orbit)
                result.append(orbit)
    return result


def _field(rng, n, bg, colors, counts, box, teaching):
    full = np.full((n, n), bg, dtype=np.int8)
    orbits = [_orbits(n, k) for k in counts]
    # A small layer has visible seeds along a different, unselected mirror
    # axis. Under that other axis it would need no hidden pixels at all. Its
    # counter therefore supplies information absent from the visible pattern.
    hinges = []
    for j in range(len(colors)):
        for alt in range(4):
            if alt == counts[j] % 4:
                continue
            eligible = []
            for orbit in orbits[j]:
                if len(orbit) != 2 or sum(_inside(p, box) for p in orbit) != 1:
                    continue
                visible = next(p for p in orbit if not _inside(p, box))
                if _mate(*visible, n, alt) == visible:
                    eligible.append(orbit)
            if eligible:
                hinges.append((j, eligible))
    if not hinges:
        return None
    hinge, eligible = hinges[int(rng.integers(len(hinges)))]
    for idx in rng.choice(len(eligible), int(rng.integers(1, min(4, len(eligible)) + 1)), replace=False):
        for p in eligible[int(idx)]:
            full[p] = colors[hinge]
    # First place one actually erased, externally witnessed orbit for each
    # teaching layer (two layers suffice on ordinary examples).
    required = list(range(len(colors))) if teaching else [hinge, int(rng.choice(
        [j for j in range(len(colors)) if j != hinge]))]
    rng.shuffle(required)
    for j in required:
        if j == hinge:
            continue
        choices = [o for o in orbits[j] if sum(_inside(p, box) for p in o) == 1
                   and len(o) == 2 and all(full[p] == bg for p in o)]
        if not choices:
            return None
        for p in choices[int(rng.integers(len(choices)))]:
            full[p] = colors[j]
    for j in rng.permutation(len(colors)):
        j = int(j)
        if j == hinge:
            continue
        style = int(rng.integers(4))
        ar, ac = rng.uniform(0, n, 2)
        br, bc = rng.uniform(0, n, 2)
        period, phase = int(rng.integers(2, 5)), int(rng.integers(4))
        scores = []
        for orbit in orbits[j]:
            if style == 0:  # Paired compact patches, potentially joined on the mirror.
                score = min((r - ar) ** 2 + (c - ac) ** 2 for r, c in orbit)
            elif style == 1:  # Mirrored stripes, bars, and teeth.
                score = min((r + phase) % period + abs(c - ac) / n for r, c in orbit)
            elif style == 2:  # Two independently located mirrored islands.
                score = min(min((r - ar) ** 2 + (c - ac) ** 2,
                                (r - br) ** 2 + (c - bc) ** 2) for r, c in orbit)
            else:
                score = float(rng.uniform(0, n * n / 3))
            scores.append(score + float(rng.random()) * .7)
        target = max(4, int(n * n * rng.uniform(.045, .22)))
        area = int(np.sum(full == colors[j]))
        for idx in np.argsort(scores, kind="stable"):
            orbit = orbits[j][int(idx)]
            # A content orbit entirely erased would have no visible evidence.
            if all(_inside(p, box) for p in orbit) or not all(full[p] == bg for p in orbit):
                continue
            for p in orbit:
                full[p] = colors[j]
            area += len(orbit)
            if area >= target:
                break
    if not all(np.any(full == color) for color in colors):
        return None
    return full


def _bar_starts(blocked, length, orientation):
    h, w = blocked.shape
    bh, bw = (1, length) if orientation == "horizontal" else (length, 1)
    good = np.ones((h - bh + 1, w - bw + 1), dtype=bool)
    for i in range(length):
        dy, dx = (0, i) if bh == 1 else (i, 0)
        good &= ~blocked[dy:dy + good.shape[0], dx:dx + good.shape[1]]
    return np.argwhere(good)


def _layout(rng, board, bg, frame, colors, counts, forced):
    n, b = len(board), len(board) + 2
    for attempt in range(18):
        h = w = 30 if attempt >= 15 else 0
        if attempt < 15:
            h, w = int(rng.integers(max(b + 4, 16), 31)), int(rng.integers(max(b + 4, 16), 31))
        y, x = (0, 0) if attempt >= 15 else (int(rng.integers(h - b + 1)), int(rng.integers(w - b + 1)))
        a = np.full((h, w), bg, dtype=np.int8)
        a[y:y + b, x:x + b] = frame
        a[y + 1:y + b - 1, x + 1:x + b - 1] = board
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0, y - 1):min(h, y + b + 1), max(0, x - 1):min(w, x + b + 1)] = True
        success = True
        order = sorted((j for j, k in enumerate(counts) if k), key=lambda j: -counts[j])
        for j in order:
            possibilities = []
            for orientation in ("horizontal", "vertical"):
                if j in forced and forced[j] != orientation:
                    continue
                starts = _bar_starts(blocked, counts[j], orientation)
                if len(starts):
                    possibilities.append((orientation, starts))
            if not possibilities:
                success = False
                break
            orientation, starts = possibilities[int(rng.integers(len(possibilities)))]
            r, c = map(int, starts[int(rng.integers(len(starts)))])
            for i in range(counts[j]):
                u, v = (r, c + i) if orientation == "horizontal" else (r + i, c)
                a[u, v] = colors[j]
                blocked[max(0, u - 1):min(h, u + 2), max(0, v - 1):min(w, v + 2)] = True
        if success:
            return a
    return None


def _make_pair(rng, shift, test, witness, seen):
    for _ in range(180):
        n = int(rng.integers(9 if witness >= 0 else 7, 19))
        count = 4 if witness >= 0 else int(rng.integers(2, 5))
        palette = [int(v) for v in rng.choice(10, count + 3, replace=False)]
        bg, frame, mask = palette[:3]
        colors = palette[3:]
        forced = {}
        if witness == 0:
            lengths = [0, 1, 2, 3]
            # Color identities themselves are randomized, so no numeric ranking.
            forced = {2: "horizontal", 3: "vertical"}
        elif witness == 1:
            repeated = int(rng.choice([2, 6]))
            lengths = [4, int(rng.choice([5, 9])), repeated, repeated]
            forced = {2: "horizontal", 3: "vertical"}
        else:
            lengths = [int(v) for v in rng.integers(0, 10, count)]
            if test:
                lengths[int(rng.integers(count))] = int(rng.integers(5, 10))
            if len({v % 4 for v in lengths}) < 2 or not any(lengths):
                continue
        box = _window(rng, n, shift, test)
        full = _field(rng, n, bg, colors, lengths, box, witness >= 0)
        if full is None:
            continue
        r, s, c, d = box
        output = full[r:s, c:d].copy()
        if len(set(int(v) for v in output.ravel()) - {bg}) < 2:
            continue
        board = full.copy()
        board[r:s, c:d] = mask
        a = _layout(rng, board, bg, frame, colors, lengths, forced)
        if a is None or _key(a) in seen:
            continue
        try:
            info = _read_input(a)
            inferred = _complete(info)[r:s, c:d]
            if not np.array_equal(output, inferred):
                continue
            e = _evidence(info, output)
            if not (e["no_A"] and e["no_B"] and e["no_common"] and e["counter_needed"]):
                continue
            if witness >= 0:
                if len(e["affected"]) != count or not e["direction"]:
                    continue
                if any(e["unique_axes"][color] != (k % 4,) for color, k in zip(colors, lengths)):
                    continue
        except AssertionError:
            continue
        return {"input": a, "output": output}
    raise RuntimeError("Bounded turnable-mirror sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            pair = _make_pair(rng, shift, split == "test", i if split == "train" and i < 2 else -1, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, answer = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _read_input(a)
            full = _complete(info)
            r, s, c, d = info["box"]
            assert np.array_equal(answer, full[r:s, c:d]), "Incorrect erased-window crop"
            assert info["mask"] not in answer and info["frame"] not in answer
            assert len(set(int(v) for v in answer.ravel()) - {info["background"]}) >= 2
            # Reinsert output; the only changed cells are in the query window.
            visible = info["board"] != info["mask"]
            assert np.array_equal(full[visible], info["board"][visible])
            e = _evidence(info, answer)
            assert e["no_A"] and e["no_B"] and e["no_common"] and e["counter_needed"], "A parent concept became dispensable"
            if split == "train":
                training.append((info, e))
    lengths = set().union(*(e["lengths"] for _, e in training))
    assert {0, 1, 2, 3, 4}.issubset(lengths)
    assert any(k in lengths for k in (5, 9)), "Need an explicit repeated cycle"
    assert any(e["direction"] for _, e in training)
    assert any(len(e["axes"]) == 4 and len(e["affected"]) == 4
               and all(len(v) == 1 for v in e["unique_axes"].values()) for _, e in training)
    assert any(any(len({info["bars"][color]["orientation"] for color in e["affected"]
                        if color in info["bars"] and info["bars"][color]["length"] == k}) == 2
                   for k in range(2, 10)) for info, e in training), "Demonstrate orientation independence"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds; streaming aggregate measurements, bounded retries, no file I/O."""
    # A literal fixture outside the sampler's teaching scaffold. A diagonal
    # recovery and an unmarked axial recovery share one query. Repeated cycles
    # change the controls while preserving the expected pixels.
    for raw, axial in ((1, 0), (5, 4), (9, 8)):
        board = np.zeros((7, 7), dtype=np.int8)
        board[5, 6], board[1, 4] = 1, 2
        board[:2, :3] = 9
        a = np.zeros((17, 17), dtype=np.int8)
        a[:9, :9] = 8
        a[1:8, 1:8] = board
        a[11, :raw] = 1
        a[14, :axial] = 2
        assert np.array_equal(solve(a), np.array([[0, 1, 0], [0, 0, 2]], dtype=np.int8))
    broken = a.copy()
    broken[7, 6] = 2  # Its required axial mate is visible background.
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Conflicting visible mirror evidence accepted")
    signatures, shapes, outputs, palettes, backgrounds, sizes, areas = set(), set(), set(), set(), set(), set(), set()
    lengths, layers, geometries = set(), set(), set()
    default_pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(_digest(p))
        if seed in (0, HOST_SEEDS[1]):
            assert _digest(p) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                default_pairs += 1
                a, b = pair["input"], pair["output"]
                assert np.array_equal(solve(a), b)
                info = _read_input(a)
                full = _complete(info)
                shapes.add(a.shape)
                outputs.add(b.shape)
                sizes.add(info["size"])
                backgrounds.add(info["background"])
                palettes.add((info["background"], info["frame"], info["mask"], info["colors"]))
                lengths.update(v["length"] for v in info["bars"].values())
                layers.add(len(info["colors"]))
                areas.update(int(np.sum(full == color)) for color in info["colors"])
                r, s, c, d = info["box"]
                geometries.add((info["size"], info["outer"], (r, s, c, d)))
        # Each puzzle has a visible crop-geometry shift, with no new operator.
        train_boxes = [_read_input(x["input"]) for x in p["train"]]
        test_boxes = [_read_input(x["input"]) for x in p["test"]]
        tall_wide = all(x["output"].shape[0] > x["output"].shape[1] for x in p["train"]) and all(
            x["output"].shape[1] > x["output"].shape[0] for x in p["test"])
        def edge(info):
            r, s, c, d = info["box"]
            return r == 0 or c == 0 or s == info["size"] or d == info["size"]
        held_edge = not any(map(edge, train_boxes)) and all(map(edge, test_boxes))
        thin = all(min(x["output"].shape) >= 2 for x in p["train"]) and all(
            min(x["output"].shape) == 1 for x in p["test"])
        assert tall_wide or held_edge or thin
    assert len(signatures) == 42 and default_pairs == 210
    assert len(shapes) >= 35 and len(outputs) >= 20 and len(geometries) >= 150
    assert sizes == set(range(7, 19)) and backgrounds == set(range(10))
    assert lengths == set(range(1, 10)) and layers == {2, 3, 4}
    assert len(palettes) >= 150 and len(areas) >= 30
    # Every legal explicit count combination, plus type and boundary failures.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(104729 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal example counts accepted")
    # Labels and visible reflection witnesses are checked, not trusted.
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted label accepted")
    # Arbitrary literal-color bijections keep the locally encoded rule intact.
    perm = np.random.default_rng(812).permutation(10).astype(np.int8)
    recolored = {s: [{k: perm[v] for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(recolored)
    # Reversing the board and counters together changes absolute mirror axes;
    # it is deliberately not declared a family-preserving augmentation.
    return {"seeds": 42, "pairs": default_pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(outputs),
            "source_sizes": sorted(sizes), "layer_counts": sorted(layers),
            "bar_lengths": sorted(lengths), "layer_area_range": [min(areas), max(areas)],
            "geometries": len(geometries), "palettes": len(palettes)}
