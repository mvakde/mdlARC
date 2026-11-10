"""Length-matched stencil threading: directed strips fill equal-length mask runs.

Only standard-library modules and NumPy are used. No parent generator is imported.
The input-only parser deliberately accepts a wider fitting family than the sampler.
"""
from collections import Counter, defaultdict, deque
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    """Read geometry and identify fast arm by its reuse as the mask signal."""
    _require(min(a.shape) >= 5, "no room for a border key")
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            q = a[::-1] if bottom else a
            q = q[:, ::-1] if right else q
            origin, horizontal, vertical, bg = map(int, (q[0, 0], q[0, 1], q[1, 0], q[-1, -1]))
            if len({origin, horizontal, vertical, bg}) != 4:
                continue
            if not (np.all(q[0, 1:] == horizontal) and np.all(q[1:, 0] == vertical)
                    and np.all(q[-1, 1:] == bg) and np.all(q[1:, -1] == bg)):
                continue
            interior = q[1:, 1:]
            present = [c for c in (horizontal, vertical) if np.any(interior == c)]
            if len(present) != 1 or np.any(interior == origin):
                continue
            fast = present[0]
            axis = int(fast == vertical)
            rotation = (3 if bottom else 1) if axis else (2 if right else 0)
            candidates.append((rotation, fast, vertical if not axis else horizontal, bg, origin))
    _require(len(candidates) == 1, "key and signal must have one interpretation")
    return candidates[0]


def _components(binary):
    remaining = set(map(tuple, np.argwhere(binary)))
    boxes = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, cells = [first], [first]
        while todo:
            r, c = todo.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    todo.append(p)
        r0, c0 = min(p[0] for p in cells), min(p[1] for p in cells)
        r1, c1 = max(p[0] for p in cells)+1, max(p[1] for p in cells)+1
        _require(len(cells) == (r1-r0)*(c1-c0), "source strip is not a full rectangle")
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


def _runs(mask, up=False):
    result = []
    for r in (range(mask.shape[0]-1, -1, -1) if up else range(mask.shape[0])):
        c = 0
        while c < mask.shape[1]:
            if not mask[r, c]:
                c += 1
                continue
            end = c+1
            while end < mask.shape[1] and mask[r, end]:
                end += 1
            result.append((r, c, end-c))
            c = end
    return result


def _read(input_grid):
    a = _grid(input_grid)
    rotation, signal, slow, bg, origin = _key(a)
    q = np.rot90(a, rotation).copy()  # proper rotation only
    up = bool(q[-1, 0] == origin)
    # Fast is now rightward. The origin is at the left end of its top/bottom arm.
    _require(int(q[-1 if up else 0, 0]) == origin, "bad normalized key")
    interior = q[:, 1:].copy()
    interior = interior[:-1] if up else interior[1:]
    _require(np.all(interior[0 if up else -1] == bg) and np.all(interior[:, -1] == bg), "missing far rim")
    _require(np.all(interior[-1 if up else 0] == bg) and np.all(interior[:, 0] == bg), "payload touches key")
    rr, cc = np.nonzero(interior == signal)
    _require(len(rr) > 0, "empty mask")
    mr, mc, mh, mw = int(rr.min()), int(cc.min()), int(rr.max()-rr.min()+1), int(cc.max()-cc.min()+1)
    _require(2*mh <= 30 and 2*mw <= 30, "expanded mask exceeds 30")
    mask_crop = interior[mr:mr+mh, mc:mc+mw]
    _require(np.all((mask_crop == bg) | (mask_crop == signal)), "strip overlaps mask rectangle")
    mask = mask_crop == signal
    boxes = _components((interior != bg) & (interior != signal))
    _require(len(boxes) > 0, "no strips")
    _require(not np.any(interior == slow) and not np.any(interior == origin), "key colors in payload")
    lanes = sorted(set(b[0] for b in boxes))
    _require(all(b-a >= 3 for a, b in zip(lanes, lanes[1:])), "staggered source lanes")
    for r0, c0, r1, c1 in boxes:
        _require(r1-r0 == 2 and c1-c0 >= 2 and (c1-c0) % 2 == 0, "strip must be two thick and even long")
        _require(r0 >= 1 and c0 >= 1 and r1 < interior.shape[0] and c1 < interior.shape[1], "strip rim missing")
        # At least one blank cell between the complete mask box and each strip.
        _require(r1 < mr or mr+mh < r0 or c1 < mc or mc+mw < c0, "mask/strip separation missing")
    order = sorted(boxes, key=lambda b: (-b[0] if up else b[0], b[1]))
    strips = [interior[r0:r1, c0:c1].copy() for r0, c0, r1, c1 in order]
    runs = _runs(mask, up)
    _require(Counter(b.shape[1]//2 for b in strips) == Counter(n for _, _, n in runs), "run/strip length inventories differ")
    return {"mask": mask, "runs": runs, "strips": strips, "boxes": order,
            "background": bg, "signal": signal, "slow": slow, "origin": origin,
            "up": up, "rotation": rotation, "normalized": q, "mask_position": (mr, mc)}


def _thread(info, strips=None, runs=None):
    queues = defaultdict(deque)
    for strip in info["strips"] if strips is None else strips:
        queues[strip.shape[1]//2].append(strip)
    mask = info["mask"]
    output = np.full((2*mask.shape[0], 2*mask.shape[1]), info["background"], dtype=np.int8)
    for r, c, n in info["runs"] if runs is None else runs:
        _require(bool(queues[n]), "missing matching strip")
        output[2*r:2*r+2, 2*c:2*(c+n)] = queues[n].popleft()
    _require(not any(queues.values()), "unused source strip")
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive everything from this input, including signal, directions and ties."""
    return _thread(_read(input_grid))


def _sample_mask(rng, witness, regime, test):
    if witness:
        h = int(rng.integers(3, 6))
        repeat = int(rng.integers(1, 4))
        w = int(rng.integers(max(5, 2*repeat+1), 9))
        other = int(rng.choice([n for n in range(1, min(4, w)+1) if n != repeat]))
        split_row = int(rng.integers(h))
        rest = [r for r in range(h) if r != split_row]
        rng.shuffle(rest)
        mask = np.zeros((h, w), dtype=bool)
        mask[split_row, :repeat] = mask[split_row, -repeat:] = True
        for index, r in enumerate(rest):
            n = repeat if index == 0 else other if index == 1 else int(rng.integers(1, 5))
            start = int(rng.integers(w-n+1))
            mask[r, start:start+n] = True
        return mask
    if regime == "height" and test:
        h, w = int(rng.integers(6, 10)), int(rng.integers(3, 8))
    elif regime == "length" and test:
        h, w = int(rng.integers(2, 6)), int(rng.integers(7, 13))
    else:
        h, w = int(rng.integers(2, 6)), int(rng.integers(3, 7))
    mask = np.zeros((h, w), dtype=bool)
    family = str(rng.choice(["bars", "diagonal", "sparse", "ring", "ladder", "dense", "islands", "full"]))
    if family in ("bars", "diagonal"):
        for r in range(h):
            if 0 < r < h-1 and rng.random() < .18:
                continue
            n = 1 if family == "diagonal" else int(rng.integers(1, min(w, 6)+1))
            start = int(round(r*(w-n)/(h-1))) if family == "diagonal" else int(rng.integers(w-n+1))
            mask[r, start:start+n] = True
    elif family == "ring":
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    elif family == "ladder":
        mask[::2] = True
        mask[:, int(rng.integers(w))] = True
    elif family == "islands":
        mask[::2, ::2] = True
    elif family == "full":
        mask[:] = True
    else:
        mask = rng.random((h, w)) < (.72 if family == "dense" else .28)
    # Establish tight bounds without an invisible padding convention.
    mask[0, 0] = mask[-1, -1] = True
    if regime == "length" and test:
        r = int(rng.integers(h))
        n = int(rng.integers(5, min(w, 10)+1))
        start = int(rng.integers(w-n+1))
        mask[r, start:start+n] = True
    return mask


def _pack(rng, mask, lengths, witness):
    """Pack a mask box and full strip rectangles into shared, separated space.

    Lanes can lie above, below, or alongside the mask; there is no split panel.
    """
    mh, mw = mask.shape
    for attempt in range(36):
        if attempt >= 30:
            h = w = 29
        else:
            h = int(rng.integers(max(9 if witness else 7, mh+2), 30))
            w = int(rng.integers(max(8, mw+2, max(lengths)+2), 30))
        mr, mc = int(rng.integers(1, h-mh)), int(rng.integers(1, w-mw))
        starts, cursor = [], int(rng.integers(1, min(4, h-3)))
        while cursor+2 < h:
            starts.append(cursor)
            cursor += int(rng.integers(3, 7))
        if witness:
            lane_count = min(len(starts), int(rng.integers(2, 4)))
        else:
            lane_count = min(len(starts), int(rng.integers(1, min(len(lengths), len(starts))+1)))
        selected = rng.choice(starts, size=lane_count, replace=False).tolist()
        free = []
        for r in selected:
            if r+2 <= mr-1 or r >= mr+mh+1:
                intervals = [(1, w-1)]
            else:
                intervals = [(1, mc-1), (mc+mw+1, w-1)]
            free.extend((r, a, b) for a, b in intervals if b-a >= 2)
        placements = []
        successful = True
        # Long first is a packing method, not the final reading order.
        indices = sorted(range(len(lengths)), key=lambda i: (-lengths[i], float(rng.random())))
        for i in indices:
            n = lengths[i]
            options = [j for j, (_, a, b) in enumerate(free) if b-a >= n]
            if not options:
                successful = False
                break
            j = int(rng.choice(options))
            r, a, b = free.pop(j)
            c = int(rng.integers(a, b-n+1))
            placements.append((r, c, n))
            # A blank cell separates all neighboring strips in a lane.
            if c-1-a >= 2:
                free.append((r, a, c-1))
            if b-(c+n+1) >= 2:
                free.append((r, c+n+1, b))
        if successful:
            return (h, w), (mr, mc), placements
    return None


def _payload(rng, length, palette, witness):
    mode = 3 if witness else int(rng.integers(4))
    if mode == 0:
        return np.full((2, length), rng.choice(palette), dtype=np.int8)
    if mode == 1:
        return np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
    if mode == 2:
        period = int(rng.integers(1, min(4, length)+1))
        return np.tile(rng.choice(palette, size=(2, period)), (1, length))[:, :length].astype(np.int8)
    return rng.choice(palette, size=(2, length)).astype(np.int8)


def _paint(rng, mask, shape, position, placements, up, rotation, colors, witness):
    bg, signal, slow, origin = colors[:4]
    palette = colors[4:4+int(rng.integers(3 if witness else 1, 7))]
    h, w = shape
    scene = np.full(shape, bg, dtype=np.int8)
    mr, mc = position
    scene[mr:mr+mask.shape[0], mc:mc+mask.shape[1]] = np.where(mask, signal, bg)
    placed = []
    for r, c, n in placements:
        block = _payload(rng, n, palette, witness)
        scene[r:r+2, c:c+n] = block
        placed.append((r, c, block))
    # Authored ground truth routes construction records, without calling solve.
    sources = sorted(placed, key=lambda p: (-p[0] if up else p[0], p[1]))
    matching = defaultdict(deque)
    for _, _, block in sources:
        matching[block.shape[1]//2].append(block)
    output = np.full((2*mask.shape[0], 2*mask.shape[1]), bg, dtype=np.int8)
    for r, c, n in _runs(mask, up):
        output[2*r:2*r+2, 2*c:2*(c+n)] = matching[n].popleft()
    a = np.full((h+1, w+1), bg, dtype=np.int8)
    if up:
        a[:-1, 1:] = scene
    else:
        a[1:, 1:] = scene
    a[:, 0] = slow
    a[-1 if up else 0, :] = signal
    a[-1 if up else 0, 0] = origin
    return {"input": np.rot90(a, rotation).copy(), "output": output}


def _source_alternative(info, kind):
    ids = list(range(len(info["strips"])))
    boxes = info["boxes"]
    if kind == "fast_major":
        ids.sort(key=lambda i: (boxes[i][1], -boxes[i][0] if info["up"] else boxes[i][0]))
    elif kind in ("reverse_fast", "joint_fast"):
        ids.sort(key=lambda i: (-boxes[i][0] if info["up"] else boxes[i][0], -boxes[i][1]))
    elif kind in ("reverse_slow", "joint_slow"):
        ids.sort(key=lambda i: (boxes[i][0] if info["up"] else -boxes[i][0], boxes[i][1]))
    runs = info["runs"]
    if kind in ("joint_fast", "destination_fast"):
        runs = sorted(runs, key=lambda p: (-p[0] if info["up"] else p[0], -p[1]))
    elif kind in ("joint_slow", "destination_slow"):
        runs = sorted(runs, key=lambda p: (p[0] if info["up"] else -p[0], p[1]))
    return _thread(info, [info["strips"][i] for i in ids], runs)


def _asymmetric(block):
    return all(not np.array_equal(block, q) for q in (block[::-1], block[:, ::-1], block[::-1, ::-1]))


def _witness(info):
    lengths = [b.shape[1]//2 for b in info["strips"]]
    if len(set(lengths)) < 2 or lengths == [n for _, _, n in info["runs"]]:
        return False
    if not any(_asymmetric(b) for b in info["strips"]):
        return False
    if not any(b.shape[1] > 2 and any(not np.array_equal(b[:, :2], b[:, c:c+2])
                                     for c in range(2, b.shape[1], 2))
               for b in info["strips"]):
        return False  # distinguish threading from repeating one 2x2 stamp
    expected = _thread(info)
    return all(not np.array_equal(expected, _source_alternative(info, k))
               for k in ("fast_major", "reverse_fast", "reverse_slow", "destination_fast",
                         "destination_slow", "joint_fast", "joint_slow"))


def _make_pair(rng, witness, regime, test, up, rotation):
    for _ in range(160):
        mask = _sample_mask(rng, witness, regime, test)
        runs = _runs(mask, up)
        lengths = [2*n for _, _, n in runs]
        if not (2 <= len(runs) <= 12 and int(mask.sum()) <= 45 and max(lengths) <= 20):
            continue
        if regime == "length" and not test and max(lengths) > 8:
            continue
        packed = _pack(rng, mask, lengths, witness)
        if packed is None:
            continue
        shape, position, placements = packed
        colors = rng.permutation(10).astype(np.int8)
        pair = _paint(rng, mask, shape, position, placements, up, rotation, colors, witness)
        info = _read(pair["input"])
        if witness and not _witness(info):
            continue
        return pair
    raise RuntimeError("bounded construction could not find a readable witness")


def _count(name, value, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    _require(ntrain > ntest, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(["recombine", "height", "length"]))
    first_rotation, first_up = int(rng.integers(4)), bool(rng.integers(2))
    configurations = [(first_up, first_rotation), (not first_up, (first_rotation+1) % 4)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            witness = split == "train" and index < 2
            if split == "train":
                up, rotation = configurations[index % 2]
            elif regime == "recombine":
                up, rotation = configurations[index % 2]
                up = not up  # familiar direction paired with the other handedness
            else:
                up, rotation = configurations[int(rng.integers(2))]
            for _ in range(16):
                pair = _make_pair(rng, witness, regime, split == "test", up, rotation)
                key = _array_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate input construction")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _array_key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count("num_train", len(puzzle["train"]), 2, 10)
    _count("num_test", len(puzzle["test"]), 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "bad pair counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _array_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _read(inp)
            expected = _thread(d)
            _require(np.array_equal(expected, out), "incorrect threading")
            # Check actual coarse-cell occupancy, not only equality to solve.
            mh, mw = d["mask"].shape
            occupied = np.any(out.reshape(mh, 2, mw, 2) != d["background"], axis=(1, 3))
            _require(np.array_equal(occupied, d["mask"]), "macro-cell support differs from mask")
            _require(set(map(int, out.flat)) <= {d["background"]} | set(map(int, np.concatenate(d["strips"], axis=1).flat)), "uncopied color")
            _require(not np.any(out == d["signal"]), "mask signal leaked")
            if split == "train":
                train.append(d)
    _require(sum(_witness(d) for d in train) >= 2, "two directed matching witnesses needed")
    _require({d["up"] for d in train} == {False, True}, "both handedness conventions need evidence")
    _require({d["rotation"] % 2 for d in train} == {0, 1}, "both strip axes need evidence")
    _require(any(np.any(~d["mask"]) for d in train), "show inactive macro-cells")


def _fingerprint(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _cell_regression():
    """Small literal cell expectations, distinct from generation and prediction."""
    mask = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]])
    a = np.array([[4, 5, 6, 4], [6, 4, 5, 5]], dtype=np.int8)
    b = np.array([[7, 8], [8, 7]], dtype=np.int8)
    c = np.array([[9, 4, 9, 5], [5, 9, 4, 9]], dtype=np.int8)
    d = np.array([[6, 7, 8, 9], [9, 8, 7, 6]], dtype=np.int8)
    expected = np.array([
        [4, 5, 6, 4, 0, 0, 0, 0], [6, 4, 5, 5, 0, 0, 0, 0],
        [0, 0, 9, 4, 9, 5, 0, 0], [0, 0, 5, 9, 4, 9, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 8], [0, 0, 0, 0, 0, 0, 8, 7],
        [0, 0, 0, 0, 6, 7, 8, 9], [0, 0, 0, 0, 9, 8, 7, 6]], dtype=np.int8)
    for up in (False, True):
        scene = np.zeros((14, 24), dtype=np.int8)
        scene[8:12, 2:6] = mask
        for r, col, block in ((1, 8, a), (1, 13, b), (1, 17, c), (4, 6, d)):
            scene[r:r+2, col:col+block.shape[1]] = block
        inp = np.zeros((15, 25), dtype=np.int8)
        if up:
            inp[:-1, 1:] = scene
        else:
            inp[1:, 1:] = scene
        inp[:, 0] = 2
        inp[-1 if up else 0] = 1
        inp[-1 if up else 0, 0] = 3
        answer = expected.copy()
        if up:
            answer[0:2, 0:4], answer[2:4, 2:6] = c, a
        original = inp.copy()
        for k in range(4):
            assert np.array_equal(solve(np.rot90(inp, k)), answer)
        assert np.array_equal(inp, original), "solve mutated input"
        info = _read(inp)
        if not up:
            assert all(not np.array_equal(answer, _source_alternative(info, kind))
                       for kind in ("fast_major", "reverse_fast", "reverse_slow", "joint_fast", "joint_slow"))
        # Removing length routing would pair the short B with a long run.
        assert [x.shape[1]//2 for x in info["strips"]] != [n for _, _, n in info["runs"]]
        # Both equal-length occurrences survive, even when their payloads match.
        repeated = inp.copy()
        row_offset = 0 if up else 1
        repeated[1+row_offset:3+row_offset, 18:22] = a
        repeated_answer = answer.copy()
        repeated_answer[0:2, 0:4] = a
        if not up:
            repeated_answer[2:4, 2:6] = a
        assert np.array_equal(solve(repeated), repeated_answer)
        damaged = inp.copy()
        damaged[10+row_offset, 5] = 1  # the one-cell run becomes two cells
        try:
            solve(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError("unmatched mask/source inventory accepted")
        damaged = inp.copy()
        damaged[1+row_offset, 9] = 0  # hole in full source rectangle
        try:
            solve(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError("damaged full strip accepted")


def _family_boundaries():
    """Check fitting semantic cases outside the sampler's ordinary ranges."""
    def with_key(scene):
        inp = np.zeros((scene.shape[0]+1, scene.shape[1]+1), dtype=np.int8)
        inp[1:, 1:] = scene
        inp[0, 1:], inp[1:, 0], inp[0, 0] = 1, 2, 3
        return inp

    block = np.array([[4, 5], [6, 4]], dtype=np.int8)
    scene = np.zeros((5, 8), dtype=np.int8)
    scene[1, 1] = 1
    scene[1:3, 4:6] = block
    assert np.array_equal(solve(with_key(scene)), block)  # one matched occurrence

    scene = np.zeros((19, 20), dtype=np.int8)
    scene[1:16, 1] = 1
    blocks = []
    for r in (1, 4, 7, 10, 13):
        for c in (4, 7, 10):
            b = np.full((2, 2), 4+len(blocks) % 6, dtype=np.int8)
            scene[r:r+2, c:c+2] = b
            blocks.append(b)
    assert np.array_equal(solve(with_key(scene)), np.concatenate(blocks, axis=0))

    scene = np.zeros((14, 22), dtype=np.int8)
    scene[10, 1:16:2] = 1
    answer = np.zeros((2, 30), dtype=np.int8)
    index = 0
    for r in (1, 4):
        for c in (1, 4, 7, 10):
            b = np.full((2, 2), 4+index % 6, dtype=np.int8)
            scene[r:r+2, c:c+2] = b
            answer[:, 4*index:4*index+2] = b
            index += 1
    assert np.array_equal(solve(with_key(scene)), answer)  # tight blank columns


def self_test():
    """Bounded 42-seed diversity suite, count audit and semantic regressions."""
    _cell_regression()
    _family_boundaries()
    fingerprints, shapes, mask_patterns, strip_patterns, configs = set(), set(), set(), set(), set()
    backgrounds, counts, maximums, lane_counts = set(), set(), set(), set()
    blank_lines = full = repeated = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                d = _read(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add((np.asarray(pair["input"]).shape, np.asarray(pair["output"]).shape))
                mask_patterns.add(_array_key(d["mask"]))
                mapping = {d["background"]: 0}
                signature = []
                for b in d["strips"]:
                    signature.append((b.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in b.flat)))
                strip_patterns.add(tuple(signature))
                configs.add((d["up"], d["rotation"]))
                backgrounds.add(d["background"])
                counts.add(len(d["runs"]))
                maximums.add(max(n for _, _, n in d["runs"]))
                lane_counts.add(len(set(b[0] for b in d["boxes"])))
                blank_lines += int(np.any(~d["mask"].any(axis=1)) or np.any(~d["mask"].any(axis=0)))
                full += int(np.all(d["mask"]))
                repeated += int(len(set(_array_key(b) for b in d["strips"])) < len(d["strips"]))
    assert len(fingerprints) == 42 and len(shapes) >= 100
    assert len(mask_patterns) >= 100 and len(strip_patterns) >= 150
    assert len(configs) == 8 and backgrounds == set(range(10))
    assert len(counts) >= 7 and len(maximums) >= 6 and len(lane_counts) >= 4
    assert blank_lines >= 3 and full >= 3 and repeated >= 3
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    assert _fingerprint(generate_puzzle(7)) == _fingerprint(generate_puzzle(7))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(37)
    pair = p["train"][0]
    # Whole-input proper rotation leaves the normalized answer invariant.
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), pair["output"])
    # A global palette permutation changes only the same copied color roles.
    palette = np.array([9, 5, 8, 6, 4, 3, 1, 7, 0, 2], dtype=np.int8)
    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    out = p["test"][0]["output"]
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "puzzles": len(fingerprints), "shape_pairs": len(shapes),
            "mask_patterns": len(mask_patterns), "strip_patterns": len(strip_patterns),
            "key_configurations": len(configs), "run_counts": sorted(counts),
            "maximum_run_lengths": sorted(maximums), "source_lane_counts": sorted(lane_counts),
            "internal_blank_line_pairs": blank_lines, "full_mask_pairs": full,
            "repeated_strip_pairs": repeated}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
