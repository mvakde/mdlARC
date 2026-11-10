"""Key-ordered ribbon pieces sewn by unique exact, possibly offset seams.

Only standard Python and NumPy. No disk writes, caches, or saved traces.
Construction labels come from planned coordinates; solve reads pixels only.
"""
from itertools import combinations
import hashlib
import numpy as np


HOST_SEEDS = (0, 299176803, 397430435, 424385073, 269157092, 672526503,
              209175235, 783698697, 789683041, 30273488, 105879950, 242883302,
              887965665, 42911626, 521759943, 180914464, 364555978, 838351599,
              849639541, 894242975, 211188780, 66224626, 400176914, 412924071,
              134092349, 562786651, 446700724, 284042735, 416932073, 69780955,
              879639348, 35566783, 676694922, 89152843, 365744344, 14721857,
              678800183, 435493774, 685414149, 135680712, 199419384, 748302939)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer color grid within ARC bounds")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _boxes(mask):
    pending = set(zip(*np.nonzero(mask)))
    boxes = []
    while pending:
        p = min(pending)
        pending.remove(p)
        points, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in pending:
                    pending.remove(q)
                    points.append(q)
                    stack.append(q)
        r0, c0 = map(min, zip(*points))
        r1, c1 = (max(r for r, c in points)+1, max(c for r, c in points)+1)
        if len(points) != (r1-r0)*(c1-c0):
            raise ValueError("A piece is not a filled rectangle")
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


def _decode(value):
    """Read key geometry and proper-rotation-normalized strips, without labels."""
    a = _grid(value)
    keys = []
    if min(a.shape) < 5:
        raise ValueError("No room for a key and separated pieces")
    for axis in (0, 1):
        for bottom in (False, True):
            for right in (False, True):
                g = a[::-1] if bottom else a
                g = g[:, ::-1] if right else g
                g = g.T if axis else g
                o, f, s, bg = map(int, (g[0, 0], g[0, 1], g[1, 0], g[-1, -1]))
                if len({o, f, s, bg}) != 4:
                    continue
                arm = np.where(np.arange(1, g.shape[1]) % 2, f, o)
                if (np.array_equal(g[0, 1:], arm) and np.all(g[1:, 0] == s)
                        and np.all(g[-1, 1:] == bg) and np.all(g[1:, -1] == bg)):
                    keys.append((g, (axis, bottom, right), (bg, o, f, s)))
    if len(keys) != 1:
        raise ValueError("The visible key must be unique")
    g, config, roles = keys[0]
    bg = roles[0]
    mask = g != bg
    mask = mask.copy()
    mask[0] = False
    mask[:, 0] = False
    boxes = _boxes(mask)
    if len(boxes) < 2:
        raise ValueError("Need at least two ribbon pieces")
    lanes = {}
    for r0, c0, r1, c1 in boxes:
        if not (r0 >= 2 and c0 >= 2 and r1 < g.shape[0]
                and c1 < g.shape[1] and r1-r0 == 2 and c1-c0 >= 2):
            raise ValueError("Pieces must be separated two-cell lane rectangles")
        lanes.setdefault(r0, []).append((r0, c0, r1, c1))
    rows = sorted(lanes)
    if any(y-x < 3 for x, y in zip(rows, rows[1:])):
        raise ValueError("Lane bands overlap or are staggered")
    blocks, ordered_boxes, counts = [], [], []
    # A reflection of the coordinate frame must not reflect the payload itself.
    reverse_thickness = sum(config) % 2 == 1
    for row in rows:
        group = sorted(lanes[row], key=lambda b: b[1])
        counts.append(len(group))
        for left, right_box in zip(group, group[1:]):
            if right_box[1] - left[3] < 1:
                raise ValueError("No separator between pieces")
        for r0, c0, r1, c1 in group:
            b = g[r0:r1, c0:c1]
            blocks.append((b[::-1] if reverse_thickness else b).copy())
            ordered_boxes.append((r0, c0, r1, c1))
    return {"blocks": blocks, "background": bg, "roles": roles,
            "configuration": config, "counts": tuple(counts), "boxes": ordered_boxes}


def _cells(block, position):
    r0, c0 = position
    return {(r0+r, c0+c): int(block[r, c])
            for r in range(2) for c in range(block.shape[1])}


def _add(union, block, position):
    for p, color in _cells(block, position).items():
        if p in union and union[p] != color:
            raise ValueError("Conflicting ribbon cells")
        union[p] = color


def _candidates(previous, incoming, position=(0, 0), union=None):
    """All proper suffix-prefix seams; never rank or optimize the candidates."""
    result = []
    for d in (-1, 0, 1):
        for k in range(1, min(previous.shape[1], incoming.shape[1])):
            if (2-abs(d))*k < 2:
                continue
            a = previous[max(0, d):min(2, 2+d), -k:]
            b = incoming[max(0, -d):min(2, 2-d), :k]
            if not np.array_equal(a, b):
                continue
            pos = position[0]+d, position[1]+previous.shape[1]-k
            if union is not None and any(p in union and union[p] != color
                                         for p, color in _cells(incoming, pos).items()):
                continue
            result.append((d, k, pos))
    return result


def _render(union, background):
    r0, c0 = map(min, zip(*union))
    h, w = max(r for r, c in union)-r0+1, max(c for r, c in union)-c0+1
    if max(h, w) > 30:
        raise ValueError("Assembled ribbon exceeds ARC bounds")
    out = np.full((h, w), background, dtype=np.int8)
    for (r, c), color in union.items():
        out[r-r0, c-c0] = color
    return out


def _assemble(blocks, background):
    union = _cells(blocks[0], (0, 0))
    positions, seams = [(0, 0)], []
    for previous, incoming in zip(blocks, blocks[1:]):
        candidates = _candidates(previous, incoming, positions[-1], union)
        if len(candidates) != 1:
            raise ValueError("Every directed seam must have exactly one placement")
        d, k, pos = candidates[0]
        seams.append((d, k))
        positions.append(pos)
        _add(union, incoming, pos)
    return _render(union, background), positions, seams


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    return _assemble(d["blocks"], d["background"])[0]


def _asymmetric(b):
    return all(not np.array_equal(b, other)
               for other in (b[::-1], b[:, ::-1], b[::-1, ::-1]))


def _strip(rng, length, palette):
    b = rng.choice(palette, size=(2, length)).astype(np.int8)
    mode = int(rng.integers(8))
    if mode == 0:
        b[1] = b[0]
    elif mode == 1:
        b[:] = rng.choice(palette, size=(2, 1))
    elif mode == 2:
        period = int(rng.integers(1, min(length, 4)))
        b = np.tile(b[:, :period], (1, length))[:, :length].copy()
    return b


def _common_ports(blocks):
    end = blocks[0][:, :1]
    if end[0, 0] == end[1, 0] or not all(
            np.array_equal(b[:, :1], end) and np.array_equal(b[:, -1:], end) for b in blocks):
        return False
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            if i != j and [(d, k) for d, k, p in _candidates(a, b)] != [(0, 1)]:
                return False
    return True


def _ribbon(rng, n, palette, background, common=False, full_motion=False):
    """Bounded constructive plans. Labels are the explicit planned cell union."""
    for attempt in range(96):
        if common:
            end = rng.choice(palette, size=(2, 1), replace=False)
            blocks = []
            for i in range(n):
                for trial in range(64):
                    b = _strip(rng, int(rng.integers(4, 9)), palette)
                    b[:, :1] = b[:, -1:] = end
                    proposed = blocks + [b]
                    if (_asymmetric(b) and all(_key(b) != _key(a) for a in blocks)
                            and _common_ports(proposed)):
                        blocks.append(b)
                        break
                else:
                    break
            if len(blocks) != n:
                continue
            steps = [(0, 1)] * (n-1)
        else:
            motions = [int(rng.integers(-1, 2)) for _ in range(n-1)]
            if full_motion:
                motions[:3] = [-1, 0, 1]
                rng.shuffle(motions)
            blocks = [_strip(rng, int(rng.integers(3, 9)), palette)]
            union = _cells(blocks[0], (0, 0))
            positions, steps = [(0, 0)], []
            for index, motion in enumerate(motions):
                previous = blocks[-1]
                options = [(length, k) for length in range(3, 9)
                           for k in range(1 if motion == 0 else 2,
                                          min(4, previous.shape[1]-1, length-1)+1)
                           if positions[-1][1]+previous.shape[1]-k+length
                           <= 30-(n-index-2)]
                if not options:
                    break
                for trial in range(96):
                    length, k = options[int(rng.integers(len(options)))]
                    pos = positions[-1][0]+motion, positions[-1][1]+previous.shape[1]-k
                    b = _strip(rng, length, palette)
                    for r in range(2):
                        for c in range(length):
                            p = pos[0]+r, pos[1]+c
                            if p in union:
                                b[r, c] = union[p]
                    candidates = _candidates(previous, b, positions[-1], union)
                    if candidates == [(motion, k, pos)]:
                        blocks.append(b)
                        steps.append((motion, k))
                        positions.append(pos)
                        _add(union, b, pos)
                        break
                else:
                    break
            if len(blocks) != n or not any(_asymmetric(b) for b in blocks):
                continue
            if full_motion and len({k for d, k in steps}) < 2:
                continue
        # Explicit planned coordinates provide labels, separate from matching.
        expected_union, planned = {}, [(0, 0)]
        _add(expected_union, blocks[0], (0, 0))
        for a, b, (motion, k) in zip(blocks, blocks[1:], steps):
            last = planned[-1]
            pos = last[0]+motion, last[1]+a.shape[1]-k
            planned.append(pos)
            _add(expected_union, b, pos)
        try:
            expected = _render(expected_union, background)
            actual, recovered, recovered_steps = _assemble(blocks, background)
        except ValueError:
            continue
        if not np.array_equal(actual, expected) or recovered != planned or recovered_steps != steps:
            raise AssertionError("Construction and pixel matching disagree")
        if common and np.array_equal(_assemble(blocks[::-1], background)[0], expected):
            continue
        return blocks, expected
    raise RuntimeError("Bounded ribbon construction exhausted")


def _gaps(rng, count, total):
    # count mandatory one-cell gaps; retain only O(count) small integers.
    return (1 + rng.multinomial(total-count, rng.dirichlet(np.ones(count)))).tolist()


def _pack(rng, blocks, roles, config, witness=False):
    n = len(blocks)
    lengths = [b.shape[1] for b in blocks]
    partitions = []
    min_lanes = 2 if witness else 1
    for lanes in range(min_lanes, min(n, 8)+1):
        for cuts in combinations(range(1, n), lanes-1):
            bounds = (0,) + cuts + (n,)
            groups = [lengths[bounds[i]:bounds[i+1]] for i in range(lanes)]
            needed = max(sum(g)+len(g)+1 for g in groups)
            if needed <= 29 and (not witness or max(map(len, groups)) >= 2):
                partitions.append((bounds, needed))
    bounds, needed = partitions[int(rng.integers(len(partitions)))]
    counts = [b-a for a, b in zip(bounds, bounds[1:])]
    lanes = len(counts)
    # Choose compact or sparse margins without exceeding the 30-cell canvas.
    fast_extent = int(rng.integers(needed, 30))
    slow_min = 3*lanes+1
    slow_extent = int(rng.integers(slow_min, 30))
    bg, origin, fast, slow = roles
    grid = np.full((slow_extent+1, fast_extent+1), bg, dtype=np.int8)
    grid[0, 0] = origin
    grid[0, 1:] = np.where(np.arange(1, fast_extent+1) % 2, fast, origin)
    grid[1:, 0] = slow
    vgaps = _gaps(rng, lanes+1, slow_extent-2*lanes)
    v, index = 1+vgaps[0], 0
    reverse_thickness = sum(config) % 2 == 1
    for lane, count in enumerate(counts):
        group = blocks[index:index+count]
        ugaps = _gaps(rng, count+1, fast_extent-sum(b.shape[1] for b in group))
        u = 1+ugaps[0]
        for j, block in enumerate(group):
            grid[v:v+2, u:u+block.shape[1]] = block[::-1] if reverse_thickness else block
            u += block.shape[1]+ugaps[j+1]
        index += count
        v += 2+vgaps[lane+1]
    axis, bottom, right = config
    if axis:
        grid = grid.T
    if bottom:
        grid = grid[::-1]
    if right:
        grid = grid[:, ::-1]
    return grid.copy()


def _integer(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    configs = [(a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True)]
    first = configs[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    held = [(first[0], second[1], second[2]), (second[0], first[1], first[2])]
    available = [c for c in configs if c not in held]
    train_configs = [first, second] + [available[int(rng.integers(len(available)))]
                                      for _ in range(nt-2)]
    puzzle, used = {"train": [], "test": []}, set()
    first_background = None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            config = train_configs[i] if split == "train" else held[int(rng.integers(2))]
            common = split == "train" and (i == 0 or (i >= 2 and rng.random() < .25))
            n = int(rng.integers(3, 6) if split == "train" else rng.integers(6, 9))
            if split == "train" and i == 1:
                n = int(rng.integers(4, 6))
            for retry in range(12):
                roles = tuple(map(int, rng.choice(10, 4, replace=False)))
                if split == "train" and i == 1 and roles[0] == first_background:
                    roles = ((roles[0]+1) % 10,) + roles[1:]
                    if len(set(roles)) != 4:
                        continue
                size = int(rng.integers(3, 7))
                palette = rng.choice([c for c in range(10) if c != roles[0]], size, replace=False)
                blocks, out = _ribbon(rng, n, palette, roles[0], common,
                                      full_motion=not common and (split == "test" or i == 1))
                inp = _pack(rng, blocks, roles, config, witness=common)
                if _key(inp) not in used:
                    used.add(_key(inp))
                    puzzle[split].append({"input": inp, "output": out})
                    if split == "train" and i == 0:
                        first_background = roles[0]
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _ordered_witness(descriptor, output):
    blocks = descriptor["blocks"]
    if len(descriptor["counts"]) < 2 or max(descriptor["counts"]) < 2:
        return False
    if not _common_ports(blocks) or not all(_asymmetric(b) for b in blocks):
        return False
    bg = descriptor["background"]
    groups, start = [], 0
    for count in descriptor["counts"]:
        groups.append(blocks[start:start+count])
        start += count
    alternatives = [blocks[::-1], [b for group in groups for b in group[::-1]],
                    [b for group in groups[::-1] for b in group]]
    return all(not np.array_equal(_assemble(alt, bg)[0], output) for alt in alternatives)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, descriptors, motion_witness, order_witness = set(), {"train": [], "test": []}, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            d = _decode(inp)
            expected, positions, seams = _assemble(d["blocks"], d["background"])
            assert np.array_equal(out, expected), "Incorrect sewn union or crop"
            assert not np.array_equal(inp, out)
            payload_colors = set().union(*(set(map(int, b.flat)) for b in d["blocks"]))
            assert set(map(int, out.flat)) - {d["background"]} == payload_colors
            # Every piece retains its pixels at its recovered coordinates.
            minr, minc = min(r for r, c in positions), min(c for r, c in positions)
            for block, (r, c) in zip(d["blocks"], positions):
                assert np.array_equal(out[r-minr:r-minr+2, c-minc:c-minc+block.shape[1]], block)
            d["seams"] = seams
            descriptors[split].append(d)
            if split == "train":
                order_witness |= _ordered_witness(d, out)
                motion_witness |= ({x for x, k in seams} == {-1, 0, 1}
                                   and len({k for x, k in seams}) >= 2
                                   and any(_asymmetric(b) for b in d["blocks"]))
    assert order_witness, "Training must expose keyed order among interchangeable seams"
    assert motion_witness, "Training must expose all three displacements and varied seams"
    tr, te = descriptors["train"], descriptors["test"]
    assert {d["configuration"][0] for d in tr} == {0, 1}
    assert {d["configuration"][1] for d in tr} == {False, True}
    assert {d["configuration"][2] for d in tr} == {False, True}
    assert len({d["background"] for d in tr}) >= 2
    assert max(len(d["blocks"]) for d in tr) < min(len(d["blocks"]) for d in te)
    assert not ({d["configuration"] for d in tr} & {d["configuration"] for d in te})


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and all 30 legal counts; retain only aggregate metrics."""
    shapes, out_shapes, configs, counts, lanes, lengths, seam_types = (set() for _ in range(7))
    backgrounds, palettes, signatures, contours = (set() for _ in range(4))
    mixed_parity = older_hits = total = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                d = _decode(pair["input"])
                actual, positions, seams = _assemble(d["blocks"], d["background"])
                assert np.array_equal(actual, pair["output"])
                shapes.add(pair["input"].shape)
                out_shapes.add(actual.shape)
                configs.add(d["configuration"])
                counts.add(len(d["blocks"]))
                lanes.add(d["counts"])
                widths = tuple(b.shape[1] for b in d["blocks"])
                lengths.update(widths)
                mixed_parity += len({w % 2 for w in widths}) > 1
                seam_types.update(seams)
                backgrounds.add(d["background"])
                palettes.add(len(set().union(*(set(map(int, b.flat)) for b in d["blocks"]))))
                contours.add(tuple(r for r, c in positions))
                for i in range(2, len(positions)):
                    cells = set(_cells(d["blocks"][i], positions[i]))
                    older_hits += any(cells & set(_cells(d["blocks"][j], positions[j]))
                                      for j in range(i-1))
        # Global rotations of the input rotate the key too, not the answer.
        if seed == 0:
            for p in puzzle["train"] + puzzle["test"]:
                for k in range(4):
                    assert np.array_equal(solve(np.rot90(p["input"], k)), p["output"])
                mapping = np.array([6, 4, 9, 0, 7, 1, 8, 3, 5, 2], dtype=np.int8)
                assert np.array_equal(solve(mapping[p["input"]]), mapping[p["output"]])
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(out_shapes) >= 35
    assert len(configs) == 8 and counts == set(range(3, 9))
    assert len(lanes) >= 25 and lengths == set(range(3, 9))
    assert backgrounds == set(range(10)) and len(palettes) >= 4
    assert len(contours) >= 60 and mixed_parity >= 100
    assert {d for d, k in seam_types} == {-1, 0, 1}
    assert {k for d, k in seam_types} >= {1, 2, 3, 4}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    assert _puzzle_digest(generate_puzzle(23, 6, 3)) == _puzzle_digest(generate_puzzle(23, 6, 3))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for bad in (-1, True, 1.0):
        try:
            generate_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal seed accepted")
    # Explicit one-row and two-row seams, independent of random construction.
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
    b = np.array([[7, 8, 2], [4, 1, 6]], dtype=np.int8)
    assert [(d, k) for d, k, p in _candidates(a, b)] == [(1, 2)]
    out, positions, seams = _assemble([a, b], 0)
    assert positions == [(0, 0), (1, 2)] and out.shape == (3, 5)
    assert np.array_equal(out, np.array([[1, 2, 3, 4, 0], [5, 6, 7, 8, 2],
                                       [0, 0, 4, 1, 6]], dtype=np.int8))
    up = np.array([[9, 9, 6], [3, 4, 2]], dtype=np.int8)
    level = np.array([[4, 9, 3], [8, 1, 2]], dtype=np.int8)
    fixtures = [([a, b], out),
                ([a, up], np.array([[0, 0, 9, 9, 6], [1, 2, 3, 4, 2],
                                    [5, 6, 7, 8, 0]], dtype=np.int8)),
                ([a, level], np.array([[1, 2, 3, 4, 9, 3],
                                       [5, 6, 7, 8, 1, 2]], dtype=np.int8)),
                ([np.array([[1, 1], [2, 2]], dtype=np.int8)]*2,
                 np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int8))]
    fixture_rng = np.random.default_rng(9182)
    for blocks, expected in fixtures:
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    inp = _pack(fixture_rng, blocks, (0, 9, 8, 7), (axis, bottom, right))
                    assert np.array_equal(solve(inp), expected)
    # A pairwise match can be disallowed by material from an older piece.
    older = np.array([[9, 5, 6, 7], [1, 1, 1, 9]], dtype=np.int8)
    assert [(d, k) for d, k, p in _candidates(older, a)] == [(-1, 3)]
    try:
        _assemble([older, a, b], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("A conflict with older material was ignored")
    # Parsing and matching support lengths beyond the sampled 3..8 subset.
    wide = fixture_rng.integers(1, 10, size=(2, 27), dtype=np.int8)
    for trial in range(64):
        extension = fixture_rng.integers(1, 10, size=(2, 4), dtype=np.int8)
        extension[:, :3] = wide[:, -3:]
        if [(d, k) for d, k, p in _candidates(wide, extension)] == [(0, 3)]:
            break
    else:
        raise AssertionError("Wide fixture construction failed")
    expected = np.concatenate((wide, extension[:, 3:]), axis=1)
    for axis in (0, 1):
        inp = _pack(fixture_rng, [wide, extension], (0, 9, 8, 7), (axis, True, False))
        assert np.array_equal(solve(inp), expected)
    try:
        _assemble([np.ones((2, 4), dtype=np.int8)]*2, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous repeated bars accepted")
    broken = generate_puzzle(101, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt label accepted")
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(out_shapes),
            "configurations": len(configs), "piece_counts": sorted(counts),
            "lane_populations": len(lanes), "contours": len(contours),
            "seam_types": sorted(seam_types), "backgrounds": sorted(backgrounds),
            "palette_sizes": sorted(palettes), "mixed_parity_pairs": mixed_parity,
            "nonconsecutive_intersections": older_hits}
