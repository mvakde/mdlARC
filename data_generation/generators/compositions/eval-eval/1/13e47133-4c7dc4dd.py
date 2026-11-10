"""Contour rhythm transplants: a visible swatch teaches a target's shell word.

No parent modules, files, seeds, or generation metadata are used by solve().
"""
from collections import Counter, deque
from itertools import product

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    unseen = mask.copy()
    result = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if not unseen[sr, sc]:
            continue
        component = np.zeros(mask.shape, bool)
        queue = [(sr, sc)]
        unseen[sr, sc] = False
        for r, c in queue:
            component[r, c] = True
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and unseen[nr, nc]:
                    unseen[nr, nc] = False
                    queue.append((nr, nc))
        result.append(component)
    return result


def _erosion_depth(mask, diagonal=True):
    """Input-reading path: simultaneous stripping, including off-grid boundary."""
    active = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    level = 0
    while active.any():
        padded = np.pad(active, 1)
        inner = active.copy()
        for dr, dc in DIR8 if diagonal else DIR4:
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[active & ~inner] = level
        active = inner
        level += 1
    return depth


def _bfs_depth(mask):
    """Construction path, checked against iterative erosion in self_test."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                    and mask[r + dr, c + dc]) for dr, dc in DIR8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and depth[nr, nc] < 0:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))
    return depth


def _paint(body, wall, palette, word, depth_fn=_bfs_depth):
    result = body.copy()
    colors = np.asarray(palette, dtype=np.int8)[list(word)]
    for region in _components(body != wall):
        depth = depth_fn(region)
        result[region] = colors[depth[region] % len(word)]
    return result


def _read_panel(grid, mask, wall, neutral_hint):
    rr, cc = np.nonzero(mask)
    r, c = int(rr.min()), int(cc.min())
    h, w = int(rr.max() - r + 1), int(cc.max() - c + 1)
    if h < 5 or w < 5:
        return None
    p = grid[r:r + h, c:c + w]
    if not (np.all(p[0] == wall) and np.all(p[-1] == wall)
            and np.all(p[:, 0] == wall) and np.all(p[:, -1] == wall)
            and np.all(p[2] == wall)):
        return None
    header = p[1, 1:-1]
    possibilities = []
    # The palette is anchored at either end; its padding identifies read direction.
    for row in (header, header[::-1]):
        neutral = int(row[-1])
        if neutral == wall or neutral != neutral_hint:
            continue
        occupied = np.flatnonzero(row != neutral)
        k = len(occupied)
        if k in (2, 3) and np.array_equal(occupied, np.arange(k)):
            palette = tuple(map(int, row[:k]))
            if wall not in palette and len(set(palette)) == k:
                possibilities.append((neutral, palette))
    if len(possibilities) != 1:
        return None
    neutral, palette = possibilities[0]
    body = p[3:-1, 1:-1].copy()
    complete = np.all(np.isin(body, palette))
    blank = np.all(np.isin(body, (neutral, wall))) and np.any(body == neutral)
    if complete == blank:
        return None
    return dict(box=(r, c, h, w), body_box=(r + 3, c + 1, h - 4, w - 2),
                body=body, palette=palette, neutral=neutral, complete=bool(complete))


def _scene(input_grid):
    original = _grid(input_grid)
    candidates = []
    for turn in range(4):
        grid = np.rot90(original, turn)
        for wall in map(int, np.unique(grid)):
            walls = _components(grid == wall)
            if len(walls) != 2:
                continue
            outside = np.ones(grid.shape, bool)
            disjoint = True
            for mask in walls:
                rr, cc = np.nonzero(mask)
                r, c, h, w = int(rr.min()), int(cc.min()), int(rr.max() + 1), int(cc.max() + 1)
                if not outside[r:h, c:w].all():
                    disjoint = False
                outside[r:h, c:w] = False
            exterior = np.unique(grid[outside])
            if not disjoint or len(exterior) != 1:
                continue
            neutral = int(exterior[0])
            panels = [_read_panel(grid, m, wall, neutral) for m in walls]
            if any(p is None for p in panels):
                continue
            if (panels[0]["neutral"] != panels[1]["neutral"]
                    or len(panels[0]["palette"]) != len(panels[1]["palette"])
                    or sum(p["complete"] for p in panels) != 1):
                continue
            ref = next(p for p in panels if p["complete"])
            target = next(p for p in panels if not p["complete"])
            candidates.append(dict(grid=grid.copy(), turn=turn, wall=wall,
                                   neutral=neutral, ref=ref, target=target))
    if len(candidates) != 1:
        raise ValueError(f"expected one readable swatch/target scene, got {len(candidates)}")
    scene = candidates[0]
    ref = scene["ref"]
    depth = _erosion_depth(np.ones(ref["body"].shape, bool))
    sequence = []
    for d in range(int(depth.max()) + 1):
        colors = np.unique(ref["body"][depth == d])
        if len(colors) != 1:
            raise ValueError("swatch colors must be constant on each contour layer")
        sequence.append(ref["palette"].index(int(colors[0])))
    words = [tuple(sequence[:p]) for p in range(1, 5)
             if 2 * p <= len(sequence)
             and all(sequence[d] == sequence[d % p] for d in range(len(sequence)))]
    if not words or set(sequence) != set(range(len(ref["palette"]))):
        raise ValueError("swatch must show two cycles of a length 2..4 word using every role")
    # All compatible words must agree indefinitely, not just inside the swatch.
    expansions = {tuple(word[d % len(word)] for d in range(12)) for word in words}
    if len(expansions) != 1:
        raise ValueError("ambiguous contour rhythm")
    scene["word"] = min(words, key=len)
    return scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _scene(input_grid)
    target = scene["target"]
    body = _paint(target["body"], scene["wall"], target["palette"],
                  scene["word"], _erosion_depth)
    return np.rot90(body, -scene["turn"]).copy()


def _words(k):
    result = []
    for p in range(k, 5):
        for word in product(range(k), repeat=p):
            if set(word) != set(range(k)):
                continue
            if any(p % q == 0 and all(word[d] == word[d % q] for d in range(p))
                   for q in range(1, p)):
                continue
            result.append(word)
    return result


def _pocket(rng, mask, rooms):
    options = []
    for room in rooms:
        for transpose in (False, True):
            a, b, c, d = room if not transpose else (room[2], room[3], room[0], room[1])
            if b - a >= 7 and d - c >= 5:
                for side in (-1, 1):
                    options.append((transpose, a, b, c, d, side))
    if not options:
        return False
    transpose, a, b, c, d, side = options[int(rng.integers(len(options)))]
    work = mask.T if transpose else mask
    top = int(rng.integers(a + 1, b - 4))
    bottom = int(rng.integers(top + 4, b))
    distance = int(rng.integers(3, min(6, d - c - 1) + 1))
    edge = c if side == 1 else d
    far = edge + side * distance
    lo, hi = sorted((edge, far))
    if work[top:bottom + 1, lo:hi + 1].any():
        return False
    work[top, lo:hi + 1] = work[bottom, lo:hi + 1] = True
    work[top:bottom + 1, far] = True
    return True


def _step(rng, mask):
    for _ in range(35):
        work = mask.T if rng.integers(2) else mask
        points = np.argwhere(work)
        if not len(points):
            return
        r, c = map(int, points[int(rng.integers(len(points)))])
        end = r + int(rng.integers(3, 8))
        shift = int(rng.choice((-2, -1, 1, 2)))
        lo, hi = sorted((c, c + shift))
        if r < 1 or end + 1 >= work.shape[0] or lo < 1 or hi + 1 >= work.shape[1]:
            continue
        block = work[r - 1:end + 2, lo - 1:hi + 2]
        expected = np.zeros_like(block)
        expected[:, c - lo + 1] = True
        if np.array_equal(block, expected):
            work[r + 1:end, c] = False
            work[r, lo:hi + 1] = work[end, lo:hi + 1] = True
            work[r:end + 1, c + shift] = True
            return


def _target_walls(rng, shape, test=False, diagnostic=False):
    mask = np.zeros(shape, bool)
    h, w = shape
    rooms = [(0, h - 1, 0, w - 1)]
    count = 1 if diagnostic else int(rng.integers(1, 5 if test else 4))
    while len(rooms) < count:
        choices = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, extent in ((0, b - a + 1), (1, d - c + 1)) if extent >= 7]
        if not choices:
            break
        i, axis = choices[int(rng.integers(len(choices)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + 3, b - 2))
            mask[cut, c:d + 1] = True
            rooms.extend(((a, cut - 1, c, d), (cut + 1, b, c, d)))
        else:
            cut = int(rng.integers(c + 3, d - 2))
            mask[a:b + 1, cut] = True
            rooms.extend(((a, b, c, cut - 1), (a, b, cut + 1, d)))
    pocket_budget = 1 if diagnostic else int(rng.integers(0, 4 if test else 3))
    for _ in range(pocket_budget):
        _pocket(rng, mask, rooms)
    for _ in range(int(rng.integers(0, 4 if test else 3))):
        _step(rng, mask)
    if rng.integers(2):
        mask = mask[::-1].copy()
    if rng.integers(2):
        mask = mask[:, ::-1].copy()
    return mask


def _geometry_witness(body, wall, palette, word):
    colors = np.asarray(palette)[list(word)]
    rectangular = manhattan = False
    for region in _components(body != wall):
        rr, cc = np.nonzero(region)
        depth = _erosion_depth(region)
        box = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                 cc - cc.min(), cc.max() - cc))
        truth = colors[depth[region] % len(word)]
        rectangular |= bool(np.any(truth != colors[box % len(word)]))
        four_depth = _erosion_depth(region, False)
        manhattan |= bool(np.any(truth != colors[four_depth[region] % len(word)]))
    return rectangular and manhattan


def _roles(rng, k):
    colors = list(map(int, rng.permutation(10)))
    neutral, wall = colors[:2]
    ref = tuple(colors[2:2 + k])
    mode = int(rng.integers(3))
    if mode == 0:
        target = ref
    elif mode == 1:
        target = (ref[0], *colors[2 + k:2 + k + k - 1])
    else:
        target = tuple(colors[2 + k:2 + 2 * k])
    return neutral, wall, ref, tuple(target)


def _assemble(rng, ref_shape, target_mask, roles, words):
    neutral, wall, ref_palette, target_palette = roles
    rh, rw = ref_shape
    th, tw = target_mask.shape
    sizes = [(rh + 4, rw + 2), (th + 4, tw + 2)]
    gap = int(rng.integers(1, 31 - sum(s[1] for s in sizes)))
    width_min = sum(s[1] for s in sizes) + gap
    height_min = max(s[0] for s in sizes)
    height = int(rng.integers(height_min, 31))
    width = int(rng.integers(width_min, 31))
    x = int(rng.integers(width - width_min + 1))
    order = list(map(int, rng.permutation(2)))
    positions = [None, None]
    for i in order:
        positions[i] = (int(rng.integers(height - sizes[i][0] + 1)), x)
        x += sizes[i][1] + gap
    base = np.full((height, width), neutral, dtype=np.int8)
    bodies = [np.full(ref_shape, neutral, dtype=np.int8),
              np.where(target_mask, wall, neutral).astype(np.int8)]
    for i, ((r, c), (h, w), palette) in enumerate(zip(positions, sizes, (ref_palette, target_palette))):
        base[r:r + h, c:c + w] = neutral
        base[r, c:c + w] = base[r + h - 1, c:c + w] = wall
        base[r:r + h, c] = base[r:r + h, c + w - 1] = wall
        base[r + 2, c:c + w] = wall
        base[r + 1, c + 1:c + 1 + len(palette)] = palette
        base[r + 3:r + h - 1, c + 1:c + w - 1] = bodies[i]
    turn, reflect = int(rng.integers(4)), bool(rng.integers(2))
    def orient(a):
        return np.rot90(a[:, ::-1] if reflect else a, turn).copy()
    pairs = []
    r, c = positions[0]
    for word in words:
        inp = base.copy()
        inp[r + 3:r + 3 + rh, c + 1:c + 1 + rw] = _paint(bodies[0], wall, ref_palette, word)
        output = _paint(bodies[1], wall, target_palette, word)
        pairs.append(dict(input=orient(inp), output=orient(output)))
    return pairs


def _make_pairs(rng, test=False, diagnostic=False):
    k = 2 if diagnostic else int(rng.integers(2, 4))
    roles = _roles(rng, k)
    words = [(0, 0, 1), (0, 1, 1)] if diagnostic else [_words(k)[int(rng.integers(len(_words(k))))]]
    base = 4 * len(words[0]) - 1  # min side that exposes two complete cycles
    rw = int(rng.integers(base, base + 3))
    rh = int(rng.integers(base, base + 7))
    tw = int(rng.integers(9 if diagnostic else 7, 26 - rw))
    th = int(rng.integers(19 if test else 13 if diagnostic else 9,
                          27 if test else 21 if diagnostic else 23))
    for _ in range(40):
        mask = _target_walls(rng, (th, tw), test, diagnostic)
        body = np.where(mask, roles[1], roles[0]).astype(np.int8)
        if not diagnostic or all(_geometry_witness(body, roles[1], roles[3], word) for word in words):
            return _assemble(rng, (rh, rw), mask, roles, words)
    # Guaranteed diagnostic geometry; this is still a pocket in the same family.
    mask = np.zeros((18, 12), bool)
    mask[4, :4] = mask[12, :4] = True
    mask[4:13, 3] = True
    body = np.where(mask, roles[1], roles[0]).astype(np.int8)
    assert all(_geometry_witness(body, roles[1], roles[3], word) for word in words)
    return _assemble(rng, (13, 11), mask, roles, words)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": _make_pairs(rng, diagnostic=True), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", nt - 2), ("test", ne)):
        for _ in range(count):
            for _attempt in range(30):
                pair = _make_pairs(rng, test=split == "test")[0]
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded search exhausted for distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_scenes = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            s = _scene(inp)
            target = s["target"]
            canonical_out = np.rot90(out, s["turn"])
            assert canonical_out.shape == target["body"].shape
            assert np.array_equal(canonical_out == s["wall"], target["body"] == s["wall"])
            assert set(map(int, np.unique(out))) <= set(target["palette"]) | {s["wall"]}
            assert np.array_equal(solve(inp), out), "wrong transferred contour colors"
            # Check the result layer by layer, including each independently reset chamber.
            role_word = np.asarray(target["palette"])[list(s["word"])]
            for region in _components(target["body"] != s["wall"]):
                depth = _erosion_depth(region)
                for d in range(int(depth.max()) + 1):
                    assert np.all(canonical_out[region & (depth == d)] == role_word[d % len(role_word)])
            if split == "train":
                train_scenes.append(s)
            else:
                assert 19 <= target["body"].shape[0] <= 26, "missing test scale shift"
    a, b = train_scenes[:2]
    assert a["turn"] == b["turn"] and a["ref"]["box"] == b["ref"]["box"]
    allowed = np.zeros(a["grid"].shape, bool)
    r, c, h, w = a["ref"]["body_box"]
    allowed[r:r + h, c:c + w] = True
    assert np.array_equal(a["grid"][~allowed], b["grid"][~allowed]), "twins must change only the swatch"
    assert a["word"] != b["word"]
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])
    for s in (a, b):
        assert _geometry_witness(s["target"]["body"], s["wall"], s["target"]["palette"], s["word"])
        raw = tuple(range(len(s["target"]["palette"])))
        assert not np.array_equal(_paint(s["target"]["body"], s["wall"], s["target"]["palette"], raw),
                                  np.rot90(puzzle["train"][0 if s is a else 1]["output"], s["turn"]))


def self_test():
    shapes, geometries, palettes, words, turns, key_sizes, signatures = (set() for _ in range(7))
    counts, concave, extrapolated = Counter(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for split in ("train", "test") for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _scene(pair["input"])
                target, ref = s["target"], s["ref"]
                mask = target["body"] != s["wall"]
                shapes.add((pair["input"].shape, pair["output"].shape))
                geometries.add(_key(mask.astype(np.int8)))
                palettes.add((s["neutral"], s["wall"], ref["palette"], target["palette"]))
                words.add(s["word"])
                turns.add(s["turn"])
                key_sizes.add(len(target["palette"]))
                regions = _components(mask)
                counts[len(regions)] += 1
                max_depth = -1
                for region in regions:
                    depth = _erosion_depth(region)
                    assert np.array_equal(depth, _bfs_depth(region))
                    rr, cc = np.nonzero(region)
                    concave += region.sum() < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
                    max_depth = max(max_depth, int(depth.max()))
                extrapolated += max_depth > (min(ref["body"].shape) - 1) // 2
    assert len(signatures) == 42 and len(shapes) >= 35 and len(geometries) >= 80
    assert len(palettes) >= 80 and len(words) >= 12 and len(counts) >= 4
    assert turns == {0, 1, 2, 3} and key_sizes == {2, 3} and concave >= 80
    assert extrapolated > 0
    # All legal explicit counts, determinism, and invalid types/counts.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(9127, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    p = generate_puzzle(0, 2, 1)
    q = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for split in p for a, b in zip(p[split], q[split]) for k in a)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Geometry and key reading must remain equivariant under every dihedral view.
    for pair in p["train"] + p["test"]:
        for turn in range(4):
            for flip in (False, True):
                def orient(a):
                    return np.rot90(a[:, ::-1] if flip else a, turn).copy()
                assert np.array_equal(solve(orient(pair["input"])), orient(pair["output"]))
    # Shallow bodies are valid even though the normal sampler prefers larger plans.
    for k, shape, word in ((2, (1, 3), (1, 0)), (3, (1, 4), (2, 0, 1))):
        roles = (0, 1, tuple(range(2, 2 + k)), tuple(range(5, 5 + k)))
        side = 4 * len(word) - 1
        pair = _assemble(np.random.default_rng(400 + k), (side, side),
                         np.zeros(shape, bool), roles, [word])[0]
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # Returned-data corruption check, rather than trusting construction metadata.
    damaged = {split: [{k: np.array(v).copy() for k, v in pair.items()} for pair in p[split]] for split in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("changed output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shape_pairs=len(shapes),
                target_geometries=len(geometries), palettes=len(palettes), rhythms=len(words),
                region_counts=dict(sorted(counts.items())), concave_regions=int(concave),
                extrapolated_pairs=int(extrapolated))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
