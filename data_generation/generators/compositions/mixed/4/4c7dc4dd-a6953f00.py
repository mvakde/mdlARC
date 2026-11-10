"""Framed parity-crop analogies, implementing the saved proposal literally.

All semantics are input-visible. See composition.md for the unavoidable limit:
the required fixed parity mapping permits a reference-bypassing shortcut.
No file I/O occurs during generation, validation, solving, or self_test.
"""
from itertools import permutations
import hashlib

import numpy as np


HOST_SEEDS = (
    0, 170617409, 756978981, 649338518, 215726548, 971198450,
    730023604, 221846388, 612857927, 118186517, 547519342,
    676199911, 858227516, 818823239, 947141776, 975454875,
    536779605, 201406002, 261447189, 357083096, 542358735,
    843822040, 611821489, 698744109, 114886284, 812959804,
    906478019, 976978371, 841695657, 784235329, 612313036,
    165971019, 241784045, 666396623, 626267593, 177439506,
    252137674, 840819213, 947710166, 532900717, 922144950,
    182468127,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color outside 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _corner(a, left):
    return a[:2, :2].copy() if left else a[:2, -2:].copy()


def _literal_crop(a):
    # Used to construct labels and to measure the saved proposal's shortcut.
    return _corner(a, bool(a.shape[0] % 2))


def _outline(a, r, c, h, w, color):
    a[r, c:c+w] = a[r+h-1, c:c+w] = color
    a[r:r+h, c] = a[r:r+h, c+w-1] = color


def _rings(a):
    """Exact isolated monochrome outline components, independent of color IDs."""
    h, w = a.shape
    visited = np.zeros(a.shape, dtype=bool)
    rings = []
    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0]:
                continue
            color = int(a[r0, c0])
            visited[r0, c0] = True
            stack, points = [(r0, c0)], []
            while stack:
                r, c = stack.pop()
                points.append((r, c))
                for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= y < h and 0 <= x < w and not visited[y, x]
                            and a[y, x] == color):
                        visited[y, x] = True
                        stack.append((y, x))
            top = min(p[0] for p in points)
            bottom = max(p[0] for p in points)
            left = min(p[1] for p in points)
            right = max(p[1] for p in points)
            bh, bw = bottom-top+1, right-left+1
            if (bh >= 4 and bw >= 4 and len(points) == 2*bh + 2*bw - 4
                    and all(r in (top, bottom) or c in (left, right) for r, c in points)):
                rings.append((top, left, bh, bw, color))
    return rings


def _inside(inner, outer):
    r, c, h, w = inner[:4]
    y, x, bh, bw = outer[:4]
    return y < r and x < c and r+h < y+bh and c+w < x+bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r+h <= y or y+bh <= r or c+w <= x or x+bw <= c


def _interior(a, box):
    r, c, h, w = box[:4]
    return a[r+1:r+h-1, c+1:c+w-1].copy()


def _extract(a):
    """Find the unique three-group, six-panel association in visible pixels."""
    rings = _rings(a)
    scenes = []
    for panel_color in sorted({b[4] for b in rings}):
        panels = [b for b in rings if b[4] == panel_color and b[2] == b[3]]
        # Structural colors are reserved; all six panel frames share one color.
        if len(panels) != 6:
            continue
        for group_color in sorted({b[4] for b in rings} - {panel_color}):
            boxes = [b for b in rings if b[4] == group_color]
            if len(boxes) != 3 or any(not _disjoint(x, y) for x, y in permutations(boxes, 2)):
                continue
            groups, used, neutrals = [], [], []
            for box in boxes:
                members = [p for p in panels if _inside(p, box)]
                small = [p for p in members if p[2:4] == (4, 4)]
                large = [p for p in members if 5 <= p[2] <= 16]
                if (len(members) != 2 or len(small) != 1 or len(large) != 1
                        or not _disjoint(*members)):
                    break
                padding = np.ones((box[2]-2, box[3]-2), dtype=bool)
                for p in members:
                    r, c = p[0]-box[0]-1, p[1]-box[1]-1
                    padding[r:r+p[2], c:c+p[3]] = False
                values = np.unique(_interior(a, box)[padding])
                if len(values) != 1 or int(values[0]) in (panel_color, group_color):
                    break
                neutrals.append(int(values[0]))
                groups.append(dict(box=box, source_box=large[0], companion_box=small[0],
                                   source=_interior(a, large[0]), companion=_interior(a, small[0])))
                used.extend(members)
            if len(groups) != 3 or len(set(used)) != 6 or len(set(neutrals)) != 1:
                continue
            neutral = neutrals[0]
            missing = [i for i, g in enumerate(groups) if np.all(g["companion"] == neutral)]
            if len(missing) != 1:
                continue
            structural = {panel_color, group_color}
            if any(any(int(v) in structural for v in np.unique(g[k]))
                   for g in groups for k in ("source", "companion")):
                continue
            scenes.append(dict(groups=groups, target=missing[0], neutral=neutral,
                               panel_color=panel_color, group_color=group_color))
    _require(len(scenes) == 1, f"expected one frame association, found {len(scenes)}")
    return scenes[0]


def _infer(scene):
    """Infer both code entries from the two complete in-input examples."""
    code = {}
    for i, group in enumerate(scene["groups"]):
        if i == scene["target"]:
            continue
        a, b = group["source"], group["companion"]
        matches = [left for left in (False, True) if np.array_equal(_corner(a, left), b)]
        _require(len(matches) == 1, "reference must identify exactly one top corner")
        parity = a.shape[0] % 2
        _require(parity not in code, "reference examples must cover both parities")
        code[parity] = matches[0]
    _require(code == {0: False, 1: True}, "reference code conflicts with odd-left/even-right")
    return code


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read frames, infer the witnessed parity code, and reconstruct the 2x2 blank."""
    scene = _extract(_grid(input_grid))
    code = _infer(scene)
    target = scene["groups"][scene["target"]]["source"]
    return _corner(target, code[target.shape[0] % 2])


def _grow(a):
    """Insert a duplicate middle row/column, retaining both top corner blocks."""
    cut = max(2, a.shape[0] // 2)
    b = np.concatenate((a[:, :cut], a[:, cut-1:cut], a[:, cut:]), axis=1)
    return np.concatenate((b[:cut], b[cut-1:cut], b[cut:]), axis=0)


def _texture(rng, side, palette, mode):
    rr, cc = np.indices((side, side))
    k = len(palette)
    if mode == 0:
        labels = rng.integers(k, size=(side, side))
    elif mode == 1:
        tile = rng.integers(k, size=(int(rng.integers(1, 4)), int(rng.integers(1, 4))))
        labels = tile[rr % tile.shape[0], cc % tile.shape[1]]
    elif mode == 2:
        axis = rr if rng.integers(2) else cc
        labels = (axis // int(rng.integers(1, 4)) + int(rng.integers(k))) % k
    elif mode == 3:
        labels = np.minimum.reduce((rr, cc, side-1-rr, side-1-cc)) % k
    elif mode == 4:
        labels = np.full((side, side), int(rng.integers(k)))
        for _ in range(int(rng.integers(1, 5))):
            h, w = map(int, rng.integers(1, min(4, side)+1, size=2))
            r, c = int(rng.integers(side-h+1)), int(rng.integers(side-w+1))
            labels[r:r+h, c:c+w] = int(rng.integers(k))
    elif mode == 5:
        labels = (rr + int(rng.integers(1, k+1))*cc + rr//2) % k
    else:
        labels = rng.integers(k, size=(side, side))
        labels[side//2:] = labels[:side-side//2][::-1]
    return np.asarray(palette[labels], dtype=np.int8)


def _pattern(rng, side, palette, mode, profile):
    a = _texture(rng, side, palette, mode)
    # No side=3 corner overwrites: the overlap must remain consistent.
    if side == 3 and 1 in profile and max(profile) > 1:
        # Overlapping candidates can still contrast a solid and a richer crop.
        if profile[0] == 1:
            a[:2, :2] = palette[0]
            a[:2, -1] = palette[1]
        else:
            a[:2, -2:] = palette[0]
            a[:2, 0] = palette[1]
    elif side >= 4:
        for left, richness in ((True, profile[0]), (False, profile[1])):
            chosen = rng.choice(palette, size=min(richness, len(palette)), replace=False)
            block = np.full(4, chosen[0], dtype=np.int8)
            block[:len(chosen)] = chosen
            rng.shuffle(block)
            c = 0 if left else side-2
            a[:2, c:c+2] = block.reshape(2, 2)
    if np.array_equal(a[:2, :2], a[:2, -2:]):
        # The outermost right cell never overlaps the upper-left block at n>=3.
        old = int(a[0, -1])
        a[0, -1] = next(int(x) for x in palette if x != old)
    return a


def _group(rng, source, companion, neutral, panel_color, group_color,
           orientation, minimal=False):
    """Build one logical association, varying spacing and member order."""
    arrays = [source, companion]
    if rng.integers(2):
        arrays.reverse()
    sizes = [x.shape[0]+2 for x in arrays]
    gap = 1 if minimal else int(rng.integers(1, 4))
    margin = 1 if minimal else int(rng.integers(1, 3))
    extra = 0 if minimal else int(rng.integers(3))
    span, cross = sum(sizes)+gap+2*margin, max(sizes)+2*margin+extra
    h, w = (cross, span) if orientation == 0 else (span, cross)
    a = np.full((h, w), neutral, dtype=np.int8)
    _outline(a, 0, 0, h, w, group_color)
    along = margin
    for content, size in zip(arrays, sizes):
        across = int(rng.integers(margin, cross-margin-size+1))
        r, c = (across, along) if orientation == 0 else (along, across)
        _outline(a, r, c, size, size, panel_color)
        a[r+1:r+size-1, c+1:c+size-1] = content
        along += size+gap
    return a


def _join_shape(a, b, axis, gap):
    return ((a[0]+b[0]+gap, max(a[1], b[1])) if axis == 0 else
            (max(a[0], b[0]), a[1]+b[1]+gap))


def _join(rng, a, b, axis, gap, neutral):
    shape = _join_shape(a.shape, b.shape, axis, gap)
    result = np.full(shape, neutral, dtype=np.int8)
    cross = 1-axis
    oa = int(rng.integers(shape[cross]-a.shape[cross]+1))
    ob = int(rng.integers(shape[cross]-b.shape[cross]+1))
    if axis == 0:
        result[:a.shape[0], oa:oa+a.shape[1]] = a
        result[a.shape[0]+gap:, ob:ob+b.shape[1]] = b
    else:
        result[oa:oa+a.shape[0], :a.shape[1]] = a
        result[ob:ob+b.shape[0], a.shape[1]+gap:] = b
    return result


def _pack(rng, groups, neutral):
    candidates = []
    for order in permutations(range(3)):
        for first_axis in (0, 1):
            for second_axis in (0, 1):
                for gap in (1, 2):
                    s = _join_shape(groups[order[0]].shape, groups[order[1]].shape, first_axis, gap)
                    s = _join_shape(s, groups[order[2]].shape, second_axis, gap)
                    if max(s) <= 30:
                        candidates.append((order, first_axis, second_axis, gap))
    if not candidates:
        return None
    order, ax1, ax2, gap = candidates[int(rng.integers(len(candidates)))]
    first = _join(rng, groups[order[0]], groups[order[1]], ax1, gap, neutral)
    core = _join(rng, first, groups[order[2]], ax2, gap, neutral)
    # Small independently allocated exterior margins, still within 30x30.
    dh = int(rng.integers(min(3, 30-core.shape[0])+1))
    dw = int(rng.integers(min(3, 30-core.shape[1])+1))
    r, c = int(rng.integers(dh+1)), int(rng.integers(dw+1))
    canvas = np.full((core.shape[0]+dh, core.shape[1]+dw), neutral, dtype=np.int8)
    canvas[r:r+core.shape[0], c:c+core.shape[1]] = core
    return canvas


def _make_pair(rng, target_side, mode, profile):
    colors = rng.permutation(10).astype(np.int8)
    neutral, panel_color, group_color = map(int, colors[:3])
    palette = colors[3:3+int(rng.integers(2, 8))]
    base_side = int(rng.integers(3, 6))
    ref = _pattern(rng, base_side, palette, mode, profile)
    other = _grow(ref)
    target_mode = int(rng.integers(7))
    target = _pattern(rng, target_side, palette, target_mode, profile)
    sources = [ref, other, target]
    companions = [_literal_crop(ref), _literal_crop(other), np.full((2, 2), neutral, dtype=np.int8)]
    # Geometry retries are bounded and never write intermediate scenes.
    for attempt in range(25):
        minimal = attempt == 24
        orientations = [0, 0, 0] if minimal else list(map(int, rng.integers(2, size=3)))
        groups = [_group(rng, a, b, neutral, panel_color, group_color, axis, minimal)
                  for a, b, axis in zip(sources, companions, orientations)]
        canvas = _pack(rng, groups, neutral)
        if canvas is not None:
            break
    else:
        raise RuntimeError("bounded geometry construction exhausted")
    # Outside the group outlines only, add a simple periodic carrier. It uses
    # data colors, never either reserved frame color, and is unrelated to crop.
    boxes = [b for b in _rings(canvas) if b[4] == group_color]
    exterior = np.ones(canvas.shape, dtype=bool)
    for r, c, h, w, _ in boxes:
        exterior[r:r+h, c:c+w] = False
    if rng.random() < .65:
        rr, cc = np.indices(canvas.shape)
        field = np.where((rr+cc) % int(rng.integers(2, 5)) == 0, palette[0], neutral)
        canvas[exterior] = field[exterior]
    return {"input": canvas, "output": _literal_crop(target)}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return exactly train/test pairs; hold out larger target-source squares."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    sides = [int(rng.choice([3, 5, 7])), int(rng.choice([4, 6]))]
    if a >= 4:
        sides += [int(rng.choice([v for v in (3, 5, 7) if v != sides[0]])),
                  int(rng.choice([v for v in (4, 6) if v != sides[1]]))]
    while len(sides) < a:
        sides.append(int(rng.integers(3, 8)))
    rng.shuffle(sides)
    test_sides = [int(rng.choice([8, 10, 12, 14])), int(rng.choice([9, 11, 13]))][:b]
    while len(test_sides) < b:
        test_sides.append(int(rng.integers(8, 15)))
    if b == 1:
        test_sides = [int(rng.integers(8, 15))]
    rng.shuffle(test_sides)
    puzzle, used = {"train": [], "test": []}, set()
    for split, sizes in (("train", sides), ("test", test_sides)):
        for index, side in enumerate(sizes):
            # Deliberate, opposed corner richness examples in the first two
            # training pairs; selected color count cannot be the selector.
            if split == "train":
                selected, decoy = (1, 4) if index % 2 == 0 else (4, 1)
                profile = (selected, decoy) if side % 2 else (decoy, selected)
            else:
                profile = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
            mode = int(rng.integers(7))
            for _ in range(16):
                pair = _make_pair(rng, side, mode, profile)
                signature = _key(pair["input"])
                if signature not in used:
                    used.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
    return puzzle


def _reference_check(scene):
    refs = [g["source"] for i, g in enumerate(scene["groups"]) if i != scene["target"]]
    small, large = sorted(refs, key=lambda a: a.shape[0])
    _require(large.shape[0] == small.shape[0]+1, "reference scales must be consecutive")
    _require(np.array_equal(large, _grow(small)), "reference content must survive middle insertion")
    _require(not np.array_equal(small[:2, :2], small[:2, -2:]), "distinct reference corners required")
    for left in (False, True):
        _require(np.array_equal(_corner(small, left), _corner(large, left)), "reference corners changed")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_parities, train_sizes, test_sizes = set(), set(), {0: set(), 1: set()}, []
    richness_signs = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            scene = _extract(grid)
            _reference_check(scene)
            _infer(scene)
            _require(all(not np.any(g["source"] == scene["neutral"]) for g in scene["groups"]),
                     "sources use data colors, not blank color")
            target = scene["groups"][scene["target"]]["source"]
            _require(out.shape == (2, 2), "output is entire 2x2 companion interior")
            _require(np.array_equal(out, solve(grid)), "incorrect answer")
            _require(not np.array_equal(_corner(target, True), _corner(target, False)), "target corner witness")
            _require(not np.any(target == scene["neutral"]), "source uses data colors, not blank color")
            if split == "train":
                train_parities.add(target.shape[0] % 2)
                train_sizes[target.shape[0] % 2].add(target.shape[0])
                decoy = _corner(target, not bool(target.shape[0] % 2))
                richness_signs.add(int(np.sign(len(np.unique(out))-len(np.unique(decoy)))))
            else:
                test_sizes.append(target.shape[0])
    _require(train_parities == {0, 1}, "both target parities must be demonstrated")
    _require({-1, 1} <= richness_signs, "training must defeat both corner-richness shortcuts")
    _require(max(s for sizes in train_sizes.values() for s in sizes) < min(test_sizes), "held-out target scale")
    if len(puzzle["train"]) >= 4:
        _require(all(len(sizes) >= 2 for sizes in train_sizes.values()), "two training scales per parity")
    if len(puzzle["test"]) >= 2:
        _require({s % 2 for s in test_sizes} == {0, 1}, "both test parities required")


def _digest(puzzle):
    d = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for k in ("input", "output"):
                a = np.asarray(pair[k], dtype=np.uint8)
                d.update(bytes(a.shape))
                d.update(a.tobytes())
    return d.hexdigest()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(x), len(roles)) for x in a.flat)


def self_test():
    """42 supplied seeds, bounded diversity, ablations and malformed cases."""
    shapes, sizes, palettes, structures, signatures, orders = set(), set(), set(), set(), set(), set()
    mixed = uniform = zero_free = odd_tests = even_tests = bypass = 0
    canvas_parity_failures = reference_copy_failures = 0
    constant_corner_failures = {"left": 0, "right": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                scene = _extract(g)
                target = scene["groups"][scene["target"]]["source"]
                shapes.add(g.shape)
                sizes.add(target.shape[0])
                palettes.add(tuple(map(int, np.unique(g))))
                structures.add(_normalized(g))
                zero_free += 0 not in g
                uniform += len(np.unique(out)) == 1
                if split == "test":
                    odd_tests += target.shape[0] % 2 == 1
                    even_tests += target.shape[0] % 2 == 0
                directions = []
                for group in scene["groups"]:
                    s, c = group["source_box"], group["companion_box"]
                    horizontal = s[1]+s[3] <= c[1] or c[1]+c[3] <= s[1]
                    directions.append(horizontal)
                    orders.add((horizontal, (s[1] < c[1]) if horizontal else (s[0] < c[0])))
                mixed += len(set(directions)) > 1
                # Omitting B by taking a fixed corner fails at every instance
                # of the opposite parity; actual contrasting crops are checked.
                opposite = _corner(target, not bool(target.shape[0] % 2))
                _require(not np.array_equal(out, opposite), "parity ablation is invisible")
                for name, left in (("left", True), ("right", False)):
                    constant_corner_failures[name] += not np.array_equal(out, _corner(target, left))
                canvas_parity_failures += not np.array_equal(out, _corner(target, bool(g.shape[0] % 2)))
                refs = [x["companion"] for i, x in enumerate(scene["groups"]) if i != scene["target"]]
                reference_copy_failures += all(not np.array_equal(out, ref) for ref in refs)
                # Honest A ablation: the mandated fixed code DOES bypass the
                # complete reference after segmentation; quantify, don't hide.
                bypass += np.array_equal(out, _literal_crop(target))
    _require(len(signatures) == 42, "42-seed puzzle diversity")
    _require(len(shapes) >= 30 and len(structures) >= 180, "structural diversity")
    _require(sizes == set(range(3, 15)), "target scale diversity")
    _require(len(palettes) >= 30 and min(mixed, uniform, zero_free, odd_tests, even_tests) > 0, "role/geometry diversity")
    _require(len(orders) == 4, "panel member placement diversity")
    _require(bypass == 210, "fixed-code limitation measurement changed")
    _require(min(*constant_corner_failures.values(), canvas_parity_failures,
                 reference_copy_failures) > 0, "shortcut counterexamples absent")
    # All explicit legal count combinations, streamed one puzzle at a time.
    count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(71000+10*a+b, a, b)
            validate_generated_puzzle(puzzle)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "counts")
            count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Incorrect labels and damaged in-input reference evidence must be rejected.
    p = generate_puzzle(123, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    p = generate_puzzle(124, 2, 1)
    damaged = p["test"][0]["input"].copy()
    scene = _extract(damaged)
    ref = next(g for i, g in enumerate(scene["groups"]) if i != scene["target"])
    r, c, h, w, _ = ref["companion_box"]
    damaged[r+1:r+h-1, c+1:c+w-1] = _corner(ref["source"], not bool(ref["source"].shape[0] % 2))
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("contradictory reference accepted")
    # Palette relabeling and irrelevant exterior changes must not change the
    # inferred geometry. This also checks that solve leaves its argument alone.
    p = generate_puzzle(125, 2, 1)
    permutation = np.array([8, 3, 0, 7, 1, 9, 4, 2, 6, 5], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        original = pair["input"].copy()
        recolored = permutation[original]
        _require(np.array_equal(solve(recolored), permutation[pair["output"]]), "palette dependence")
        _require(np.array_equal(recolored, permutation[original]), "solve mutated its input")
        scene = _extract(original)
        exterior = np.ones(original.shape, dtype=bool)
        for group in scene["groups"]:
            r, c, h, w = group["box"][:4]
            exterior[r:r+h, c:c+w] = False
        original[exterior] = scene["neutral"]
        _require(np.array_equal(solve(original), pair["output"]), "wallpaper changed the answer")
    # Exercise the last geometry attempt directly at both size boundaries,
    # rather than relying on random attempts to reach the guaranteed fit.
    for side in (3, 14):
        rng = np.random.default_rng(500 + side)
        ref = np.array([[3, 4, 3, 4, 5]] * 5, dtype=np.int8)
        sources = [ref, _grow(ref), _pattern(rng, side, np.array([3, 4, 5]), 0, (1, 3))]
        companions = [_literal_crop(ref), _literal_crop(sources[1]), np.zeros((2, 2), dtype=np.int8)]
        groups = [_group(rng, a, b, 0, 1, 2, 0, minimal=True)
                  for a, b in zip(sources, companions)]
        canvas = _pack(rng, groups, 0)
        _require(canvas is not None, "minimal geometry failed to fit")
        _require(np.array_equal(solve(canvas), _literal_crop(sources[2])), "minimal geometry prediction")
    return dict(seeds=42, pairs=210, count_combinations=count_calls,
                canvas_shapes=len(shapes), normalized_inputs=len(structures),
                target_sides=sorted(sizes), palettes=len(palettes),
                mixed_orientations=int(mixed), uniform_outputs=int(uniform),
                odd_tests=int(odd_tests), even_tests=int(even_tests),
                zero_free_inputs=int(zero_free),
                constant_corner_failures={k: int(v) for k, v in constant_corner_failures.items()},
                canvas_parity_failures=int(canvas_parity_failures),
                both_reference_copies_fail=int(reference_copy_failures),
                palette_and_wallpaper_checks=3, minimal_geometry_checks=2,
                fixed_code_reference_bypass=int(bypass),
                limitation="A inference is demonstrated but not information-theoretically necessary")
