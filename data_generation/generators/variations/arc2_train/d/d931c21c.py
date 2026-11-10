"""Generate scenes governed by closure and one-cell contour halos."""

import hashlib
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_PAIR_DIGESTS = {
    "800fbbd099b7481c3b68062861575b0ddbb126a67ee0288cc7cee46f0d46d8f6",
    "e8e7d142ffea8231ac5146ebcffa54064bb8766d9cf88e005244ac18aaa53d4e",
    "20c9ba31d939366140a2f43d28adee041e88927ecbe63ccf4539faab3d0eaf04",
    "c0f8092d4a07c7c3c993f5662c9dbb1174da3f013bea27484ebf33cb9855d7cc",
    "e2ccdcd93241adf9164ecda5b53a042b42771226c71b45dd5a04ff19b945aa25",
}


def _count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _counts(rng, num_train, num_test):
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    answer = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            cells = [(r, c)]
            seen[r, c] = True
            for x, y in cells:
                for dr, dc in _N4:
                    xx, yy = x + dr, y + dc
                    if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                        seen[xx, yy] = True
                        cells.append((xx, yy))
            component = np.zeros_like(mask)
            for x, y in cells:
                component[x, y] = True
            answer.append(component)
    return answer


def _dilate8(mask):
    h, w = mask.shape
    result = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            a, b = max(0, -dr), min(h, h - dr)
            x, y = max(0, -dc), min(w, w - dc)
            result[a + dr:b + dr, x + dc:y + dc] |= mask[a:b, x:y]
    return result


def _topology(foreground):
    h, w = foreground.shape
    background = ~foreground
    exterior = np.zeros_like(background)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if background[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if background[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and background[x, y] and not exterior[x, y]:
                exterior[x, y] = True
                stack.append((x, y))
    cavities = background & ~exterior
    fg_components = _components(foreground)
    cavity_components = _components(cavities)
    qualifying = []
    closed = np.zeros_like(foreground)
    for component in fg_components:
        neighboring = False
        for dr, dc in _N4:
            neighboring |= bool((_shift(component, dr, dc) & cavities).any())
        qualifying.append(neighboring)
        if neighboring:
            closed |= component
    band = _dilate8(closed) & background
    return {
        "background": background, "exterior": exterior, "cavities": cavities,
        "foreground_components": fg_components, "cavity_components": cavity_components,
        "qualifying": qualifying, "closed": closed,
        "outer": band & exterior, "inner": band & cavities,
    }


def _shift(mask, dr, dc):
    h, w = mask.shape
    result = np.zeros_like(mask)
    a, b = max(0, -dr), min(h, h - dr)
    x, y = max(0, -dc), min(w, w - dc)
    result[a + dr:b + dr, x + dc:y + dc] = mask[a:b, x:y]
    return result


def _draw_lines(points):
    cells = []
    for (r, c), (x, y) in zip(points, points[1:]):
        if r != x and c != y:
            raise AssertionError("nonorthogonal segment")
        dr = (x > r) - (x < r)
        dc = (y > c) - (y < c)
        while (r, c) != (x, y):
            cells.append((r, c))
            r += dr
            c += dc
    cells.append(points[-1])
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    h = max(r for r, _ in cells) - r0 + 1
    w = max(c for _, c in cells) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _degree_turns(mask):
    turns = 0
    for r, c in np.argwhere(mask):
        neighbors = []
        for dr, dc in _N4:
            x, y = int(r) + dr, int(c) + dc
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y]:
                neighbors.append((dr, dc))
        if len(neighbors) != 2:
            return -1
        if neighbors[0][0] != -neighbors[1][0] or neighbors[0][1] != -neighbors[1][1]:
            turns += 1
    return turns


def _closed_mask(rng, h, w, style="any"):
    for _ in range(40):
        kind = style if style != "any" else str(rng.choice(("rectangle", "corner", "step", "multi")))
        if kind == "rectangle" or h < 6 or w < 6:
            mask = _draw_lines([(0, 0), (0, w - 1), (h - 1, w - 1),
                                (h - 1, 0), (0, 0)])
        elif kind == "corner":
            depth = int(rng.integers(1, min(3, h - 3) + 1))
            reach = int(rng.integers(2, min(4, w - 3) + 1))
            mask = _draw_lines([(0, reach), (0, w - 1), (h - 1, w - 1),
                                (h - 1, 0), (depth, 0), (depth, reach), (0, reach)])
        else:
            sides = list(map(int, rng.permutation(4)[:(2 if kind == "multi" else 1)]))
            points = [(0, 0)]
            for side, length in enumerate((w, h, w, h)):
                if side in sides and length >= 7:
                    a = int(rng.integers(2, length - 3))
                    b = int(rng.integers(a + 2, length - 1))
                    depth = int(rng.choice((-1, 1)))
                    if side == 0:
                        points += [(0, a), (depth, a), (depth, b), (0, b)]
                    elif side == 1:
                        points += [(a, w - 1), (a, w - 1 - depth),
                                   (b, w - 1 - depth), (b, w - 1)]
                    elif side == 2:
                        points += [(h - 1, w - 1 - a), (h - 1 - depth, w - 1 - a),
                                   (h - 1 - depth, w - 1 - b), (h - 1, w - 1 - b)]
                    else:
                        points += [(h - 1 - a, 0), (h - 1 - a, depth),
                                   (h - 1 - b, depth), (h - 1 - b, 0)]
                points.append(((0, w - 1), (h - 1, w - 1),
                               (h - 1, 0), (0, 0))[side])
            mask = _draw_lines(points)
        topo = _topology(np.pad(mask, 1))
        if (_degree_turns(mask) >= 4 and len(topo["foreground_components"]) == 1
                and len(topo["cavity_components"]) == 1):
            return mask
    return _closed_mask(rng, h, w, "rectangle")


def _open_mask(rng, h, w, motif):
    if motif == "dot":
        return np.ones((1, 1), dtype=bool)
    if motif == "arc":
        choices = (
            [(0, w - 1), (0, 0), (h - 1, 0)],
            [(0, 0), (h - 1, 0), (h - 1, w - 1)],
            [(h - 1, 0), (h - 1, w - 1), (0, w - 1)],
        )
        return _draw_lines(choices[int(rng.integers(0, len(choices)))])
    loop = _closed_mask(rng, h, w, str(rng.choice(("rectangle", "corner", "step"))))
    candidates = []
    for r, c in np.argwhere(loop):
        if 0 < r < loop.shape[0] - 1 and 0 < c < loop.shape[1] - 1:
            continue
        candidates.append((int(r), int(c)))
    rng.shuffle(candidates)
    for r, c in candidates:
        for length in map(int, rng.permutation([1, 2, 3, 4])):
            trial = loop.copy()
            if r in (0, loop.shape[0] - 1):
                if c + length > loop.shape[1] - 1:
                    continue
                trial[r, c:c + length] = False
            else:
                if r + length > loop.shape[0] - 1:
                    continue
                trial[r:r + length, c] = False
            topo = _topology(np.pad(trial, 1))
            if trial.any() and not topo["cavity_components"] and len(topo["foreground_components"]) == 1:
                return trial
    return _open_mask(rng, h, w, "arc")


def _dimension(rng, regime):
    choice = float(rng.random())
    if choice < 0.10:
        return 8
    if choice < 0.27:
        return 30
    if choice < (0.47 if regime == "source_train" else 0.40):
        return int(rng.integers(8, 13))
    if choice < (0.78 if regime == "source_train" else 0.61):
        return int(rng.integers(13, 23))
    return int(rng.integers(23, 31))


def _place_scene(rng, h, w, specs):
    # Place by actual occupied cells. A two-cell gap prevents competing halos
    # and keeps unchanged open neighborhoods outside colored bands.
    for _ in range(18):
        scene = np.zeros((h, w), dtype=bool)
        masks = []
        success = True
        for spec in specs:
            placed = False
            for _ in range(110):
                kind = spec["kind"]
                max_h = min(h, spec.get("max_h", 13))
                max_w = min(w, spec.get("max_w", 13))
                min_h = spec.get("min_h", 3 if kind == "closed" else 1)
                min_w = spec.get("min_w", 3 if kind == "closed" else 1)
                if h < min_h or w < min_w:
                    break
                fh = int(rng.integers(min_h, max(min_h, max_h) + 1))
                fw = int(rng.integers(min_w, max(min_w, max_w) + 1))
                if kind == "closed":
                    mask = _closed_mask(rng, fh, fw, spec.get("style", "any"))
                else:
                    motif = spec.get("motif", "broken")
                    mask = _open_mask(rng, fh, fw, motif)
                mh, mw = mask.shape
                if mh > h or mw > w:
                    continue
                edge = spec.get("edge")
                if edge == "top":
                    r = 0
                elif edge == "bottom":
                    r = h - mh
                else:
                    r = int(rng.integers(0, h - mh + 1))
                if edge == "left":
                    c = 0
                elif edge == "right":
                    c = w - mw
                else:
                    c = int(rng.integers(0, w - mw + 1))
                candidate = np.zeros_like(scene)
                candidate[r:r + mh, c:c + mw] = mask
                if (candidate & _dilate8(_dilate8(scene))).any():
                    continue
                joined = scene | candidate
                topo = _topology(joined)
                if (len(topo["cavity_components"]) !=
                    sum(s["kind"] == "closed" for s in specs[:len(masks) + 1])):
                    continue
                if len(topo["foreground_components"]) != len(masks) + 1:
                    continue
                desired = [s["kind"] == "closed" for s in specs[:len(masks) + 1]]
                if any(
                    not any(bool((component & object_mask).any()) and qualifies == want
                            for component, qualifies in zip(topo["foreground_components"],
                                                            topo["qualifying"]))
                    for object_mask, want in zip(masks + [candidate], desired)
                ):
                    continue
                scene = joined
                masks.append(candidate)
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            return scene
    return None


def _pair(scene, palette):
    background, contour, outer, inner = palette
    grid = np.full(scene.shape, background, dtype=np.int8)
    grid[scene] = contour
    topo = _topology(scene)
    out = grid.copy()
    out[topo["outer"]] = outer
    out[topo["inner"]] = inner
    return {"input": grid, "output": out}


def _palette(rng, halos, base, vary):
    if not vary:
        return (base[0], base[1], halos[0], halos[1])
    available = [c for c in range(10) if c not in halos]
    colors = rng.choice(available, size=2, replace=False)
    return (int(colors[0]), int(colors[1]), halos[0], halos[1])


def _make_pair(rng, index, regime, palette):
    if regime == "source_train" and index == 0:
        h, w = int(rng.integers(21, 28)), int(rng.integers(21, 29))
        specs = [
            {"kind": "closed", "style": str(rng.choice(("corner", "step", "multi"))),
             "min_h": 10, "min_w": 10, "max_h": 14, "max_w": 15},
            {"kind": "open", "motif": "dot", "min_h": 1, "min_w": 1},
        ]
    elif regime == "source_train" and index == 1:
        h, w = int(rng.integers(17, 26)), int(rng.integers(17, 27))
        specs = [
            {"kind": "closed", "style": "rectangle", "min_h": 3, "min_w": 3,
             "max_h": 5, "max_w": 5,
             "edge": str(rng.choice(("top", "bottom", "left", "right")))},
            {"kind": "open", "motif": str(rng.choice(("broken", "arc"))),
             "min_h": 4, "min_w": 4, "max_h": 8, "max_w": 8},
        ]
    else:
        h, w = _dimension(rng, regime), _dimension(rng, regime)
        area = h * w
        capacity = 1 if area < 110 else 2 if area < 210 else 3 if area < 360 else 5 if area < 600 else 7
        if regime == "target_test" and area >= 500 and rng.random() < 0.6:
            count = int(rng.integers(min(4, capacity), capacity + 1))
        else:
            count = int(rng.integers(1, capacity + 1))
        no_op = rng.random() < (0.17 if regime == "target_test" else 0.12)
        nclosed = 0 if no_op else int(rng.integers(1, count + 1))
        kinds = ["closed"] * nclosed + ["open"] * (count - nclosed)
        rng.shuffle(kinds)
        edge_positions = list(map(int, rng.permutation(count)[:int(rng.integers(0, min(3, count) + 1))]))
        specs = []
        for j, kind in enumerate(kinds):
            max_size = min(15 if kind == "closed" else 17, max(4, min(h, w) - 2))
            spec = {"kind": kind, "min_h": 3 if kind == "closed" else 1,
                    "min_w": 3 if kind == "closed" else 1,
                    "max_h": max_size, "max_w": max_size}
            if kind == "open":
                spec["motif"] = str(rng.choice(("dot", "arc", "broken"), p=(0.22, 0.34, 0.44)))
                if spec["motif"] != "dot":
                    spec["min_h"] = 3
                    spec["min_w"] = 3
            if j in edge_positions:
                spec["edge"] = str(rng.choice(("top", "bottom", "left", "right")))
            specs.append(spec)
    for _ in range(8):
        scene = _place_scene(rng, h, w, specs)
        if scene is not None and scene.sum() < scene.size / 2:
            return _pair(scene, palette)
        if len(specs) > (2 if regime == "source_train" and index < 2 else 1):
            specs = specs[:-1]
    # Deterministic fallback keeps legal seeds total, while retaining topology.
    if regime == "source_train" and index < 2:
        raise RuntimeError("failed to place training demonstration")
    scene = _place_scene(rng, h, w, [{"kind": "closed", "style": "rectangle",
                                      "min_h": 3, "min_w": 3, "max_h": 3, "max_w": 3}])
    if scene is None:
        raise RuntimeError("failed to place fallback contour")
    return _pair(scene, palette)


def _signature(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    nt, ns = _counts(rng, num_train, num_test)
    colors = list(map(int, rng.choice(10, size=4, replace=False)))
    base, halos = colors[:2], colors[2:]
    vary = bool(rng.random() < 0.58)
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, split, count in (("source_train", "train", nt), ("target_test", "test", ns)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
            for attempt in range(35):
                palette = _palette(pair_rng, halos, base,
                                   vary and (index > 0 or regime == "target_test"))
                pair = _make_pair(pair_rng, index, regime, palette)
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain unique input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    background = int(values[np.argmax(counts)])
    contour = int(values[np.argmin(counts)])
    return background, contour, _topology(grid == contour)


def _source_digest(pair):
    a = pair["input"].astype(np.int8)
    b = pair["output"].astype(np.int8)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes() + b.tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    analyses = []
    seen_inputs, seen_pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        a, b = pair["input"], pair["output"]
        for grid in (a, b):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert bool(np.all((0 <= grid) & (grid <= 9)))
        assert a.shape == b.shape
        assert _source_digest(pair) not in _SOURCE_PAIR_DIGESTS
        key = _signature(a)
        pair_key = (key, _signature(b))
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        bg, fg, topo = _infer_input(a)
        assert len(topo["cavity_components"]) == sum(topo["qualifying"])
        claims = np.zeros(a.shape, dtype=np.int16)
        open_mask = np.zeros(a.shape, dtype=bool)
        for component, qualifies in zip(topo["foreground_components"], topo["qualifying"]):
            if qualifies:
                assert _degree_turns(component) >= 4
                claims += (_dilate8(component) & topo["background"]).astype(np.int16)
            else:
                assert not (_dilate8(component) & topo["cavities"]).any()
                open_mask |= component
        assert int(claims.max()) <= 1
        assert not (_dilate8(open_mask) & (topo["outer"] | topo["inner"])).any()
        analyses.append((bg, fg, topo))
    outer_colors, inner_colors = set(), set()
    positive = 0
    saw_open = saw_core = saw_clip = saw_irregular = False
    for pair, (bg, fg, topo) in zip(puzzle["train"], analyses[:nt]):
        if topo["outer"].any():
            outer_colors.update(map(int, np.unique(pair["output"][topo["outer"]])))
        if topo["inner"].any():
            inner_colors.update(map(int, np.unique(pair["output"][topo["inner"]])))
        positive += bool(topo["inner"].any())
        saw_open |= any(not x for x in topo["qualifying"])
        saw_core |= bool((topo["cavities"] & ~topo["inner"]).any())
        for component, qualifies in zip(topo["foreground_components"], topo["qualifying"]):
            if qualifies:
                saw_clip |= bool(component[0].any() or component[-1].any()
                                 or component[:, 0].any() or component[:, -1].any())
                saw_irregular |= _degree_turns(component) > 4
    assert len(outer_colors) == len(inner_colors) == 1
    outer, inner = next(iter(outer_colors)), next(iter(inner_colors))
    assert outer != inner and positive >= 2
    assert saw_open and saw_core and saw_clip and saw_irregular
    for pair, (bg, fg, topo) in zip(puzzle["train"] + puzzle["test"], analyses):
        assert len({bg, fg, outer, inner}) == 4
        expected = pair["input"].copy()
        expected[topo["outer"]] = outer
        expected[topo["inner"]] = inner
        assert np.array_equal(pair["output"], expected)


def _puzzles_equal(a, b):
    return all(np.array_equal(x[k], y[k])
               for split in ("train", "test")
               for x, y in zip(a[split], b[split])
               for k in ("input", "output")) and all(
                   len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, profiles, palettes = set(), set(), set(), set(), set()
    six_turn = small_cavity = core = no_op = dot = edge_sides = 0
    eight_turn = many_turn = long_cavity = 0
    small_train = small_test = large_train = large_test = full_size = elongated = 0
    early_noop = test_noop = edge_free_test = multi_edge_test = 0
    large_open = left_open = dense_mixture = 0
    varying_palettes = fixed_palettes = varying_counts = 0
    touched_sides = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_colors, puzzle_counts = set(), set()
        for split, index, pair in (
            (split, index, pair)
            for split in ("train", "test")
            for index, pair in enumerate(puzzle[split])
        ):
            bg, fg, topo = _infer_input(pair["input"])
            h, w = pair["input"].shape
            shapes.add((h, w))
            palettes.add((bg, fg))
            puzzle_colors.add((bg, fg))
            closed_count = sum(topo["qualifying"])
            open_count = len(topo["qualifying"]) - closed_count
            puzzle_counts.add((closed_count, open_count))
            profiles.add((len(topo["foreground_components"]), len(topo["cavity_components"]),
                          pair["input"].shape))
            unchanged = bool(np.array_equal(pair["input"], pair["output"]))
            no_op += unchanged
            early_noop += int(split == "train" and index < len(puzzle["train"]) - 1 and unchanged)
            test_noop += int(split == "test" and unchanged)
            small_train += int(split == "train" and max(h, w) <= 12)
            small_test += int(split == "test" and max(h, w) <= 12)
            large_train += int(split == "train" and min(h, w) >= 23)
            large_test += int(split == "test" and min(h, w) >= 23)
            full_size += int((h, w) == (30, 30))
            elongated += int(max(h, w) / min(h, w) >= 1.6)
            dense_mixture += int(closed_count >= 3 and open_count >= 2)
            core += bool((topo["cavities"] & ~topo["inner"]).any())
            for cavity in topo["cavity_components"]:
                small_cavity += int(cavity.sum() == 1)
                rows, cols = np.where(cavity)
                ch, cw = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
                long_cavity += int(max(ch, cw) >= 2 * min(ch, cw))
            clipped = 0
            for component, qualifies in zip(topo["foreground_components"], topo["qualifying"]):
                if qualifies:
                    turns = _degree_turns(component)
                    six_turn += int(turns == 6)
                    eight_turn += int(turns == 8)
                    many_turn += int(turns >= 10)
                    sides = (component[0].any(), component[-1].any(),
                             component[:, 0].any(), component[:, -1].any())
                    clipped += int(any(sides))
                    edge_sides += int(any(sides))
                    touched_sides.update(i for i, yes in enumerate(sides) if yes)
                else:
                    dot += int(component.sum() == 1)
                    rows, cols = np.where(component)
                    large_open += int(rows.max() - rows.min() >= 8
                                      and cols.max() - cols.min() >= 8)
                    left_open += int(component[:, 0].any())
            edge_free_test += int(split == "test" and clipped == 0)
            multi_edge_test += int(split == "test" and clipped >= 2)
        varying_palettes += int(len(puzzle_colors) > 1)
        fixed_palettes += int(len(puzzle_colors) == 1)
        varying_counts += int(len(puzzle_counts) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(profiles) >= 50 and len(palettes) >= 25
    assert six_turn > 0 and small_cavity > 0 and core > 0 and no_op > 0 and dot > 0
    assert edge_sides > 0
    assert eight_turn > 0 and many_turn > 0 and long_cavity > 0
    assert small_train > 0 and small_test > 0 and large_train > 0 and large_test > 0
    assert full_size > 0 and elongated > 0 and touched_sides == {0, 1, 2, 3}
    assert early_noop > 0 and test_noop > 0 and edge_free_test > 0 and multi_edge_test > 0
    assert large_open > 0 and left_open > 0 and dense_mixture > 0
    assert varying_palettes > 0 and fixed_palettes > 0 and varying_counts == 42
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + nt + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000 + (nt or 0) + (ns or 0), nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    assert _puzzles_equal(generate_puzzle(3000), generate_puzzle(3000))
    assert not _puzzles_equal(generate_puzzle(3000), generate_puzzle(3001))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("accepted illegal counts")
