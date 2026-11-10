"""Generate ARC grids from four corner cues and outward color signals."""

import hashlib

import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")
_ARMS = ("top", "bottom", "left", "right")
_SOURCE_PAIR_HASHES = {
    "13560a7cec6ff6fd61b7303eb7bef98a45164e8fa2b3f7b7140dbb4463bb8100",
    "683a806679b5eee27a617f1f522fc08f87a96394c49ccf9dd4437620745413e4",
    "fef9469125a464ea364638199483ee94fd9cc34757d10160fb4914ae1fbbc35e",
    "acf5d43175bf429df3b4fa16efece0f7440d9e0fbf06190833208ccefb61effd",
}


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt is None and ns is None:
        nt = int(rng.integers(2, 11))
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _l_cells(corner, top, bottom, left, right):
    if corner == "TL":
        return ((top, left), (top, left + 1), (top + 1, left))
    if corner == "TR":
        return ((top, right), (top, right - 1), (top + 1, right))
    if corner == "BL":
        return ((bottom, left), (bottom, left + 1), (bottom - 1, left))
    if corner == "BR":
        return ((bottom, right), (bottom, right - 1), (bottom - 1, right))
    raise AssertionError("unknown corner")


def _perimeter(top, bottom, left, right):
    cells = {(top, c) for c in range(left, right + 1)}
    cells.update((bottom, c) for c in range(left, right + 1))
    cells.update((r, left) for r in range(top, bottom + 1))
    cells.update((r, right) for r in range(top, bottom + 1))
    return cells


def _corner_candidates(grid, background):
    h, w = grid.shape
    result = {corner: [] for corner in _CORNERS}
    for corner in _CORNERS:
        for r in range(h):
            for c in range(w):
                top = r if corner[0] == "T" else r - 1
                bottom = r + 1 if corner[0] == "T" else r
                left = c if corner[1] == "L" else c - 1
                right = c + 1 if corner[1] == "L" else c
                if min(top, left) < 0 or bottom >= h or right >= w:
                    continue
                cells = _l_cells(corner, top, bottom, left, right)
                color = int(grid[r, c])
                if color != background and all(int(grid[y, x]) == color for y, x in cells):
                    result[corner].append((r, c, color))
    return result


def _analyze_input(grid):
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    ls = _corner_candidates(grid, background)
    matches = []
    for top, left, tl_color in ls["TL"]:
        for tr_top, right, tr_color in ls["TR"]:
            if tr_top != top or right - left < 4:
                continue
            for bottom, bl_left, bl_color in ls["BL"]:
                if bl_left != left or bottom - top < 4:
                    continue
                for br_bottom, br_right, br_color in ls["BR"]:
                    if (br_bottom, br_right) != (bottom, right):
                        continue
                    corner_colors = (tl_color, tr_color, bl_color, br_color)
                    for structure in set(corner_colors):
                        if corner_colors.count(structure) != 3:
                            continue
                        exceptional = next(k for k, col in enumerate(corner_colors) if col != structure)
                        fill = corner_colors[exceptional]
                        cue = set()
                        for corner in _CORNERS:
                            cue.update(_l_cells(corner, top, bottom, left, right))
                        if any((int(grid[y, x]) != background) != ((y, x) in cue)
                               for y, x in _perimeter(top, bottom, left, right)):
                            continue
                        if np.any(grid[top + 1:bottom, left + 1:right] != background):
                            continue
                        matches.append((top, bottom, left, right, structure, fill,
                                        _CORNERS[exceptional], background))
    if len(matches) != 1:
        raise AssertionError(f"expected one four-corner frame, found {len(matches)}")
    top, bottom, left, right, structure, fill, corner, background = matches[0]
    active = {arm: [] for arm in _ARMS}
    inert = []
    cue = set().union(*(_l_cells(k, top, bottom, left, right) for k in _CORNERS))
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == background or (r, c) in cue:
                continue
            if left <= c <= right and r < top:
                active["top"].append((r, c, color))
            elif left <= c <= right and r > bottom:
                active["bottom"].append((r, c, color))
            elif top <= r <= bottom and c < left:
                active["left"].append((r, c, color))
            elif top <= r <= bottom and c > right:
                active["right"].append((r, c, color))
            else:
                inert.append((r, c, color))
    return {"background": background, "structure": structure, "fill": fill,
            "corner": corner, "top": top, "bottom": bottom, "left": left,
            "right": right, "active": active, "inert": inert, "cue": cue}


def _solve_from_input(grid):
    info = _analyze_input(grid)
    h, w = grid.shape
    bg, fill, structure = info["background"], info["fill"], info["structure"]
    top, bottom, left, right = (info[k] for k in ("top", "bottom", "left", "right"))
    out = grid.copy()
    for r in range(h):
        for c in range(w):
            if (top <= r <= bottom or left <= c <= right) and int(grid[r, c]) == bg:
                out[r, c] = fill
    for r, c in _perimeter(top, bottom, left, right):
        out[r, c] = structure
    for c in range(left, right + 1):
        current = fill
        for r in range(top - 1, -1, -1):
            if int(grid[r, c]) != bg:
                current = int(grid[r, c])
            out[r, c] = current
        current = fill
        for r in range(bottom + 1, h):
            if int(grid[r, c]) != bg:
                current = int(grid[r, c])
            out[r, c] = current
    for r in range(top, bottom + 1):
        current = fill
        for c in range(left - 1, -1, -1):
            if int(grid[r, c]) != bg:
                current = int(grid[r, c])
            out[r, c] = current
        current = fill
        for c in range(right + 1, w):
            if int(grid[r, c]) != bg:
                current = int(grid[r, c])
            out[r, c] = current
    return out


def _geometry(rng, showcase, regime):
    if showcase:
        h = int(rng.integers(21, 31))
        w = int(rng.integers(21, 31))
        hh = int(rng.integers(5, min(10, h - 8) + 1))
        hw = int(rng.integers(5, min(10, w - 8) + 1))
        top = int(rng.integers(4, h - hh - 3))
        left = int(rng.integers(4, w - hw - 3))
        return h, w, top, top + hh - 1, left, left + hw - 1

    low = 12 if regime == "source_train" else 14
    h = int(rng.integers(low, 31))
    w = int(rng.integers(low, 31))
    aspect = int(rng.integers(0, 4))
    hh = int(rng.integers(5, min(13, h) + 1))
    if aspect == 0:
        hw = max(5, min(w, hh + int(rng.integers(-1, 2))))
    elif aspect == 1:
        hw = max(5, min(w, hh - int(rng.integers(2, 6))))
    elif aspect == 2:
        hw = max(5, min(w, hh + int(rng.integers(2, 6))))
    else:
        hw = int(rng.integers(5, min(14, w) + 1))
    top = int(rng.integers(0, h - hh + 1))
    left = int(rng.integers(0, w - hw + 1))
    return h, w, top, top + hh - 1, left, left + hw - 1


def _arm_cells(arm, h, w, top, bottom, left, right):
    if arm == "top":
        return [(r, c) for r in range(top) for c in range(left, right + 1)]
    if arm == "bottom":
        return [(r, c) for r in range(bottom + 1, h) for c in range(left, right + 1)]
    if arm == "left":
        return [(r, c) for r in range(top, bottom + 1) for c in range(left)]
    return [(r, c) for r in range(top, bottom + 1) for c in range(right + 1, w)]


def _put_markers(rng, grid, bounds, fill, structure, showcase, regime):
    h, w, top, bottom, left, right = bounds
    bg = int(grid[0, 0]) if (0, 0) not in _perimeter(top, bottom, left, right) else None
    if bg is None:
        bg = int(np.bincount(grid.ravel(), minlength=10).argmax())
    available = [x for x in range(10) if x not in (bg, structure)]
    rng.shuffle(available)
    palette_size = int(rng.integers(2, 9))
    palette = available[:palette_size]
    if int(rng.integers(0, 3)) != 0 and fill not in palette:
        palette[-1] = fill

    used = set()
    if showcase:
        vertical = ("top", "bottom")[int(rng.integers(0, 2))]
        horizontal = ("left", "right")[int(rng.integers(0, 2))]
        distinct = [x for x in available if x != fill][:4]
        if vertical == "top":
            c = int(rng.integers(left, right + 1))
            positions = ((1, c), (top - 1, c))
        else:
            c = int(rng.integers(left, right + 1))
            positions = ((h - 2, c), (bottom + 1, c))
        for pos, color in zip(positions, distinct[:2]):
            grid[pos] = color
            used.add(pos)
        if horizontal == "left":
            r = int(rng.integers(top, bottom + 1))
            positions = ((r, 1), (r, left - 1))
        else:
            r = int(rng.integers(top, bottom + 1))
            positions = ((r, w - 2), (r, right + 1))
        for pos, color in zip(positions, distinct[2:4]):
            grid[pos] = color
            used.add(pos)
        for arm in _ARMS:
            if arm in (vertical, horizontal):
                continue
            options = [p for p in _arm_cells(arm, *bounds) if p not in used]
            pos = options[int(rng.integers(0, len(options)))]
            grid[pos] = fill if len(used) == 4 else int(palette[int(rng.integers(0, len(palette)))])
            used.add(pos)
        target_active = int(rng.integers(8, 18))
        selected = list(_ARMS)
    else:
        mode = int(rng.integers(0, 12))
        if mode == 0:
            selected = []
        elif mode <= 4:
            selected = list(rng.choice(_ARMS, size=int(rng.integers(1, 3)), replace=False))
        else:
            selected = list(rng.choice(_ARMS, size=int(rng.integers(2, 5)), replace=False))
        selected = [a for a in selected if _arm_cells(a, *bounds)]
        target_active = int(rng.integers(len(selected), 19)) if selected else 0
        if regime == "target_test" and selected and int(rng.integers(0, 3)) == 0:
            target_active = int(rng.integers(10, 21))
        for arm in selected:
            options = _arm_cells(arm, *bounds)
            pos = options[int(rng.integers(0, len(options)))]
            grid[pos] = int(palette[int(rng.integers(0, len(palette)))])
            used.add(pos)

    # Extra signals are allocated without a line-spacing constraint. Repeated
    # lines and adjacent points are part of the source family.
    options = [p for arm in selected for p in _arm_cells(arm, *bounds) if p not in used]
    rng.shuffle(options)
    for pos in options[:max(0, target_active - len(used))]:
        grid[pos] = int(palette[int(rng.integers(0, len(palette)))])
        used.add(pos)

    inert = [(r, c) for r in range(h) for c in range(w)
             if not top <= r <= bottom and not left <= c <= right and int(grid[r, c]) == bg]
    rng.shuffle(inert)
    count = int(rng.integers(0, 16))
    if int(rng.integers(0, 9)) == 0:
        count = 0
    for pos in inert[:count]:
        grid[pos] = int(palette[int(rng.integers(0, len(palette)))])


def _generate_pair(rng, corner, showcase, regime, avoid_roles=()):
    for attempt in range(80):
        h, w, top, bottom, left, right = _geometry(rng, showcase, regime)
        colors = list(range(10))
        rng.shuffle(colors)
        bg, structure, fill = colors[:3]
        if avoid_roles and (bg, structure, fill) == avoid_roles:
            continue
        grid = np.full((h, w), bg, dtype=np.int8)
        for key in _CORNERS:
            for r, c in _l_cells(key, top, bottom, left, right):
                grid[r, c] = fill if key == corner else structure
        _put_markers(rng, grid, (h, w, top, bottom, left, right),
                     fill, structure, showcase, regime)
        try:
            info = _analyze_input(grid)
            if (info["top"], info["bottom"], info["left"], info["right"],
                    info["corner"], info["structure"], info["fill"]) != (
                    top, bottom, left, right, corner, structure, fill):
                continue
            output = _solve_from_input(grid)
            if np.array_equal(grid, output):
                continue
            return {"input": grid, "output": output}
        except AssertionError:
            continue
    raise AssertionError("could not construct an unambiguous pair")


def _pair_hash(pair):
    a = pair["input"].astype(np.uint8, copy=False)
    b = pair["output"].astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The transformation is the only shared convention. Color roles, scale,
    # marker palette, and exceptional corner are all visible pair properties.
    regime = ("heldout", "mixed", "familiar")[int(rng.integers(0, 3))]
    heldout = _CORNERS[int(rng.integers(0, 4))]
    train_corners = [c for c in _CORNERS if c != heldout]
    rng.shuffle(train_corners)
    mixed_order = list(_CORNERS)
    rng.shuffle(mixed_order)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    previous_roles = ()
    for split, count in (("train", train_count), ("test", test_count)):
        for i in range(count):
            if regime == "heldout":
                corner = train_corners[i % 3] if split == "train" else heldout
            elif regime == "mixed" and test_count == 4:
                corner = mixed_order[i % 4]
            elif regime == "mixed":
                corner = _CORNERS[int(rng.integers(0, 4))]
            elif split == "train":
                corner = train_corners[i % 3]
            else:
                corner = train_corners[int(rng.integers(0, min(train_count, 3)))]
            for _ in range(20):
                pair = _generate_pair(rng, corner, split == "train" and i == 0,
                                      "source_train" if split == "train" else "target_test",
                                      previous_roles)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in used_inputs and _pair_hash(pair) not in _SOURCE_PAIR_HASHES:
                    break
            else:
                raise AssertionError("duplicate pair construction")
            used_inputs.add(key)
            info = _analyze_input(pair["input"])
            previous_roles = (info["background"], info["structure"], info["fill"])
            puzzle[split].append(pair)
    return puzzle


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs = set(), set()
    train_arms = set()
    train_multiline = False
    test_multiline = False
    train_color_changes = False
    test_color_changes = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            a, b = pair["input"], pair["output"]
            assert a.shape == b.shape and not np.array_equal(a, b)
            ikey = _grid_key(a)
            pkey = (ikey, _grid_key(b))
            assert ikey not in inputs and pkey not in pairs
            inputs.add(ikey)
            pairs.add(pkey)
            assert _pair_hash(pair) not in _SOURCE_PAIR_HASHES
            info = _analyze_input(a)
            assert len(info["cue"]) == 12
            assert len({info[k] for k in ("background", "structure", "fill")}) == 3
            assert np.array_equal(b, _solve_from_input(a))
            assert all(int(b[r, c]) == int(a[r, c]) for r, c, _ in info["inert"])
            lines = []
            colors_by_line = {}
            for arm, markers in info["active"].items():
                if markers and split == "train":
                    train_arms.add(arm)
                for r, c, color in markers:
                    line = (arm, c if arm in ("top", "bottom") else r)
                    lines.append(line)
                    colors_by_line.setdefault(line, set()).add(color)
            if len(lines) != len(set(lines)):
                if split == "train":
                    train_multiline = True
                else:
                    test_multiline = True
            if any(len(colors) > 1 for colors in colors_by_line.values()):
                if split == "train":
                    train_color_changes = True
                else:
                    test_color_changes = True
    assert train_arms == set(_ARMS)
    assert train_multiline or not test_multiline
    assert train_color_changes or not test_color_changes


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    observations = {name: 0 for name in (
        "corner_only", "multiline", "close", "empty_arm", "few_arms",
        "no_signal", "no_inert", "adjacent_inert", "boundary_hub",
        "small_canvas", "large_canvas", "tall_hub", "wide_hub",
        "square_hub", "rich_active", "rich_inert", "fill_signal",
    )}
    role_triplets = set()
    layouts = set()
    seen_corners = set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                h, w = grid.shape
                top, bottom, left, right = (info[k] for k in
                    ("top", "bottom", "left", "right"))
                active = [m for ls in info["active"].values() for m in ls]
                inert = info["inert"]
                lines = [(arm, c if arm in ("top", "bottom") else r)
                    for arm, ls in info["active"].items() for r, c, _ in ls]
                flags = {
                    "corner_only": len(info["cue"]) == 12 and
                        len(_perimeter(top, bottom, left, right)) > 12,
                    "multiline": len(lines) > len(set(lines)),
                    "close": any(r in (top - 1, bottom + 1) or
                        c in (left - 1, right + 1) for r, c, _ in active),
                    "empty_arm": any(not ls for ls in info["active"].values()),
                    "few_arms": 1 <= sum(bool(ls) for ls in info["active"].values()) <= 2,
                    "no_signal": not active,
                    "no_inert": not inert,
                    "adjacent_inert": any(abs(r - y) + abs(c - x) == 1
                        for j, (r, c, _) in enumerate(inert)
                        for y, x, _ in inert[j + 1:]),
                    "boundary_hub": 0 in (top, left, h - bottom - 1, w - right - 1),
                    "small_canvas": h <= 16 and w <= 16,
                    "large_canvas": h >= 27 and w >= 27,
                    "tall_hub": bottom - top > right - left,
                    "wide_hub": bottom - top < right - left,
                    "square_hub": bottom - top == right - left,
                    "rich_active": len(active) >= 14,
                    "rich_inert": len(inert) >= 12,
                    "fill_signal": any(color == info["fill"] for _, _, color in active),
                }
                for name, present in flags.items():
                    observations[name] += bool(present)
                role_triplets.add((info["background"], info["structure"], info["fill"]))
                layouts.add((grid.shape, top, bottom, left, right, info["corner"]))
                seen_corners.add(info["corner"])
    assert all(observations.values()), observations
    assert len(role_triplets) >= 100 and len(layouts) >= 150
    assert seen_corners == set(_CORNERS)
    explicit = generate_puzzle(9173, 7, 3)
    validate_generated_puzzle(explicit)
    assert len(explicit["train"]) == 7 and len(explicit["test"]) == 3
    for args in ((2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(4200, *args))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(6000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    def fingerprint(puzzle):
        return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                                   for p in puzzle[split])) for split in ("train", "test"))
    assert fingerprint(generate_puzzle(11)) == fingerprint(generate_puzzle(11))
    assert fingerprint(generate_puzzle(11)) != fingerprint(generate_puzzle(12))
    assert len({fingerprint(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
