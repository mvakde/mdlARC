"""Motion-selected stride-two change maps. Standard library and NumPy only."""
from __future__ import annotations

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a


def _read(input_grid):
    """Parse only visible input evidence; never use an output shape or palette."""
    grid = _grid(input_grid)
    height, width = grid.shape
    if height % 2 == 0 and width % 4 == 1 and width >= 9:
        middle = width // 2
        first, second = grid[:, :middle], grid[:, middle + 1:]
        stripe, orientation = grid[:, middle], "vertical"
    elif width % 2 == 0 and height % 4 == 1 and height >= 9:
        middle = height // 2
        first, second = grid[:middle], grid[middle + 1:]
        stripe, orientation = grid[middle], "horizontal"
    else:
        raise ValueError("not two even-sized panels and a central divider")
    if len(stripe) < 4:
        raise ValueError("divider needs two endpoints and an interior")
    departure, spine, arrival = int(stripe[0]), int(stripe[1]), int(stripe[-1])
    if len({departure, spine, arrival}) != 3 or not np.all(stripe[1:-1] == spine):
        raise ValueError("malformed endpoint key")
    if any(np.any(panel == color) for panel in (first, second)
           for color in (departure, spine, arrival)):
        raise ValueError("key colors may not occur in the panels")
    values, counts = np.unique(np.concatenate((first.ravel(), second.ravel())),
                               return_counts=True)
    background = int(values[np.argmax(counts)])
    if any(np.count_nonzero(panel == background) <= panel.size // 2
           for panel in (first, second)):
        raise ValueError("background must be a strict majority in each panel")
    changed = np.argwhere(first != second)
    if not len(changed):
        raise ValueError("motion is required to reveal the phase")
    phases = np.unique(changed % 2, axis=0)
    if len(phases) != 1:
        raise ValueError("changes must share one stride-two phase")
    phase = tuple(int(v) for v in phases[0])
    if np.any((first != second) & (first != background) & (second != background)):
        raise ValueError("conflicting foreground overlap")
    pr, pc = phase
    a, b = first[pr::2, pc::2], second[pr::2, pc::2]
    return dict(first=first, second=second, a=a, b=b, phase=phase,
                background=background, spine=spine, departure=departure,
                arrival=arrival, orientation=orientation)


def _change_map(a, b, background, departure, arrival):
    result = a.copy()
    result[(a != background) & (b == background)] = departure
    result[(a == background) & (b != background)] = arrival
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    return _change_map(info["a"], info["b"], info["background"],
                       info["departure"], info["arrival"])


def _join(first, second, roles, orientation):
    _, spine, departure, arrival = roles
    length = first.shape[0 if orientation == "vertical" else 1]
    key = np.full(length, spine, dtype=np.int8)
    key[0], key[-1] = departure, arrival
    if orientation == "vertical":
        return np.concatenate((first, key[:, None], second), axis=1)
    return np.concatenate((first, key[None, :], second), axis=0)


def _place_event(rng, first, second, background, palette, reserved, forbidden, budget,
                 force_overlap=False):
    height, width = first.shape
    for attempt in range(250):
        kind = "bar" if force_overlap else str(rng.choice(
            ("point", "bar", "square", "elbow"), p=(.40, .30, .15, .15)))
        axis = int(rng.integers(2))
        if kind == "point":
            shape = [(0, 0)]
        elif kind == "bar":
            length = int(rng.integers(2, min(4, height if axis == 0 else width)))
            shape = [(i, 0) if axis == 0 else (0, i) for i in range(length)]
        elif kind == "square":
            shape = [(0, 0), (0, 1), (1, 0), (1, 1)]
        else:
            shape = [(0, 0), (1, 0), (1, 1)]
        row, col = int(rng.integers(height)), int(rng.integers(width))
        if force_overlap:
            sign = int(rng.choice((-1, 1)))
            dr, dc = (sign, 0) if axis == 0 else (0, sign)
        else:
            dr = int(rng.integers(1 - height, height))
            dc = int(rng.integers(1 - width, width))
            if dr == dc == 0:
                continue
        old = {(row + r, col + c) for r, c in shape}
        new = {(r + dr, c + dc) for r, c in old}
        union = old | new
        if (union & (reserved | forbidden)
                or any(not (0 <= r < height and 0 <= c < width) for r, c in union)
                or np.count_nonzero(first != background) + len(old) > budget):
            continue
        color = int(rng.choice(palette))
        for cell in old:
            first[cell] = color
        for cell in new:
            second[cell] = color
        reserved.update(union)
        return True
    return False


def _latent_pair(rng, shape, background, palette, blank_band, overlap):
    first = np.full(shape, background, dtype=np.int8)
    second = np.full_like(first, background)
    height, width = shape
    forbidden, reserved = set(), set()
    if blank_band:
        if rng.integers(2):
            band = int(rng.integers(1, height - 1))
            forbidden = {(band, c) for c in range(width)}
        else:
            band = int(rng.integers(1, width - 1))
            forbidden = {(r, band) for r in range(height)}
    budget = max(3, int(first.size * .28))
    event_count = int(rng.integers(1, 4))
    for event in range(event_count):
        success = _place_event(rng, first, second, background, palette, reserved, forbidden,
                               budget - 1, force_overlap=overlap and event == 0)
        if not success:
            if event == 0:
                raise RuntimeError("event placement failed")
            break
    # Stationary motifs are independent of the moving footprints.
    desired = int(rng.integers(1, max(2, budget - int(np.count_nonzero(first != background)) + 1)))
    free = [(r, c) for r in range(height) for c in range(width)
            if (r, c) not in reserved | forbidden]
    rng.shuffle(free)
    motif = str(rng.choice(("scatter", "line", "patch", "repeat")))
    center = free[0]
    if motif == "line":
        free.sort(key=lambda p: (p[0] != center[0], abs(p[1] - center[1])))
    elif motif == "patch":
        free.sort(key=lambda p: max(abs(p[0] - center[0]), abs(p[1] - center[1])))
    elif motif == "repeat":
        free.sort(key=lambda p: (p[1] != center[1], p[0] % 2))
    color = int(rng.choice(palette))
    for cell in free[:desired]:
        first[cell] = second[cell] = (int(rng.choice(palette)) if motif == "scatter" else color)
    return first, second


def _camouflage(rng, shape, background, palette, count):
    height, width = shape
    cells = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(cells)
    mode = str(rng.choice(("scatter", "bands", "patch", "repeat")))
    anchor = cells[0]
    if mode == "bands":
        axis = int(rng.integers(2))
        cells.sort(key=lambda p: (abs(p[axis] - anchor[axis]), p[1-axis]))
    elif mode == "patch":
        cells.sort(key=lambda p: max(abs(p[0] - anchor[0]), abs(p[1] - anchor[1])))
    elif mode == "repeat":
        cells.sort(key=lambda p: ((p[0] - anchor[0]) % 2, (p[1] - anchor[1]) % 3))
    result = np.full(shape, background, dtype=np.int8)
    colors = [int(rng.choice(palette)) for _ in range(3)]
    for r, c in cells[:count]:
        result[r, c] = int(rng.choice(palette)) if mode == "scatter" else colors[c % 3]
    return result


def _shape(rng, orientation, scale):
    if scale == "small":
        return int(rng.integers(4, 7)), int(rng.integers(4, 7))
    if scale == "large":
        long, short = int(rng.integers(9, 16)), int(rng.integers(4, 8))
    else:
        long, short = int(rng.integers(4, 13)), int(rng.integers(4, 8))
    return (long, short) if orientation == "vertical" else (short, long)


def _make_pair(rng, shape, phase, roles, orientation, anchor=False, overlap=False):
    background = roles[0]
    available = [c for c in range(10) if c not in roles]
    palette = [int(c) for c in rng.choice(available, size=int(rng.integers(1, 5)), replace=False)]
    a, b = _latent_pair(rng, shape, background, palette, anchor, overlap)
    height, width = shape
    first = np.full((2 * height, 2 * width), background, dtype=np.int8)
    second = np.full_like(first, background)
    wrong = [(r, c) for r in range(2) for c in range(2) if (r, c) != phase]
    densest = int(rng.integers(3))
    occupied = int(np.count_nonzero(a != background))
    for index, (pr, pc) in enumerate(wrong):
        count = max(1, int(a.size * float(rng.uniform(.08, .43))))
        if anchor and index == densest:
            count = int(rng.integers(occupied + 1, max(occupied + 2, int(a.size * .44) + 1)))
        elif anchor and index == (densest + 1) % 3:
            count = int(rng.integers(1, occupied))
        texture = _camouflage(rng, shape, background, palette, count)
        first[pr::2, pc::2] = texture
        second[pr::2, pc::2] = texture
    pr, pc = phase
    first[pr::2, pc::2], second[pr::2, pc::2] = a, b
    return {"input": _join(first, second, roles, orientation),
            "output": _change_map(a, b, background, roles[2], roles[3])}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.permutation(10)[:4])
    regime = str(rng.choice(("phase", "growth", "shrinkage")))
    layout = int(rng.integers(3))
    if layout == 0:
        orientations = ["vertical"] * (num_train + num_test)
    elif layout == 1:
        orientations = ["horizontal"] * (num_train + num_test)
    else:
        orientations = ["vertical", "horizontal"] + [
            str(rng.choice(("vertical", "horizontal")))
            for _ in range(num_train + num_test - 2)]
    first_phase = (int(rng.integers(2)), int(rng.integers(2)))
    complement = (1 - first_phase[0], 1 - first_phase[1])
    remaining = [(r, c) for r in range(2) for c in range(2)
                 if (r, c) not in (first_phase, complement)]
    rng.shuffle(remaining)
    train_pool = [first_phase, complement] + (remaining[:1] if num_train >= 3 else [])
    test_pool = remaining[1:] if num_train >= 3 else remaining
    phases = [first_phase, complement]
    phases += [train_pool[i % len(train_pool)] if regime == "phase" else
               (int(rng.integers(2)), int(rng.integers(2))) for i in range(2, num_train)]
    phases += [test_pool[i % len(test_pool)] if regime == "phase" else
               (int(rng.integers(2)), int(rng.integers(2))) for i in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(num_train + num_test):
        is_train = index < num_train
        if regime == "growth":
            scale = "small" if is_train else "large"
        elif regime == "shrinkage":
            scale = "large" if is_train else "small"
        else:
            scale = "mixed"
        for _ in range(64):
            pair = _make_pair(rng, _shape(rng, orientations[index], scale), phases[index],
                              roles, orientations[index], anchor=index < 2,
                              overlap=index == 1 or rng.random() < .30)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if is_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not create distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _overlap(a, b, background):
    height, width = a.shape
    for row, col in np.argwhere((a != background) & (b == background)):
        color = int(a[row, col])
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            for length in (2, 3, 4):
                end = int(row + length * dr), int(col + length * dc)
                if not (0 <= end[0] < height and 0 <= end[1] < width):
                    continue
                if (a[end] == background and b[end] == color and all(
                    a[row + k * dr, col + k * dc] == color
                    and b[row + k * dr, col + k * dc] == color
                    for k in range(1, length))):
                    return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, roles, info = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _read(inp)
            a, b, bg = d["a"], d["b"], d["background"]
            assert out.shape == a.shape and np.array_equal(solve(inp), out)
            roles.add((bg, d["spine"], d["departure"], d["arrival"]))
            content_colors = set(int(c) for c in np.unique(d["first"])) - {bg}
            assert 1 <= len(content_colors) <= 4
            old, new = (a != bg) & (b == bg), (a == bg) & (b != bg)
            assert int(old.sum()) == int(new.sum()) > 0
            assert np.any((a == b) & (a != bg)) and np.any((a == bg) & (b == bg))
            assert np.all(out[old] == d["departure"]) and np.all(out[new] == d["arrival"])
            assert np.array_equal(out[a == b], a[a == b])
            assert not np.array_equal(out, a) and not np.array_equal(out, b)
            for color in np.unique(d["first"]):
                assert np.count_nonzero(d["first"] == color) == np.count_nonzero(d["second"] == color)
            occupied, wrong_counts = int(np.count_nonzero(a != bg)), []
            for pr in range(2):
                for pc in range(2):
                    if (pr, pc) == d["phase"]:
                        continue
                    x, y = d["first"][pr::2, pc::2], d["second"][pr::2, pc::2]
                    assert np.array_equal(x, y) and np.any(x != bg)
                    assert not np.array_equal(x, out)  # B-free phase selection cannot fit.
                    wrong_counts.append(int(np.count_nonzero(x != bg)))
            full_map = _change_map(d["first"], d["second"], bg, d["departure"], d["arrival"])
            assert full_map.shape != out.shape  # A-free comparison leaves camouflage.
            d["denser_wrong"] = max(wrong_counts) > occupied
            d["sparser_wrong"] = min(wrong_counts) < occupied
            d["blank_band"] = bool(np.any(np.all((a == bg) & (b == bg), axis=0))
                                    or np.any(np.all((a == bg) & (b == bg), axis=1)))
            d["overlap"] = _overlap(a, b, bg)
            info[split].append(d)
    assert len(roles) == 1
    train, test = info["train"], info["test"]
    train_phases = {d["phase"] for d in train}
    assert any((1-r, 1-c) in train_phases for r, c in train_phases)
    assert any(d["blank_band"] for d in train)
    assert any(d["overlap"] for d in train)
    assert sum(d["denser_wrong"] and d["sparser_wrong"] for d in train) >= 2
    assert {d["orientation"] for d in test} <= {d["orientation"] for d in train}
    new_phase = all(d["phase"] not in train_phases for d in test)
    bigger = min(max(d["a"].shape) for d in test) > max(max(d["a"].shape) for d in train)
    smaller = max(max(d["a"].shape) for d in test) < min(max(d["a"].shape) for d in train)
    assert new_phase or bigger or smaller


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed coverage plus input-only metamorphic and API checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(123, 7, 3)) == _puzzle_key(generate_puzzle(123, 7, 3))
    phases, orientations, shapes, palettes, changes, structures = set(), set(), set(), set(), set(), set()
    phase_shifts = growth = shrinkage = 0
    for puzzle in puzzles:
        train_info = [_read(pair["input"]) for pair in puzzle["train"]]
        test_info = [_read(pair["input"]) for pair in puzzle["test"]]
        phase_shifts += all(d["phase"] not in {t["phase"] for t in train_info} for d in test_info)
        growth += min(max(d["a"].shape) for d in test_info) > max(max(t["a"].shape) for t in train_info)
        shrinkage += max(max(d["a"].shape) for d in test_info) < min(max(t["a"].shape) for t in train_info)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read(pair["input"])
                bg, pr, pc = d["background"], *d["phase"]
                phases.add(d["phase"])
                orientations.add(d["orientation"])
                shapes.add(d["a"].shape)
                palettes.add(tuple(sorted(set(int(c) for c in np.unique(d["first"])) - {bg})))
                changes.add(int(np.count_nonzero(d["a"] != d["b"])))
                mapping, normalized = {}, []
                for c in pair["output"].flat:
                    normalized.append(mapping.setdefault(int(c), len(mapping)))
                structures.add((pair["output"].shape, tuple(normalized)))
        # Move common off-phase ink while preserving its palette and occupancy.
        pair = puzzle["train"][0]
        d = _read(pair["input"])
        roles = d["background"], d["spine"], d["departure"], d["arrival"]
        first, second = d["first"].copy(), d["second"].copy()
        for r in range(2):
            for c in range(2):
                if (r, c) != d["phase"]:
                    texture = first[r::2, c::2].copy()
                    occupied = tuple(np.argwhere(texture != roles[0])[0])
                    empty = tuple(np.argwhere(texture == roles[0])[0])
                    texture[empty], texture[occupied] = texture[occupied], roles[0]
                    first[r::2, c::2] = second[r::2, c::2] = texture
        assert np.array_equal(solve(_join(first, second, roles, d["orientation"])), pair["output"])
        # A new visible key, without changing panels, swaps exactly the markers.
        swapped = (roles[0], roles[1], roles[3], roles[2])
        expected = pair["output"].copy()
        expected[pair["output"] == roles[2]] = roles[3]
        expected[pair["output"] == roles[3]] = roles[2]
        assert np.array_equal(solve(_join(first, second, swapped, d["orientation"])), expected)
        # Fine-grid translation changes phase; a crossed boundary becomes a coarse roll.
        rolled = _join(np.roll(d["first"], (1, 1), (0, 1)),
                       np.roll(d["second"], (1, 1), (0, 1)), roles, d["orientation"])
        assert np.array_equal(solve(rolled), np.roll(pair["output"], d["phase"], (0, 1)))
    assert len(phases) == 4 and len(orientations) == 2 and len(shapes) >= 30
    assert len(palettes) >= 30 and len(changes) >= 8 and len(structures) >= 150
    assert min(phase_shifts, growth, shrinkage) >= 5
    # Hand-derived witness, with an overlapping bar, stationary 4, and blank bands.
    a = np.array([[3, 3, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]])
    b = np.array([[0, 3, 3, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]])
    expected = np.array([[2, 3, 1, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]])
    roles = (0, 5, 2, 1)
    for pr in range(2):
        for pc in range(2):
            first, second = np.zeros((8, 8), dtype=int), np.zeros((8, 8), dtype=int)
            for r in range(2):
                for c in range(2):
                    if (r, c) != (pr, pc):
                        first[r, c] = second[r, c] = 4
            first[pr::2, pc::2], second[pr::2, pc::2] = a, b
            for orientation in ("vertical", "horizontal"):
                assert np.array_equal(solve(_join(first, second, roles, orientation)), expected)
    malformed_key = _join(first, second, roles, "vertical")
    malformed_key[2, 8] = 2
    mixed_phase = second.copy()
    mixed_phase[1-pr, 1-pc] = 0
    for invalid in (malformed_key, _join(first, first, roles, "vertical"),
                    _join(first, mixed_phase, roles, "vertical")):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or malformed input accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1000 + 10 * nt + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "phases": len(phases), "orientations": len(orientations),
            "output_shapes": len(shapes), "content_palettes": len(palettes),
            "change_counts": len(changes), "normalized_outputs": len(structures),
            "phase_shift_puzzles": phase_shifts, "growth_puzzles": growth,
            "shrinkage_puzzles": shrinkage}


if __name__ == "__main__":
    print(self_test())
