"""Color counters expose blank windows. See task_spec.md for the family contract.

No parent module, file, seed, or generation state is consulted by solve().
Generation constructs compatible color layers directly, never parent puzzles.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HIGHLIGHT = 1
HOST_SEEDS = (0,352825948,681584266,957242497,646826428,868878146,
              227527395,829920445,604993497,386617659,554884881,423754740,
              204924525,452618977,449145196,890404411,336945934,342682864,
              338710786,813516181,5388217,166404358,425511240,44228800,
              831915831,844142873,158092656,28794057,512411757,114876583,
              260567979,947785577,40977728,685897097,507411595,965473994,
              244197933,389581389,329558956,977113609,463476458,377496381)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < low or
            (high is not None and value > high)):
        raise ValueError(f"invalid {name}: {value!r}")
    return int(value)


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _window_union(square, background):
    """Simultaneous union, computed before recoloring, including overlaps."""
    h, w = square.shape
    union = np.zeros((h, w), dtype=bool)
    if min(h, w) < 3:
        return union, []
    starts = np.ones((h - 2, w - 2), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            starts &= square[dy:dy + h - 2, dx:dx + w - 2] == background
    for dy in range(3):
        for dx in range(3):
            union[dy:dy + h - 2, dx:dx + w - 2] |= starts
    return union, [tuple(map(int, p)) for p in np.argwhere(starts)]


def _rotate_layers(source, background, counts):
    result = np.full(source.shape, background, dtype=np.int8)
    used = np.zeros(source.shape, dtype=bool)
    for color in np.unique(source):
        color = int(color)
        if color == background:
            continue
        mask = np.rot90(source == color, -counts.get(color, 0) % 4)
        if np.any(mask & used):
            raise ValueError("rotated color masks overlap")
        result[mask] = color
        used |= mask
    return result


def _candidates(inp):
    """All tight square crops with a clear halo and matching exterior bars.

    An occupancy integral image searches every position and every possible size.
    Source layers need not be connected or have a solid border. This input-only
    grammar is adapted from the localization principle in parent B's final code.
    """
    if np.any(inp == HIGHLIGHT):
        raise ValueError("highlight color 1 is reserved for output")
    values, frequencies = np.unique(inp, return_counts=True)
    if np.sum(frequencies == frequencies.max()) != 1:
        raise ValueError("background must be the unique most frequent input color")
    bg = int(values[frequencies.argmax()])
    occupied = inp != bg
    total = int(occupied.sum())
    h, w = inp.shape
    prefix = np.pad(occupied.astype(np.int16), ((1, 0), (1, 0)))
    prefix = prefix.cumsum(0).cumsum(1)

    def area(r, c, e, f):
        return prefix[e, f] - prefix[r, f] - prefix[e, c] + prefix[r, c]

    found = []
    for n in range(3, min(h, w) + 1):
        rr, cc = np.indices((h - n + 1, w - n + 1))
        ee, ff = rr + n, cc + n
        inside = area(rr, cc, ee, ff)
        valid = (inside >= 2) & (inside < total)
        valid &= area(rr, cc, rr + 1, ff) > 0
        valid &= area(ee - 1, cc, ee, ff) > 0
        valid &= area(rr, cc, ee, cc + 1) > 0
        valid &= area(rr, ff - 1, ee, ff) > 0
        valid &= area(np.maximum(0, rr - 1), np.maximum(0, cc - 1),
                      np.minimum(h, ee + 1), np.minimum(w, ff + 1)) == inside
        for r, c in np.argwhere(valid):
            r, c = int(r), int(c)
            source = inp[r:r + n, c:c + n]
            colors = tuple(int(x) for x in np.unique(source) if x != bg)
            if len(colors) < 2:
                continue
            exterior = occupied.copy()
            exterior[r:r + n, c:c + n] = False
            markers, marker_cells = {}, []
            good = True
            for color in np.unique(inp[exterior]):
                color = int(color)
                coords = np.argwhere(exterior & (inp == color))
                ys, xs = coords.T
                horizontal = np.ptp(ys) == 0 and np.ptp(xs) + 1 == len(xs)
                vertical = np.ptp(xs) == 0 and np.ptp(ys) + 1 == len(ys)
                if color not in colors or not (horizontal or vertical):
                    good = False
                    break
                orientation = "single" if len(xs) == 1 else "h" if horizontal else "v"
                markers[color] = {"length": len(xs), "orientation": orientation,
                                  "coords": tuple(map(tuple, coords.tolist()))}
                marker_cells.extend((color, int(y), int(x)) for y, x in coords)
            if not good or not markers:
                continue
            if any(a != b and max(abs(y - u), abs(x - v)) <= 1
                   for i, (a, y, x) in enumerate(marker_cells)
                   for b, u, v in marker_cells[i + 1:]):
                continue
            found.append({"background": bg, "source": source.copy(),
                          "row": r, "col": c, "size": n,
                          "colors": colors, "markers": markers})
    return found


def _parse(inp):
    found = _candidates(inp)
    if len(found) != 1:
        raise ValueError(f"expected one identifiable square, found {len(found)}")
    return found[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the square after counted rotations and blank-window highlighting."""
    info = _parse(_grid(input_grid))
    counts = {c: m["length"] for c, m in info["markers"].items()}
    square = _rotate_layers(info["source"], info["background"], counts)
    footprint, _ = _window_union(square, info["background"])
    square[footprint] = HIGHLIGHT
    return square


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    orbits = []
    while remaining:
        r, c = min(remaining)
        orbit = list(dict.fromkeys(((r, c), (c, n - 1 - r),
                                   (n - 1 - r, n - 1 - c), (n - 1 - c, r))))
        remaining.difference_update(orbit)
        orbits.append(orbit)
    return orbits


def _dilate(mask):
    padded = np.pad(mask, 1)
    result = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy:dy + h, dx:dx + w]
    return result


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _target_field(rng, n, background, colors, profile):
    """Compose a fresh final field; allocate whole rotation orbits to colors.

    Owning an orbit does not mean filling it. Partial, asymmetric occupancy
    exposes different empty windows when the independently rotated layers move.
    The orbit ownership is only a construction device, never a solver rule.
    """
    blank = np.zeros((n, n), dtype=bool)
    patches = ([(3, 3)] if profile == 0 else [(3, 4), (3, 3)] if profile == 1
               else [((3, 3), (3, 4), (4, 4), (3, 5), (4, 4))[int(rng.integers(5))]
                     for _ in range(int(rng.integers(1, 4)))])
    for index, shape in enumerate(patches):
        ph, pw = shape
        if rng.random() < .5:
            ph, pw = pw, ph
        reserved = _dilate(blank)
        options = [(r, c) for r in range(n - ph + 1) for c in range(n - pw + 1)
                   if not reserved[r:r + ph, c:c + pw].any()]
        if not options:
            return None
        edges = [(r, c) for r, c in options if r in (0, n - ph) or c in (0, n - pw)]
        pool = edges if edges and rng.random() < .25 else options
        r, c = pool[int(rng.integers(len(pool)))]
        patch = np.ones((ph, pw), dtype=bool)
        # Later examples can have the union of two offset windows, rather than
        # only rectangular holes. Both 3x3 windows are explicitly part of it.
        if profile >= 2 and ph == pw == 4 and rng.random() < .5:
            patch[:] = False
            patch[:3, :3] = True
            patch[1:, 1:] = True
        blank[r:r + ph, c:c + pw] |= patch
    # A narrow attached tail must remain unhighlighted; connected-component
    # filling is therefore an observable wrong rule, even in small curricula.
    tails = []
    for r, c in np.argwhere(blank):
        r, c = int(r), int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            points = [(r + dr, c + dc), (r + 2 * dr, c + 2 * dc)]
            if all(0 <= y < n and 0 <= x < n and not blank[y, x] for y, x in points):
                tails.append(points)
    if tails:
        for p in tails[int(rng.integers(len(tails)))]:
            blank[p] = True
    # An eight-of-nine blank near miss, separated from the real blank patches.
    reserved = _dilate(blank)
    options = [(r, c) for r in range(n - 2) for c in range(n - 2)
               if not reserved[r:r + 3, c:c + 3].any()]
    if not options:
        return None
    r, c = options[int(rng.integers(len(options)))]
    near = np.ones((3, 3), dtype=bool)
    near[int(rng.integers(3)), int(rng.integers(3))] = False
    blank[r:r + 3, c:c + 3] |= near
    ring = _dilate(blank) & ~blank

    orbits = _orbits(n)
    depth = np.array([min(r, c, n - 1 - r, n - 1 - c) for (r, c), *_ in orbits])
    # Balanced but variably sized territories: radial bands, folded spatial
    # patches, or dispersed orbits. None is a visible source locator.
    mode = int(rng.integers(3))
    if mode == 0:
        scores = depth + rng.random(len(orbits)) * .8
    elif mode == 1:
        ar, ac = rng.uniform(0, n / 2, 2)
        scores = np.array([min((r - ar)**2 + (c - ac)**2 for r, c in orbit)
                           for orbit in orbits]) + rng.random(len(orbits))
    else:
        scores = rng.random(len(orbits))
    order = np.argsort(scores)
    allocations = np.full(len(colors), 2)
    allocations += rng.multinomial(len(orbits) - int(allocations.sum()),
                                   rng.dirichlet(np.full(len(colors), 3.0)))
    owners = np.zeros((n, n), dtype=np.int8)
    cursor = 0
    for k in rng.permutation(len(colors)):
        for i in order[cursor:cursor + allocations[k]]:
            for p in orbits[int(i)]:
                owners[p] = k
        cursor += int(allocations[k])
    density = float(rng.uniform(.45, .80))
    occupied = rng.random((n, n)) < density
    occupied[ring] = True
    occupied[blank] = False
    # Full occupancy of just four related edge cells determines the common
    # square extent without imposing a frame or a connected source component.
    extent = [o for o, d in zip(orbits, depth) if d == 0 and not any(blank[p] for p in o)]
    if not extent:
        return None
    for p in extent[int(rng.integers(len(extent)))]:
        occupied[p] = True
    target = np.full((n, n), background, dtype=np.int8)
    for k, color in enumerate(colors):
        target[occupied & (owners == k)] = color
    if any(np.sum(target == color) < 3 for color in colors):
        return None
    return target


def _place_bars(rng, source, background, counts, forced):
    """Bounded packing, followed by a guaranteed right/bottom-lane layout."""
    n = len(source)
    for attempt in range(19):
        if attempt == 18:
            h = w = 30
            r = c = 0
        else:
            h, w = (int(rng.integers(n + 3, 31)) for _ in range(2))
            r, c = int(rng.integers(h - n + 1)), int(rng.integers(w - n + 1))
        inp = np.full((h, w), background, dtype=np.int8)
        inp[r:r + n, c:c + n] = source
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0, r - 1):min(h, r + n + 1), max(0, c - 1):min(w, c + n + 1)] = True
        success = True
        for index, (color, length) in enumerate(sorted(counts.items(), key=lambda p: -p[1])):
            if attempt == 18:
                orientation = forced.get(color, "h")
                if orientation == "h":
                    cells = [(2 * index, n + 2 + i) for i in range(length)]
                else:
                    cells = [(n + 2 + i, 2 * index) for i in range(length)]
            else:
                groups = []
                for orientation in ("h", "v"):
                    if color in forced and forced[color] != orientation:
                        continue
                    bh, bw = (1, length) if orientation == "h" else (length, 1)
                    yy, xx = np.indices((h - bh + 1, w - bw + 1))
                    good = np.ones(yy.shape, dtype=bool)
                    for i in range(length):
                        good &= ~blocked[yy + (i if orientation == "v" else 0),
                                         xx + (i if orientation == "h" else 0)]
                    for side in (yy + bh <= r - 1, yy >= r + n + 1,
                                 xx + bw <= c - 1, xx >= c + n + 1):
                        starts = np.argwhere(good & side)
                        if len(starts):
                            groups.append((orientation, starts))
                if not groups:
                    success = False
                    break
                orientation, starts = groups[int(rng.integers(len(groups)))]
                y, x = map(int, starts[int(rng.integers(len(starts)))])
                cells = [(y + (i if orientation == "v" else 0),
                          x + (i if orientation == "h" else 0)) for i in range(length)]
            for y, x in cells:
                inp[y, x] = color
                blocked[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)] = True
        if not success:
            continue
        values, frequencies = np.unique(inp, return_counts=True)
        if int(values[frequencies.argmax()]) != background or np.sum(frequencies == frequencies.max()) != 1:
            continue
        candidates = _candidates(inp)
        if len(candidates) == 1 and (candidates[0]["row"], candidates[0]["col"], candidates[0]["size"]) == (r, c, n):
            return inp
    return None


def _make_pair(rng, witness, test):
    layer_count = 3 if witness == 0 else 4 if witness == 1 else int(rng.integers(2, 6))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != HIGHLIGHT])))
    background, colors = palette[0], palette[1:1 + layer_count]
    if witness == 0:
        lengths = [1, 2, 2]
    elif witness == 1:
        lengths = [3, 0, 5, 4]
    else:
        lengths = list(map(int, rng.integers(0, 10 if test else 6, layer_count)))
        if test:
            lengths[int(rng.integers(layer_count))] = int(rng.integers(6, 10))
        if not any(k % 4 for k in lengths):
            lengths[0] = int(rng.integers(1, 4))
    counts = {c: k for c, k in zip(colors, lengths) if k}
    forced = {colors[1]: "h", colors[2]: "v"} if witness == 0 else {}
    for _ in range(160):
        n = int(rng.integers(9, 15)) if witness == 0 else int(rng.integers(13, 20)) if witness == 1 else int(rng.integers(10, 20))
        target = _target_field(rng, n, background, colors, max(0, witness) if witness >= 0 else 2)
        if target is None:
            continue
        source = _rotate_layers(target, background, {c: -k for c, k in counts.items()})
        if witness >= 0 and not all(_asymmetric(source == c) for c in colors):
            continue
        footprint, starts = _window_union(target, background)
        before, _ = _window_union(source, background)
        if not starts or np.array_equal(footprint, before):
            continue
        if not np.any(footprint & (source != background)):
            continue
        if witness == 0 and (len(starts) != 1 or footprint.sum() != 9):
            continue
        if witness == 1 and not (len(starts) == 3 and footprint.sum() == 21):
            continue
        if not np.any((target == background) & ~footprint):
            continue
        inp = _place_bars(rng, source, background, counts, forced)
        if inp is None:
            continue
        out = target.copy()
        out[footprint] = HIGHLIGHT
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction could not produce an informative pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(8):
                pair = _make_pair(rng, index if split == "train" and index < 2 else -1, split == "test")
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Coordinate and scalar-window audit, independent of generated target."""
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise ValueError("pair must have exactly input and output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(inp)
    bg, source, n = info["background"], info["source"], info["size"]
    assert out.shape == (n, n)
    rotated = np.full((n, n), bg, dtype=np.int8)
    for color in info["colors"]:
        turns = info["markers"].get(color, {}).get("length", 0) % 4
        for r, c in np.argwhere(source == color):
            r, c = int(r), int(c)
            for _ in range(turns):
                r, c = c, n - 1 - r
            assert rotated[r, c] == bg, "conflicting masks"
            rotated[r, c] = color
        assert np.sum(source == color) == np.sum(out == color), "layer area changed"
    union, starts = np.zeros((n, n), dtype=bool), []
    for r in range(n - 2):
        for c in range(n - 2):
            if all(rotated[r + dy, c + dx] == bg for dy in range(3) for dx in range(3)):
                union[r:r + 3, c:c + 3] = True
                starts.append((r, c))
    expected = rotated.copy()
    expected[union] = HIGHLIGHT
    assert np.array_equal(out, expected), "incorrect rotations, blank union, or crop"
    assert np.array_equal(out == HIGHLIGHT, union)
    assert starts and np.any((rotated == bg) & ~union)
    # Deliberate near misses are background-rich but must not be painted.
    assert any(not union[r:r + 3, c:c + 3].any() and
               np.sum(rotated[r:r + 3, c:c + 3] == bg) == 8
               for r in range(n - 2) for c in range(n - 2)), "missing preserved near miss"
    before, _ = _window_union(source, bg)
    assert not np.array_equal(before, union), "B does not change the blank footprint"
    assert np.any(union & (source != bg)), "no newly exposed target cells"
    assert np.any(out != rotated), "A does not change the answer"
    info.update(rotated=rotated, footprint=union, starts=starts)
    return info


def validate_generated_puzzle(puzzle) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, training, test_lengths = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            assert key not in used, "duplicate input"
            used.add(key)
            info = _check_pair(pair)
            assert 2 <= len(info["colors"]) <= 5
            assert all(1 <= m["length"] <= 9 for m in info["markers"].values())
            if split == "train":
                training.append(info)
            else:
                test_lengths.append(max(m["length"] for m in info["markers"].values()))
    assert all(k >= 6 for k in test_lengths), "missing long-counter shift"
    lengths, unmarked, repeated_orientations = set(), False, False
    for info in training:
        same_length = {}
        for color in info["colors"]:
            if not _asymmetric(info["source"] == color):
                continue
            marker = info["markers"].get(color)
            if marker is None:
                unmarked = True
            else:
                length = marker["length"]
                lengths.add(length)
                same_length.setdefault(length, set()).add(marker["orientation"])
        repeated_orientations |= any({"h", "v"} <= directions for directions in same_length.values())
    assert {1, 2, 3, 4, 5} <= lengths and unmarked and repeated_orientations
    assert any(len(i["starts"]) == 1 and i["footprint"].sum() == 9 for i in training)
    assert any(i["footprint"].sum() < 9 * len(i["starts"]) for i in training)
    # A blank component containing a target also has a narrow unpainted tail.
    assert any(np.any(_dilate(i["footprint"]) & (i["rotated"] == i["background"]) &
                         ~i["footprint"]) for i in training), "component filling is not distinguished"
    assert all(m["length"] <= 5 for i in training for m in i["markers"].values())


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
    """42 host seeds, streaming metrics, no files, no retained grid archive."""
    shapes, sizes, backgrounds, layer_counts, footprints, hashes = (set() for _ in range(6))
    structures, lengths, orientations, sides = set(), set(), set(), set()
    overlap = edge_windows = remaining_blanks = pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprint = _puzzle_digest(puzzle)
        assert fingerprint == _puzzle_digest(generate_puzzle(seed))
        hashes.add(fingerprint)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                info = _check_pair(pair)
                assert np.array_equal(solve(inp.copy()), pair["output"])
                bg, n, r, c = (info[k] for k in ("background", "size", "row", "col"))
                shapes.add(inp.shape)
                sizes.add(n)
                backgrounds.add(bg)
                layer_counts.add(len(info["colors"]))
                footprints.add(int(info["footprint"].sum()))
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized).hexdigest())
                overlap += info["footprint"].sum() < 9 * len(info["starts"])
                edge_windows += any(y in (0, n - 3) or x in (0, n - 3) for y, x in info["starts"])
                remaining_blanks += int(np.sum((info["rotated"] == bg) & ~info["footprint"]))
                for m in info["markers"].values():
                    lengths.add(m["length"])
                    orientations.add(m["orientation"])
                    ys, xs = np.array(m["coords"]).T
                    sides.update(name for name, yes in (("top", max(ys) < r), ("bottom", min(ys) >= r + n),
                                                       ("left", max(xs) < c), ("right", min(xs) >= c + n)) if yes)
                pairs += 1
    assert len(hashes) == 42 and len(structures) >= 200
    assert len(shapes) >= 35 and len(sizes) >= 9 and len(backgrounds) == 9
    assert layer_counts == {2, 3, 4, 5} and lengths == set(range(1, 10))
    assert orientations == {"single", "h", "v"} and len(sides) == 4
    assert overlap >= 42 and edge_windows > 0 and len(footprints) >= 12
    # Exhaust the entire explicit count API at one seed, holding one puzzle.
    legal_counts = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(71, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            legal_counts += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corruption must not pass the semantic checker.
    bad = generate_puzzle(17, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Full family boundary semantics, beyond the sampled presentation range.
    field = np.full((5, 5), 7, dtype=np.int8)
    union, starts = _window_union(field, 7)
    assert len(starts) == 9 and union.all()
    field[2, 2] = 2
    assert not _window_union(field, 7)[0].any()
    # A hand-constructed compatible input outside the sampling grammar: only
    # two foreground cells, shared orbit ownership, side three, and a ten-cell
    # counter. This also exercises an empty highlight union and no mutation.
    small = np.zeros((13, 13), dtype=np.int8)
    small[1, 1], small[3, 3] = 2, 3
    small[:5, 12] = 2
    small[9, 1:11] = 3
    preserved = small.copy()
    wanted = np.array([[3, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
    assert np.array_equal(solve(small), wanted) and np.array_equal(small, preserved)
    try:
        _rotate_layers(small[1:4, 1:4], 0, {2: 1, 3: 3})
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting rotations accepted")
    return {"seeds": 42, "pairs": pairs, "legal_count_combinations": legal_counts,
            "distinct_puzzles": len(hashes), "input_shapes": len(shapes),
            "square_sizes": sorted(sizes), "backgrounds": sorted(backgrounds),
            "layer_counts": sorted(layer_counts), "color_normalized_inputs": len(structures),
            "highlight_areas": sorted(footprints), "overlapping_window_pairs": int(overlap),
            "edge_window_pairs": int(edge_windows), "unhighlighted_blank_cells": remaining_blanks,
            "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
