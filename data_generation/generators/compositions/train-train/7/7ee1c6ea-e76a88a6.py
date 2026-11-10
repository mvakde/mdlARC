"""Transport a visible motif between the alternating phases of rectangular frames.

Only standard-library Python and NumPy are used. The solver parses its input;
it has no access to generation parameters, example pairs, or random state.
"""

from collections import deque
import numpy as np


_TILE_SHAPES = tuple((h, w) for h in range(3, 6) for w in range(3, 6))


def _orientation_alternatives(tile):
    moved = [np.fliplr(tile), np.flipud(tile), np.rot90(tile, 2)]
    if tile.shape[0] == tile.shape[1]:
        moved += [tile.T, np.rot90(tile), np.rot90(tile, 3), np.rot90(tile.T, 2)]
    return moved


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=False)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < height and 0 <= xx < width
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        result.append(cells)
    return result


def _fields(shape, frames):
    outline = np.zeros(shape, dtype=bool)
    depth = np.zeros(shape, dtype=int)
    for t, l, b, r in frames:  # inclusive bounds
        outline[t, l:r + 1] = True
        outline[b, l:r + 1] = True
        outline[t:b + 1, l] = True
        outline[t:b + 1, r] = True
        depth[t + 1:b, l + 1:r] += 1
    return outline, depth


def _separate(a, b, h, w):
    return (a[0] + h < b[0] or b[0] + h < a[0]
            or a[1] + w < b[1] or b[1] + w < a[1])


def _corners_clear(outline, y, x, h, w):
    return not (outline[y, x] or outline[y + h - 1, x]
                or outline[y, x + w - 1] or outline[y + h - 1, x + w - 1])


def _parse(a):
    """Recover roles, complete source, interrupted targets, and enclosure depths."""
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() * 2 <= a.size:
        raise ValueError("the neutral background must occupy more than half the grid")
    frame_sets = []
    for color in colors:
        color = int(color)
        if color == bg:
            continue
        boxes = []
        components = _components(a == color)
        for cells in components:
            yy, xx = np.array(cells).T
            t, l, b, r = int(yy.min()), int(xx.min()), int(yy.max()), int(xx.max())
            if b - t + 1 < 7 or r - l + 1 < 7:
                continue
            expected, _ = _fields(a.shape, [(t, l, b, r)])
            if len(cells) == int(expected.sum()) and all(expected[y, x] for y, x in cells):
                boxes.append((t, l, b, r))
        if boxes and len(boxes) == len(components):
            frame_sets.append((color, tuple(sorted(boxes))))
    if len(frame_sets) != 1:
        raise ValueError("expected one dedicated color making complete separated frames")
    frame_color, frames = frame_sets[0]
    if not 1 <= len(frames) <= 3:
        raise ValueError("expected one to three frames")
    outline, depth = _fields(a.shape, frames)
    if not np.array_equal(outline, a == frame_color):
        raise ValueError("frame color may occur only on the outlines")
    remaining = set(map(int, colors)) - {bg, frame_color}
    if len(remaining) != 3:
        raise ValueError("expected a blank color and two payload colors")
    possibilities = []
    for blank in sorted(remaining):
        payload = tuple(sorted(remaining - {blank}))
        source_mask = np.isin(a, payload)
        ys, xs = np.nonzero(source_mask)
        if not len(ys):
            continue
        sy, sx = int(ys.min()), int(xs.min())
        h, w = int(ys.max()) - sy + 1, int(xs.max()) - sx + 1
        if (h, w) not in _TILE_SHAPES or h * w != len(ys):
            continue
        tile = a[sy:sy + h, sx:sx + w].copy()
        if len(np.unique(depth[sy:sy + h, sx:sx + w])) != 1:
            continue
        targets = []
        coverage = np.zeros(a.shape, dtype=int)
        for y in range(a.shape[0] - h + 1):
            for x in range(a.shape[1] - w + 1):
                if not (a[y, x] == a[y + h - 1, x]
                        == a[y, x + w - 1] == a[y + h - 1, x + w - 1] == blank):
                    continue
                patch = a[y:y + h, x:x + w]
                if np.all((patch == blank) | (patch == frame_color)):
                    targets.append((y, x))
                    coverage[y:y + h, x:x + w] += patch == blank
        if (not 1 <= len(targets) <= 5
                or not np.array_equal(coverage, (a == blank).astype(int))):
            continue
        positions = [(sy, sx)] + targets
        if not all(_separate(p, q, h, w)
                   for i, p in enumerate(positions) for q in positions[i + 1:]):
            continue
        possibilities.append(dict(background=bg, frame_color=frame_color,
                                  blank=blank, payload=payload, frames=frames,
                                  outline=outline, depth=depth, source=(sy, sx),
                                  tile=tile, targets=tuple(targets),
                                  source_depth=int(depth[sy, sx])))
    if len(possibilities) != 1:
        raise ValueError("source and congruent blank rectangles must be unique")
    return possibilities[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _parse(a)
    out = a.copy()
    tile = info["tile"]
    h, w = tile.shape
    first, second = info["payload"]
    for y, x in info["targets"]:
        reverse = ((info["depth"][y:y + h, x:x + w] - info["source_depth"]) % 2) == 1
        stamp = np.where(reverse, np.where(tile == first, second, first), tile)
        patch = out[y:y + h, x:x + w]
        writable = a[y:y + h, x:x + w] == info["blank"]
        patch[writable] = stamp[writable]
    return out


def _make_frames(rng, kind, h, w):
    if kind == "single":
        height, width = map(int, rng.integers(19, 31, size=2))
        fh = int(rng.integers(max(8, h + 2), height - 2))
        fw = int(rng.integers(max(8, w + 2), width - 2))
        t = int(rng.integers(height - fh + 1))
        l = int(rng.integers(width - fw + 1))
        frames = [(t, l, t + fh - 1, l + fw - 1)]
    elif kind == "nested":
        height, width = map(int, rng.integers(25, 31, size=2))
        # An exterior strip and a wide annulus support the two teaching sources.
        t = h + int(rng.integers(1, 3))
        l = int(rng.integers(0, 3))
        b = height - 1 - int(rng.integers(0, 3))
        r = width - 1 - int(rng.integers(0, 3))
        inner = (t + h + 1, l + w + 1,
                 b - int(rng.integers(2, 5)), r - int(rng.integers(2, 5)))
        frames = [(t, l, b, r), inner]
    elif kind == "triple":
        height, width = map(int, rng.integers(27, 31, size=2))
        box = (int(rng.integers(0, 2)), int(rng.integers(0, 2)),
               height - 1 - int(rng.integers(0, 2)),
               width - 1 - int(rng.integers(0, 2)))
        frames = [box]
        for _ in range(2):
            dt, dl, db, dr = map(int, rng.integers(2, 5, size=4))
            t, l, b, r = frames[-1]
            frames.append((t + dt, l + dl, b - db, r - dr))
    elif kind == "disjoint":
        height, width = int(rng.integers(20, 31)), int(rng.integers(24, 31))
        widths = [int(rng.integers(7, 11)), int(rng.integers(7, 11))]
        gap = int(rng.integers(1, 4))
        spare = width - sum(widths) - gap
        left = int(rng.integers(spare + 1))
        frames = []
        for fw in widths:
            fh = int(rng.integers(max(8, h + 2), height + 1))
            t = int(rng.integers(height - fh + 1))
            frames.append((t, left, t + fh - 1, left + fw - 1))
            left += fw + gap
    else:
        raise ValueError("unknown layout kind")
    if any(b - t + 1 < max(7, h + 2) or r - l + 1 < max(7, w + 2)
           for t, l, b, r in frames):
        return None
    # Reflections and transposition remove any privileged side of the scene.
    if rng.random() < .5:
        frames = [(height - 1 - b, l, height - 1 - t, r) for t, l, b, r in frames]
    if rng.random() < .5:
        frames = [(t, width - 1 - r, b, width - 1 - l) for t, l, b, r in frames]
    if rng.random() < .5:
        height, width = width, height
        frames = [(l, t, r, b) for t, l, b, r in frames]
    return (height, width), frames


def _motif(rng, h, w, payload, asymmetric):
    yy, xx = np.indices((h, w))
    for _ in range(100):
        mode = int(rng.integers(6))
        if mode == 0:
            bits = rng.random((h, w)) < rng.uniform(.2, .8)
        elif mode == 1:
            bits = yy + xx <= int(rng.integers(1, h + w - 3))
        elif mode == 2:
            bits = yy < int(rng.integers(1, h))
        elif mode == 3:
            bits = (yy < int(rng.integers(1, h))) & (xx < int(rng.integers(1, w)))
        elif mode == 4:
            bits = (yy + xx) % 2 == 0
        else:
            bits = np.zeros((h, w), dtype=bool)
            bits.flat[rng.choice(h * w, size=int(rng.integers(2, h * w - 1)), replace=False)] = True
        if rng.random() < .5:
            bits = np.fliplr(bits)
        if rng.random() < .5:
            bits = np.flipud(bits)
        if not 2 <= bits.sum() <= h * w - 2:
            continue
        if asymmetric and any(np.array_equal(bits, v) for v in _orientation_alternatives(bits)):
            continue
        if rng.random() < .5:
            bits = ~bits
        return np.where(bits, payload[1], payload[0]).astype(int)
    raise RuntimeError("could not make an informative motif")


def _corner_cross(position, h, w, frame):
    y, x = position
    t, l, b, r = frame
    return any(y < cy < y + h - 1 and x < cx < x + w - 1
               for cy, cx in ((t, l), (t, r), (b, l), (b, r)))


def _reference_output(a, frames, source, targets, tile, roles):
    """Direct per-cell construction, separate from the vectorized solver."""
    _, frame_color, blank, first, second = roles
    sy, sx = source
    source_depth = sum(t < sy < b and l < sx < r for t, l, b, r in frames)
    out = a.copy()
    h, w = tile.shape
    for y, x in targets:
        for i in range(h):
            for j in range(w):
                if a[y + i, x + j] != blank:
                    assert a[y + i, x + j] == frame_color
                    continue
                d = sum(t < y + i < b and l < x + j < r for t, l, b, r in frames)
                color = int(tile[i, j])
                if (d - source_depth) % 2:
                    color = second if color == first else first
                out[y + i, x + j] = color
    return out


def _make_pair(rng, shape, kind, source_depth, count, roles, teaching):
    h, w = shape
    bg, frame_color, blank, first, second = roles
    for _ in range(160):
        layout = _make_frames(rng, kind, h, w)
        if layout is None:
            continue
        canvas, frames = layout
        outline, depth = _fields(canvas, frames)
        positions = [(y, x) for y in range(canvas[0] - h + 1)
                     for x in range(canvas[1] - w + 1)]
        sources = [p for p in positions
                   if not np.any(outline[p[0]:p[0] + h, p[1]:p[1] + w])
                   and np.all(depth[p[0]:p[0] + h, p[1]:p[1] + w] == source_depth)]
        if not sources:
            continue
        source = sources[int(rng.integers(len(sources)))]
        available = [p for p in positions if _separate(p, source, h, w)
                     and _corners_clear(outline, *p, h, w)]
        # The first destination demonstrates both unchanged and inverted parts.
        boundary = frames[-1] if kind in ("nested", "triple") else None
        crossing = [p for p in available
                    if len(set(depth[p[0]:p[0] + h, p[1]:p[1] + w]
                               [~outline[p[0]:p[0] + h, p[1]:p[1] + w]].flat)) >= 2
                    and (boundary is None or _corner_cross(p, h, w, boundary))]
        if not crossing:
            continue
        target = crossing[int(rng.integers(len(crossing)))]
        targets = [target]
        for index in range(1, count):
            candidates = [p for p in available if all(_separate(p, q, h, w) for q in targets)]
            if teaching in (0, 1) and index == 1:
                candidates = [p for p in candidates
                              if not np.any(outline[p[0]:p[0] + h, p[1]:p[1] + w])
                              and (teaching == 0 or np.all(
                                  depth[p[0]:p[0] + h, p[1]:p[1] + w] == 0))]
            elif rng.random() < .4:
                uncut = [p for p in candidates
                         if not np.any(outline[p[0]:p[0] + h, p[1]:p[1] + w])]
                if uncut:
                    candidates = uncut
            if not candidates:
                break
            targets.append(candidates[int(rng.integers(len(candidates)))])
        if len(targets) != count:
            continue
        tile = _motif(rng, h, w, (first, second), asymmetric=teaching == 0)
        a = np.full(canvas, bg, dtype=int)
        sy, sx = source
        a[sy:sy + h, sx:sx + w] = tile
        for y, x in targets:
            a[y:y + h, x:x + w] = blank
        a[outline] = frame_color
        if np.count_nonzero(a == bg) * 2 <= a.size:
            continue
        out = _reference_output(a, frames, source, targets, tile, roles)
        # Reject any accidental ambiguous amodal rectangle or role assignment.
        try:
            info = _parse(a)
        except ValueError:
            continue
        if info["source"] != source or set(info["targets"]) != set(targets):
            continue
        if teaching == 0:
            # Visible destinations must distinguish reflection and half-turns.
            good = True
            for moved in _orientation_alternatives(tile):
                wrong = _reference_output(a, frames, source, targets, moved, roles)
                good &= not np.array_equal(wrong, out)
            if not good:
                continue
        pair = {"input": a, "output": out}
        try:
            _analyze_pair(pair)
        except AssertionError:
            # Rarely, foreground outline pixels hide every occurrence of a
            # sparse source accent in the selected destinations. Resample.
            continue
        return pair
    raise RuntimeError("bounded construction failed for a phase-transport scene")


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    fixed_roles = tuple(palette[:3])
    varying_roles = rng.random() < .3
    withheld = _TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))]
    train_shapes = [s for s in _TILE_SHAPES if s != withheld]
    seen = set()
    normalized_motifs = set()
    puzzle = {"train": [], "test": []}
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            for _ in range(20):
                shape = (withheld if split == "test" and index == 0
                         else train_shapes[int(rng.integers(len(train_shapes)))])
                if split == "train" and index == 0:
                    kind, sd, teaching = "single", 0, 0
                    count = int(rng.integers(2, 5))
                elif split == "train" and index == 1:
                    kind, sd, teaching = "nested", 1, 1
                    count = int(rng.integers(2, 5))
                else:
                    teaching = -1
                    if split == "test" and index == 0:
                        kind = "triple" if rng.random() < .45 else "nested"
                        sd = 3 if kind == "triple" else 2
                    else:
                        kind = str(rng.choice(("single", "nested", "triple", "disjoint")))
                        maximum = {"single": 1, "nested": 2, "triple": 3, "disjoint": 1}[kind]
                        sd = int(rng.integers(maximum + 1))
                        # Three tightly spaced rings need not have a wide exterior/annulus.
                        if kind == "triple":
                            sd = 3
                    count = int(rng.integers(1, 6))
                base = fixed_roles
                if varying_roles and not (split == "train" and index == 0):
                    base = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
                pool = [c for c in range(10) if c not in base]
                payload = tuple(int(c) for c in rng.choice(pool, size=2, replace=False))
                roles = (*base, *payload)
                try:
                    pair = _make_pair(rng, shape, kind, sd, count, roles, teaching)
                except RuntimeError:
                    continue
                a = pair["input"]
                key = a.shape, a.tobytes()
                tile = _parse(a)["tile"]
                motif_key = tile.shape, (tile != tile.flat[0]).tobytes()
                if key in seen or (split == "train" and index == 1 and motif_key in normalized_motifs):
                    continue
                seen.add(key)
                normalized_motifs.add(motif_key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded puzzle construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    info = _parse(a)
    roles = (info["background"], info["frame_color"], info["blank"], *info["payload"])
    expected = _reference_output(a, info["frames"], info["source"], info["targets"], info["tile"], roles)
    assert np.array_equal(out, expected), "incorrect phase-correct copy"
    assert np.array_equal(solve(a), out)
    writable = a == info["blank"]
    assert np.array_equal(a[~writable], out[~writable]), "only blank cells change"
    assert np.all(np.isin(out[writable], info["payload"]))
    assert not np.any(out == info["blank"])
    h, w = info["tile"].shape
    raw = a.copy()
    copied_offsets, destination_depths, relative_phases = set(), set(), set()
    for y, x in info["targets"]:
        patch = raw[y:y + h, x:x + w]
        mask = a[y:y + h, x:x + w] == info["blank"]
        patch[mask] = info["tile"][mask]
        for i, j in np.argwhere(mask):
            d = int(info["depth"][y + i, x + j])
            destination_depths.add(d)
            copied_offsets.add(int(info["tile"][i, j]))
            relative_phases.add((d - info["source_depth"]) % 2)
    assert relative_phases == {0, 1}, "show both phases of copying"
    assert not np.array_equal(raw, out), "omitting A must change the answer"
    assert copied_offsets == set(info["payload"]), "both source colors must be used"
    # A-only recoloring cannot consume any blank marker; B is indispensable.
    no_copy = a.copy()
    odd = info["depth"] % 2 == 1
    first, second = info["payload"]
    no_copy[odd & (a == first)] = second
    no_copy[odd & (a == second)] = first
    assert not np.array_equal(no_copy, out), "omitting B must change the answer"
    info["destination_depths"] = destination_depths
    info["crossed_targets"] = sum(np.any(info["outline"][y:y + h, x:x + w]) for y, x in info["targets"])
    info["corner_targets"] = sum(any(_corner_cross(p, h, w, f) for f in info["frames"])
                                  for p in info["targets"])
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _grid(pair["input"])
            key = a.shape, a.tobytes()
            assert key not in seen, "duplicate inputs"
            seen.add(key)
            info[split].append(_analyze_pair(pair))
    first, second = info["train"][:2]
    assert first["source_depth"] == 0 and len(first["frames"]) == 1
    assert 1 <= first["crossed_targets"] < len(first["targets"])
    assert all(not np.array_equal(first["tile"], moved)
               for moved in _orientation_alternatives(first["tile"]))
    assert second["source_depth"] == 1 and len(second["frames"]) == 2
    assert second["corner_targets"] >= 1
    assert {0, 1, 2} <= second["destination_depths"], "teach nesting cancellation"
    # A union-of-interiors shortcut disagrees on the depth-two destination.
    pair = puzzle["train"][1]
    a, expected = np.asarray(pair["input"]), np.asarray(pair["output"])
    union_copy = a.copy()
    h, w = second["tile"].shape
    p, q = second["payload"]
    for y, x in second["targets"]:
        reverse = (second["depth"][y:y + h, x:x + w] > 0) != (second["source_depth"] > 0)
        stamp = np.where(reverse, np.where(second["tile"] == p, q, p), second["tile"])
        patch = union_copy[y:y + h, x:x + w]
        mask = a[y:y + h, x:x + w] == second["blank"]
        patch[mask] = stamp[mask]
    assert not np.array_equal(union_copy, expected)
    motifs = {(a["tile"].shape, (a["tile"] != a["tile"].flat[0]).tobytes()) for a in info["train"]}
    assert len(motifs) >= 2
    assert info["test"][0]["tile"].shape not in {a["tile"].shape for a in info["train"]}
    assert info["test"][0]["source_depth"] >= 2


def _equal(a, b):
    return all(np.array_equal(p[k], q[k]) for s in ("train", "test")
               for p, q in zip(a[s], b[s]) for k in ("input", "output"))


def self_test():
    """Bounded semantic, counterfactual, determinism, and 42-seed diversity checks."""
    shapes, canvases, motifs, layouts, palettes = set(), set(), set(), set(), set()
    depths, target_counts, frame_counts = set(), set(), set()
    corner = crossed = edge = nested = disjoint = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = np.asarray(pair["input"])
                info = _analyze_pair(pair)
                total += 1
                shapes.add(info["tile"].shape)
                canvases.add(a.shape)
                tile = info["tile"]
                motifs.add((tile.shape, (tile != tile.flat[0]).tobytes()))
                layouts.add((a.shape, info["frames"], info["source"], info["targets"]))
                palettes.add((info["background"], info["frame_color"], info["blank"], info["payload"]))
                depths.add(info["source_depth"])
                frame_counts.add(len(info["frames"]))
                target_counts.add(len(info["targets"]))
                corner += info["corner_targets"]
                crossed += info["crossed_targets"]
                edge += any(t == 0 or l == 0 or b == a.shape[0] - 1 or r == a.shape[1] - 1
                            for t, l, b, r in info["frames"])
                nested += int(info["depth"].max()) >= 2
                disjoint += len(info["frames"]) > 1 and int(info["depth"].max()) == 1
    assert shapes == set(_TILE_SHAPES)
    assert depths == {0, 1, 2, 3} and frame_counts == {1, 2, 3}
    assert target_counts == {1, 2, 3, 4, 5}
    assert len(canvases) >= 35 and len(motifs) >= 60
    assert len(layouts) >= 200 and len(palettes) >= 60
    assert corner >= 80 and crossed >= total and edge >= 20
    assert nested >= 80 and disjoint >= 5
    assert _equal(generate_puzzle(123), generate_puzzle(123))
    assert _equal(generate_puzzle(777, 10, 4), generate_puzzle(777, 10, 4))
    for counts in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(999, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    # Metamorphic check: palette renaming commutes with solving, even when zero moves.
    p = generate_puzzle(101, 2, 1)
    rename = np.array([8, 6, 4, 2, 0, 9, 7, 5, 3, 1])
    for pair in p["train"] + p["test"]:
        assert np.array_equal(solve(rename[pair["input"]]), rename[pair["output"]])
        for turns in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], turns)),
                                  np.rot90(pair["output"], turns))
    return dict(seeds=42, pairs=total, tile_shapes=len(shapes), canvas_shapes=len(canvases),
                motifs=len(motifs), layouts=len(layouts), palettes=len(palettes),
                crossed_targets=int(crossed), corner_targets=int(corner),
                edge_frames=int(edge), nested_pairs=int(nested), disjoint_pairs=int(disjoint))


if __name__ == "__main__":
    print(self_test())
