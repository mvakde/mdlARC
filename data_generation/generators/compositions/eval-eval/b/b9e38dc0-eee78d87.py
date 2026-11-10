"""Wall-registered patterned spotlights.

A marked stencil lives in the rear of an open funnel. Repeat its occupancy
outward, registering each column against the continued counterclockwise rail.
The single marker is both the cyclic phase origin and the emitted color.
Foreign objects cast straight shadows through the entire geometric beam,
including its visually blank pattern gaps. The source socket is preserved.

Only standard Python and NumPy are used. No parent module or evidence is read.
"""

from __future__ import annotations

import itertools
import numpy as np


_TILE_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5)
                     if h * w >= 6)
_GROUPS = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fit_terminal(rows, sign):
    """Infer period and corner phase from the longest regular terminal rail."""
    rows = np.asarray(rows, dtype=int)
    xs = np.arange(len(rows)) - len(rows) + 1
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            expected = rows[-1] + sign * ((xs + phase) // period)
            suffix = 0
            for equal in (expected == rows)[::-1]:
                if not equal:
                    break
                suffix += 1
            if suffix >= 2 * period + 1 and len(set(rows[-suffix:])) >= 3:
                candidates.append((suffix, period, phase))
    if not candidates:
        raise ValueError("rail must show at least two complete widening steps")
    longest = max(item[0] for item in candidates)
    best = [(p, f) for length, p, f in candidates if length == longest]
    if len(best) != 1:
        raise ValueError("ambiguous terminal rail")
    return best[0]


def _decode(input_grid):
    """Read every convention from a single input; no output or RNG is consulted."""
    original = _grid(input_grid)
    counts = np.bincount(original.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= original.size // 2:
        raise ValueError("background must be a strict majority")
    matches = []
    for rotation in range(4):
        a = np.rot90(original, -rotation)
        for wall in np.flatnonzero(counts >= 17):
            wall = int(wall)
            if wall == background:
                continue
            ys, xs = np.nonzero(a == wall)
            rear, mouth = int(xs.min()), int(xs.max())
            if mouth - rear < 7 or mouth >= a.shape[1] - 1:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 5 or np.any(np.diff(cap) != 1):
                continue
            upper, lower = [], []
            for x in range(rear + 1, mouth + 1):
                lanes = np.flatnonzero(a[:, x] == wall)
                if len(lanes) != 2 or lanes[1] - lanes[0] < 4:
                    break
                upper.append(int(lanes[0]))
                lower.append(int(lanes[1]))
            if len(upper) != mouth - rear:
                continue
            if cap[0] != upper[0] or cap[-1] != lower[0]:
                continue
            if np.any(np.abs(np.diff(upper)) > 1) or np.any(np.abs(np.diff(lower)) > 1):
                continue
            try:
                up_fit = _fit_terminal(upper, -1)
                low_fit = _fit_terminal(lower, 1)
            except ValueError:
                continue
            chamber = np.zeros(a.shape, dtype=bool)
            for j, (u, l) in enumerate(zip(upper, lower), rear + 1):
                chamber[u + 1:l, j] = True
            occupied = chamber & (a != background)
            colors, amounts = np.unique(a[occupied], return_counts=True)
            if len(colors) != 2 or sorted(int(n) for n in amounts)[0] != 1:
                continue
            marker = int(colors[np.argmin(amounts)])
            material = int(colors[np.argmax(amounts)])
            if int(amounts.max()) < 3:
                continue
            # These source colors occur nowhere else; foreign objects use other colors.
            source = (a == material) | (a == marker)
            if np.any(source & ~chamber):
                continue
            coords = np.argwhere(source)
            top, left = (int(v) for v in coords.min(axis=0))
            bottom, right = (int(v) for v in coords.max(axis=0))
            shape = (bottom - top + 1, right - left + 1)
            if shape not in _TILE_SHAPES:
                continue
            tile = source[top:bottom + 1, left:right + 1].copy()
            if not 4 <= int(tile.sum()) < tile.size:
                continue
            anchor = tuple(int(v) for v in np.argwhere(a == marker)[0] - (top, left))
            extras = (a != background) & (a != wall) & ~source
            if np.any(extras & chamber):
                continue
            if any(np.count_nonzero(a == c) > 16 for c in np.unique(a[extras])):
                continue
            all_upper = np.full(a.shape[1], upper[-1], dtype=int)
            all_lower = np.full(a.shape[1], lower[-1], dtype=int)
            all_upper[rear + 1:mouth + 1] = upper
            all_lower[rear + 1:mouth + 1] = lower
            for x in range(mouth + 1, a.shape[1]):
                all_upper[x] = upper[-1] - (x - mouth + up_fit[1]) // up_fit[0]
                all_lower[x] = lower[-1] + (x - mouth + low_fit[1]) // low_fit[0]
            matches.append(dict(grid=a, rotation=rotation, background=background,
                                wall=wall, marker=marker, material=material,
                                rear=rear, mouth=mouth, upper=all_upper,
                                lower=all_lower, fits=(up_fit, low_fit),
                                tile=tile, anchor=anchor, origin=(top, left),
                                extras=extras, chamber=chamber))
    if len(matches) != 1:
        raise ValueError("input must contain exactly one interpretable funnel and marked stencil")
    return matches[0]


def _masks(info, registration="wall", anchor=None):
    a, mouth = info["grid"], info["mouth"]
    y, x = np.indices(a.shape)
    upper, lower = info["upper"][None, :], info["lower"][None, :]
    footprint = (x > mouth) & (y > upper) & (y < lower)
    ay, ax = info["anchor"] if anchor is None else anchor
    if registration == "wall":
        lateral = y - upper - 1
    elif registration == "fixed":
        lateral = y - info["upper"][mouth + 1] - 1
    elif registration == "clipped":
        lateral = y - np.maximum(upper, -1) - 1
    else:
        raise ValueError("unknown registration")
    h, w = info["tile"].shape
    texture = info["tile"][(lateral + ay) % h, (x - mouth - 1 + ax) % w]
    return footprint, texture


def _render(info, *, shadows=True, solid=False, registration="wall", anchor=None,
            gap_blockers=True):
    """Construction uses mask unions; solve below uses an axial lane scan."""
    a = info["grid"]
    footprint, texture = _masks(info, registration, anchor)
    shadow = np.zeros(a.shape, dtype=bool)
    if shadows:
        blockers = footprint & info["extras"]
        if not gap_blockers:
            blockers &= texture
        for y, x in np.argwhere(blockers):
            shadow[int(y), int(x) + 1:] = True
    paint = footprint & (a == info["background"]) & ~shadow
    if not solid:
        paint &= texture
    result = a.copy()
    result[paint] = info["marker"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict from input geometry, stencil, phase marker and opaque objects."""
    info = _decode(input_grid)
    a, out = info["grid"], info["grid"].copy()
    h, w = info["tile"].shape
    ay, ax = info["anchor"]
    for y in range(a.shape[0]):
        blocked = False
        for x in range(info["mouth"] + 1, a.shape[1]):
            if not info["upper"][x] < y < info["lower"][x]:
                continue
            if info["extras"][y, x]:
                blocked = True
            elif not blocked and a[y, x] == info["background"]:
                tr = (y - info["upper"][x] - 1 + ay) % h
                tc = (x - info["mouth"] - 1 + ax) % w
                if info["tile"][tr, tc]:
                    out[y, x] = info["marker"]
    return np.rot90(out, info["rotation"]).copy()


def _sample_tile(rng, shape):
    """Full-box occupancy with holes; the marker can occupy any filled cell."""
    h, w = shape
    for _ in range(80):
        size = int(rng.integers(4, h * w))
        tile = np.zeros(h * w, dtype=bool)
        tile[rng.choice(h * w, size=size, replace=False)] = True
        tile = tile.reshape(shape)
        if all((tile[0].any(), tile[-1].any(), tile[:, 0].any(), tile[:, -1].any())):
            choices = np.argwhere(tile)
            anchor = tuple(int(v) for v in choices[int(rng.integers(len(choices)))])
            return tile, anchor
    # A finite, valid fallback, with no outside seed or stored puzzle.
    tile = np.zeros(shape, dtype=bool)
    tile[0, 0] = tile[0, -1] = tile[-1, 0] = tile[-1, -1] = True
    return tile, (0, w - 1)


def _geometry(rng, tile_shape, projection_range):
    th, tw = tile_shape
    runs = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    phases = tuple(int(rng.integers(p)) for p in runs)
    reservoir = tw + int(rng.integers(1, 4))
    shoulder = int(rng.integers(3))
    start = (th // 2 + 1 + int(rng.integers(3)),
             th - th // 2 + 1 + int(rng.integers(3)))
    profiles = [list(itertools.repeat(v, reservoir)) for v in start]
    for _ in range(shoulder):
        for profile in profiles:
            profile.append(max(2, profile[-1] + int(rng.integers(-1, 2))))
    flare = 2 * max(runs) + 2
    for profile, period, phase in zip(profiles, runs, phases):
        end = profile[-1]
        profile.extend(end + (t + phase) // period for t in range(1, flare + 1))
    # The phase at the mouth advances with the length of the visible flare.
    phases = tuple((flare + phase) % period for period, phase in zip(runs, phases))
    length = len(profiles[0])
    pmin, pmax = projection_range
    rear_max = min(4, 30 - length - pmin - 1)
    rear = int(rng.integers(rear_max + 1))
    mouth = rear + length
    projection = int(rng.integers(pmin, min(pmax, 29 - mouth) + 1))
    width = mouth + 1 + projection
    span = max(profiles[0]) + max(profiles[1]) + 1
    height = int(rng.integers(span, min(30, span + 7) + 1))
    center = int(rng.integers(max(profiles[0]), height - max(profiles[1])))
    upper = np.full(width, center - profiles[0][-1], dtype=int)
    lower = np.full(width, center + profiles[1][-1], dtype=int)
    upper[rear + 1:mouth + 1] = center - np.array(profiles[0])
    lower[rear + 1:mouth + 1] = center + np.array(profiles[1])
    for x in range(mouth + 1, width):
        upper[x] = upper[mouth] - (x - mouth + phases[0]) // runs[0]
        lower[x] = lower[mouth] + (x - mouth + phases[1]) // runs[1]
    return (height, width), rear, mouth, reservoir, upper, lower, runs, phases


def _place_group(rng, info, offsets, color, *, hole=False, straddle=False, used_lanes=()):
    a = info["grid"]
    footprint, texture = _masks(info)
    candidates = []
    # At least four following columns make both gaps and colored shadow sites visible.
    for x in range(info["mouth"] + 1, a.shape[1] - 4):
        for y in range(a.shape[0]):
            cells = [(y + dy, x + dx) for dy, dx in offsets]
            if any(yy >= a.shape[0] or xx >= a.shape[1] or a[yy, xx] != info["background"]
                   for yy, xx in cells):
                continue
            inside = [bool(footprint[yy, xx]) for yy, xx in cells]
            if not any(inside) or (straddle and all(inside)):
                continue
            if any(yy in used_lanes for yy, xx in cells if footprint[yy, xx]):
                continue
            if hole and (len(cells) != 1 or texture[y, x]):
                continue
            # Each group must remove at least two actual motif sites downstream.
            effect = sum(int(np.count_nonzero(footprint[yy, xx + 1:] & texture[yy, xx + 1:]
                                             & (a[yy, xx + 1:] == info["background"])))
                         for (yy, xx), inside_cell in zip(cells, inside) if inside_cell)
            if effect >= 2:
                candidates.append(cells)
    if not candidates:
        return False
    cells = candidates[int(rng.integers(len(candidates)))]
    for y, x in cells:
        a[y, x] = color
        info["extras"][y, x] = True
    return True


def _build_pair(rng, shape, colors, rotation, projection_range, style):
    tile, anchor = _sample_tile(rng, shape)
    grid_shape, rear, mouth, reservoir, upper, lower, runs, phases = _geometry(
        rng, shape, projection_range)
    bg, wall, material, marker = (int(c) for c in colors[:4])
    a = np.full(grid_shape, bg, dtype=np.int8)
    a[upper[rear + 1]:lower[rear + 1] + 1, rear] = wall
    for x in range(rear + 1, mouth + 1):
        a[upper[x], x] = a[lower[x], x] = wall
    top = int(rng.integers(upper[rear + 1] + 1, lower[rear + 1] - shape[0] + 1))
    left = int(rng.integers(rear + 1, rear + reservoir - shape[1] + 2))
    a[top:top + shape[0], left:left + shape[1]][tile] = material
    a[top + anchor[0], left + anchor[1]] = marker
    info = dict(grid=a, rotation=rotation, background=bg, wall=wall, material=material,
                marker=marker, rear=rear, mouth=mouth, upper=upper, lower=lower,
                fits=tuple(zip(runs, phases)), tile=tile, anchor=anchor,
                origin=(top, left), extras=np.zeros(grid_shape, dtype=bool))
    footprint, _ = _masks(info)
    if style != "clean":
        if not _place_group(rng, info, _GROUPS[0], int(colors[4]), hole=True):
            return None
        occupied_lanes = set(int(y) for y, x in np.argwhere(info["extras"] & footprint))
        if not _place_group(rng, info, _GROUPS[1], int(colors[5]), used_lanes=occupied_lanes):
            return None
        if style == "dense":
            _place_group(rng, info, _GROUPS[int(rng.integers(2, 5))], int(colors[4]),
                         straddle=bool(rng.integers(2)))
        elif style == "repeat":
            # Later objects in the same lane leave the shadow width unchanged.
            options = [(int(y), xx) for y, x in np.argwhere(info["extras"] & footprint)
                       for xx in range(int(x) + 1, a.shape[1] - 1)
                       if footprint[y, xx] and a[y, xx] == bg]
            if options:
                y, x = options[int(rng.integers(len(options)))]
                a[y, x] = int(colors[5])
                info["extras"][y, x] = True
    # The same blocker colors can also occur outside the beam, where they are inert.
    for _ in range(int(rng.integers(1, 3))):
        y, x = np.indices(a.shape)
        chamber = ((x > rear) & (x <= mouth) & (y > upper[None, :]) & (y < lower[None, :]))
        candidates = np.argwhere(~footprint & ~chamber & (a == bg))
        if len(candidates):
            yy, xx = candidates[int(rng.integers(len(candidates)))]
            a[yy, xx] = int(colors[4 + int(rng.integers(2))])
            info["extras"][yy, xx] = True
    out = _render(info)
    # Inputs must expose texture, widening-wall registration and opacity when used.
    if np.count_nonzero(out != a) < 8:
        return None
    if np.array_equal(out, _render(info, solid=True)):
        return None
    if np.array_equal(out, _render(info, registration="fixed")):
        return None
    if style != "clean" and np.array_equal(out, _render(info, shadows=False)):
        return None
    pair = {"input": np.rot90(a, rotation).copy(), "output": np.rot90(out, rotation).copy()}
    try:
        decoded = _decode(pair["input"])
    except ValueError:
        return None
    if not np.array_equal(solve(pair["input"]), pair["output"]):
        raise AssertionError("construction and input-only lane scan disagree")
    return pair, decoded


def _training_alternatives(infos):
    """Concrete rivals that demonstrations must refute, evaluated on actual grids."""
    rivals = dict(solid=False, no_shadows=False, ink_only_occlusion=False, fixed_registration=False,
                  clipped_registration=False,
                  no_phase_shift=False, mirrored_transverse=False, mirrored_axial=False)
    for info in infos:
        expected = _render(info)
        rivals["solid"] |= not np.array_equal(expected, _render(info, solid=True))
        rivals["no_shadows"] |= not np.array_equal(expected, _render(info, shadows=False))
        rivals["ink_only_occlusion"] |= not np.array_equal(expected, _render(info, gap_blockers=False))
        rivals["fixed_registration"] |= not np.array_equal(expected, _render(info, registration="fixed"))
        rivals["clipped_registration"] |= not np.array_equal(expected, _render(info, registration="clipped"))
        rivals["no_phase_shift"] |= not np.array_equal(expected, _render(info, anchor=(0, 0)))
        for name, axis in (("mirrored_transverse", 0), ("mirrored_axial", 1)):
            modified = dict(info)
            modified["tile"] = np.flip(info["tile"], axis=axis)
            anchor = list(info["anchor"])
            anchor[axis] = info["tile"].shape[axis] - 1 - anchor[axis]
            modified["anchor"] = tuple(anchor)
            rivals[name] |= not np.array_equal(expected, _render(modified))
    return rivals


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < 0.3)
    palette = rng.permutation(10)
    regime = int(rng.integers(3))  # longer projection / grouped obstacles / rotated recombination
    preferred = int(rng.integers(4))
    first_shape = _TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))]
    # A period-two axis equals its own reversed cyclic indexing. The second
    # witness needs >=3 on BOTH axes, regardless of symmetries of the first.
    eligible = [s for s in _TILE_SHAPES if s != first_shape and min(s) >= 3]
    shapes = [first_shape, eligible[int(rng.integers(len(eligible)))]]
    puzzle, seen, train_infos = {"train": [], "test": []}, set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                shape = shapes[index] if index < 2 else _TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))]
                style = "clean" if index == 0 else "teaching" if index == 1 else str(
                    rng.choice(("clean", "teaching", "dense", "repeat")))
                projection_range = (6, 9)
                rotation = preferred if index == 0 else (preferred + int(rng.choice((1, 2)))) % 4
            else:
                # New occupancy on a demonstrated template size; no unshown size law.
                shape = train_infos[int(rng.integers(len(train_infos)))]["tile"].shape
                style = ("dense" if regime == 1 else "repeat" if regime == 2
                         else str(rng.choice(("clean", "teaching", "dense"))))
                projection_range = (10, 14) if regime == 0 else (6, 12)
                rotation = (preferred + 3) % 4 if regime == 2 else int(rng.integers(4))
            colors = palette if shared_palette else rng.permutation(10)
            accepted = None
            for _ in range(120):
                built = _build_pair(rng, shape, colors, rotation, projection_range, style)
                if built is None:
                    continue
                pair, info = built
                if _key(pair["input"]) in seen:
                    continue
                if split == "train" and index == 1 and not all(
                        _training_alternatives(train_infos + [info]).values()):
                    continue
                if split == "train" and index == 1 and (
                        _features(info)["silhouette"] == _features(train_infos[0])["silhouette"]):
                    continue
                if split == "test" and any(
                        info["tile"].shape == other["tile"].shape
                        and np.array_equal(info["tile"], other["tile"])
                        for other in train_infos):
                    continue
                accepted = built
                break
            if accepted is None:
                raise RuntimeError(f"bounded sampler exhausted: seed={seed}, split={split}, index={index}, shape={shape}")
            pair, info = accepted
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                train_infos.append(info)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    footprint, texture = _masks(info)
    a, out = info["grid"], _render(info)
    no_shadow = _render(info, shadows=False)
    effective = footprint & info["extras"]
    blocked_holes = effective & ~texture
    hole_shadow = False
    for y, x in np.argwhere(blocked_holes):
        hole_shadow |= bool(np.any((no_shadow[y, x + 1:] != out[y, x + 1:])))
    silhouette = tuple(zip(info["upper"][info["rear"] + 1:info["mouth"] + 1]
                          - info["upper"][info["rear"] + 1],
                          info["lower"][info["rear"] + 1:info["mouth"] + 1]
                          - info["upper"][info["rear"] + 1]))
    return dict(rotation=info["rotation"], tile_shape=info["tile"].shape,
                tile=_key(info["tile"]), occupancy=int(info["tile"].sum()),
                anchor=info["anchor"], palette=tuple(info[k] for k in
                ("background", "wall", "material", "marker")), fits=info["fits"],
                silhouette=silhouette, projection=a.shape[1] - info["mouth"] - 1,
                effective=int(effective.sum()), inert=int((info["extras"] & ~footprint).sum()),
                shadow=int(np.count_nonzero(no_shadow != out)), hole_shadow=hole_shadow,
                repeated_lane=bool(np.any(effective.sum(axis=1) > 1)),
                clipped=bool(footprint[0].any() or footprint[-1].any()),
                changed=int(np.count_nonzero(a != out)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            infos[split].append(info)
            expected = solve(inp)
            assert np.array_equal(expected, out), "input-only solver mismatch"
            a, canonical_out = info["grid"], np.rot90(out, -info["rotation"])
            assert np.array_equal(canonical_out, _render(info)), "lane scan / mask union mismatch"
            footprint, texture = _masks(info)
            changed = a != canonical_out
            assert changed.sum() >= 8
            assert np.all(a[changed] == info["background"])
            assert np.all(canonical_out[changed] == info["marker"])
            assert np.all((footprint & texture)[changed])
            assert np.array_equal(a[~footprint], canonical_out[~footprint])
            assert np.array_equal(a[a != info["background"]], canonical_out[a != info["background"]])
            assert np.any(footprint & ~texture & (a == info["background"]))
            assert not np.array_equal(_render(info, solid=True), canonical_out)
            assert not np.array_equal(_render(info, registration="fixed"), canonical_out)
    features = [_features(info) for info in infos["train"]]
    assert any(f["effective"] == 0 for f in features), "need an unobstructed texture witness"
    assert any(f["hole_shadow"] for f in features), "a pattern gap must still conduct an occludable ray"
    assert any(len(np.unique(info["grid"][info["extras"] & _masks(info)[0]])) >= 2
               for info in infos["train"])
    assert all(_training_alternatives(infos["train"]).values()), "training leaves a concrete shortcut unresolved"
    assert len({info["tile"].shape for info in infos["train"]}) >= 2
    assert len({f["silhouette"] for f in features}) >= 2
    for query in infos["test"]:
        assert any(query["tile"].shape == t["tile"].shape for t in infos["train"])
        assert not any(query["tile"].shape == t["tile"].shape and np.array_equal(query["tile"], t["tile"])
                       for t in infos["train"])


def self_test():
    """Bounded grammar, counterfactual, metamorphic and 42-seed variation checks."""
    records, signatures, shapes, structures = [], set(), set(), set()
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
    for seed in range(42):
        a, b = legal[seed % len(legal)]
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s]
                          for k in ("input", "output"))
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                records.append((split, _features(info)))
                shapes.add(pair["input"].shape)
                # Normalize literal colors independently for each actual input.
                mapping = {}
                structures.add((pair["input"].shape,
                                tuple(mapping.setdefault(int(c), len(mapping))
                                      for c in pair["input"].ravel())))
        if seed in (0, 13, 41):
            again = generate_puzzle(seed, a, b)
            assert signature == tuple(_key(p[k]) for s in ("train", "test") for p in again[s]
                                      for k in ("input", "output"))
    fs = [f for _, f in records]
    metrics = dict(seeds=42, pairs=len(fs), distinct_puzzles=len(signatures),
                   input_shapes=len(shapes), color_normalized_inputs=len(structures),
                   stencil_masks=len({f["tile"] for f in fs}),
                   funnel_profiles=len({f["silhouette"] for f in fs}),
                   tile_shapes=sorted({f["tile_shape"] for f in fs}),
                   palette_roles=len({f["palette"] for f in fs}),
                   rail_period_pairs=sorted({tuple(p for p, phase in f["fits"]) for f in fs}),
                   anchor_positions=sorted({f["anchor"] for f in fs}),
                   shadowed_pairs=sum(f["shadow"] > 0 for f in fs),
                   hole_shadow_pairs=sum(f["hole_shadow"] for f in fs),
                   repeated_lane_pairs=sum(f["repeated_lane"] for f in fs),
                   clipped_pairs=sum(f["clipped"] for f in fs))
    assert len(signatures) == 42 and len(shapes) >= 45 and len(structures) >= 180
    assert metrics["stencil_masks"] >= 80 and metrics["funnel_profiles"] >= 100
    assert set(tuple(s) for s in metrics["tile_shapes"]) == set(_TILE_SHAPES)
    assert metrics["palette_roles"] >= 70 and len(metrics["rail_period_pairs"]) == 9
    assert metrics["hole_shadow_pairs"] >= 42 and metrics["repeated_lane_pairs"] >= 8
    assert metrics["clipped_pairs"] >= 20
    for split in ("train", "test"):
        sub = [f for s, f in records if s == split]
        assert {f["rotation"] for f in sub} == {0, 1, 2, 3}
    # Rotation and full palette permutation must commute with the rule.
    fixture = generate_puzzle(91, 2, 1)["train"][0]
    inp, out = fixture["input"], fixture["output"]
    permutation = np.array([7, 4, 9, 1, 6, 2, 0, 8, 3, 5], dtype=np.int8)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(out, rotation))
        assert np.array_equal(solve(permutation[np.rot90(inp, rotation)]),
                              permutation[np.rot90(out, rotation)])
    # Explicit tiny rail fixtures settle thin virtual corners and their phase.
    assert _fit_terminal((9, 8, 7, 6, 5), -1) == (1, 0)
    assert _fit_terminal((9, 9, 8, 8, 7, 7), -1) == (2, 1)
    assert _fit_terminal((9, 9, 8, 8, 7), -1) == (2, 0)
    assert _fit_terminal((9, 9, 9, 8, 8, 8, 7, 7, 7), -1) == (3, 2)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A shadow-ignoring label and a wall-ignoring label must be rejected.
    for rival in ("shadows", "registration"):
        damaged = generate_puzzle(92, 2, 1)
        pair = next(p for p in damaged["train"] if _features(_decode(p["input"]))["shadow"] > 0)
        info = _decode(pair["input"])
        bad = _render(info, shadows=False) if rival == "shadows" else _render(info, registration="fixed")
        pair["output"] = np.rot90(bad, info["rotation"]).copy()
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("counterfactual label accepted")
    return metrics
