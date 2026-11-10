"""Solvent chambers: local lookup selects soluble cargo; the key points gravity.

Only standard library and NumPy. No filesystem access or persistent generation state.
"""
import hashlib
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = D8 if diagonal else D4
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, queue = {start}, [start]
        while queue:
            r, c = queue.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    queue.append(q)
        result.append(frozenset(cells))
    return result


def _inside_regions(walls):
    """Four-connected complement components which cannot reach the canvas edge."""
    h, w = walls.shape
    return [s for s in _components(~walls)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in s)]


def _rod(cells, color):
    return (frozenset(cells), int(color))


def _translate(cells, direction):
    dr, dc = direction
    return frozenset((r + dr, c + dc) for r, c in cells)


def _settle(rods, allowed, direction=(1, 0)):
    """Greatest simultaneously movable set, with rigid rods and fixed walls.

    A rod is held if ANY cell would hit a fixed boundary or a held rod. A stack
    can descend together. There is no rotation, friction, balancing or bonding.
    """
    result = list(rods)
    for _ in range(30):
        targets = [_translate(s, direction) for s, color in result]
        moving = {i for i, s in enumerate(targets) if s <= allowed}
        while True:
            stationary = set(range(len(result))) - moving
            blocked = {i for i in moving if any(targets[i] & result[j][0]
                                                for j in stationary)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [(targets[i], color) if i in moving else (s, color)
                  for i, (s, color) in enumerate(result)]
    raise AssertionError("gravity failed to terminate within the canvas")


def _read(grid):
    """Recover all mechanics from ink, ignoring cargo when testing enclosure."""
    a = _array(grid)
    values, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    bg = int(values[np.argmax(counts)])
    h, w = a.shape
    keys = []
    for cells in _components(a != bg, True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw or len({int(a[q]) for q in cells}) < 2:
            continue
        if sh == 2 and sw >= 2 and 0 < c0 <= c1 < w - 1:
            if r0 == 0:
                keys.append((cells, tuple(map(int, a[0, c0:c1+1])),
                             tuple(map(int, a[1, c0:c1+1])), (1, 0)))
            if r1 == h - 1:
                keys.append((cells, tuple(map(int, a[-1, c0:c1+1])),
                             tuple(map(int, a[-2, c0:c1+1])), (-1, 0)))
        if sw == 2 and sh >= 2 and 0 < r0 <= r1 < h - 1:
            if c0 == 0:
                keys.append((cells, tuple(map(int, a[r0:r1+1, 0])),
                             tuple(map(int, a[r0:r1+1, 1])), (0, 1)))
            if c1 == w - 1:
                keys.append((cells, tuple(map(int, a[r0:r1+1, -1])),
                             tuple(map(int, a[r0:r1+1, -2])), (0, -1)))
    if len(keys) != 1:
        raise ValueError("expected one isolated, non-corner edge key")
    key, sources, targets, direction = keys[0]
    if len(set(sources)) != len(sources):
        raise ValueError("key sources must be distinct")
    mapping = dict(zip(sources, targets))
    walls = np.zeros(a.shape, dtype=bool)
    wall_ids = np.full(a.shape, -1, dtype=np.int16)
    rods, contours = [], []
    for color in values:
        if color == bg:
            continue
        mask = a == color
        for q in key:
            mask[q] = False
        for cells in _components(mask, True):
            rr, cc = zip(*cells)
            straight = len(set(rr)) == 1 or len(set(cc)) == 1
            if straight:
                if len(cells) < 2:
                    raise ValueError("cargo rods have length at least two")
                rods.append(_rod(cells, color))
            else:
                index = len(contours)
                contours.append(_rod(cells, color))
                for q in cells:
                    walls[q] = True
                    wall_ids[q] = index
    regions = []
    for allowed in _inside_regions(walls):
        boundary = {q for r, c in allowed for dr, dc in D4
                    if 0 <= (q := (r + dr, c + dc))[0] < h
                    and 0 <= q[1] < w and walls[q]}
        owners = {int(wall_ids[q]) for q in boundary}
        if len(owners) != 1:
            raise ValueError("a chamber needs one fixed contour owner")
        owner = next(iter(owners))
        color = contours[owner][1]
        regions.append({"cells": allowed, "wall": color, "owner": owner,
                        "target": mapping.get(color)})
    membership = []
    for cells, color in rods:
        hits = [i for i, region in enumerate(regions) if cells & region["cells"]]
        if len(hits) > 1 or (hits and not cells <= regions[hits[0]]["cells"]):
            raise ValueError("a rod may not cross a chamber boundary")
        membership.append(hits[0] if hits else -1)
    return {"grid": a, "background": bg, "key": key, "mapping": mapping,
            "direction": direction, "walls": walls, "contours": contours,
            "regions": regions, "rods": rods, "membership": membership}


def _transform(scene, settle=True, recursive=False, global_delete=False):
    a = scene["grid"]
    out = a.copy()
    bg = scene["background"]
    for i, region in enumerate(scene["regions"]):
        if region["target"] is None:
            continue
        target = region["target"]
        if recursive:
            visited = {region["wall"]}
            while target in scene["mapping"] and target not in visited:
                visited.add(target)
                target = scene["mapping"][target]
        local = [p for p, m in zip(scene["rods"], scene["membership"]) if m == i]
        for cells, color in local:
            for q in cells:
                out[q] = bg
        survivors = [p for p in local if p[1] != target]
        final = _settle(survivors, region["cells"], scene["direction"]) if settle else survivors
        for cells, color in final:
            for q in cells:
                out[q] = color
    if global_delete:
        # A deliberate competing rule: forget local enclosure/ownership.
        destinations = set(scene["mapping"].values())
        for (cells, color), m in zip(scene["rods"], scene["membership"]):
            if m == -1 and color in destinations:
                for q in cells:
                    out[q] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution. Keys, containment, rod geometry and force are visible."""
    return _transform(_read(input_grid))


def _paint(shape, bg, key_data, frames):
    a = np.full(shape, bg, dtype=np.int8)
    sources, targets, start = key_data
    a[0, start:start+len(sources)] = sources
    a[1, start:start+len(sources)] = targets
    for frame in frames:
        for q in frame["walls"]:
            a[q] = frame["color"]
        for cells, color in frame["rods"]:
            for q in cells:
                a[q] = color
    return a


def _geometry(rng, target, other_colors, kind, small=False):
    """One procedural cavity, with a stable support network and uneven floor."""
    ih = int(rng.integers(8, 11 if small else 14))
    iw = int(rng.integers(5, 8 if small else 11))
    top = [1 + int(rng.integers(0, 2)) for _ in range(iw)]
    bottom = [ih - int(rng.integers(0, 3)) for _ in range(iw)]
    # Flat and stepped ceilings/floors are both legitimate structures.
    if rng.random() < .3:
        top = [1] * iw
    if rng.random() < .25:
        bottom = [ih] * iw
    cavity = frozenset((r, c + 1) for c in range(iw)
                       for r in range(top[c], bottom[c] + 1))
    # Choose an arbitrary broad beam span and a floor-supported soluble leg.
    left = int(rng.integers(1, max(2, iw - 2)))
    right = int(rng.integers(left + 2, iw + 1))
    columns = list(range(left, right + 1))
    floor = min(bottom[c-1] for c in columns)
    length = int(rng.integers(3, min(5, floor - max(top[c-1] for c in columns)) + 1))
    row = floor - length
    x = int(rng.choice([c for c in columns if bottom[c-1] == floor]))
    y = max(columns, key=lambda c: abs(c - x))
    beam_color, post_color, cap_color = other_colors[:3]
    rods = [_rod(((r, x) for r in range(row + 1, bottom[x-1] + 1)), target)]
    if kind == "redundant":
        # A differently colored full-height leg keeps the broad beam up.
        rods.append(_rod(((r, y) for r in range(row + 1, bottom[y-1] + 1)), post_color))
    elif kind in ("landing", "cascade"):
        rods.append(_rod(((r, y) for r in range(bottom[y-1] - 1, bottom[y-1] + 1)), post_color))
    elif kind == "double":
        # Two disjoint same-identity legs vanish together.
        rods.append(_rod(((r, y) for r in range(row + 1, bottom[y-1] + 1)), target))
    rods.append(_rod(((row, c) for c in columns), beam_color))
    # A second level makes a real support cascade, without a hidden junction.
    upper = row - max(top[c-1] for c in columns)
    if upper >= 2 and (kind == "cascade" or rng.random() < .6):
        u = int(rng.choice(columns))
        n = min(upper, int(rng.integers(2, 4)))
        rods.append(_rod(((r, u) for r in range(row - n, row)), cap_color))
    return cavity, rods, (ih + 2, iw + 2)


def _frame(rng, wall, target, colors, kind="landing", multi=False, opened=False, small=False):
    # Cargo never has its own wall's color. Other chambers may use that same
    # literal color as cargo, giving genuine directed mapping chains.
    choices = [c for c in colors if c not in (wall, target)]
    rng.shuffle(choices)
    if len(choices) < 3:
        raise AssertionError("insufficient cargo palette")
    cavity, rods, shape = _geometry(rng, target, choices, kind, small)
    cavities = [cavity]
    if multi:
        second, more, _ = _geometry(rng, target, choices,
                                    "redundant" if kind != "redundant" else "landing", True)
        # Shared wall, with either vertically or horizontally adjacent cavities.
        vertical = bool(rng.integers(2))
        dr, dc = (shape[0] - 1, 0) if vertical else (0, shape[1] - 1)
        second = frozenset((r + dr, c + dc) for r, c in second)
        more = [(_translate_n(s, dr, dc), color) for s, color in more]
        cavities.append(second)
        rods += more
    all_inside = frozenset().union(*cavities)
    walls = {(r + dr, c + dc) for r, c in all_inside for dr, dc in D4} - set(all_inside)
    # Join separate ring silhouettes if profile notches left them disconnected.
    # Bounding outer rails preserve the intended cargo domains but can enclose
    # additional empty cavities, which the input-only ownership checks retain.
    if multi:
        components = _set_components(walls)
        if len(components) > 1:
            # In this construction, bounding rails are outside both cargo holes.
            maxr = max(r for r, c in walls)
            maxc = max(c for r, c in walls)
            walls |= {(0, c) for c in range(maxc+1)} | {(maxr, c) for c in range(maxc+1)}
            walls |= {(r, 0) for r in range(maxr+1)} | {(r, maxc) for r in range(maxr+1)}
    if opened:
        # Remove a complete vertical chimney from one cavity's ceiling. No
        # lower support is altered; four-neighbor escape is now explicit.
        first = cavities[0]
        c = int(rng.choice(sorted({c for r, c in first})))
        r = min(r for r, cc in first if cc == c)
        walls = {q for q in walls if not (q[1] == c and q[0] < r)}
    maxr = max(r for r, c in walls)
    maxc = max(c for r, c in walls)
    return {"walls": frozenset(walls), "color": wall, "rods": rods,
            "cavities": cavities, "opened": opened, "shape": (maxr + 1, maxc + 1)}


def _set_components(cells):
    h = max(r for r, c in cells) + 1
    w = max(c for r, c in cells) + 1
    a = np.zeros((h, w), dtype=bool)
    for q in cells:
        a[q] = True
    return _components(a, True)


def _translate_n(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _place(rng, frames, key_length):
    """Bounded flexible rectangle packing, with no fixed per-object slots."""
    min_h = 3 + max(f["shape"][0] for f in frames)
    min_w = max(key_length + 2, max(f["shape"][1] for f in frames))
    if min_h > 30 or min_w > 30:
        return None
    for attempt in range(18):
        h = int(rng.integers(min_h, 31)) if attempt < 12 else 30
        w = int(rng.integers(min_w, 31)) if attempt < 12 else 30
        occupied = np.zeros((h, w), dtype=bool)
        occupied[:3] = True
        placed = []
        for frame in sorted(frames, key=lambda f: -f["shape"][0] * f["shape"][1]):
            fh, fw = frame["shape"]
            positions = [(r, c) for r in range(3, h - fh + 1) for c in range(w - fw + 1)
                         if not occupied[r:r+fh, c:c+fw].any()]
            if not positions:
                break
            r, c = positions[int(rng.integers(len(positions)))]
            placed.append({**frame,
                           "walls": _translate_n(frame["walls"], r, c),
                           "rods": [(_translate_n(s, r, c), v) for s, v in frame["rods"]],
                           "cavities": [_translate_n(s, r, c) for s in frame["cavities"]]})
            occupied[max(3, r-1):min(h, r+fh+1), max(0, c-1):min(w, c+fw+1)] = True
        else:
            return (h, w), placed
    return None


def _make_pair(rng, lesson, turns):
    for attempt in range(60):
        palette = list(map(int, rng.permutation(10)))
        bg, wall_a, wall_b, solvent, beam, post, cap, extra, inert, spare = palette
        # Only the chain lesson constrains lookup structure. Other examples
        # permit shared targets, cycles, and destinations outside the sources.
        mapping = ({wall_a: wall_b, wall_b: solvent} if lesson == "chain" else
                   {wall_a: int(rng.choice([wall_b, solvent, beam, extra])),
                    wall_b: int(rng.choice([wall_a, solvent, post, extra]))})
        target_a = mapping[wall_a]
        for c in list(map(int, rng.permutation([beam, post, cap, extra])))[:int(rng.integers(0, 4))]:
            mapping[c] = int(rng.choice([wall_a, wall_b, solvent, extra, c]))
        colors = [solvent, beam, post, cap, extra, inert, spare]
        if lesson == "scope":
            # Identical soluble identity in a sealed and an opened contour.
            frames = [_frame(rng, wall_a, target_a, colors, "landing", small=True),
                      _frame(rng, wall_a, target_a, colors, "floor", opened=True, small=True)]
        elif lesson == "chain":
            frames = [_frame(rng, wall_a, wall_b, colors, "cascade", small=True),
                      _frame(rng, wall_b, solvent, [wall_a, beam, post, cap, extra, inert, spare],
                             "redundant", small=True)]
            # Force the chain destination's destination to be visible surviving
            # cargo in the first chamber, not merely an unused key entry.
            first = frames[0]
            # Rod 1 is the surviving short landing post. Recolor just that
            # rod, preserving the separate identities at every contact.
            first["rods"][1] = (first["rods"][1][0], solvent)
        else:
            n = 1 if lesson == "partition" else int(rng.integers(1, 4))
            frames = []
            for i in range(n):
                wc = int(rng.choice([wall_a, wall_b, inert]))
                tc = mapping.get(wc, solvent)
                candidates = [c for c in palette if c not in (bg, wc, tc)]
                mode = str(rng.choice(["floor", "landing", "cascade", "redundant", "double"]))
                frames.append(_frame(rng, wc, tc, candidates, mode,
                                     multi=n == 1 and bool(rng.integers(2)),
                                     opened=i > 0 and rng.random() < .25, small=n >= 2 or attempt >= 35))
            # Every query contains an active closed chamber. Extra scenes vary
            # in effects; inert and open chambers do not replace the core lesson.
            frames[0] = _frame(rng, wall_a, target_a, colors,
                                str(rng.choice(["landing", "cascade", "double", "floor"])),
                                multi=n == 1 and (lesson == "partition" or bool(rng.integers(2))),
                                small=n >= 2 or attempt >= 35)
        packed = _place(rng, frames, len(mapping))
        if packed is None:
            continue
        shape, frames = packed
        order = list(map(int, rng.permutation(list(mapping))))
        key_data = order, [mapping[c] for c in order], int(rng.integers(1, shape[1] - len(mapping)))
        inp = _paint(shape, bg, key_data, frames)
        # Construct labels from the latent, visible chamber geometry. solve()
        # later reconstructs it independently from this rendered input.
        final_frames = []
        for frame in frames:
            if frame["opened"] or frame["color"] not in mapping:
                final_frames.append(frame)
                continue
            final_rods = []
            for allowed in frame["cavities"]:
                local = [p for p in frame["rods"] if p[0] <= allowed]
                final_rods += _settle([p for p in local if p[1] != mapping[frame["color"]]], allowed)
            final_frames.append({**frame, "rods": final_rods})
        out = _paint(shape, bg, key_data, final_frames)
        inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
        try:
            s = _read(inp)
            assert np.array_equal(_transform(s), out)
            assert all(_settle([p for p, m in zip(s["rods"], s["membership"]) if m == i],
                               region["cells"], s["direction"]) ==
                       [p for p, m in zip(s["rods"], s["membership"]) if m == i]
                       for i, region in enumerate(s["regions"]))
            # The input drawing must preserve every independent cargo rod.
            def rotated(cells):
                mask = np.zeros(shape, dtype=bool)
                for q in cells:
                    mask[q] = True
                return frozenset(map(tuple, np.argwhere(np.rot90(mask, turns)).tolist()))
            assert set(s["rods"]) == {(rotated(cells), color)
                                      for frame in frames for cells, color in frame["rods"]}
            assert set(s["contours"]) == {(rotated(frame["walls"]), frame["color"])
                                          for frame in frames}
            f = _features(s, out)
            if lesson == "scope":
                assert f["exterior_target"] and f["moved"] and f["landing"]
            if lesson == "chain":
                assert f["chain_witness"] and f["redundant"]
            assert f["deleted"] and f["active"]
        except (ValueError, AssertionError):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def _features(s, out):
    deleted, distances, landing, redundant = 0, [], False, False
    active = 0
    for i, region in enumerate(s["regions"]):
        local = [p for p, m in zip(s["rods"], s["membership"]) if m == i]
        target = region["target"]
        if target is None:
            continue
        active += 1
        survivors = [p for p in local if p[1] != target]
        removed = len(local) - len(survivors)
        deleted += removed
        final = _settle(survivors, region["cells"], s["direction"])
        for (before, color), (after, _) in zip(survivors, final):
            r0, c0 = min(before)
            r1, c1 = min(after)
            distance = abs(r1-r0) + abs(c1-c0)
            distances.append(distance)
            if distance and _translate(after, s["direction"]) <= region["cells"]:
                landing = True
            if removed and not distance and _translate(before, s["direction"]) <= region["cells"]:
                redundant = True
    exterior_target = sum(m == -1 and color in s["mapping"].values()
                          for (cells, color), m in zip(s["rods"], s["membership"]))
    chain_witness = not np.array_equal(_transform(s, recursive=True), out)
    return {"active": active, "deleted": deleted, "moved": sum(d > 0 for d in distances),
            "distances": sorted(set(distances) - {0}), "landing": landing,
            "redundant": redundant, "exterior_target": exterior_target,
            "chain_witness": chain_witness, "cavities": len(s["regions"]),
            "rods": len(s["rods"]), "contours": len(s["contours"]),
            "direction": s["direction"], "background": s["background"],
            "key_length": len(s["mapping"])}


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError("illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    num_train, num_test = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(4))
    # Two adjacent gravity directions teach the edge-to-interior convention.
    train_turns = [orientation, (orientation + 1) % 4]
    train_turns += [int(rng.choice(train_turns)) for _ in range(num_train - 2)]
    novel_direction = bool(rng.integers(2))
    test_turns = [((orientation + int(rng.integers(2, 4))) % 4 if novel_direction
                   else int(rng.choice(train_turns))) for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, turns_list in (("train", train_turns), ("test", test_turns)):
        for i, turns in enumerate(turns_list):
            lesson = ("scope" if i == 0 else "chain") if split == "train" and i < 2 else "recombine"
            if split == "test" and not novel_direction:
                lesson = "partition"
            for _ in range(12):
                pair = _make_pair(rng, lesson, turns)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _array(pair["input"]), _array(pair["output"])
            assert a.shape == out.shape
            key = _grid_key(a)
            assert key not in seen
            seen.add(key)
            s = _read(a)
            assert np.array_equal(_transform(s), out)
            bg = s["background"]
            fixed = s["walls"].copy()
            for q in s["key"]:
                fixed[q] = True
            assert np.array_equal(a[fixed], out[fixed])
            for (cells, color), m in zip(s["rods"], s["membership"]):
                if m == -1 or s["regions"][m]["target"] is None:
                    assert all(out[q] == color for q in cells)
            for i, region in enumerate(s["regions"]):
                local = [p for p, m in zip(s["rods"], s["membership"]) if m == i]
                assert all(color != region["wall"] for cells, color in local)
                assert _settle(local, region["cells"], s["direction"]) == local
                if region["target"] is not None:
                    survivors = [p for p in local if p[1] != region["target"]]
                    final = _settle(survivors, region["cells"], s["direction"])
                    assert _settle(final, region["cells"], s["direction"]) == final
                    # No recoloring, loss, stretching or overlap among survivors.
                    assert [len(p[0]) for p in survivors] == [len(p[0]) for p in final]
                    assert sum(len(p[0]) for p in final) == len(set().union(*(p[0] for p in final)))
                    assert all(out[q] != region["target"] for q in region["cells"])
            f = _features(s, out)
            assert f["active"] and f["deleted"]
            if split == "train":
                train.append((s, f, out))
    assert len({f["direction"] for s, f, out in train}) >= 2
    assert any(f["landing"] for s, f, out in train)
    assert any(f["redundant"] for s, f, out in train)
    assert any(f["chain_witness"] for s, f, out in train)
    assert any(f["exterior_target"] for s, f, out in train)
    assert any(not np.array_equal(_transform(s, settle=False), out) for s, f, out in train)
    assert any(not np.array_equal(_transform(s, global_delete=True), out) for s, f, out in train)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _mechanics_test():
    # Exact rigid-support outcomes, specified without scene generation.
    allowed = frozenset((r, c) for r in range(9) for c in range(9))
    leg_a = _rod(((r, 1) for r in range(5, 9)), 1)
    leg_b = _rod(((r, 5) for r in range(5, 9)), 3)
    beam = _rod(((4, c) for c in range(7)), 2)
    assert _settle([leg_a, leg_b, beam], allowed) == [leg_a, leg_b, beam]
    assert _settle([leg_b, beam], allowed) == [leg_b, beam]
    assert _settle([beam], allowed) == [_rod(((8, c) for c in range(7)), 2)]
    short = _rod(((r, 5) for r in (7, 8)), 3)
    upper = _rod(((r, 3) for r in (2, 3)), 4)
    assert _settle([short, beam, upper], allowed) == [
        short, _rod(((6, c) for c in range(7)), 2),
        _rod(((r, 3) for r in (4, 5)), 4)]
    stepped = frozenset((r, c) for c in range(9) for r in range(9 if c < 4 else 7))
    assert _settle([_rod(((2, c) for c in range(1, 8)), 2)], stepped) == [
        _rod(((6, c) for c in range(1, 8)), 2)]

    # Hand-specified closed/open pair. Diagonal contour corners enclose under
    # four-neighbor leakage. The map is 1 -> 2 -> 3, applied exactly once.
    a = np.zeros((14, 22), dtype=np.int8)
    a[:2, 2:4] = [[1, 2], [2, 3]]
    for offset in (0, 11):
        cavity = frozenset((r, c + offset) for r in range(5, 12) for c in range(2, 9))
        walls = {(r + dr, c + dc) for r, c in cavity for dr, dc in D4} - set(cavity)
        if offset:
            walls.remove((4, 5 + offset))
        for q in walls:
            a[q] = 1
        a[9:12, 3 + offset] = 2
        a[8, 3 + offset:8 + offset] = 3
        a[10:12, 7 + offset] = 4
    expected = a.copy()
    expected[9:12, 3] = 0
    expected[8, 3:8] = 0
    expected[9, 3:8] = 3
    assert len(_read(a)["regions"]) == 1
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(a, turns)), np.rot90(expected, turns))
    assert np.array_equal(a[:, 12:], expected[:, 12:])


def self_test():
    """42 puzzles streamed in memory, bounded counts; no logs or trace files."""
    _mechanics_test()
    shapes, backgrounds, directions, lengths, counts, structures, fingerprints = [set() for _ in range(7)]
    stats = {"multi": 0, "unequal_falls": 0, "inert": 0, "novel": 0, "familiar": 0,
             "shared_targets": 0, "active_cycles": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        local_directions = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                original = a.copy()
                assert np.array_equal(solve(a), out) and np.array_equal(a, original)
                s = _read(a)
                f = _features(s, out)
                shapes.add(a.shape)
                backgrounds.add(f["background"])
                directions.add(f["direction"])
                local_directions[split].add(f["direction"])
                lengths.add(f["key_length"])
                counts.add(f["rods"])
                # Geometry signature excludes literal palette and includes cargo.
                structures.add((a.shape, (a != s["background"]).tobytes()))
                owners = [r["owner"] for r in s["regions"]]
                stats["multi"] += len(set(owners)) < len(owners)
                stats["unequal_falls"] += len(f["distances"]) > 1
                stats["inert"] += any(r["target"] is None for r in s["regions"])
                stats["shared_targets"] += len(set(s["mapping"].values())) < len(s["mapping"])
                active = {r["wall"] for r in s["regions"]}
                stats["active_cycles"] += any(s["mapping"].get(s["mapping"].get(c)) == c
                                               and s["mapping"].get(c) != c for c in active)
                # Every rule is equivariant to rotation/reflection and a full
                # palette permutation; this probes solver conventions directly.
                assert np.array_equal(solve(np.rot90(a)), np.rot90(out))
                assert np.array_equal(solve((a + 3) % 10), (out + 3) % 10)
                assert np.array_equal(solve(np.fliplr(a)), np.fliplr(out))
        stats["novel"] += bool(local_directions["test"] - local_directions["train"])
        stats["familiar"] += bool(local_directions["test"] & local_directions["train"])
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len(shapes) >= 40 and len(backgrounds) == 10 and len(directions) == 4
    assert lengths == set(range(2, 6)) and len(counts) >= 8
    assert all(value > 0 for value in stats.values()), stats
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(10000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    assert _fingerprint(generate_puzzle(123)) == _fingerprint(generate_puzzle(123))
    assert _fingerprint(generate_puzzle(123)) != _fingerprint(generate_puzzle(124))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(12, 2, 1)
    q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    scene = _read(q["test"][0]["input"])
    q["test"][0]["output"][min(scene["key"])] = scene["background"]
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed key accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "structures": len(structures), "rod_counts": sorted(counts), **stats}
