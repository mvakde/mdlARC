"""Locator-addressed transport of independently rotated 3x3 color layers.

All conventions are visible in one input. Nothing is read from disk by solving
or generation. self_test retains only aggregate statistics and small hashes.
"""
from itertools import combinations
import hashlib

import numpy as np


HOST_SEEDS = (0,405787093,983008106,625514188,109863840,101249189,235390907,962406223,315990682,846245661,84583629,380796670,81248890,757596735,107108279,167880757,747695068,586286121,84785523,500091247,368572777,343341713,858661978,834301837,591348150,994095572,127992093,889444404,181864350,803089539,750775259,271493348,377914591,446781504,945080945,656001358,665431388,123123789,413455372,285642024,655396214,535679895)
POSITIONS = tuple((r, c) for r in range(3) for c in range(3))
LATTICE_MASK = np.zeros((11, 11), dtype=bool)
LATTICE_MASK[[3, 7], :] = True
LATTICE_MASK[:, [3, 7]] = True


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("grid must be two dimensional with sides in 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors in 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.tobytes()


def _panel(position):
    r, c = position
    return slice(4*r, 4*r+3), slice(4*c, 4*c+3)


def _coord(r, c, turns):
    for _ in range(turns % 4):
        r, c = c, 2-r
    return r, c


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _symmetry(mask):
    if np.array_equal(mask, np.rot90(mask)):
        return "quarter"
    if np.array_equal(mask, np.rot90(mask, 2)):
        return "half"
    return "asymmetric"


def _parse(input_grid):
    """Recover all roles, the original address, and bars from input alone."""
    a = _grid(input_grid)
    values, frequencies = np.unique(a, return_counts=True)
    if np.count_nonzero(frequencies == frequencies.max()) != 1:
        raise ValueError("background must be uniquely modal")
    bg = int(values[frequencies.argmax()])
    lattices = []
    # Divider occurs exactly on the forty lattice cells, nowhere else. Its
    # coordinate extrema determine the candidate origin without a crop oracle.
    for color in values[frequencies == 40]:
        points = np.argwhere(a == color)
        r, c = map(int, points.min(axis=0))
        if tuple(points.max(axis=0)) != (r+10, c+10):
            continue
        crop = a[r:r+11, c:c+11]
        if crop.shape == (11, 11) and np.array_equal(crop == color, LATTICE_MASK):
            lattices.append((r, c, int(color), crop))
    if len(lattices) != 1:
        raise ValueError("expected exactly one 11x11 divider lattice")
    r, c, divider, lattice = lattices[0]
    if divider == bg:
        raise ValueError("background and divider must differ")
    panel_counts = np.bincount(lattice[~LATTICE_MASK], minlength=10)
    locators = [int(x) for x in np.flatnonzero(panel_counts == 1)
                if x not in (bg, divider)]
    if len(locators) != 1:
        raise ValueError("exactly one lattice content color must occur once")
    locator = locators[0]
    y, x = map(int, np.argwhere(lattice == locator)[0])
    source = y//4, x//4
    address = y % 4, x % 4
    if address not in POSITIONS:
        raise ValueError("locator cannot occupy a divider")
    payload = lattice[_panel(source)].copy()
    palette = tuple(int(x) for x in np.unique(payload) if x != bg)
    if len(palette) < 2:
        raise ValueError("payload must contain locator and another color")
    exterior = a != bg
    exterior = exterior.copy()
    exterior[r:r+11, c:c+11] = False
    markers = {}
    bar_points = []
    for color in np.unique(a[exterior]):
        color = int(color)
        points = np.argwhere(exterior & (a == color))
        yy, xx = points.T
        horizontal = np.ptp(yy) == 0 and np.ptp(xx)+1 == len(points)
        vertical = np.ptp(xx) == 0 and np.ptp(yy)+1 == len(points)
        if color not in palette or not 1 <= len(points) <= 9 or not (horizontal or vertical):
            raise ValueError("each exterior color must form one matching straight bar of length 1..9")
        for y, x in points:
            if r-1 <= y <= r+11 and c-1 <= x <= c+11:
                raise ValueError("bar must have a background gap from the entire lattice")
        orientation = "single" if len(points) == 1 else ("horizontal" if horizontal else "vertical")
        markers[color] = {"length": len(points), "orientation": orientation,
                          "points": tuple(map(tuple, points.tolist()))}
        bar_points.extend((color, int(y), int(x)) for y, x in points)
    if not markers:
        raise ValueError("at least one counter bar is required")
    for i, (color, y, x) in enumerate(bar_points):
        if any(color != other and max(abs(y-v), abs(x-u)) <= 1
               for other, v, u in bar_points[i+1:]):
            raise ValueError("different bars must be separated even diagonally")
    return {"array": a, "background": bg, "divider": divider, "origin": (r, c),
            "lattice": lattice, "locator": locator, "source": source,
            "address": address, "payload": payload, "palette": palette, "markers": markers}


def _empty_canvas(info):
    out = np.full(info["array"].shape, info["background"], dtype=np.int8)
    r, c = info["origin"]
    out[r:r+11, c:c+11][LATTICE_MASK] = info["divider"]
    return out


def _place(out, info, payload, destination):
    r, c = info["origin"]
    out[r:r+11, c:c+11][_panel(destination)] = payload
    return out


def _coordinate_payload(info):
    """Coordinate implementation used by the public input-only solver."""
    bg = info["background"]
    result = np.full((3, 3), bg, dtype=np.int8)
    for color in info["palette"]:
        k = info["markers"].get(color, {}).get("length", 0)
        for r, c in np.argwhere(info["payload"] == color):
            y, x = _coord(int(r), int(c), k)
            if result[y, x] != bg:
                raise ValueError("independently rotated layers conflict")
            result[y, x] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    transformed = _coordinate_payload(info)
    return _place(_empty_canvas(info), info, transformed, info["address"])


def _array_transform(payload, bg, counts):
    """Array implementation used for generation and validation."""
    result = np.full((3, 3), bg, dtype=np.int8)
    occupied = np.zeros((3, 3), dtype=bool)
    for color in np.unique(payload):
        color = int(color)
        if color == bg:
            continue
        mask = np.rot90(payload == color, -counts.get(color, 0) % 4)
        if np.any(mask & occupied):
            raise ValueError("rotated masks overlap; no overwrite priority exists")
        occupied |= mask
        result[mask] = color
    return result


def _make_payload(rng, bg, colors, address, counts, required_asym):
    # At most 96 tiny attempts. Single-cell fallback is guaranteed: with at
    # most three placed cells, union of source/target exclusions has size <=6.
    target_locator = _coord(*address, counts.get(colors[0], 0))
    for attempt in range(97):
        payload = np.full((3, 3), bg, dtype=np.int8)
        payload[address] = colors[0]
        occupied = {target_locator}
        good = True
        order = [colors[int(i)] for i in rng.permutation(np.arange(1, len(colors)))]
        for j, color in enumerate(order):
            allowed = [p for p in POSITIONS if payload[p] == bg
                       and _coord(*p, counts.get(color, 0)) not in occupied]
            if attempt == 96:
                wanted = 1
            else:
                reserve = len(order)-j-1
                cap = min(5, len(allowed), int(np.count_nonzero(payload == bg))-reserve)
                if cap < 1:
                    good = False
                    break
                wanted = int(rng.integers(1, cap+1))
            candidates = list(combinations(allowed, wanted))
            rng.shuffle(candidates)
            chosen = None
            for points in candidates:
                mask = np.zeros((3, 3), dtype=bool)
                for p in points:
                    mask[p] = True
                if color not in required_asym or _asymmetric(mask):
                    chosen = points
                    break
            if chosen is None:
                good = False
                break
            for p in chosen:
                payload[p] = color
                occupied.add(_coord(*p, counts.get(color, 0)))
        if good and all(_asymmetric(payload == color) for color in required_asym):
            return payload
    raise RuntimeError("could not construct payload")


def _fallback_canvas(lattice, bg, counts, orientations):
    """Guaranteed slots for up to four bars of either orientation, length <=9."""
    a = np.full((30, 30), bg, dtype=np.int8)
    a[9:20, 9:20] = lattice
    horizontal = vertical = 0
    for i, (color, length) in enumerate(counts.items()):
        orientation = orientations.get(color, "horizontal" if i % 2 == 0 else "vertical")
        if orientation == "horizontal":
            a[2*horizontal, :length] = color
            horizontal += 1
        else:
            a[:length, 22+2*vertical] = color
            vertical += 1
    _parse(a)
    return a, (9, 9)


def _put_bars(rng, lattice, bg, counts, orientations):
    # Ordinary attempts retain varied offsets, margins, rectangular geometry,
    # orientations and sides; the last layout has a packing proof.
    for _ in range(20):
        h, w = map(int, rng.integers(15, 31, size=2))
        r, c = int(rng.integers(h-10)), int(rng.integers(w-10))
        a = np.full((h, w), bg, dtype=np.int8)
        a[r:r+11, c:c+11] = lattice
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0,r-1):min(h,r+12), max(0,c-1):min(w,c+12)] = True
        success = True
        for color in sorted(counts, key=lambda x: -counts[x]):
            length = counts[color]
            options = []
            for orientation in ("horizontal", "vertical"):
                if color in orientations and orientations[color] != orientation:
                    continue
                dh, dw = (1, length) if orientation == "horizontal" else (length, 1)
                yy, xx = np.indices((h-dh+1, w-dw+1))
                clear = np.ones(yy.shape, dtype=bool)
                for k in range(length):
                    clear &= ~blocked[yy+(k if dh > 1 else 0), xx+(k if dw > 1 else 0)]
                starts = np.argwhere(clear)
                if len(starts):
                    options.append((orientation, starts))
            if not options:
                success = False
                break
            orientation, starts = options[int(rng.integers(len(options)))]
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            for k in range(length):
                v, u = y+(k if orientation == "vertical" else 0), x+(k if orientation == "horizontal" else 0)
                a[v, u] = color
                blocked[max(0,v-1):min(h,v+2), max(0,u-1):min(w,u+2)] = True
        if success:
            try:
                _parse(a)
            except ValueError:
                continue
            return a, (r, c)
    return _fallback_canvas(lattice, bg, counts, orientations)


def _address_plan(rng, num_train, num_test):
    corners = [p for p in POSITIONS if p[0] != 1 and p[1] != 1]
    edges = [p for p in POSITIONS if (p[0] == 1) != (p[1] == 1)]
    a = corners[int(rng.integers(4))]
    choices = [p for p in edges if p[0] != a[0] and p[1] != a[1]]
    b = choices[int(rng.integers(len(choices)))]
    addresses = [a, b]
    rng.shuffle(addresses)
    if num_train >= 3:
        addresses.append((next(iter({0,1,2}-{a[0],b[0]})), next(iter({0,1,2}-{a[1],b[1]}))))
    addresses += [POSITIONS[int(rng.integers(9))] for _ in range(num_train-len(addresses))]
    pool = [(r,c) for r in sorted({p[0] for p in addresses})
            for c in sorted({p[1] for p in addresses})]
    rng.shuffle(pool)
    test_addresses = sorted(pool, key=lambda p: p in addresses)[:num_test]
    sources = []
    for i, address in enumerate(addresses):
        options = [p for p in POSITIONS if (i >= 2 or p != address)
                   and (i != 1 or p != sources[0])]
        sources.append(options[int(rng.integers(len(options)))])
    test_sources = []
    used = set(zip(sources, addresses))
    for i, address in enumerate(test_addresses):
        options = [p for p in POSITIONS if (p, address) not in used and (i != 0 or p != address)]
        p = options[int(rng.integers(len(options)))]
        test_sources.append(p)
        used.add((p, address))
    return sources, addresses, test_sources, test_addresses


def _draw_pair(rng, source, address, witness, regime, odd):
    layers = 3 if witness >= 0 else int(rng.integers(2, 5))
    role_colors = [int(x) for x in rng.choice(10, 2+layers, replace=False)]
    bg, divider = role_colors[:2]
    colors = role_colors[2:]  # first is the unique locator within the lattice
    orientations = {}
    if witness == 0:
        lengths = [odd, 2, 2]
        orientations[colors[1]], orientations[colors[2]] = ("horizontal", "vertical") if rng.random() < .5 else ("vertical", "horizontal")
        required_asym = set(colors)
    elif witness == 1:
        lengths = [4-odd, 0, 4 if regime == 0 else int(rng.choice([4, 8]))]
        required_asym = set(colors)
    else:
        lengths = [0]*layers
        for i in rng.choice(layers, int(rng.integers(1, layers+1)), replace=False):
            lengths[int(i)] = int(rng.integers(1, 5 if regime == 0 and witness == -1 else 10))
        required_asym = set()
        if witness == -2 and regime == 0:
            # Extrapolate unary counts after the demonstrated full cycle.
            lengths[1] = int(rng.choice([5, 6, 7, 9]))
            required_asym.add(colors[1])
    counts = {color: k for color, k in zip(colors, lengths) if k}
    payload = _make_payload(rng, bg, colors, address, counts, required_asym)
    transformed = _array_transform(payload, bg, counts)
    lattice = np.full((11, 11), bg, dtype=np.int8)
    lattice[LATTICE_MASK] = divider
    lattice[_panel(source)] = payload
    extras = [c for c in range(10) if c not in role_colors]
    rng.shuffle(extras)
    auxiliaries = colors[1:]+extras[:int(rng.integers(0, min(3, len(extras))+1))]
    distractors = [p for p in POSITIONS if p != source]
    for p in distractors:
        roll = rng.random()
        count = 0 if roll < .17 else (1 if roll < .34 else int(rng.integers(2, 7)))
        cells = rng.choice(9, count, replace=False)
        panel = lattice[_panel(p)]
        for k in cells:
            panel.flat[int(k)] = auxiliaries[int(rng.integers(len(auxiliaries)))]
    # Rarity supplies an input-visible locator, unlike A's learned color. Every
    # other lattice content color appears >=2 times; locator bars don't count.
    free = [(p, k) for p in distractors for k in range(9) if lattice[_panel(p)].flat[k] == bg]
    rng.shuffle(free)
    for color in auxiliaries:
        missing = max(0, 2-int(np.count_nonzero(lattice == color)))
        for _ in range(missing):
            p, k = free.pop()
            lattice[_panel(p)].flat[k] = color
    if not any(np.any(lattice[_panel(p)] != bg) for p in distractors):
        lattice[_panel(distractors[0])].flat[:2] = colors[1]
    a, origin = _put_bars(rng, lattice, bg, counts, orientations)
    out = np.full(a.shape, bg, dtype=np.int8)
    r, c = origin
    out_lattice = out[r:r+11, c:c+11]
    out_lattice[LATTICE_MASK] = divider
    out_lattice[_panel(address)] = transformed
    return {"input": a, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(2))
    odd = int(rng.choice([1, 3]))
    sources, addresses, test_sources, test_addresses = _address_plan(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, ss, aa in (("train", sources, addresses), ("test", test_sources, test_addresses)):
        for i, (source, address) in enumerate(zip(ss, aa)):
            witness = i if split == "train" and i < 2 else (-1 if split == "train" else -2)
            for _ in range(16):
                pair = _draw_pair(rng, source, address, witness, regime, odd)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _parse(inp)
    bg, payload = info["background"], info["payload"]
    counts = {c: bar["length"] for c, bar in info["markers"].items()}
    transformed = _array_transform(payload, bg, counts)
    expected = _place(_empty_canvas(info), info, transformed, info["address"])
    assert np.array_equal(expected, out), "incorrect destination, rotation, clearing or lattice"
    assert np.array_equal(solve(inp), out), "coordinate/array implementations disagree"
    assert np.count_nonzero(out == info["locator"]) == 1
    r, c = info["origin"]
    crop = out[r:r+11, c:c+11]
    assert all(np.count_nonzero(payload == color) == np.count_nonzero(crop == color)
               for color in info["palette"])
    assert any(np.any(info["lattice"][_panel(p)] != bg) for p in POSITIONS if p != info["source"])
    info["transformed"] = transformed
    return info


def _evidence(infos):
    lengths, repeated_orientations = set(), False
    independent, unmarked, direction, original_address = False, False, False, False
    ablate_a, ablate_b = False, False
    for info in infos:
        payload, transformed = info["payload"], info["transformed"]
        residues = set()
        by_length = {}
        for color in info["palette"]:
            if not _asymmetric(payload == color):
                continue
            marker = info["markers"].get(color)
            if marker is None:
                unmarked = True
            else:
                k = marker["length"]
                lengths.add(k)
                residues.add(k % 4)
                direction |= k % 2 == 1
                by_length.setdefault(k, set()).add(marker["orientation"])
        repeated_orientations |= any({"horizontal", "vertical"} <= v for v in by_length.values())
        independent |= len(residues) >= 2
        ablate_a |= info["source"] != info["address"]
        ablate_b |= not np.array_equal(payload, transformed)
        locator_target = tuple(map(int, np.argwhere(transformed == info["locator"])[0]))
        original_address |= locator_target != info["address"]
    return {"short_counts": {1,2,3} <= lengths, "cycle": bool({4,8} & lengths),
            "orientation": repeated_orientations, "independent": independent,
            "unmarked": unmarked, "clockwise": direction,
            "address_before_rotation": original_address, "A_necessary": ablate_a,
            "B_necessary": ablate_b}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(info["array"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    assert all(_evidence(train).values()), "missing instructional witness"
    addresses = {i["address"] for i in train}
    sources = {i["source"] for i in train}
    assert len(sources) >= 2
    rows, cols = {p[0] for p in addresses}, {p[1] for p in addresses}
    assert len(rows) >= 2 and len(cols) >= 2
    # Among all eight square symmetries only direct local-to-macro addressing
    # agrees with the examples. This catches transposed/reflected address plans.
    compatible = []
    for flip in (False, True):
        for turns in range(4):
            if all(_coord(p[0], 2-p[1] if flip else p[1], turns) == p for p in addresses):
                compatible.append((flip, turns))
    assert compatible == [(False, 0)]
    test_addresses = [i["address"] for i in infos["test"]]
    assert len(test_addresses) == len(set(test_addresses))
    combinations_seen = {(i["source"], i["address"]) for i in train}
    for info in infos["test"]:
        assert info["address"][0] in rows and info["address"][1] in cols
        assert (info["source"], info["address"]) not in combinations_seen


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 specified seeds, bounded in-memory aggregates; writes no artifacts."""
    shapes, backgrounds, positions, addresses, areas, symmetries = set(), set(), set(), set(), set(), set()
    palettes, fingerprints, structure_hashes, lengths, orientations, sides = set(), set(), set(), set(), set(), set()
    totals = {"pairs": 0, "stationary": 0, "moving_locator": 0, "empty_distractor": 0,
              "dense_distractor": 0, "symmetric_control": 0, "long_test": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                a, bg = info["array"], info["background"]
                shapes.add(a.shape)
                backgrounds.add(bg)
                positions.add(info["source"])
                addresses.add(info["address"])
                palettes.add(info["palette"])
                totals["pairs"] += 1
                totals["stationary"] += info["source"] == info["address"]
                totals["moving_locator"] += tuple(np.argwhere(info["transformed"] == info["locator"])[0]) != info["address"]
                mapping = {}
                normalized = bytes(mapping.setdefault(int(x), len(mapping)) for x in a.flat)
                structure_hashes.add(hashlib.sha256(bytes(a.shape)+normalized).hexdigest())
                for color in info["palette"]:
                    mask = info["payload"] == color
                    areas.add(int(mask.sum()))
                    sym = _symmetry(mask)
                    symmetries.add(sym)
                    totals["symmetric_control"] += sym != "asymmetric" and color in info["markers"]
                rr, cc = info["origin"]
                for marker in info["markers"].values():
                    lengths.add(marker["length"])
                    orientations.add(marker["orientation"])
                    totals["long_test"] += split == "test" and marker["length"] > 4
                    pts = np.array(marker["points"])
                    y, x = pts.T
                    for side, condition in (("top", max(y) < rr), ("bottom", min(y) >= rr+11),
                                            ("left", max(x) < cc), ("right", min(x) >= cc+11)):
                        if condition:
                            sides.add(side)
                for p in POSITIONS:
                    if p != info["source"]:
                        mass = int(np.count_nonzero(info["lattice"][_panel(p)] != bg))
                        totals["empty_distractor"] += mass == 0
                        totals["dense_distractor"] += mass >= 5
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and backgrounds == set(range(10))
    assert positions == addresses == set(POSITIONS)
    assert len(palettes) >= 50 and len(structure_hashes) == totals["pairs"]
    assert {1,2,3,4,5} <= areas
    assert symmetries == {"asymmetric", "half", "quarter"}
    assert lengths == set(range(1,10))
    assert orientations == {"single", "horizontal", "vertical"}
    assert sides == {"top", "bottom", "left", "right"}
    assert all(totals[k] > 0 for k in totals)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(42, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.0,1),(4,None),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutation checks exercise both layer geometry and exact clearing.
    p = generate_puzzle(19, 2, 1)
    bad = {k: v.copy() for k, v in p["test"][0].items()}
    bad["output"][0,0] = (int(bad["output"][0,0])+1) % 10
    try:
        _analyze_pair(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Counterfactuals: omitting either mechanism must change training answers.
    for pair in p["train"]:
        info = _parse(pair["input"])
        no_a = _place(_empty_canvas(info), info, _coordinate_payload(info), info["source"])
        no_b = _place(_empty_canvas(info), info, info["payload"], info["address"])
        assert not np.array_equal(no_a, pair["output"])
        assert not np.array_equal(no_b, pair["output"])
    # Directly exercise guaranteed packing with the maximum four long bars.
    lattice = np.zeros((11, 11), dtype=np.int8)
    lattice[LATTICE_MASK] = 9
    lattice[:3, :3] = [[1,2,2],[3,3,4],[0,4,0]]
    for forced in ({c: "horizontal" for c in (1,2,3,4)},
                   {c: "vertical" for c in (1,2,3,4)},
                   {1: "horizontal", 2: "vertical"}):
        a, origin = _fallback_canvas(lattice, 0, {c: 9 for c in (1,2,3,4)}, forced)
        assert origin == (9,9) and a.shape == (30,30)
        info = _parse(a)
        assert np.array_equal(_coordinate_payload(info), np.rot90(lattice[:3,:3], -1))
    # Reject an unmatched counter and an ambiguous rarity-based locator.
    info = _parse(p["train"][0]["input"])
    a = info["array"].copy()
    y, x = next(iter(info["markers"].values()))["points"][0]
    other = next(c for c in range(10) if c not in info["palette"] and c != info["background"])
    a[y, x] = other
    try:
        solve(a)
    except ValueError:
        pass
    else:
        raise AssertionError("unmatched counter accepted")
    ambiguous = lattice.copy()
    ambiguous[8,8] = 5
    try:
        _parse(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous locator accepted")
    return {"seeds": 42, "shapes": len(shapes), "palettes": len(palettes),
            "structures": len(structure_hashes), "layer_areas": sorted(areas),
            "symmetries": sorted(symmetries), "bar_lengths": sorted(lengths), "totals": totals}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
