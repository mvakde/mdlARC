"""Recover two occluded rectangular loops, then weave their corners at a pin.

Only NumPy and the standard library are used. Generation renders complete latent
loops before applying a mask; solve reconstructs them from pixels alone. No files
are read or written by this module.
"""

import hashlib
import numpy as np


HOST_SEEDS = (0,795371253,204820100,919730896,615760846,771343721,794111108,
              206603375,154231475,186443861,602729513,226523940,443954515,
              477151908,631233623,274075015,65090940,583745929,335375677,
              224007938,864421611,371726611,467620254,506034183,625982198,
              727117528,46755301,421808609,394752042,484389015,563843289,
              253212506,398643707,669187125,91886732,793495185,182803557,
              545651478,55349055,977247980,683544683,680037029)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape), "invalid grid shape")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid grid colors")
    return grid


def _walk(box):
    """Clockwise cells, NW first, with each corner appearing exactly once."""
    t, b, l, r = box
    return ([(t, x) for x in range(l, r)] + [(y, r) for y in range(t, b)]
            + [(b, x) for x in range(r, l, -1)] + [(y, l) for y in range(b, t, -1)])


def _corners(box):
    t, b, l, r = box
    return ((t, l), (t, r), (b, r), (b, l))


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _inside(point, box):
    y, x = point
    t, b, l, r = box
    return t <= y <= b and l <= x <= r


def _components(points):
    remaining, groups = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _complete(grid, box, visible):
    """Require all supported, doubly witnessed short words to agree."""
    cells = _walk(box)
    predictions, fits = set(), []
    observed = [(i, int(grid[p])) for i, p in enumerate(cells) if p in visible]
    for period in range(1, 7):
        word, counts = [None] * period, [0] * period
        for i, color in observed:
            j = i % period
            if word[j] is not None and word[j] != color:
                break
            word[j] = color
            counts[j] += 1
        else:
            if min(counts) >= 2:
                predictions.add(tuple(word[i % period] for i in range(len(cells))))
                fits.append(period)
    _require(len(predictions) == 1, "missing or ambiguous perimeter repetition")
    return dict(zip(cells, next(iter(predictions)))), tuple(fits)


def _weave(outer, inner, center):
    return np.array(((outer[0], inner[0], outer[1]),
                     (inner[3], center, inner[1]),
                     (outer[3], inner[2], outer[2])), dtype=np.int8)


def _place(frame, center, fill, motif):
    t, b, l, r = frame
    cy, cx = center
    result = np.full((b-t-1, r-l-1), fill, dtype=np.int8)
    y, x = cy-t-1, cx-l-1
    result[y-1:y+2, x-1:x+2] = motif
    return result


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(bool(np.all(grid[0] == bg) and np.all(grid[-1] == bg)
                  and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)),
             "an exterior background margin is required")
    frames = []
    for color in np.unique(grid):
        if color == bg:
            continue
        points = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        box = _bbox(points)
        t, b, l, r = box
        if b-t < 4 or r-l < 4 or points != set(_walk(box)):
            continue
        interior = grid[t+1:b, l+1:r]
        values, counts = np.unique(interior, return_counts=True)
        fill = int(values[int(np.argmax(counts))])
        pin = np.argwhere(interior != fill)
        if len(pin) != 1:
            continue
        cy, cx = (int(pin[0, 0])+t+1, int(pin[0, 1])+l+1)
        if t+2 <= cy <= b-2 and l+2 <= cx <= r-2:
            frames.append((box, int(color), fill, (cy, cx)))
    _require(len(frames) == 1, "expected one filled frame with one center pin")
    frame, frame_color, fill, center = frames[0]
    cy, cx = center
    center_color = int(grid[center])
    _require(center_color not in (bg, fill, frame_color), "invalid pin color")
    visible = {tuple(map(int, p)) for p in np.argwhere(grid != bg)
               if not _inside(tuple(map(int, p)), frame)}
    groups = _components(visible)
    _require(len(groups) == 2, "expected two separated exposed loops")
    rings = []
    for group in groups:
        ry = max(abs(y-cy) for y, x in group)
        rx = max(abs(x-cx) for y, x in group)
        _require(ry >= 3 and rx >= 3, "ring radius too small")
        box = (cy-ry, cy+ry, cx-rx, cx+rx)
        t, b, l, r = box
        _require(0 < t < b < grid.shape[0]-1 and 0 < l < r < grid.shape[1]-1,
                 "ring reaches canvas boundary")
        full = set(_walk(box))
        _require({p for p in full if not _inside(p, frame)} == group,
                 "visible geometry is not a centered rectangular perimeter")
        colors = {int(grid[p]) for p in group}
        _require(colors.isdisjoint((bg, fill, frame_color)), "role colors overlap")
        completed, fits = _complete(grid, box, group)
        corners = _corners(box)
        hidden = [p for p in corners if _inside(p, frame)]
        _require(len(hidden) in (1, 2), "each loop must hide one or two corners")
        ft, fb, fl, fr = frame
        _require(all(ft < y < fb and fl < x < fr for y, x in hidden),
                 "hidden corners must lie strictly inside the frame")
        rings.append({"box": box, "radii": (ry, rx), "values": completed,
                      "visible": group, "fits": fits, "hidden_corners": tuple(hidden),
                      "points": tuple(completed[p] for p in corners)})
    rings.sort(key=lambda ring: ring["radii"][0], reverse=True)
    outer, inner = rings
    _require(all(a-b >= 2 for a, b in zip(outer["radii"], inner["radii"])),
             "loops must nest with a blank orthogonal gap")
    _require(len(outer["hidden_corners"]) == len(inner["hidden_corners"]),
             "both rings must share an occlusion regime")
    motif = _weave(outer["points"], inner["points"], center_color)
    output = _place(frame, center, fill, motif)
    scene = {"frame": frame, "center": center, "fill": fill, "background": bg,
             "frame_color": frame_color, "center_color": center_color,
             "rings": rings, "shape": tuple(grid.shape)}
    return output, scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible geometry and repeated colors only."""
    return _read(input_grid)[0]


def _signature(scene):
    return (scene["shape"], scene["frame"], scene["center"],
            tuple(r["radii"] for r in scene["rings"]))


def _roles(scene):
    return (*scene["rings"][0]["points"], *scene["rings"][1]["points"],
            scene["center_color"])


def _geometry(rng, kind, first, testing):
    if first:
        oy = ox = int(rng.choice([6, 7, 8]))
        iy = ix = int(rng.choice([3, 4]))
    else:
        low, high = (8, 11) if testing else (6, 10)
        oy, ox = (int(x) for x in rng.integers(low, high, size=2))
        minimum = 4 if kind == "half" else 3
        iy = int(rng.integers(minimum, oy-1))
        ix = int(rng.integers(minimum, ox-1))
        if oy == ox and iy == ix:
            return None
    # Coordinates are initially relative to the shared pin at (0,0).
    dx = int(rng.integers(2, ix))
    dy = int(rng.integers(2, iy))
    ey, ex = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    if kind == "quadrant":
        frame = (-dy, oy+ey, -dx, ox+ex)
    elif bool(rng.integers(2)):
        frame = (-oy-ey, oy+ey, -dx, ox+ex)
    else:
        frame = (-dy, oy+ey, -ox-ex, ox+ex)
    t, b, l, r = frame
    if bool(rng.integers(2)):
        t, b = -b, -t
    if bool(rng.integers(2)):
        l, r = -r, -l
    top, bottom, left, right = (int(x) for x in rng.integers(1, 3, size=4))
    ymin, ymax, xmin, xmax = min(-oy, t), max(oy, b), min(-ox, l), max(ox, r)
    cy, cx = top-ymin, left-xmin
    shape = (ymax-ymin+1+top+bottom, xmax-xmin+1+left+right)
    frame = (t+cy, b+cy, l+cx, r+cx)
    boxes = [(cy-a, cy+a, cx-bb, cx+bb) for a, bb in ((oy, ox), (iy, ix))]
    return shape, frame, (cy, cx), boxes


def _make_pair(rng, index, testing, backgrounds, previous_roles=None):
    first = index == 0 and not testing
    second = index == 1 and not testing
    kind = "quadrant" if first else "half" if second else str(rng.choice(["quadrant", "half"]))
    for _ in range(512):
        geometry = _geometry(rng, kind, first, testing)
        if geometry is None:
            continue
        shape, frame, center, boxes = geometry
        palette = [int(x) for x in rng.permutation([c for c in range(10) if c != backgrounds])]
        bg, border, fill, pin = backgrounds, palette[0], palette[1], palette[2]
        if first or (not second and bool(rng.integers(2))):
            fill = bg
        ring_palette = palette[3:]
        if not first and not second and bool(rng.integers(2)):
            pin = int(rng.choice(ring_palette))
        under = np.full(shape, bg, dtype=np.int8)
        streams = []
        for box in boxes:
            period = 5 if first or second else int(rng.integers(1, 7))
            word = [int(x) for x in rng.permutation(ring_palette)[:period]]
            if not first and not second and period >= 3 and rng.random() < 0.35:
                word[-1] = word[0]
            phase = int(rng.integers(period))
            for i, p in enumerate(_walk(box)):
                under[p] = word[(i+phase) % period]
            streams.append(tuple(int(under[p]) for p in _corners(box)))
        if first or second:
            if any(len(set(stream)) != 4 for stream in streams):
                continue
            current_roles = (*streams[0], *streams[1], pin)
            if previous_roles is not None and len(set(zip(previous_roles, current_roles))) != 9:
                continue
        # Truth uses complete latent loops, before any opaque pixels exist.
        output = _place(frame, center, fill, _weave(*streams, pin))
        inp = under.copy()
        t, b, l, r = frame
        inp[t:b+1, l:r+1] = fill
        for p in _walk(frame):
            inp[p] = border
        inp[center] = pin
        try:
            recovered, scene = _read(inp)
        except ValueError:
            continue
        if not np.array_equal(recovered, output):
            continue
        return {"input": inp, "output": output}, scene
    raise RuntimeError("bounded sampling exhausted while seeking visible repetition witnesses")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (seed >= 0 and 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need nonnegative seed, 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    backgrounds = [int(x) for x in rng.permutation(10)]
    puzzle, seen_inputs, seen_geometry = {"train": [], "test": []}, set(), set()
    first_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            number = len(seen_inputs)
            for _ in range(32):
                pair, scene = _make_pair(rng, index, split == "test", backgrounds[number % 10],
                                         first_roles if split == "train" and index == 1 else None)
                key = _key(pair["input"])
                signature = _signature(scene)
                if key not in seen_inputs and signature not in seen_geometry:
                    break
            else:
                raise RuntimeError("bounded uniqueness sampling exhausted")
            seen_inputs.add(key)
            seen_geometry.add(signature)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first_roles = _roles(scene)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return (tuple(a.shape), a.tobytes())


def _ablations(grid, scene):
    frame, center, fill = scene["frame"], scene["center"], scene["fill"]
    outer, inner = scene["rings"]
    # Omitting A: feed observed, still erased corner colors to B.
    streams = [tuple(int(grid[p]) for p in _corners(r["box"])) for r in (outer, inner)]
    no_reconstruction = _place(frame, center, fill, _weave(*streams, scene["center_color"]))
    # Omitting B: return the ordinary restored A-style crop plus the visible pin.
    t, b, l, r = frame
    no_interleave = np.full((b-t-1, r-l-1), fill, dtype=np.int8)
    for ring in (outer, inner):
        for (y, x), color in ring["values"].items():
            if t < y < b and l < x < r:
                no_interleave[y-t-1, x-l-1] = color
    no_interleave[center[0]-t-1, center[1]-l-1] = scene["center_color"]
    return no_reconstruction, no_interleave


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, geometry, scenes = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            predicted, scene = _read(inp)
            assert np.array_equal(predicted, out), "wrong reconstructed corner weave"
            sig = _signature(scene)
            assert sig not in geometry, "geometry must vary within a puzzle"
            geometry.add(sig)
            for alternative in _ablations(inp, scene):
                assert not np.array_equal(alternative, out), "one parent is dispensable"
            scenes[split].append(scene)
    first, second = scenes["train"][:2]
    assert first["background"] != second["background"]
    assert first["fill"] == first["background"] and second["fill"] != second["background"]
    assert len(first["rings"][0]["hidden_corners"]) == 1
    assert len(second["rings"][0]["hidden_corners"]) == 2
    assert all(len(set(r["points"])) == 4 for s in (first, second) for r in s["rings"])
    assert len(set(zip(_roles(first), _roles(second)))) == 9, "demonstrations alias point roles"
    assert all(y == x for y, x in (r["radii"] for r in first["rings"]))
    assert any(y != x for y, x in (r["radii"] for r in second["rings"]))


def self_test():
    """Bounded, streaming checks; returns compact descriptive measurements."""
    shapes, layouts, normalized, words, bgs, fills, hidden, outputs = set(), set(), set(), set(), set(), set(), set(), set()
    hashes = set()
    pairs, rectangles, reused_pin = 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                predicted, scene = _read(inp)
                assert np.array_equal(predicted, out)
                digest.update(repr(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                shapes.add(inp.shape)
                outputs.add(out.shape)
                layouts.add(_signature(scene))
                roles = {}
                normalized.add((inp.shape, bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                bgs.add(scene["background"])
                fills.add(scene["fill"] == scene["background"])
                hidden.add(len(scene["rings"][0]["hidden_corners"]))
                for ring in scene["rings"]:
                    words.add(min(ring["fits"]))
                rectangles += any(a != b for a, b in (r["radii"] for r in scene["rings"]))
                reused_pin += scene["center_color"] in {c for r in scene["rings"] for c in r["values"].values()}
                pairs += 1
        hashes.add(digest.hexdigest())
    assert len(hashes) == 42 and len(shapes) >= 35 and len(layouts) >= 170
    assert len(normalized) >= 170 and len(outputs) >= 30
    assert bgs == set(range(10)) and fills == {False, True} and hidden == {1, 2}
    assert words == set(range(1, 7)) and rectangles >= 100 and reused_pin >= 15
    # Whole-puzzle determinism, explicit counts, invalid types/counts, and corruption.
    one, two = generate_puzzle(101, 10, 4), generate_puzzle(101, 10, 4)
    assert all(_key(a[k]) == _key(b[k]) for s in one for a, b in zip(one[s], two[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(314159+a*10+b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,False),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(102, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(shapes),
            "output_shapes": len(outputs), "geometries": len(layouts),
            "normalized_inputs": len(normalized), "periods": sorted(words),
            "rectangular_pairs": rectangles, "pin_reuse_pairs": reused_pin,
            "independent_solver": False}
