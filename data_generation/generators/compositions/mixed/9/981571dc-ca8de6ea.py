"""Repair diagonal reflection orbits, then interleave two concentric rings.

Only standard library and NumPy are used. No files, traces, or caches are written.
"""
from __future__ import annotations

from hashlib import sha256
import numpy as np


_PERIMETER = ((0, 0), (0, 1), (0, 2), (1, 2),
              (2, 2), (2, 1), (2, 0), (1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _corners(cy, cx, radius):
    """UL, UR, LR, LL: clockwise in grid coordinates."""
    return ((cy - radius, cx - radius), (cy - radius, cx + radius),
            (cy + radius, cx + radius), (cy + radius, cx - radius))


def _orbit(point, center):
    """Both perpendicular diagonal mirrors and their composition."""
    r, c = point
    cy, cx = center
    dr, dc = r - cy, c - cx
    return tuple(sorted({point, (cy + dc, cx + dr),
                         (cy - dc, cx - dr), (cy - dr, cx - dc)}))


def _inspect_input(input_grid):
    """Infer background, center, radii, and unanimous witnesses from input only.

    The visible center must be a foreground cell. Every other visible point
    lies on one of its diagonals, at one of exactly two positive radii.
    At least one point survives in each of the four two-point ring orbits.
    """
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if counts.max() * 2 <= a.size:
        raise ValueError("background must occupy more than half the canvas")
    bg = int(colors[int(np.argmax(counts))])
    visible = [tuple(int(v) for v in p) for p in np.argwhere(a != bg)]
    if not 5 <= len(visible) <= 8:
        raise ValueError("expected center and partially erased eight-point rings")
    candidates = []
    for cy, cx in visible:
        radii = set()
        for r, c in visible:
            if (r, c) == (cy, cx):
                continue
            if abs(r - cy) != abs(c - cx) or r == cy:
                break
            radii.add(abs(r - cy))
        else:
            if len(radii) != 2:
                continue
            inner_radius, outer_radius = sorted(radii)
            if not (0 <= cy - outer_radius and cy + outer_radius < a.shape[0]
                    and 0 <= cx - outer_radius and cx + outer_radius < a.shape[1]):
                continue
            rings = (_corners(cy, cx, outer_radius),
                     _corners(cy, cx, inner_radius))
            repaired, values, orbits, missing_roles = a.copy(), [], [], []
            valid = True
            for ring_index, ring in enumerate(rings):
                for diagonal in range(2):
                    orbit = _orbit(ring[diagonal], (cy, cx))
                    witnesses = {int(a[p]) for p in orbit if a[p] != bg}
                    if len(orbit) != 2 or len(witnesses) != 1:
                        valid = False
                        break
                    color = next(iter(witnesses))
                    role = 2 * ring_index + diagonal
                    if any(a[p] == bg for p in orbit):
                        missing_roles.append(role)
                    for p in orbit:
                        repaired[p] = color
                    orbits.append(orbit)
                if not valid:
                    break
                ring_values = tuple(int(repaired[p]) for p in ring)
                if ring_values[0] == ring_values[1]:
                    valid = False
                    break
                values.append(ring_values)
            if valid:
                candidates.append({"background": bg, "center": (cy, cx),
                                   "radii": (outer_radius, inner_radius),
                                   "rings": rings, "values": tuple(values),
                                   "orbits": tuple(orbits),
                                   "missing_roles": tuple(missing_roles),
                                   "restored": repaired})
    if len(candidates) != 1:
        raise ValueError("input does not identify one recoverable ring motif")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _inspect_input(input_grid)
    outer, inner = info["values"]
    out = np.empty((3, 3), dtype=np.int8)
    for q in range(4):
        out[_PERIMETER[2 * q]] = outer[q]
        out[_PERIMETER[2 * q + 1]] = inner[q]
    out[1, 1] = info["restored"][info["center"]]
    return out


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _sample_geometry(rng, outer_low, outer_high, margin_limit=6):
    outer = int(rng.integers(outer_low, outer_high + 1))
    inner = int(rng.integers(1, outer))
    budget = 30 - (2 * outer + 1)
    options = [(a, b) for a in range(margin_limit + 1)
               for b in range(margin_limit + 1) if a + b <= budget]
    top, bottom = options[int(rng.integers(len(options)))]
    left, right = options[int(rng.integers(len(options)))]
    return outer, inner, top, bottom, left, right


def _roles(rng, bg, mode):
    p = [int(v) for v in rng.permutation([c for c in range(10) if c != bg])]
    if mode == "distinct":
        return tuple(p[:5])
    if mode == "reuse":
        # Equalities cross radial depth and may include the retained center.
        if rng.random() < 0.5:
            return p[0], p[1], p[0], p[2], p[int(rng.integers(3))]
        return p[0], p[1], p[1], p[0], p[int(rng.integers(2))]
    pool = p[:int(rng.integers(2, 6))]
    outer = [int(v) for v in rng.choice(pool, 2, replace=False)]
    inner = [int(v) for v in rng.choice(pool, 2, replace=False)]
    return *outer, *inner, int(rng.choice(pool))


def _make_pair(rng, geometry, bg, roles, missing_roles):
    outer, inner, top, bottom, left, right = geometry
    cy, cx = top + outer, left + outer
    inp = np.full((2 * outer + 1 + top + bottom,
                   2 * outer + 1 + left + right), bg, dtype=np.int8)
    outer_values = (roles[0], roles[1], roles[0], roles[1])
    inner_values = (roles[2], roles[3], roles[2], roles[3])
    for radius, values in ((outer, outer_values), (inner, inner_values)):
        for point, color in zip(_corners(cy, cx, radius), values):
            inp[point] = color
    inp[cy, cx] = roles[4]
    # Ground truth is computed before erasure, independently of solve().
    out = np.array(((roles[0], roles[2], roles[1]),
                    (roles[3], roles[4], roles[3]),
                    (roles[1], roles[2], roles[0])), dtype=np.int8)
    for role in missing_roles:
        ring = _corners(cy, cx, outer if role < 2 else inner)
        diagonal = role % 2
        inp[ring[diagonal + 2 * int(rng.integers(2))]] = bg
    return {"input": inp, "output": out}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _geometry_key(info, shape):
    cy, cx = info["center"]
    return *shape, cy, cx, *info["radii"]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    # None is accepted as an omitted count for the supplied sampling utility.
    if num_test is None:
        num_test = 1
    if num_train is None:
        num_train = max(4, int(num_test) + 1) if isinstance(num_test, (int, np.integer)) else 4
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen, geometries = {"train": [], "test": []}, set(), set()
    first_role = int(rng.integers(4))
    first_bg = int(rng.integers(10))
    larger_test = bool(rng.integers(2))
    for index in range(nt + ns):
        training = index < nt
        if index == 0:
            missing, mode, limits = (first_role,), "distinct", (2, 4)
            bg = first_bg
        elif index == 1:
            missing = tuple(k for k in range(4) if k != first_role)
            mode, limits = "reuse", (5, 8)
            bg = int(rng.choice([c for c in range(10) if c != first_bg]))
        else:
            count = int(rng.integers(1, 4)) if training else (
                4 if index == nt else int(rng.integers(2, 5)))
            missing = tuple(int(v) for v in rng.choice(4, count, replace=False))
            mode = ("distinct", "reuse", "mixed")[int(rng.integers(3))]
            limits = (2, 8) if training else ((9, 12) if larger_test else (3, 8))
            bg = int(rng.integers(10))
        for _ in range(64):
            geometry = _sample_geometry(rng, *limits, margin_limit=2 if index == 0 else 6)
            if geometry in geometries:
                continue
            pair = _make_pair(rng, geometry, bg, _roles(rng, bg, mode), missing)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-geometry sampling exhausted")
        geometries.add(geometry)
        seen.add(key)
        puzzle["train" if training else "test"].append(pair)
    # The pedagogical roles are not encoded by demonstration index.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _inspect_input(inp)
            bg, clean = info["background"], info["restored"]
            center = info["center"]
            _require(np.count_nonzero(clean != bg) == 9, "repair must yield nine points")
            _require(np.array_equal(clean[inp != bg], inp[inp != bg]), "visible cells changed")
            _require(clean[center] == inp[center] != bg, "center must survive unchanged")
            _require(np.count_nonzero(clean != inp) == len(info["missing_roles"]),
                     "one missing member per damaged orbit")
            _require(not np.any(out == bg), "background must be discarded")
            _require(out.shape == (3, 3), "compact output must be 3x3")
            # Direct observed-position checks, separate from solve's perimeter loop.
            outer, inner = info["rings"]
            for source, target in zip(outer, ((0, 0), (0, 2), (2, 2), (2, 0))):
                _require(out[target] == clean[source], "outer corner correspondence")
            for source, target in zip(inner, ((0, 1), (1, 2), (2, 1), (1, 0))):
                _require(out[target] == clean[source], "clockwise inner correspondence")
            _require(out[1, 1] == inp[center], "center transfer")
            _require(np.array_equal(solve(inp.copy()), out), "input-only solve mismatch")
            # Omitting A leaves at least one blank in the compact perimeter.
            raw = np.array(((inp[outer[0]], inp[inner[0]], inp[outer[1]]),
                            (inp[inner[3]], inp[center], inp[inner[1]]),
                            (inp[outer[3]], inp[inner[2]], inp[outer[2]])))
            _require(not np.array_equal(raw, out), "reflection repair is unnecessary")
            # Omitting B leaves a sparse, larger two-ring field.
            _require(clean.shape != out.shape, "interleaving is unnecessary")
            features[split].append((info, _geometry_key(info, inp.shape)))
    train = [info for info, _ in features["train"]]
    tests = [info for info, _ in features["test"]]
    _require(set().union(*(set(i["missing_roles"]) for i in train)) == set(range(4)),
             "training must demonstrate repair of both diagonals on both rings")
    _require(all(1 <= len(i["missing_roles"]) <= 3 for i in train), "training mask range")
    _require(any(len(i["missing_roles"]) == 1 and len(set(i["restored"][i["restored"] != i["background"]])) == 5
                 for i in train), "need a five-role, lightly erased demonstration")
    _require(any(len(i["missing_roles"]) == 3 for i in train), "need a three-erasure demonstration")
    _require(any(len(i["missing_roles"]) == 4 for i in tests), "test must combine all four repairs")
    _require(len({i["background"] for i in train}) >= 2, "background role must vary")
    _require(len({i["radii"] for i in train}) >= 2, "training must vary radial geometry")
    train_geometries = {g for _, g in features["train"]}
    _require(all(g not in train_geometries for _, g in features["test"]), "test geometry must be new")


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed study: stream puzzles; retain hashes and small sets only."""
    hashes, shapes, geometries, masks, backgrounds, payload_counts = set(), set(), set(), set(), set(), set()
    larger = reused = offset = wide = tall = total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        train_radii = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _inspect_input(inp)
                digest.update(bytes(inp.shape))
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                shapes.add(inp.shape)
                geometries.add(_geometry_key(info, inp.shape))
                bg, center = info["background"], info["center"]
                backgrounds.add(bg)
                payload_counts.add(len(np.unique(out)))
                mask = tuple(tuple(int(inp[p] == bg) for p in ring) for ring in info["rings"])
                masks.add(mask)
                outer, inner = info["values"]
                reused += bool(set(outer) & set(inner)) or int(out[1, 1]) in outer + inner
                cy, cx = center
                offset += (2 * cy != inp.shape[0] - 1 or 2 * cx != inp.shape[1] - 1)
                wide += inp.shape[1] > inp.shape[0]
                tall += inp.shape[0] > inp.shape[1]
                if split == "train":
                    train_radii.append(info["radii"][0])
                else:
                    larger += info["radii"][0] > max(train_radii)
                total += 1
        hashes.add(digest.hexdigest())
    _require(len(hashes) == 42, "seed diversity")
    _require(len(shapes) >= 60 and len(geometries) >= 180, "geometric diversity")
    _require(len(masks) >= 35 and len(backgrounds) == 10, "erasure/palette diversity")
    _require({2, 3, 4, 5} <= payload_counts, "cross-role color reuse range")
    _require(min(larger, reused, offset, wide, tall) >= 10, "structural variation")
    a = generate_puzzle(839390319, 7, 3)
    _require(_signature(a) == _signature(generate_puzzle(839390319, 7, 3)), "determinism")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1700 + 10 * nt + ns, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "count fidelity")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Reject wrong perimeter/center values and contradictory reflection evidence.
    for target in ((0, 1), (1, 1)):
        bad = {s: [{k: v.copy() for k, v in q.items()} for q in a[s]] for s in a}
        out = bad["test"][0]["output"]
        out[target] = (int(out[target]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted an incorrect output")
    info = _inspect_input(a["train"][0]["input"])
    clean = info["restored"]
    damaged = clean.copy()
    orbit = info["orbits"][0]
    damaged[orbit[0]] = next(c for c in range(10) if c not in (int(clean[orbit[0]]), info["background"]))
    damaged[info["orbits"][1][0]] = info["background"]
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("contradictory orbit accepted")
    # Input-only boundary coverage outside the sampler's radius envelope.
    # There are 3^4-1=80 masks: each opposite pair is intact, loses its first
    # member, or loses its second. Retain one 30x30 motif, not 80 payloads.
    boundary_rng = np.random.default_rng(912)
    base = _make_pair(boundary_rng, (14, 13, 0, 1, 1, 0), 0,
                      (1, 2, 3, 4, 5), ())
    ring_positions = (_corners(14, 15, 14), _corners(14, 15, 13))
    for code in range(1, 81):
        inp, digits = base["input"].copy(), code
        for role in range(4):
            state, digits = digits % 3, digits // 3
            if state:
                inp[ring_positions[role // 2][role % 2 + 2 * (state - 1)]] = 0
        _require(np.array_equal(solve(inp), base["output"]), "boundary/mask coverage")
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(hashes),
            "input_shapes": len(shapes), "geometries": len(geometries),
            "erasure_patterns": len(masks), "background_colors": len(backgrounds),
            "payload_color_counts": sorted(payload_counts), "larger_radius_tests": larger,
            "cross_role_reuse_pairs": reused, "offset_pairs": offset,
            "wide_pairs": wide, "tall_pairs": tall, "boundary_masks_checked": 80}
