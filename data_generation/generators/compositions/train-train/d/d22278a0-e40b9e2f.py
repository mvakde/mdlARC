"""Pinwheel fountains: rotationally recovered, competing quarter-square waves.

Only standard-library and NumPy dependencies. All solving information is visible
in the grid; the sampling plans below change difficulty, never the rule.
"""

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be 2D with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer colors required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a


def _orbit(row, col, center2):
    """Integer sites rotated about an integer or half-integer center."""
    x, y = 2 * row - center2[0], 2 * col - center2[1]
    result = []
    for _ in range(4):
        rr, cc = center2[0] + x, center2[1] + y
        _require(rr % 2 == 0 and cc % 2 == 0, "rotation misses cell centers")
        result.append((rr // 2, cc // 2))
        x, y = -y, x
    return tuple(sorted(set(result)))


def _parse(value):
    """Identify the hub by its solid square, not by a literal marker color."""
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > a.size // 2, "input background must dominate")
    hubs = []
    for color in colors:
        if color == bg:
            continue
        cells = np.argwhere(a == color)
        lo, hi = cells.min(0), cells.max(0)
        side = int(hi[0] - lo[0] + 1)
        if (side in (2, 3) and int(hi[1] - lo[1] + 1) == side
                and len(cells) == side * side):
            hubs.append((int(color), tuple(map(int, lo)), side))
    _require(len(hubs) == 1, "one uniquely colored solid square hub required")
    hub, (top, left), side = hubs[0]
    center2 = (2 * top + side - 1, 2 * left + side - 1)
    source_colors = sorted(int(c) for c in colors if c not in (bg, hub))
    _require(2 <= len(source_colors) <= 4, "need two to four source orbits")
    visible, sites, orbits, visible_counts = [], [], {}, []
    occupied = set()
    for color in source_colors:
        cells = [tuple(map(int, p)) for p in np.argwhere(a == color)]
        full = _orbit(*cells[0], center2)
        _require(len(full) == 4 and set(cells) <= set(full),
                 "each source color must describe one rotational orbit")
        _require(1 <= len(cells) <= 4, "invalid orbit visibility")
        for r, c in full:
            _require(1 <= r < a.shape[0] - 1 and 1 <= c < a.shape[1] - 1,
                     "all recovered sources must fit inside the canvas")
            _require(min(abs(2 * r - center2[0]),
                         abs(2 * c - center2[1])) >= (4 if side == 3 else 5),
                     "sources must be diagonally outside the hub")
            _require((r, c) not in occupied, "source orbits overlap")
            occupied.add((r, c))
            sites.append((r, c, color))
        visible.extend((r, c, color) for r, c in cells)
        visible_counts.append(len(cells))
        orbits[color] = full
    _require(any(n < 4 for n in visible_counts), "some source must be missing")
    return {"grid": a, "bg": bg, "hub": hub, "top": top, "left": left,
            "side": side, "center2": center2, "colors": source_colors,
            "sites": sites, "visible": visible, "orbits": orbits,
            "visible_counts": tuple(visible_counts)}


def _render(info, sites=None, metric="manhattan", outward=True,
            even_only=True, suppress_ties=True):
    a, bg, center2 = info["grid"], info["bg"], info["center2"]
    sites = info["sites"] if sites is None else sites
    rows, cols = np.indices(a.shape)
    distances = np.stack([
        abs(rows - r) + abs(cols - c) if metric == "manhattan" else
        np.maximum(abs(rows - r), abs(cols - c)) for r, c, _ in sites
    ])
    minimum = distances.min(0)
    unique = ((distances == minimum).sum(0) == 1 if suppress_ties
              else np.ones(a.shape, dtype=bool))
    winner = distances.argmin(0)
    out = np.full(a.shape, bg, dtype=np.int64)
    for i, (r, c, color) in enumerate(sites):
        paint = unique & (winner == i)
        if outward:
            paint &= ((rows - r) * (2 * r - center2[0]) >= 0)
            paint &= ((cols - c) * (2 * c - center2[1]) >= 0)
        if even_only:
            paint &= np.maximum(abs(rows - r), abs(cols - c)) % 2 == 0
        out[paint] = color
    t, l, s = info["top"], info["left"], info["side"]
    out[t:t + s, l:l + s] = info["hub"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the emitters, then apply ownership and oriented square shells."""
    return _render(_parse(input_grid))


def _cellwise(info):
    """Slower, direct specification check used by the validator."""
    a, sites = info["grid"], info["sites"]
    out = np.full(a.shape, info["bg"], dtype=int)
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            ds = [abs(row - r) + abs(col - c) for r, c, _ in sites]
            nearest = min(ds)
            owners = [i for i, d in enumerate(ds) if d == nearest]
            if len(owners) != 1:
                continue
            r, c, color = sites[owners[0]]
            cr2, cc2 = info["center2"]
            if ((row - r) * (2 * r - cr2) >= 0
                    and (col - c) * (2 * c - cc2) >= 0
                    and max(abs(row - r), abs(col - c)) % 2 == 0):
                out[row, col] = color
    t, l, s = info["top"], info["left"], info["side"]
    out[t:t + s, l:l + s] = info["hub"]
    return out


def _facts(info, out):
    """Counterfactual witnesses, measured from the returned input alone."""
    a, bg = info["grid"], info["bg"]
    sites, center2 = info["sites"], info["center2"]
    rows, cols = np.indices(a.shape)
    ds = np.stack([abs(rows - r) + abs(cols - c) for r, c, _ in sites])
    mi = ds.min(0)
    tied = (ds == mi).sum(0) > 1
    tie_witness = np.zeros(a.shape, dtype=bool)
    occlusion = np.zeros_like(tie_witness)
    for i, (r, c, color) in enumerate(sites):
        wave = ((rows - r) * (2 * r - center2[0]) >= 0)
        wave &= ((cols - c) * (2 * c - center2[1]) >= 0)
        wave &= np.maximum(abs(rows - r), abs(cols - c)) % 2 == 0
        tie_witness |= wave & tied & (ds[i] == mi)
        occlusion |= wave & (ds[i] > mi) & (out != color)
    without_rotation = _render(info, sites=info["visible"])
    remote = np.ones(a.shape, dtype=bool)
    for r, c, _ in sites:
        remote[r, c] = False
    t, l, s = info["top"], info["left"], info["side"]
    remote[t:t + s, l:l + s] = False
    erasure = remote & (out == bg) & (without_rotation != bg)
    t2, l2 = center2
    radius = max(max(abs(2 * r - t2), abs(2 * c - l2))
                 for r, c, _ in sites) // 2
    geometry = (s, a.shape, center2, tuple(sorted(
        tuple(sorted((2 * r - t2, 2 * c - l2) for r, c in orbit))
        for orbit in info["orbits"].values())))
    return {
        "tie": bool(tie_witness.any()),
        "occlusion": bool(occlusion.any()),
        "hidden_competitor": bool(erasure.any()),
        "metric": not np.array_equal(out, _render(info, metric="chebyshev")),
        "facing": not np.array_equal(out, _render(info, outward=False)),
        "parity": not np.array_equal(out, _render(info, even_only=False)),
        "grows": all(np.count_nonzero(out == color) > 4
                     for color in info["colors"]),
        "radius": radius, "geometry": geometry,
        "source_count": len(info["colors"]),
        "center_type": "cell" if s == 3 else "half",
        "centered_square": a.shape[0] == a.shape[1]
        and center2 == (a.shape[0] - 1, a.shape[1] - 1),
    }


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _sample_input(rng, side, count, radius, centered):
    # Positive-quadrant representatives determine four rotated sources each.
    values = list(range(4 if side == 3 else 5,
                        2 * radius + (1 if side == 3 else 2), 2))
    pool = [(a, b) for a in values for b in values]
    rng.shuffle(pool)
    representatives = []
    for p in pool:
        # Avoid a second orbit that is the mirror of the first. This is a
        # sampling preference, keeping rotation vs reflection easy to see.
        if (p[1], p[0]) not in representatives:
            representatives.append(p)
        if len(representatives) == count:
            break
    if (len(representatives) != count
            or max(max(p) for p in representatives) != values[-1]
            or not any(a != b for a, b in representatives)):
        return None
    span = 2 * radius + (1 if side == 3 else 2)
    if centered:
        pads = [int(rng.integers(3, 6))] * 4
    else:
        pads = [int(rng.integers(3, 7)) for _ in range(4)]
        # Ensure a genuine finite-window shift rather than a lucky square.
        if pads[0] == pads[1] and pads[2] == pads[3]:
            pads[0] = 3 if pads[0] != 3 else 4
    top_pad, bottom_pad, left_pad, right_pad = pads
    height, width = span + top_pad + bottom_pad, span + left_pad + right_pad
    if max(height, width) > 30:
        return None
    palette = [int(x) for x in rng.choice(10, size=count + 2, replace=False)]
    bg, hub = palette[:2]
    inp = np.full((height, width), bg, dtype=np.int64)
    center2 = (2 * top_pad + span - 1, 2 * left_pad + span - 1)
    top = (center2[0] - side + 1) // 2
    left = (center2[1] - side + 1) // 2
    inp[top:top + side, left:left + side] = hub
    # At least one one-member orbit; others may already be partly/fully shown.
    visibility = [1] + [int(rng.choice([1, 2, 3, 4], p=[.35, .30, .25, .10]))
                        for _ in range(count - 1)]
    rng.shuffle(visibility)
    for (a, b), color, nvisible in zip(representatives, palette[2:], visibility):
        row, col = (center2[0] + a) // 2, (center2[1] + b) // 2
        full = _orbit(row, col, center2)
        for i in rng.choice(4, size=nvisible, replace=False):
            inp[full[int(i)]] = color
    return inp


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    # Three supported extrapolations. None changes any output convention.
    shift = ("density", "radius", "window")[int(rng.integers(3))]
    types = [2, 3] + [int(rng.choice([2, 3])) for _ in range(num_train - 2)]
    rng.shuffle(types)
    train_counts = [2, 3] + [int(rng.choice([2, 3])) for _ in range(num_train - 2)]
    rng.shuffle(train_counts)
    puzzle = {"train": [], "test": []}
    seen, geometry = set(), set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            side = types[index] if split == "train" else int(rng.choice([2, 3]))
            count = (train_counts[index] if split == "train" else
                     4 if shift == "density" else int(rng.choice([2, 3])))
            low = 4 if count == 2 else 5
            radii = list(range(low, 6 if shift == "radius" else 7))
            if split == "test" and shift == "radius":
                radii = [7, 8]
            centered = shift == "window" and split == "train"
            for _ in range(500):
                radius = int(rng.choice(radii))
                inp = _sample_input(rng, side, count, radius, centered)
                if inp is None or _key(inp) in seen:
                    continue
                info = _parse(inp)
                out = _render(info)
                facts = _facts(info, out)
                if facts["geometry"] in geometry:
                    continue
                if not all(facts[k] for k in (
                    "tie", "occlusion", "hidden_competitor", "metric",
                    "facing", "parity", "grows"
                )):
                    continue
                puzzle[split].append({"input": inp, "output": out})
                seen.add(_key(inp))
                geometry.add(facts["geometry"])
                break
            else:
                raise RuntimeError("bounded sampling could not find a witness-rich pair")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, geometries = set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions must be retained")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            _require(np.array_equal(_cellwise(info), out), "cellwise rule mismatch")
            _require(np.array_equal(solve(inp), out), "solve mismatch")
            _require(np.array_equal(inp[inp != info["bg"]], out[inp != info["bg"]]),
                     "all visible cells must be retained")
            _require(set(map(int, np.unique(inp))) == set(map(int, np.unique(out))),
                     "palette changed")
            for color, orbit in info["orbits"].items():
                _require(all(out[r, c] == color for r, c in orbit),
                         "a completed source site is missing")
            facts = _facts(info, out)
            _require(facts["geometry"] not in geometries, "recolored geometry repeated")
            geometries.add(facts["geometry"])
            for feature in ("tie", "occlusion", "hidden_competitor", "metric",
                            "facing", "parity", "grows"):
                _require(facts[feature], "missing counterexample for " + feature)
            records[split].append(facts)
    tr, te = records["train"], records["test"]
    _require({r["center_type"] for r in tr} == {"cell", "half"},
             "both center conventions must be demonstrated")
    _require({r["source_count"] for r in tr} == {2, 3},
             "training must demonstrate different orbit counts")
    density = all(r["source_count"] == 4 for r in te)
    radius = min(r["radius"] for r in te) > max(r["radius"] for r in tr)
    window = all(r["centered_square"] for r in tr) and all(
        not r["centered_square"] for r in te)
    _require(density or radius or window, "missing documented train/test shift")
    return True


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity suite plus API and equivariance checks."""
    shapes, layouts, backgrounds, hubs, palettes = set(), set(), set(), set(), set()
    sizes, orbit_counts, visibility, shifts, signatures = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        records = {s: [] for s in puzzle}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                facts = _facts(info, pair["output"])
                records[split].append(facts)
                shapes.add(info["grid"].shape)
                layouts.add(facts["geometry"])
                backgrounds.add(info["bg"])
                hubs.add(info["hub"])
                palettes.add(tuple(sorted(info["colors"])))
                sizes.add((info["side"], facts["radius"]))
                orbit_counts.add(facts["source_count"])
                visibility.update(info["visible_counts"])
        tr, te = records["train"], records["test"]
        if all(f["source_count"] == 4 for f in te):
            shifts.add("density")
        if min(f["radius"] for f in te) > max(f["radius"] for f in tr):
            shifts.add("radius")
        if all(f["centered_square"] for f in tr):
            shifts.add("window")
        # The transformation respects geometry and literal color relabeling.
        p = puzzle["test"][0]
        _require(np.array_equal(solve(np.rot90(p["input"])), np.rot90(p["output"])),
                 "rotation equivariance failed")
        _require(np.array_equal(solve(np.fliplr(p["input"])), np.fliplr(p["output"])),
                 "reflection equivariance failed")
        permutation = np.roll(np.arange(10), seed % 9 + 1)
        _require(np.array_equal(solve(permutation[p["input"]]), permutation[p["output"]]),
                 "color equivariance failed")
    _require(len(signatures) == 42 and len(layouts) >= 180, "insufficient layouts")
    _require(len(shapes) >= 40 and len(sizes) >= 8, "insufficient scale variation")
    _require(len(backgrounds) == 10 and len(hubs) == 10 and len(palettes) >= 35,
             "insufficient color-role variation")
    _require(orbit_counts == {2, 3, 4} and visibility == {1, 2, 3, 4},
             "missing source counts or visibility regimes")
    _require(shifts == {"density", "radius", "window"}, "missing shift regime")
    # Every legal explicit count combination, with different deterministic seeds.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(5000 + 10 * nt + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "wrong counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_puzzle_key(generate_puzzle(9718993)) ==
             _puzzle_key(generate_puzzle(9718993)), "not deterministic")
    return {"seeds": 42, "puzzles": len(signatures), "shapes": len(shapes),
            "layouts": len(layouts), "backgrounds": sorted(backgrounds),
            "hub_colors": sorted(hubs), "source_palettes": len(palettes),
            "side_radius_pairs": sorted(sizes), "orbit_counts": sorted(orbit_counts),
            "visible_counts": sorted(visibility), "shifts": sorted(shifts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
