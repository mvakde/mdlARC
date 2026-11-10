"""Color-indexed repair atlases, composed from palette tiling and consensus.

No parent generator is imported.  Generation constructs a motif and damage;
solve recovers both solely from the separated copies visible in its argument.
"""

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("grids must have two dimensions in 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grids must contain integer colors in 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _segments(n, lines):
    bounds = [-1] + lines + [n]
    return [(a + 1, b) for a, b in zip(bounds, bounds[1:])]


def _loss_order(motif, error_mask):
    # Boolean indexing visits local coordinates in row-major order.
    return list(dict.fromkeys(int(v) for v in motif[error_mask]))


def _analyze(input_grid):
    a = _grid(input_grid)
    rows = np.flatnonzero(np.all(a == a[:, :1], axis=1)).tolist()
    cols = np.flatnonzero(np.all(a == a[:1, :], axis=0)).tolist()
    if not (1 <= len(rows) <= 5 and 1 <= len(cols) <= 5):
        raise ValueError("expected a separated lattice with 2..6 copies per axis")
    separator = int(a[rows[0], 0])
    if not (np.all(a[rows, :] == separator) and np.all(a[:, cols] == separator)):
        raise ValueError("separator lines must have one common color")
    rs, cs = _segments(a.shape[0], rows), _segments(a.shape[1], cols)
    heights = {b - c for c, b in rs}
    widths = {b - c for c, b in cs}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("unequal panels")
    h, w = next(iter(heights)), next(iter(widths))
    if not (4 <= h <= 6 and 4 <= w <= 6):
        raise ValueError("panel sides must be 4..6")
    panels = np.stack([a[r0:r1, c0:c1] for r0, r1 in rs for c0, c1 in cs])
    if np.any(panels == separator):
        raise ValueError("separator ink may not occur inside panels")
    votes = np.stack([(panels == color).sum(axis=0) for color in range(10)])
    motif = votes.argmax(axis=0).astype(np.int8)
    if np.any(votes.max(axis=0) < len(panels) - 1):
        raise ValueError("copies do not have intact consensus")
    background = int(np.bincount(motif.ravel(), minlength=10).argmax())
    if np.count_nonzero(motif == background) <= motif.size / 2:
        raise ValueError("the panel background must occupy a strict majority")
    accents = set(int(v) for v in np.unique(motif)) - {background}
    if not (2 <= len(accents) <= 5):
        raise ValueError("expected 2..5 motif accents")
    differences = panels != motif
    defective = np.flatnonzero(differences.any(axis=(1, 2)))
    if len(defective) != 1:
        raise ValueError("expected exactly one defective panel")
    defect = int(defective[0])
    errors = differences[defect]
    if np.any(panels[defect][errors] != background) or np.any(motif[errors] == background):
        raise ValueError("defects must erase foreground into background")
    order = _loss_order(motif, errors)
    if not (1 <= len(order) <= 4 and len(order) < len(accents)):
        raise ValueError("expected 1..4 lost colors and an undamaged accent color")
    if not np.any((~errors) & np.isin(motif, order)):
        raise ValueError("some ink of a lost color must survive outside the damage")
    return {"motif": motif, "errors": errors, "order": order,
            "separator": separator, "background": background,
            "defect": defect, "copies": (len(rs), len(cs)),
            "panels": panels, "accent_count": len(accents)}


def _atlas(motif, errors, order, marker):
    h, w = motif.shape
    k = len(order)
    result = np.tile(motif, (k, k))
    for i, row_color in enumerate(order):
        for j, col_color in enumerate(order):
            selected = errors & ((motif == row_color) | (motif == col_color))
            result[i*h:(i+1)*h, j*w:(j+1)*w][selected] = marker
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the missing color groups and return their pairwise repair atlas."""
    info = _analyze(input_grid)
    return _atlas(info["motif"], info["errors"], info["order"], info["separator"])


_SHAPES = tuple((h, w) for h in range(4, 7) for w in range(4, 7))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _copy_shapes(shape, low=2, high=6):
    h, w = shape
    return [(r, c) for r in range(low, high + 1) for c in range(low, high + 1)
            if r*(h+1)-1 <= 30 and c*(w+1)-1 <= 30]


def _error_positions(rng, shape, count, layout, witness):
    h, w = shape
    available = [(r, c) for r in range(h) for c in range(w)
                 if not (witness and (r, c) == (0, 0))]
    if layout in ("row", "column"):
        groups = [[p for p in available if p[0 if layout == "row" else 1] == index]
                  for index in range(h if layout == "row" else w)]
        groups = [g for g in groups if len(g) >= count]
        if groups:
            group = _pick(rng, groups)
            # Consecutive runs and gapped stripes are both allowed.
            if rng.random() < .6:
                start = int(rng.integers(len(group) - count + 1))
                return sorted(group[start:start+count])
            return sorted(group[int(i)] for i in rng.choice(len(group), count, replace=False))
    if layout == "cluster":
        anchor = _pick(rng, available)
        jitter = {p: float(rng.random()) for p in available}
        available.sort(key=lambda p: (abs(p[0]-anchor[0]) + abs(p[1]-anchor[1]), jitter[p]))
        return sorted(available[:count])
    # This branch also supplies the explicitly distributed first demonstration.
    for _ in range(100):
        points = sorted(available[int(i)] for i in rng.choice(len(available), count, replace=False))
        if layout != "distributed" or (len({p[0] for p in points}) > 1
                                        and len({p[1] for p in points}) > 1):
            return points
    raise RuntimeError("could not construct distributed damage")


def _make_pair(rng, shape, copies, k, error_count, layout, palette=None,
               witness=False, defect=None):
    h, w = shape
    area = h*w
    if palette is None:
        accent_count = int(rng.integers(k+1, 6))
        palette = tuple(int(v) for v in rng.choice(10, accent_count+2, replace=False))
    separator, background, *accents = palette
    # Reserve at least one surviving occurrence of a lost color, and one
    # occurrence of each accent which does not occur in the damage.
    max_errors = (area-1)//2 - (len(accents)-k) - 1
    if error_count is None:
        error_count = int(rng.integers(k, min(10, max_errors)+1))
    if not k <= error_count <= max_errors:
        raise ValueError("damage and motif do not fit the chosen shape")
    points = _error_positions(rng, shape, error_count, layout, witness)
    errors = np.zeros(shape, dtype=bool)
    for point in points:
        errors[point] = True
    order = [int(v) for v in rng.choice(accents, k, replace=False)]
    if witness and order == sorted(order):
        order.reverse()  # Training explicitly rejects numeric color ordering.
    motif = np.full(shape, background, dtype=np.int8)
    for i, point in enumerate(points):
        # The first k loss positions exhibit the order, with later repetitions
        # either skewed or balanced, independently of color identity.
        index = i if i < k else (0 if rng.random() < .5 else int(rng.integers(k)))
        motif[point] = order[index]
    free = [p for p in np.ndindex(shape) if not errors[p]]
    spare = (0, 0) if witness else _pick(rng, free)
    motif[spare] = order[-1]
    free.remove(spare)
    rng.shuffle(free)
    for color in accents:
        if color not in order:
            motif[free.pop()] = color
    minimum = int(np.count_nonzero(motif != background))
    foreground = int(rng.integers(minimum, (area-1)//2 + 1))
    extra = foreground - minimum
    style = _pick(rng, ("scatter", "growth", "rows", "columns"))
    if style == "growth":
        occupied = [tuple(p) for p in np.argwhere(motif != background)]
        free.sort(key=lambda p: min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in occupied))
    elif style == "rows":
        free.sort(key=lambda p: (p[0], p[1]))
    elif style == "columns":
        free.sort(key=lambda p: (p[1], p[0]))
    for point in free[:extra]:
        neighbors = [(r, c) for r, c in ((point[0]-1, point[1]), (point[0]+1, point[1]),
                                        (point[0], point[1]-1), (point[0], point[1]+1))
                     if 0 <= r < h and 0 <= c < w and motif[r, c] != background]
        motif[point] = motif[_pick(rng, neighbors)] if neighbors and rng.random() < .55 else _pick(rng, accents)
    nr, nc = copies
    if defect is None:
        defect = int(rng.integers(nr*nc))
    inp = np.full((nr*(h+1)-1, nc*(w+1)-1), separator, dtype=np.int8)
    for index in range(nr*nc):
        panel = motif.copy()
        if index == defect:
            panel[errors] = background
        r, c = divmod(index, nc)
        inp[r*(h+1):r*(h+1)+h, c*(w+1):c*(w+1)+w] = panel
    # Construct using the generated latent motif, independently of solve's
    # extraction/consensus path. Validation will recover these from the input.
    out = _atlas(motif, errors, order, separator)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = _pick(rng, ("shape", "lattice", "damage", "factor"))
    if mode == "factor" and num_train < 3:
        mode = "shape"
    held_shape = _pick(rng, _SHAPES)
    train_shapes = [s for s in _SHAPES if mode != "shape" or s != held_shape]
    common_shape = _pick(rng, train_shapes)
    common_copies = _pick(rng, _copy_shapes(common_shape, high=3))
    # Equal palette size, error count, panel size, and copy lattice in the
    # first two examples isolate the lost-color count as the size signal.
    common_palette = tuple(int(v) for v in rng.choice(10, 6, replace=False))
    factors = [2, 3]
    for index in range(2, num_train):
        if mode == "factor":
            factors.append(1 if index == 2 else int(rng.integers(1, 4)))
        else:
            factors.append(4 if index == 2 else int(rng.integers(1, 5)))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index < 2
            k = factors[index] if split == "train" else (
                4 if mode == "factor" else int(_pick(rng, [f for f in factors if f >= 2])))
            if witness:
                shape, copies, palette = common_shape, common_copies, common_palette
                errors = 4
                layout = "distributed" if index == 0 else _pick(rng, ("row", "column"))
                defect = 0 if index == 0 else copies[0]*copies[1]-1
            else:
                if split == "train":
                    shape = common_shape if rng.random() < .45 else _pick(rng, train_shapes)
                elif mode == "shape":
                    shape = held_shape
                elif mode == "damage":
                    shape = _pick(rng, [s for s in _SHAPES if s[0]*s[1] >= 24])
                else:
                    shape = _pick(rng, _SHAPES)
                if mode == "lattice":
                    choices = _copy_shapes(shape, high=3)
                    if split == "test":
                        choices = [s for s in _copy_shapes(shape) if max(s) >= 4]
                else:
                    choices = _copy_shapes(shape)
                copies = _pick(rng, choices)
                if mode == "damage":
                    # Holding the palette to k+1 accents leaves enough room
                    # for the increased loss count even on 4x6 test panels.
                    size = k+3
                    palette = tuple(int(v) for v in rng.choice(10, size, replace=False))
                    errors = 4 if split == "train" else int(rng.integers(6, min(10, (shape[0]*shape[1]-1)//2-2)+1))
                else:
                    palette, errors = None, None
                layout = _pick(rng, ("scatter", "cluster", "row", "column"))
                defect = None
            for attempt in range(80):
                pair = _make_pair(rng, shape, copies, k, errors, layout, palette, witness, defect)
                signature = _key(pair["input"])
                if signature not in used:
                    used.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, input semantics, relational blocks, and training witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal example counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            motif, errors, order = info["motif"], info["errors"], info["order"]
            h, w = motif.shape
            k, marker = len(order), info["separator"]
            _require(out.shape == (k*h, k*w), "lost colors must control both output axes")
            mark_masks = []
            for i in range(k):
                row = []
                for j in range(k):
                    block = out[i*h:(i+1)*h, j*w:(j+1)*w]
                    marked = block == marker
                    expected_mask = errors & np.isin(motif, (order[i], order[j]))
                    _require(np.array_equal(marked, expected_mask), "wrong color-pair loss selection")
                    _require(np.array_equal(block[~marked], motif[~marked]), "unselected motif cells changed")
                    row.append(marked)
                mark_masks.append(row)
            diagonal = np.stack([mark_masks[i][i] for i in range(k)])
            _require(np.array_equal(diagonal.sum(axis=0), errors.astype(int)), "diagonals must partition the damage")
            for i in range(k):
                for j in range(k):
                    _require(np.array_equal(mark_masks[i][j], diagonal[i] | diagonal[j]), "off-diagonal union failed")
            frequency = np.stack([m for row in mark_masks for m in row]).sum(axis=0)
            _require(np.array_equal(frequency, errors*(2*k-1)), "wrong cross-index marking multiplicity")
            if split == "train":
                training.append(info)
    a, b = training[:2]
    _require([len(a["order"]), len(b["order"])] == [2, 3], "training lacks the 2/3 lost-color contrast")
    _require(a["errors"].sum() == b["errors"].sum() == 4, "training must reject counting missing cells")
    _require(a["motif"].shape == b["motif"].shape and a["copies"] == b["copies"], "first two geometries must match")
    _require(a["accent_count"] == b["accent_count"] == 4, "training must reject counting the full motif palette")
    _require(a["separator"] == b["separator"] and a["background"] == b["background"]
             and set(np.unique(a["motif"])) == set(np.unique(b["motif"])),
             "first two palettes must match")
    for info in (a, b):
        _require(info["order"] != sorted(info["order"]), "numeric ordering shortcut was not distinguished")
        full_order = list(dict.fromkeys(int(v) for v in info["motif"].ravel() if int(v) in info["order"]))
        _require(full_order != info["order"], "whole-motif ordering shortcut was not distinguished")


def _structure(a):
    labels = {}
    return a.shape, tuple(labels.setdefault(int(v), len(labels)) for v in a.ravel())


def self_test():
    """Bounded 42-seed corpus, count boundaries, equivariance, and ablations."""
    shapes, geometries, structures, factors, palettes = set(), set(), set(), set(), set()
    damage_counts, locations, signatures, regimes = set(), set(), set(), set()
    multirow = one_row = one_col = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple(_key(p[r]) for s in ("train", "test") for p in puzzle[s] for r in ("input", "output"))
        _require(signature == tuple(_key(p[r]) for s in ("train", "test") for p in again[s] for r in ("input", "output")), "nondeterminism")
        signatures.add(signature)
        train_info = [_analyze(p["input"]) for p in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                original = inp.copy()
                info = _analyze(inp)
                motif, mask, order = info["motif"], info["errors"], info["order"]
                k, marker = len(order), info["separator"]
                _require(np.array_equal(solve(inp), out), "solve disagrees")
                _require(np.array_equal(inp, original), "solve mutated its input")
                shapes.add(motif.shape)
                geometries.add((motif.shape, info["copies"]))
                structures.add(_structure(inp))
                factors.add(k)
                palettes.add(tuple(np.unique(inp)))
                damage_counts.add(int(mask.sum()))
                locations.add((info["copies"], info["defect"]))
                points = np.argwhere(mask)
                one_row += len(set(points[:, 0])) == 1
                one_col += len(set(points[:, 1])) == 1
                multirow += len(set(points[:, 0])) > 1
                if k > 1:
                    # Applying B once and uniformly tiling it loses pairwise selection.
                    uniform = motif.copy()
                    uniform[mask] = marker
                    _require(not np.array_equal(np.tile(uniform, (k, k)), out), "sequential parent shortcut survived")
                    # Ignoring B and just tiling the consensus never places markers.
                    _require(not np.array_equal(np.tile(motif, (k, k)), out), "consensus-only shortcut survived")
                # Every palette role is visible; arbitrary recoloring must commute.
                permutation = np.random.default_rng(10000+seed).permutation(10)
                _require(np.array_equal(solve(permutation[inp]), permutation[out]), "color-role equivariance failed")
                if split == "test":
                    if motif.shape not in {x["motif"].shape for x in train_info}:
                        regimes.add("shape")
                    if max(info["copies"]) > max(max(x["copies"]) for x in train_info):
                        regimes.add("lattice")
                    if int(mask.sum()) > max(int(x["errors"].sum()) for x in train_info):
                        regimes.add("damage")
                    if k not in {len(x["order"]) for x in train_info}:
                        regimes.add("factor")
    _require(len(signatures) == 42 and len(structures) >= 180, "too little structural variation")
    _require(len(shapes) == 9 and len(geometries) >= 35, "too little geometric variation")
    _require(factors == {1, 2, 3, 4} and len(palettes) >= 45, "too little palette/count variation")
    _require(len(damage_counts) >= 7 and len(locations) >= 35, "too little damage variation")
    _require(min(one_row, one_col, multirow) >= 15, "missing concentrated/distributed damage")
    _require(regimes == {"shape", "lattice", "damage", "factor"}, "missing train/test shifts")
    # Exercise the rule outside the generator's ten-loss sampling cap, with
    # dense damage, edge cells, and a minimal 2x2 source lattice.
    large = _make_pair(np.random.default_rng(9001), (6, 6), (2, 2), 3, 15,
                       "cluster", palette=(0, 1, 2, 3, 4, 5))
    _require(np.array_equal(solve(large["input"]), large["output"]),
             "larger valid damage masks must remain supported")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(0, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "panel_shapes": len(shapes),
            "geometries": len(geometries), "normalized_structures": len(structures),
            "palettes": len(palettes), "loss_counts": sorted(damage_counts),
            "factors": sorted(factors), "shifts": sorted(regimes)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
