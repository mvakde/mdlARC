import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "0400000040",
            "0000003000",
            "0060000000",
            "0000000003",
            "0005555000",
            "3000000000",
            "0006000000",
            "0400000000",
            "0000030006",
            "0300000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0006666000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0900000004",
            "0000200000",
            "0000000900",
            "4005550000",
            "0005550000",
            "0005550002",
            "0090000000",
            "2000000400",
            "0000000000",
            "0020000200",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0009990000",
            "0009990000",
            "0009990000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "8040003040",
            "0000000000",
            "3000500800",
            "0040500000",
            "0000500004",
            "0800500000",
            "0000500800",
            "0000000000",
            "0000000000",
            "4080030040",
        ),
        (
            "0000000000",
            "0000000000",
            "0000400000",
            "0000400000",
            "0000400000",
            "0000400000",
            "0000400000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0040000000",
            "0000000000",
            "0005550000",
            "0005550000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0600000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0004440000",
            "0004440000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "7000060000",
            "0090000700",
            "0000500000",
            "6005550609",
            "0055555000",
            "0005550000",
            "0000500060",
            "9000000000",
            "0000000000",
            "0000600090",
        ),
        (
            "0000000000",
            "0000000000",
            "0000600000",
            "0006660000",
            "0066666000",
            "0006660000",
            "0000600000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0300200007",
            "0000000000",
            "7000550003",
            "0005555000",
            "0005555000",
            "0200550003",
            "0000000000",
            "0000000070",
            "3070000000",
            "0003007020",
        ),
        (
            "0000000000",
            "0000000000",
            "0000220000",
            "0002222000",
            "0002222000",
            "0000220000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_count("num_train", num_train, 2, 10)
    test_count = _validate_count("num_test", num_test, 1, 4)
    code = int(seed)

    if train_count is None and test_count is None:
        test_count = 1 + code % 4
        minimum_train = test_count + 1
        train_count = (
            minimum_train
            + (code // 4) % (11 - minimum_train)
        )
    elif train_count is None:
        minimum_train = test_count + 1
        train_count = (
            minimum_train
            + code % (11 - minimum_train)
        )
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + code % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    style = int(rng.integers(3))
    colors = [int(x) for x in rng.permutation(10)]
    if style == 0:
        third = next(color for color in colors if color not in (0, 5))
        roles = (0, 5, third)
        palette = tuple(color for color in colors if color not in roles)
    else:
        palette = tuple(colors[:7])
        roles = tuple(colors[7:])
    return {
        "palette": palette,
        "role_colors": roles,
        "vary_roles": bool(rng.integers(2)) if style else False,
        "style": style,
    }


def _shape(rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "bar":
        crop = np.ones((1, int(rng.integers(3, 12))), bool)
    elif kind == "rectangle":
        crop = np.ones((int(rng.integers(2, 8)), int(rng.integers(2, 9))), bool)
    elif kind == "diamond":
        radius = int(rng.integers(1, 5))
        rr, cc = np.indices((2 * radius + 1,) * 2)
        crop = abs(rr - radius) + abs(cc - radius) <= radius
    elif kind == "bevel":
        h, w = int(rng.integers(4, 9)), int(rng.integers(4, 10))
        inset = int(rng.integers(1, max(2, (w - 1) // 2)))
        crop = np.ones((h, w), bool)
        crop[0, :inset] = crop[0, w-inset:] = False
        crop[-1, :inset] = crop[-1, w-inset:] = False
    elif kind == "l":
        h, w = int(rng.integers(4, 10)), int(rng.integers(4, 10))
        thick = int(rng.integers(1, min(h, w, 4)))
        crop = np.zeros((h, w), bool)
        crop[:, :thick] = True
        crop[-thick:, :] = True
    elif kind == "t":
        h, w = int(rng.integers(4, 10)), int(rng.integers(5, 11))
        thick = int(rng.integers(1, 3))
        stem = int(rng.integers(1, w - thick + 1))
        crop = np.zeros((h, w), bool)
        crop[:thick, :] = True
        crop[:, stem:stem+thick] = True
    elif kind == "step":
        h, w = int(rng.integers(4, 9)), int(rng.integers(5, 11))
        crop = np.zeros((h, w), bool)
        for row in range(h):
            start = min(w - 2, row * (w - 2) // h)
            end = start + int(rng.integers(2, min(5, w-start+1)))
            crop[row, min(start, w//2):max(end, w//2+1)] = True
    else:  # branch
        h, w = int(rng.integers(5, 11)), int(rng.integers(5, 11))
        crop = np.zeros((h, w), bool)
        center = int(rng.integers(1, w - 1))
        crop[:, center] = True
        for row in rng.choice(np.arange(1, h-1), size=int(rng.integers(2, 5)), replace=False):
            side = -1 if rng.integers(2) else 1
            length = int(rng.integers(2, 5))
            if side < 0:
                crop[row, max(0, center-length):center+1] = True
            else:
                crop[row, center:min(w, center+length+1)] = True
    crop = np.rot90(crop, int(rng.integers(4)))
    if rng.integers(2):
        crop = np.fliplr(crop)
    return crop.copy()


def _place(crop: np.ndarray, rng: np.random.Generator, roomy: bool = False, margin: int = 6) -> np.ndarray:
    oh, ow = crop.shape
    if roomy:
        h = oh + 2 * margin + int(rng.integers(0, 5))
        w = ow + 2 * margin + int(rng.integers(0, 5))
        if h > 30 or w > 30:
            raise RuntimeError("roomy silhouette does not fit")
        row = margin + int(rng.integers(h-oh-2*margin+1))
        col = margin + int(rng.integers(w-ow-2*margin+1))
    else:
        extra_h = int(rng.integers(5, 19))
        extra_w = int(rng.integers(5, 19))
        h, w = min(30, oh + extra_h), min(30, ow + extra_w)
        row_space, col_space = h-oh, w-ow
        edge_mode = int(rng.integers(5))
        if edge_mode == 0:
            row = int(rng.choice([0, row_space]))
            col = int(rng.choice([0, col_space]))
        elif edge_mode == 1:
            if rng.integers(2):
                row = int(rng.choice([0, row_space]))
                col = int(rng.integers(col_space+1))
            else:
                row = int(rng.integers(row_space+1))
                col = int(rng.choice([0, col_space]))
        else:
            row = int(rng.integers(row_space+1))
            col = int(rng.integers(col_space+1))
    mask = np.zeros((h, w), bool)
    mask[row:row+oh, col:col+ow] = crop
    return mask


def _allowed_marker_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    forbidden = mask.copy()
    forbidden[1:] |= mask[:-1]
    forbidden[:-1] |= mask[1:]
    forbidden[:, 1:] |= mask[:, :-1]
    forbidden[:, :-1] |= mask[:, 1:]
    return [tuple(map(int, p)) for p in np.argwhere(~forbidden)]


def _distance_sq(position: tuple[int, int], coordinates: np.ndarray) -> int:
    delta = coordinates - position
    return int(np.min(np.sum(delta * delta, axis=1)))


def _far_enough(p: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(abs(p[0]-q[0]) + abs(p[1]-q[1]) > 1 for q in chosen)


def _add_extras(rng, mask, markers, colors, target, min_distance, winner_bias=0.0):
    coordinates = np.argwhere(mask)
    candidates = _allowed_marker_positions(mask)
    rng.shuffle(candidates)
    chosen = [p for p, _ in markers]
    present = {c for _, c in markers}
    missing = [c for c in colors if c not in present]
    for color in missing:
        options = [p for p in candidates if _far_enough(p, chosen) and _distance_sq(p, coordinates) > min_distance]
        if options:
            p = options[0]
            markers.append((p, color))
            chosen.append(p)
    while len(markers) < target:
        options = [p for p in candidates if _far_enough(p, chosen) and _distance_sq(p, coordinates) > min_distance]
        if not options:
            break
        p = options[0]
        chosen.append(p)
        color = int(colors[0] if rng.random() < winner_bias else rng.choice(colors[1:]))
        markers.append((p, color))
    return markers


def _generic_markers(rng, mask, winner, colors, target, want_tie):
    coordinates = np.argwhere(mask)
    candidates = _allowed_marker_positions(mask)
    if len(candidates) < 2:
        raise RuntimeError("too few marker sites")
    distances = {p: _distance_sq(p, coordinates) for p in candidates}
    near = sorted(set(distances.values()))[:min(6, len(set(distances.values())))]
    threshold = int(rng.choice(near))
    winner_options = [p for p in candidates if distances[p] == threshold]
    rng.shuffle(winner_options)
    selected = winner_options[0]
    markers = [(selected, winner)]
    chosen = [selected]
    if want_tie:
        ties = [p for p in winner_options[1:] if _far_enough(p, chosen)]
        if ties:
            p = ties[0]
            markers.append((p, winner))
            chosen.append(p)
    rivals = [p for p in candidates if distances[p] > threshold and _far_enough(p, chosen)]
    if not rivals:
        raise RuntimeError("no rival marker")
    rivals.sort(key=lambda p: distances[p])
    closest_band = rivals[:max(1, len(rivals)//3)]
    rival = closest_band[int(rng.integers(len(closest_band)))]
    rival_color = next(c for c in colors if c != winner)
    markers.append((rival, rival_color))
    winner_bias = float(rng.choice([0.02, 1.0/len(colors), 0.65]))
    return _add_extras(rng, mask, markers, colors, target, threshold, winner_bias)


def _metric_witness(rng, winner, colors, target):
    # All choices make the Euclidean winner lose to one rival under
    # Manhattan distance and another under Chebyshev distance.
    a, b, m, c = [
        (5, 2, 6, 4), (6, 2, 7, 5), (6, 3, 7, 5),
        (6, 3, 8, 5), (7, 2, 8, 6), (7, 3, 9, 6),
        (7, 4, 10, 6),
    ][int(rng.integers(7))]
    crop = _shape(rng, str(rng.choice(["bar", "rectangle"])))
    mask = _place(crop, rng, roomy=True, margin=max(a, m, c)+1)
    rows, cols = np.where(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    vs = -1 if rng.integers(2) else 1
    hs = -1 if rng.integers(2) else 1
    anchor = (r0 if vs < 0 else r1, c0 if hs < 0 else c1)
    selected = (int(anchor[0]+a*vs), int(anchor[1]+b*hs))
    rival_m = (int(anchor[0]), int(anchor[1]+m*hs))
    rival_c = (int(anchor[0]+c*vs), int(anchor[1]+c*hs))
    allowed = set(_allowed_marker_positions(mask))
    if not {selected, rival_m, rival_c} <= allowed:
        raise RuntimeError("metric witness does not fit")
    markers = [(selected, winner), (rival_m, colors[1]), (rival_c, colors[1])]
    return mask, _add_extras(rng, mask, markers, colors, target, a*a+b*b)


def _bounding_witness(rng, winner, colors):
    for _ in range(12):
        crop = _shape(rng, str(rng.choice(["bevel", "l", "t", "step"])))
        mask = _place(crop, rng, roomy=True)
        coords = np.argwhere(mask)
        bbox = np.zeros_like(mask)
        rr, cc = np.where(mask)
        bbox[rr.min():rr.max()+1, cc.min():cc.max()+1] = True
        boxcoords = np.argwhere(bbox)
        sites = _allowed_marker_positions(mask)
        distances = {p: (_distance_sq(p, coords), _distance_sq(p, boxcoords)) for p in sites}
        options = [(p, q) for p in sites for q in sites
                   if distances[p][0] < distances[q][0]
                   and distances[p][1] > distances[q][1]
                   and _far_enough(p, [q])]
        if options:
            p, q = options[int(rng.integers(len(options)))]
            return mask, [(p, winner), (q, colors[1])]
    raise RuntimeError("cannot make bounding witness")


def _render_pair(mask, markers, background, placeholder):
    inp = np.full(mask.shape, background, dtype=np.int8)
    inp[mask] = placeholder
    for p, color in markers:
        if inp[p] != background:
            raise RuntimeError("overlapping marker")
        inp[p] = color
    coords = np.argwhere(mask)
    scored = [(_distance_sq(p, coords), c) for p, c in markers]
    nearest = min(d for d, _ in scored)
    colors = {c for d, c in scored if d == nearest}
    if len(colors) != 1:
        raise RuntimeError("ambiguous nearest color")
    output = np.full(mask.shape, background, dtype=np.int8)
    output[mask] = next(iter(colors))
    return inp, output


def _rows_match(grid: np.ndarray, rows: tuple[str, ...]) -> bool:
    return grid.shape == (len(rows), len(rows[0])) and all(
        "".join(map(str, map(int, grid[i]))) == row for i, row in enumerate(rows))


def _is_source_pair(inp, out) -> bool:
    return any(_rows_match(inp, si) and _rows_match(out, so) for si, so in _SOURCE_PAIRS)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _orthogonal_components(mask):
    seen = np.zeros_like(mask, bool)
    components = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr,nc] and not seen[nr,nc]:
                    seen[nr,nc] = True
                    stack.append((nr,nc))
        components.append(comp)
    return components


def _pair_features(inp, out):
    _require(inp.shape == out.shape, "shape changed")
    colors, counts = np.unique(out, return_counts=True)
    _require(len(colors) == 2, "output needs background and one silhouette color")
    bg = int(colors[int(np.argmax(counts))])
    painted = int(next(c for c in colors if c != bg))
    mask = out == painted
    _require(3 <= int(mask.sum()) < out.size//2, "silhouette size invalid")
    _require(int(np.count_nonzero(inp == bg)) > inp.size//2, "background must dominate")
    comps = _orthogonal_components(inp != bg)
    multi = [x for x in comps if len(x) > 1]
    singles = [x[0] for x in comps if len(x) == 1]
    _require(len(multi) == 1, "not exactly one connected silhouette")
    reconstructed = np.zeros_like(mask)
    for p in multi[0]:
        reconstructed[p] = True
    _require(np.array_equal(mask, reconstructed), "silhouette changed")
    placeholder_colors = np.unique(inp[mask])
    _require(len(placeholder_colors) == 1, "silhouette not monochrome")
    placeholder = int(placeholder_colors[0])
    _require(placeholder not in (bg, painted), "role colors collide")
    _require(len(singles) >= 2, "too few markers")
    markers = [(p, int(inp[p])) for p in singles]
    marker_colors = {c for _, c in markers}
    _require(2 <= len(marker_colors) <= 5 and placeholder not in marker_colors and bg not in marker_colors, "marker palette invalid")
    coords = np.argwhere(mask)
    ds = [(_distance_sq(p, coords), c) for p, c in markers]
    minimum = min(d for d, _ in ds)
    _require({c for d,c in ds if d == minimum} == {painted}, "nearest marker color wrong or tied")
    expected = np.full(out.shape, bg, dtype=np.int8)
    expected[mask] = painted
    _require(np.array_equal(out, expected), "markers not erased")
    rr, cc = np.where(mask)
    bbox = np.zeros_like(mask)
    bbox[rr.min():rr.max()+1, cc.min():cc.max()+1] = True
    bd = [(_distance_sq(p, np.argwhere(bbox)), c) for p,c in markers]
    bmin = min(d for d,c in bd)
    bbox_colors = {c for d,c in bd if d == bmin}
    def alt(metric):
        values = []
        for p,c in markers:
            delta = np.abs(coords - p)
            d = np.min(np.sum(delta,axis=1) if metric == "manhattan" else np.max(delta,axis=1))
            values.append((int(d), c))
        best = min(d for d,c in values)
        return {c for d,c in values if d == best}
    crop = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
    edge = rr.min() == 0 or cc.min() == 0 or rr.max() == mask.shape[0]-1 or cc.max() == mask.shape[1]-1
    return {"bg": bg, "placeholder": placeholder, "winner": painted,
            "marker_colors": marker_colors, "marker_count": len(markers),
            "mask": mask, "crop": crop, "edge": edge, "tie": sum(d == minimum for d,c in ds)>1,
            "nearest_distance_sq": minimum,
            "manhattan_gap": painted not in alt("manhattan"),
            "chebyshev_gap": painted not in alt("chebyshev"),
            "bbox_gap": painted not in bbox_colors,
            "frequency_gap": sum(c == painted for _,c in markers) < max(sum(x == c for _,x in markers) for c in marker_colors)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train","test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid counts")
    seen = set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input","output"}, "invalid pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D ndarray")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid bounds")
                _require(np.issubdtype(grid.dtype, np.integer) and bool(np.all((0 <= grid)&(grid <= 9))), "grid values")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(not _is_source_pair(inp, out), "source copy")
            infos[section].append(_pair_features(inp, out))
    train = infos["train"]
    _require(len({f["winner"] for f in train}) >= 2, "train lacks output color diversity")
    _require(any(f["manhattan_gap"] for f in train), "train lacks Manhattan witness")
    _require(any(f["chebyshev_gap"] for f in train), "train lacks Chebyshev witness")
    _require(any(f["bbox_gap"] for f in train), "train lacks cellwise silhouette witness")
    _require(any(f["frequency_gap"] for f in train), "train lacks frequency witness")
    trained_markers = set().union(*(f["marker_colors"] for f in train))
    _require(all(f["winner"] in trained_markers for f in infos["test"]), "test color absent from training markers")
    return None


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ns = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    params = _sample_puzzle_parameters(rng)
    palette = list(params["palette"])
    role_pool = list(params["role_colors"])
    train_winners = list(rng.choice(palette, size=min(nt, len(palette)), replace=False))
    train_winners += [int(rng.choice(palette)) for _ in range(nt-len(train_winners))]
    rng.shuffle(train_winners)
    if len(set(train_winners)) < 2:
        train_winners[1] = next(c for c in palette if c != train_winners[0])
    test_winners = []
    for _ in range(ns):
        pool = list(set(train_winners)) if rng.random() < 0.6 else [c for c in palette if c not in train_winners]
        if not pool:
            pool = palette
        test_winners.append(int(rng.choice(pool)))
    modes = ["metric", "bounding"] + ["generic"] * (nt-2)
    rng.shuffle(modes)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = (0, 5) if params["style"] == 0 else tuple(int(x) for x in rng.choice(role_pool, size=2, replace=False))
    for section, winners in (("train", train_winners), ("test", test_winners)):
        for index, winner in enumerate(winners):
            mode = modes[index] if section == "train" else str(rng.choice(["generic","generic","generic","metric","bounding"]))
            for attempt in range(80):
                try:
                    if params["vary_roles"]:
                        bg, placeholder = (int(x) for x in rng.choice(role_pool, size=2, replace=False))
                    else:
                        bg, placeholder = first_roles
                    available = [c for c in palette if c != winner]
                    rng.shuffle(available)
                    required = set(test_winners) - set(train_winners) if section == "train" else set()
                    colors = [winner]
                    colors.extend(c for c in required if c != winner)
                    colors.extend(c for c in available if c not in colors)
                    required_count = 1 + len(required - {winner})
                    colors = colors[:int(rng.integers(max(2, required_count), min(5,len(colors))+1))]
                    if len(colors) < 2:
                        colors = [winner, available[0]]
                    if mode == "metric":
                        mask, markers = _metric_witness(rng, winner, colors, int(rng.integers(5, 11)))
                    elif mode == "bounding":
                        mask, markers = _bounding_witness(rng, winner, colors)
                    else:
                        shape_pool = ["bar", "rectangle", "diamond", "bevel", "l", "t", "step", "branch"]
                        if section == "test" and params["style"] != 0:
                            shape_pool += ["bevel", "l", "t", "step", "branch"]
                        crop = _shape(rng, str(rng.choice(shape_pool)))
                        mask = _place(crop, rng)
                        target = int(rng.choice([2,3,4,6,9,12,14,16]))
                        markers = _generic_markers(rng, mask, winner, colors, target, bool(rng.random() < .22))
                    inp, out = _render_pair(mask, markers, bg, placeholder)
                    if _grid_key(inp) in seen or _is_source_pair(inp, out):
                        continue
                    info = _pair_features(inp, out)
                    if mode == "metric" and not (info["manhattan_gap"] and info["chebyshev_gap"] and info["frequency_gap"]):
                        continue
                    if mode == "bounding" and not info["bbox_gap"]:
                        continue
                    seen.add(_grid_key(inp))
                    puzzle[section].append({"input": inp, "output": out})
                    break
                except (RuntimeError, ValueError):
                    continue
            else:
                raise RuntimeError(f"could not generate {section} pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((section, tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[section])) for section in ("train","test"))


def _predict_from_visible_grids(train_pairs, test_input):
    # Check the demonstrated same-canvas, one-object recoloring convention.
    for pair in train_pairs:
        _pair_features(pair["input"], pair["output"])
    background = int(np.bincount(test_input.ravel(), minlength=10).argmax())
    components = _orthogonal_components(test_input != background)
    objects = [component for component in components if len(component) > 1]
    _require(len(objects) == 1, "test has no unique silhouette")
    silhouette = objects[0]
    singles = [component[0] for component in components if len(component) == 1]
    _require(len(singles) >= 2, "test has too few markers")
    scored = [
        (min((r-orow)**2 + (c-ocol)**2 for orow, ocol in silhouette), int(test_input[r, c]))
        for r, c in singles
    ]
    best = min(distance for distance, _ in scored)
    colors = {color for distance, color in scored if distance == best}
    _require(len(colors) == 1, "test nearest color ambiguous")
    prediction = np.full(test_input.shape, background, dtype=np.int8)
    for row, column in silhouette:
        prediction[row, column] = next(iter(colors))
    return prediction


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    features = [_pair_features(pair["input"],pair["output"]) for p in puzzles for section in ("train","test") for pair in p[section]]
    _require(len({(f["crop"].shape, f["crop"].tobytes()) for f in features}) >= 45, "shape diversity")
    _require(len({pair["input"].shape for p in puzzles for section in ("train","test") for pair in p[section]}) >= 40, "dimension diversity")
    _require(any(f["edge"] for f in features), "no edge objects")
    _require(any(f["tie"] for f in features), "no same-color ties")
    _require(any(f["marker_count"] >= 11 for f in features) and any(f["marker_count"] == 2 for f in features), "density range")
    _require(any(f["bbox_gap"] for f in features) and any(f["manhattan_gap"] for f in features), "witness gaps")
    _require({2, 3, 4, 5} <= {len(f["marker_colors"]) for f in features}, "marker palette range")
    _require(sum(f["edge"] for f in features) >= 20, "too few edge objects")
    _require(sum(f["tie"] for f in features) >= 5, "too few same-color ties")
    _require(sum(f["manhattan_gap"] for f in features) >= 20 and sum(f["chebyshev_gap"] for f in features) >= 20, "metric witness diversity")
    _require(len({f["nearest_distance_sq"] for f in features if f["manhattan_gap"] and f["chebyshev_gap"]}) >= 4, "metric witness distances fixed")
    _require(sum(f["bbox_gap"] for f in features) >= 20, "bounding-box witness diversity")
    _require(sum(not np.all(f["crop"]) for f in features) >= 40, "irregular silhouettes absent")
    _require(any(not np.all(f["crop"]) and f["edge"] and f["marker_count"] >= 11 for f in features), "combined variations absent")
    _require(min(int(f["mask"].sum()) for f in features) <= 4 and max(int(f["mask"].sum()) for f in features) >= 40, "silhouette size range")
    varying_backgrounds = 0
    varying_placeholders = 0
    familiar_tests = 0
    new_color_tests = 0
    metric_tests = 0
    bounding_tests = 0
    broad_train_palettes = 0
    rare_winners = 0
    common_winners = 0
    for puzzle in puzzles:
        infos = {section: [_pair_features(pair["input"], pair["output"]) for pair in puzzle[section]] for section in ("train", "test")}
        every = infos["train"] + infos["test"]
        varying_backgrounds += len({f["bg"] for f in every}) > 1
        varying_placeholders += len({f["placeholder"] for f in every}) > 1
        _require(len({f["marker_count"] for f in every}) >= 2, "marker density fixed within puzzle")
        _require(len({tuple(sorted(f["marker_colors"])) for f in every}) >= 2, "candidate palette fixed within puzzle")
        broad_train_palettes += len(set().union(*(f["marker_colors"] for f in infos["train"]))) >= 7
        train_outputs = {f["winner"] for f in infos["train"]}
        for pair, f in zip(puzzle["test"], infos["test"]):
            familiar_tests += f["winner"] in train_outputs
            new_color_tests += f["winner"] not in train_outputs
            metric_tests += f["manhattan_gap"] or f["chebyshev_gap"]
            bounding_tests += f["bbox_gap"]
            prediction = _predict_from_visible_grids(
                [{"input": item["input"].copy(), "output": item["output"].copy()} for item in puzzle["train"]],
                pair["input"].copy(),
            )
            _require(np.array_equal(prediction, pair["output"]), "visible-grid prediction failed")
        for section in ("train", "test"):
            for pair, f in zip(puzzle[section], infos[section]):
                marker_values = pair["input"][(pair["input"] != f["bg"]) & ~f["mask"]]
                selected_count = int(np.count_nonzero(marker_values == f["winner"]))
                largest_rival_count = max(int(np.count_nonzero(marker_values == color)) for color in f["marker_colors"] if color != f["winner"])
                rare_winners += selected_count == 1 and largest_rival_count >= 2
                common_winners += selected_count > largest_rival_count
    _require(varying_backgrounds >= 5 and varying_placeholders >= 5, "local role variation absent")
    _require(familiar_tests >= 10 and new_color_tests >= 10, "test color schedule too narrow")
    _require(metric_tests >= 10 and bounding_tests >= 10, "test has too few decisive comparisons")
    _require(broad_train_palettes >= 10, "training marker hues too narrow")
    _require(rare_winners >= 20 and common_winners >= 20, "winner frequency too predictable")
    for ntrain, ntest in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(2200 + (ntrain or 0)*10 + (ntest or 0), ntrain, ntest)
        validate_generated_puzzle(p)
        if ntrain is not None:
            _require(len(p["train"]) == ntrain, "train count ignored")
        if ntest is not None:
            _require(len(p["test"]) == ntest, "test count ignored")
    _require(_puzzle_key(generate_puzzle(85,7,3)) == _puzzle_key(generate_puzzle(85,7,3)), "nondeterminism")
    for args in ({"num_train":1},{"num_train":11},{"num_test":0},{"num_test":5},{"num_train":2,"num_test":2}):
        try:
            generate_puzzle(1, **args)
        except ValueError:
            continue
        raise AssertionError("invalid count accepted")
