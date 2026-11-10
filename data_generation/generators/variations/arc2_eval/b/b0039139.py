import numpy as np


_SOURCE_PAIR_TEXTS = {
    (
        "0000/0440/0400/0000/1111/0000/0303/0303/0000/1111/"
        "2222/2222/2222/2222/1111/9999/9999/9999",
        "22/29/99/22/29",
    ),
    (
        "00000100000001888881333/"
        "04440103030301888881333/"
        "00400103030301888881333/"
        "04440100000001888881333/"
        "00000100000001888881333",
        "88838883888/38333833383/88838883888",
    ),
    (
        "000001000000016666661444444/"
        "044001030300016666661444444/"
        "040001030300016666661444444/"
        "044401000000016666661444444/"
        "004001033033016666661444444/"
        "000001000000016666661444444",
        "664466446644664/"
        "644464446444644/"
        "666466646664666/"
        "464446444644464",
    ),
    (
        "00000100000012222218888/"
        "04440103030012222218888/"
        "04040103030012222218888/"
        "00000100000012222218888",
        "2228222/2828282",
    ),
    (
        "0000000/"
        "0440440/"
        "0400040/"
        "0444440/"
        "0000000/"
        "1111111/"
        "0000000/"
        "0303030/"
        "0303030/"
        "0000000/"
        "0330330/"
        "0000000/"
        "1111111/"
        "8888888/"
        "8888888/"
        "8888888/"
        "8888888/"
        "8888888/"
        "8888888/"
        "1111111/"
        "7777777/"
        "7777777/"
        "7777777/"
        "7777777/"
        "7777777/"
        "7777777",
        "88788/"
        "87778/"
        "88888/"
        "77777/"
        "88788/"
        "87778/"
        "88888/"
        "77777/"
        "88788/"
        "87778/"
        "88888/"
        "77777/"
        "88788/"
        "87778/"
        "88888/"
        "77777/"
        "88788/"
        "87778/"
        "88888",
    ),
    (
        "00000010000000155555555122222/"
        "04404010330330155555555122222/"
        "00404010000000155555555122222/"
        "00444010330330155555555122222/"
        "00404010000000155555555122222/"
        "04404010330330155555555122222/"
        "00000010000000155555555122222",
        "55252552525525255252552525525/"
        "25252252522525225252252522525/"
        "25552255522555225552255522555/"
        "25252252522525225252252522525/"
        "55252552525525255252552525525",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        grid.astype(np.int8, copy=False).tobytes(),
    )


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _validate_requested_count(
    value: int | None,
    name: str,
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
        raise ValueError(f"{name} is outside its legal range")

    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_requested_count(
        num_train,
        "num_train",
        2,
        10,
    )
    test_count = _validate_requested_count(
        num_test,
        "num_test",
        1,
        4,
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = (
            minimum_train
            + (seed // 4) % number_of_choices
        )

    elif train_count is None:
        minimum_train = max(2, int(test_count) + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + seed % number_of_choices

    else:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test

    if train_count <= test_count:
        raise ValueError("no legal completion of the pair counts")

    return int(train_count), int(test_count)




def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _crop(mask):
    points = np.argwhere(mask)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _grow(h, w, area, rng):
    """An arbitrary connected cluster, with no prescribed corner endpoints."""
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    frontier = {start}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                frontier.add((x, y))
    return mask


def _make_stencil(h, w, rng, witness=False):
    """Connected masks: random pruning, branches, holes and winding paths."""
    if min(h, w) == 1:
        return np.ones((h, w), dtype=bool)
    mode = int(rng.choice(5, p=[.42, .25, .13, .14, .06]))
    mask = np.zeros((h, w), dtype=bool)
    if mode == 0:
        mask[:] = True
        target = max(h + w - 1, int(h * w * rng.uniform(.22, .88)))
        for flat in rng.permutation(h * w):
            if int(mask.sum()) <= target:
                break
            r, c = divmod(int(flat), w)
            mask[r, c] = False
            if _crop(mask).shape != (h, w) or len(_components(mask)) != 1:
                mask[r, c] = True
    elif mode == 1:
        pivot = (int(rng.integers(h)), int(rng.integers(w)))
        endpoints = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                     (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        for r, c in endpoints:
            mask[r, c] = True
            while (r, c) != pivot:
                choices = []
                if r != pivot[0]:
                    choices.append((r + (1 if r < pivot[0] else -1), c))
                if c != pivot[1]:
                    choices.append((r, c + (1 if c < pivot[1] else -1)))
                r, c = choices[int(rng.integers(len(choices)))]
                mask[r, c] = True
        # Independent protrusions increase branch and density variation.
        for _ in range(int(h * w * rng.uniform(0, .3))):
            near = np.zeros_like(mask)
            near[1:] |= mask[:-1]
            near[:-1] |= mask[1:]
            near[:, 1:] |= mask[:, :-1]
            near[:, :-1] |= mask[:, 1:]
            choices = np.argwhere(near & ~mask)
            if not len(choices):
                break
            r, c = choices[int(rng.integers(len(choices)))]
            mask[r, c] = True
    elif mode == 2:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        if h > 3 and w > 3:
            for _ in range(int(rng.integers(0, min(h, w)))):
                r = int(rng.integers(1, h-1))
                mask[r, :int(rng.integers(1, w-1))] = True
    elif mode == 3:
        # A serpentine path with variable bends, followed by tight cropping.
        mask[::2] = True
        side = int(rng.integers(2))
        for r in range(1, h, 2):
            mask[r, (w-1 if side else 0)] = True
            side = 1-side
        if rng.random() < .5:
            mask = mask[:, ::-1].copy()
    else:
        mask[:] = True
    if witness and mask.all():
        mask[0, 0] = False
    if witness and any(np.array_equal(mask, _transform(mask, k)) for k in range(1, 8)):
        # Preserve orientation visibly even in a two-demonstration puzzle.
        # This asymmetric branch is a bounded fallback for symmetric draws.
        mask[:] = False
        mask[0, :] = mask[:, 0] = True
        mask[-1, 1] = True
    return mask


def _transform(mask, index):
    return np.rot90(mask[:, ::-1] if index >= 4 else mask, index % 4)


def _partition(length, pieces, rng):
    """Positive segments with independently varied blank gaps."""
    extras = length - (2 * pieces - 1)
    sizes = np.ones(pieces, dtype=int)
    gaps = np.ones(max(0, pieces-1), dtype=int)
    for _ in range(extras):
        k = int(rng.integers(2 * pieces - 1))
        if k < pieces:
            sizes[k] += 1
        else:
            gaps[k-pieces] += 1
    starts, offset = [], 0
    for i, size in enumerate(sizes):
        starts.append((offset, offset + int(size)))
        offset += int(size) + (int(gaps[i]) if i < len(gaps) else 0)
    return starts


def _make_counter(h, w, count, rng, area=None):
    # Choose an arrangement by geometric capacity, not a fixed seed lattice.
    options = []
    for nr in range(1, min(count, (h+1)//2)+1):
        for nc in range(1, min(count, (w+1)//2)+1):
            if count <= nr*nc <= count + max(2, count//2):
                capacity = (h-nr+1)*(w-nc+1)
                if area is None or capacity * count // (nr*nc) >= area:
                    options.append((nr, nc))
    _require(bool(options), "counter does not fit")
    # Bounded retries choose genuinely uneven partitions; a balanced layout
    # below guarantees the requested ink area for the two training witnesses.
    for attempt in range(20):
        nr, nc = options[int(rng.integers(len(options)))]
        if attempt < 19:
            rows, cols = _partition(h, nr, rng), _partition(w, nc, rng)
        else:
            nr, nc = min(options, key=lambda x: x[0]*x[1])
            def balanced(length, n):
                available = length - n + 1
                sizes = [available//n + (i < available % n) for i in range(n)]
                starts, offset = [], 0
                for size in sizes:
                    starts.append((offset, offset+size))
                    offset += size+1
                return starts
            rows, cols = balanced(h, nr), balanced(w, nc)
        boxes = [(r0, r1, c0, c1) for r0, r1 in rows for c0, c1 in cols]
        rng.shuffle(boxes)
        if area is not None:
            boxes.sort(key=lambda b: (b[1]-b[0])*(b[3]-b[2]), reverse=True)
        boxes = boxes[:count]
        capacities = [(b[1]-b[0])*(b[3]-b[2]) for b in boxes]
        if area is None or sum(capacities) >= area:
            break
    _require(area is None or sum(capacities) >= area, "ink budget cannot fit")
    areas = [1] * count
    if area is not None:
        for _ in range(area-count):
            choices = [i for i in range(count) if areas[i] < capacities[i]]
            i = choices[int(rng.integers(len(choices)))]
            areas[i] += 1
    singletons = area is None and rng.random() < .13
    result = np.zeros((h, w), dtype=bool)
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        bh, bw = r1-r0, c1-c0
        if area is not None or singletons:
            obj = _grow(bh, bw, areas[i], rng)
        else:
            oh, ow = int(rng.integers(1, bh+1)), int(rng.integers(1, bw+1))
            mode = int(rng.integers(5))
            if mode == 0:
                oh = 1  # bars may run along either coordinate
            elif mode == 1:
                ow = 1
            if mode in (0, 1, 2):
                obj = np.ones((oh, ow), dtype=bool)
            elif mode == 3:
                obj = _make_stencil(oh, ow, rng)
            else:
                obj = np.ones((oh, ow), dtype=bool)
                if oh >= 3 and ow >= 3:
                    obj[1:-1, 1:-1] = False
            r0 += int(rng.integers(bh-oh+1))
            c0 += int(rng.integers(bw-ow+1))
        result[r0:r0+obj.shape[0], c0:c0+obj.shape[1]] = obj
    if result.all():
        result[0, 0] = False
    _require(len(_components(result)) == count == len(_components(result, True)),
             "counter objects touch")
    return result


def _decode(grid):
    """Parse from input alone, including its background and orientation."""
    candidates = []
    for vertical in (False, True):
        g = grid.T if vertical else grid
        for divider in range(10):
            cuts = np.flatnonzero(np.all(g == divider, axis=0)).tolist()
            if len(cuts) != 3 or np.count_nonzero(g == divider) != 3*g.shape[0]:
                continue
            a, b, c = cuts
            if not (0 < a < b-1 < c-2 < g.shape[1]-3):
                continue
            panels = (g[:, :a], g[:, a+1:b], g[:, b+1:c], g[:, c+1:])
            s, q, p, z = panels
            if not (np.all(p == p[0, 0]) and np.all(z == z[0, 0]) and p[0, 0] != z[0, 0]):
                continue
            sc, qc = set(map(int, np.unique(s))), set(map(int, np.unique(q)))
            common = sc & qc
            if len(sc) != 2 or len(qc) != 2 or len(common) != 1:
                continue
            bg = next(iter(common))
            ink, counter_ink = next(iter(sc-{bg})), next(iter(qc-{bg}))
            if len({bg, divider, ink, counter_ink}) != 4:
                continue
            sm, cm = s == ink, q == counter_ink
            tile = _crop(sm)
            coords = np.argwhere(sm)
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            candidates.append({
                "vertical": vertical, "palette": (bg, divider, ink, counter_ink),
                "tile": tile, "counter": cm, "count": len(_components(cm)),
                "primary": int(p[0, 0]), "secondary": int(z[0, 0]),
                "lengths": tuple(x.shape[1] for x in panels),
                "margins": (int(lo[0]), int(s.shape[0]-hi[0]-1),
                            int(lo[1]), int(s.shape[1]-hi[1]-1)),
                "panel_shape": s.shape,
            })
    _require(len(candidates) == 1, "input has no unique four-panel parse")
    return candidates[0]


def _predict_from_input(grid):
    # Independent of the concatenation used to construct the output.
    d = _decode(grid)
    tile, n = d["tile"], d["count"]
    h, a = tile.shape
    out = np.full((h, n*(a+1)-1), d["secondary"], dtype=np.int8)
    for r in range(h):
        for c in range(out.shape[1]):
            if c % (a+1) < a and tile[r, c % (a+1)]:
                out[r, c] = d["primary"]
    return out.T.copy() if d["vertical"] else out


def _generate_pair(rng, regime, vertical, count, cross, axial, palette,
                   swatches, witness=False, tile=None, counter_area=None,
                   anti_shortcut=False):
    _require(regime in ("source_train", "target_test"), "unknown regime")
    if tile is None:
        tile = _make_stencil(cross, axial, rng, witness)
    cross, axial = tile.shape
    bg, divider, ink, counter_ink = palette
    # Counter capacity and all panels are allocated jointly from the input budget.
    min_cross = max(cross, 6 if witness else 3)
    if not witness:
        min_cross = max(min_cross, min(30, 2*count-1) if axial > 12 else 3)
    pad_mode = int(rng.choice(3, p=[.25, .45, .30]))
    extra = 0 if pad_mode == 0 else int(rng.integers(1, (4 if pad_mode == 1 else 10)))
    canvas_h = min(30, max(min_cross, cross+extra))
    min_counter = 6 if witness else max(1, 2*((count + (canvas_h+1)//2-1)//((canvas_h+1)//2))-1)
    max_padding = min(7, 30 - 3 - axial - min_counter - 2)
    along_padding = 0 if pad_mode == 0 else int(rng.integers(max_padding+1))
    if witness:
        along_padding = max(1, along_padding)
    if tile.all() and canvas_h == cross and along_padding == 0:
        along_padding = 1
    stencil_w = axial + along_padding
    left = int(rng.integers(along_padding+1))
    top = int(rng.integers(canvas_h-cross+1))
    stencil = np.full((canvas_h, stencil_w), bg, dtype=np.int8)
    stencil[top:top+cross, left:left+axial][tile] = ink
    remaining = 30 - 3 - stencil_w
    counter_w = int(rng.integers(min_counter, min(max(14, min_counter), remaining-2)+1))
    swatch_budget = remaining-counter_w
    # Some strips are small; others spend the full 30-cell budget.
    total = int(rng.integers(2, swatch_budget+1))
    primary_w = int(rng.integers(1, total))
    secondary_w = total-primary_w
    if rng.random() < .20 and 2*count <= swatch_budget:
        primary_w = secondary_w = count
    elif rng.random() < .25 and count < swatch_budget:
        primary_w, secondary_w = count, int(rng.integers(1, swatch_budget-count+1))
    if anti_shortcut:
        options = [(a, b) for a in range(1, swatch_budget)
                   for b in range(1, swatch_budget-a+1) if a != count and b != count]
        primary_w, secondary_w = options[int(rng.integers(len(options)))]
    counter = _make_counter(canvas_h, counter_w, count, rng, counter_area)
    primary, secondary = swatches
    parts = [stencil, np.where(counter, counter_ink, bg).astype(np.int8),
             np.full((canvas_h, primary_w), primary, dtype=np.int8),
             np.full((canvas_h, secondary_w), secondary, dtype=np.int8)]
    line = np.full((canvas_h, 1), divider, dtype=np.int8)
    inp = np.concatenate([parts[0], line, parts[1], line, parts[2], line, parts[3]], axis=1)
    colored = np.where(tile, primary, secondary).astype(np.int8)
    gap = np.full((cross, 1), secondary, dtype=np.int8)
    pieces = [x for i in range(count) for x in ((colored,) if i == count-1 else (colored, gap))]
    out = np.concatenate(pieces, axis=1)
    return {"input": inp.T.copy() if vertical else inp,
            "output": out.T.copy() if vertical else out}


def _dimensions(rng, count, cross_max, axial_max, witness=False):
    low = 3 if witness else 1
    axial = int(rng.integers(low, min(axial_max, 31//count-1, 18)+1))
    cross = int(rng.integers(low, cross_max+1))
    return cross, axial


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    # The shared semantic convention is positional, not a hidden color map.
    # Sample a reusable palette vocabulary, complexity envelope, and orientation
    # preference once; geometry and visible colors remain pair-level properties.
    palettes = [tuple(map(int, rng.choice(10, 4, replace=False))) for _ in range(int(rng.integers(2, 5)))]
    second = []
    for role in range(4):
        second.append(int(rng.choice([c for c in range(10) if c != palettes[0][role] and c not in second])))
    palettes[1] = tuple(second)
    swatch_pool = [tuple(map(int, rng.choice([c for c in range(10) if c != p[1]], 2, replace=False)))
                  for p in palettes]
    simple = rng.random() < .25
    max_count = int(rng.integers(4, 6 if simple else 10))
    max_cross = int(rng.integers(5, 9 if simple else 19))
    max_axial = int(rng.integers(4, 7 if simple else 14))
    orientation_mode = int(rng.choice(3, p=[.22, .22, .56]))
    base_vertical = bool(rng.integers(2))
    witness_counts = list(map(int, rng.choice([2, 3, 4], 2, replace=False)))
    witness_area = int(rng.integers(12, 25))
    puzzle, seen = {"train": [], "test": []}, set()
    train_meta = []
    for split, amount, regime in (("train", nt, "source_train"), ("test", nq, "target_test")):
        for index in range(amount):
            witness = split == "train" and index < 2
            tile = None
            if witness:
                n = witness_counts[index]
                cross, axial = _dimensions(rng, n, min(max_cross, 6), 5, True)
            elif split == "train":
                n = witness_counts[0] if index == 2 else int(rng.integers(1, max_count+1))
                cross, axial = _dimensions(rng, n, max_cross, max_axial)
            else:
                # Test modes are sampled independently of orientation and index.
                mode = int(rng.integers(5))
                largest_count = max(d["count"] for d in train_meta)
                largest_cross = max(d["tile"].shape[0] for d in train_meta)
                reference = train_meta[int(rng.integers(len(train_meta)))]
                if mode in (2, 4):
                    n = int(rng.integers(largest_count+1, min(15, largest_count+6)+1))
                elif mode == 0:
                    n = int(rng.integers(min(d["count"] for d in train_meta), largest_count+1))
                else:
                    n = reference["count"]
                cross, axial = _dimensions(rng, n, max_cross, max_axial)
                if mode in (3, 4):
                    cross = int(rng.integers(largest_cross+1, min(26, largest_cross+10)+1))
                elif mode == 1:
                    cross, axial = reference["tile"].shape
                    if rng.random() < (.55 if simple else .25):
                        tile = reference["tile"].copy()
                elif mode == 2:
                    eligible = [d for d in train_meta if n*(d["tile"].shape[1]+1)-1 <= 30]
                    if eligible:
                        reference = eligible[int(rng.integers(len(eligible)))]
                        cross, axial = reference["tile"].shape
            vertical = (bool(rng.integers(2)) if orientation_mode == 2 else
                        (not base_vertical if orientation_mode == 1 and split == "test" else base_vertical))
            palette_index = index if witness else int(rng.integers(len(palettes)))
            palette = palettes[palette_index]
            swatches = swatch_pool[palette_index]
            if rng.random() > (.75 if simple else .35):
                swatches = tuple(map(int, rng.choice([c for c in range(10) if c != palette[1]], 2, replace=False)))
            pair = _generate_pair(rng, regime, vertical, n, cross, axial, palette, swatches,
                                  witness, tile, witness_area if witness else None,
                                  witness and index == 1)
            # A deterministic palette fallback guarantees uniqueness without a
            # rejection loop, even when familiar shapes and palettes are reused.
            source_key = (_grid_text(pair["input"]), _grid_text(pair["output"]))
            if _grid_key(pair["input"]) in seen or source_key in _SOURCE_PAIR_TEXTS:
                original = pair
                found = False
                for variant in range(100):
                    shift, swap = divmod(variant, 10)
                    mapping = (np.arange(10)+shift) % 10
                    a, b = int(mapping[0]), int(mapping[swap])
                    mapping[0], mapping[swap] = b, a
                    candidate = {k: mapping[v].astype(np.int8) for k, v in original.items()}
                    key = (_grid_text(candidate["input"]), _grid_text(candidate["output"]))
                    if _grid_key(candidate["input"]) not in seen and key not in _SOURCE_PAIR_TEXTS:
                        pair, found = candidate, True
                        break
                _require(found, "duplicate fallback exhausted")
            puzzle[split].append(pair)
            seen.add(_grid_key(pair["input"]))
            if split == "train":
                train_meta.append(_decode(pair["input"]))
    # Witness order itself carries no information.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D ndarray")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid exceeds bounds")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal color")
            inp, out = pair["input"], pair["output"]
            _require(_grid_key(inp) not in seen, "duplicate input")
            seen.add(_grid_key(inp))
            _require((_grid_text(inp), _grid_text(out)) not in _SOURCE_PAIR_TEXTS, "source copy")
            _require(not np.array_equal(inp, out), "identity pair")
            d = _decode(inp)
            _require(len(_components(d["tile"])) == 1, "stencil fragments not supported")
            _require(d["count"] == len(_components(d["counter"], True)), "ambiguous counter connectivity")
            _require(np.array_equal(_predict_from_input(inp), out), "incorrect transformation")
            if split == "train":
                train.append(d)
    _require(len({d["count"] for d in train}) >= 2, "training needs different counts")
    _require(len({d["palette"] for d in train}) >= 2, "training needs role-based colors")
    for role in range(4):
        _require(len({d["palette"][role] for d in train}) >= 2, "structural color role is fixed")
    _require(any(not d["tile"].all() and d["count"] > 1 and sum(d["margins"]) > 0 for d in train),
             "missing crop, recoloring and separator witness")
    _require(any(int(a["counter"].sum()) == int(b["counter"].sum()) and a["count"] != b["count"]
                 for a in train for b in train), "component count is confounded with ink area")
    _require(any(int(d["counter"].sum()) > d["count"] for d in train), "pixel-counting ambiguity")
    for panel in (2, 3):
        _require(any(d["lengths"][panel] != d["count"] for d in train), "swatch-size ambiguity")
    for transform in range(1, 8):
        _require(any(not np.array_equal(d["tile"], _transform(d["tile"], transform)) for d in train),
                 "training does not establish stencil orientation")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _has_hole(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return len(_components(~padded)) > 1


def _topology(mask):
    vertical = np.zeros(mask.shape, dtype=int)
    horizontal = np.zeros(mask.shape, dtype=int)
    vertical[1:] += mask[:-1]
    vertical[:-1] += mask[1:]
    horizontal[:, 1:] += mask[:, :-1]
    horizontal[:, :-1] += mask[:, 1:]
    degree = vertical + horizontal
    start = tuple(map(int, np.argwhere(mask)[0]))
    distance, frontier = {start: 0}, [start]
    for r, c in frontier:
        for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            x, y = point
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y] and point not in distance:
                distance[point] = distance[(r, c)]+1
                frontier.append(point)
    return {"branch": bool(np.any(mask & (degree >= 3))),
            "bend": bool(np.any(mask & (degree == 2) & (vertical == 1))),
            "notch": bool(np.any(~mask & (degree >= 2))),
            "detour": any(d > abs(r-start[0])+abs(c-start[1])+2 for (r, c), d in distance.items())}


def review_metrics(puzzles):
    """Measure actual returned grids, never generation-time parameters."""
    data, per_puzzle, regimes = [], [], {k: 0 for k in ("interpolation", "familiar_count", "count_only", "size_only", "combined")}
    orientation_modes = {"same": 0, "mixed": 0, "horizontal_to_vertical": 0, "vertical_to_horizontal": 0}
    for puzzle in puzzles:
        training = [_decode(p["input"]) for p in puzzle["train"]]
        tests = [_decode(p["input"]) for p in puzzle["test"]]
        lo, hi = min(d["count"] for d in training), max(d["count"] for d in training)
        cross_max = max(d["tile"].shape[0] for d in training)
        for d in tests:
            nc, ns = d["count"] > hi, d["tile"].shape[0] > cross_max
            regimes["combined" if nc and ns else "count_only" if nc else "size_only" if ns else "interpolation"] += 1
            regimes["familiar_count"] += d["count"] in {t["count"] for t in training}
        ts, qs = {d["vertical"] for d in training}, {d["vertical"] for d in tests}
        mode = ("same" if len(ts | qs) == 1 else "horizontal_to_vertical" if ts == {False} and qs == {True}
                else "vertical_to_horizontal" if ts == {True} and qs == {False} else "mixed")
        orientation_modes[mode] += 1
        rows = []
        for split, ds in (("train", training), ("test", tests)):
            for d in ds:
                t, q = d["tile"], d["counter"]
                objects = _components(q)
                sizes = [len(c) for c in objects]
                object_masks = []
                for component in objects:
                    mask = np.zeros(q.shape, dtype=bool)
                    for point in component:
                        mask[point] = True
                    object_masks.append(_crop(mask))
                topologies = [_topology(mask) for mask in object_masks]
                stencil_topology = _topology(t)
                occupied = np.argwhere(q)
                low, high = occupied.min(axis=0), occupied.max(axis=0)
                corners = [bool(t[0, 0]), bool(t[0, -1]), bool(t[-1, 0]), bool(t[-1, -1])]
                row = {"split": split, "vertical": d["vertical"], "count": d["count"],
                       "tile_shape": tuple(t.shape), "tile_cells": int(t.sum()),
                       "density": float(t.mean()), "corners": sum(corners),
                       "opposite_corners": (corners[0] and corners[3]) or (corners[1] and corners[2]),
                       "hole": _has_hole(t), "solid": bool(t.all()), "single_cell": t.size == 1,
                       "counter_area": int(q.sum()), "component_sizes": sizes,
                       "counter_hole": _has_hole(q), "singletons": max(sizes) == 1,
                       "counter_bars": any(min(m.shape) == 1 and m.size >= 4 for m in object_masks),
                       "counter_blocks": any(min(m.shape) >= 2 and m.all() for m in object_masks),
                       "counter_bends": any(m["bend"] for m in topologies),
                       "counter_branches": any(m["branch"] for m in topologies),
                       "stencil_branch": stencil_topology["branch"],
                       "stencil_notch": stencil_topology["notch"],
                       "stencil_detour": stencil_topology["detour"],
                       "counter_margins": (int(low[0]), int(q.shape[0]-1-high[0]),
                                           int(low[1]), int(q.shape[1]-1-high[1])),
                       "unequal_objects": len(set(sizes)) > 1,
                       "counter_shape": tuple(q.shape), "lengths": d["lengths"],
                       "palette": d["palette"], "swatches": (d["primary"], d["secondary"]),
                       "swatch_reuse": d["primary"] in d["palette"] or d["secondary"] in d["palette"],
                       "margins": d["margins"], "tight": sum(d["margins"]) == 0,
                       "edge": 0 in d["margins"], "diagonal_disagreement": len(_components(q, True)) != d["count"],
                       "structure": _grid_key(t)}
                rows.append(row)
        per_puzzle.append({
            "counts": len({d["count"] for d in rows}),
            "palettes": len({d["palette"] for d in rows}),
            "layouts": len({d["lengths"] for d in rows}),
            "same_count_different_area": any(a["count"] == b["count"] and a["counter_area"] != b["counter_area"] for a in rows for b in rows),
            "same_area_different_count": any(a["count"] != b["count"] and a["counter_area"] == b["counter_area"] for a in rows for b in rows),
            "repeated_swatches": len({d["swatches"] for d in rows}) < len(rows),
            "familiar_shape": any(_grid_key(a["tile"]) == _grid_key(b["tile"]) for a in training for b in tests),
        })
        data.extend(rows)
    return {"pairs": data, "puzzles": per_puzzle, "regimes": regimes, "orientations": orientation_modes}


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "insufficient puzzle diversity")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    for seed in (0, 7, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9328, a, b)
        _require(a is None or len(p["train"]) == a, "ignored train count")
        _require(b is None or len(p["test"]) == b, "ignored test count")
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("accepted illegal counts")
    metrics = review_metrics(puzzles)
    rows, groups = metrics["pairs"], metrics["puzzles"]
    _require(not any(r["diagonal_disagreement"] for r in rows), "connectivity ambiguity")
    _require(sum(not r["opposite_corners"] for r in rows) >= 30, "corner restriction")
    _require(sum(r["hole"] for r in rows) >= 10, "missing stencil holes")
    _require(len({r["structure"] for r in rows}) >= 150, "stencil structure diversity")
    _require(any(r["single_cell"] for r in rows) and any(r["solid"] for r in rows), "simple motifs missing")
    _require(max(max(r["component_sizes"]) for r in rows) >= 20, "counter components too small")
    _require(any(r["counter_hole"] for r in rows), "missing counter rings")
    for feature in ("counter_bars", "counter_blocks", "counter_bends", "counter_branches",
                    "stencil_branch", "stencil_notch", "stencil_detour"):
        _require(sum(r[feature] for r in rows) >= 10, "missing geometry: " + feature)
    _require(len({r["counter_margins"] for r in rows}) >= 30, "counter padding collapsed")
    _require(sum(r["unequal_objects"] for r in rows) >= 60, "equal-size counter restriction")
    _require(any(r["singletons"] for r in rows), "singleton counters missing")
    _require(all(g["palettes"] >= 2 and g["counts"] >= 2 for g in groups), "within-puzzle diversity")
    _require(all(g["same_area_different_count"] for g in groups), "area ambiguity")
    _require(sum(g["same_count_different_area"] for g in groups) >= 30, "area variation")
    _require(sum(g["repeated_swatches"] for g in groups) >= 10, "forced novel swatches")
    _require(sum(g["familiar_shape"] for g in groups) >= 1, "familiar shape missing")
    _require(any(r["swatch_reuse"] for r in rows), "role color reuse missing")
    swatches = [x for r in rows for x in r["lengths"][2:]]
    _require(max(swatches) >= 12 and min(swatches) == 1, "swatches too narrow")
    _require(sum(r["count"] in r["lengths"][2:] for r in rows) >= 30, "count coincidences excluded")
    _require(any(r["lengths"][2] == r["lengths"][3] for r in rows), "equal swatches missing")
    _require(sum(r["tight"] for r in rows) >= 5 and sum(r["edge"] for r in rows) >= 100, "margin coverage")
    _require(max(max(r["margins"]) for r in rows) >= 6, "large margins missing")
    _require(min(r["count"] for r in rows) == 1 and max(r["count"] for r in rows) >= 12, "count range")
    _require(max(r["tile_shape"][1] for r in rows) >= 10 and max(r["tile_shape"][0] for r in rows) >= 20, "tile range")
    _require(all(v > 0 for v in metrics["regimes"].values()), "test regime collapsed")
    _require(all(v > 0 for v in metrics["orientations"].values()), "orientation schedule collapsed")
    for n in range(1, 9):
        _require({r["vertical"] for r in rows if r["count"] == n} == {False, True}, "count/orientation coupling")
    # Corruption checks ensure that validation is not a generation-time tautology.
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupt["test"][0]["output"].flat[0] = (int(corrupt["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        return
    raise AssertionError("validator accepted a corrupted label")
