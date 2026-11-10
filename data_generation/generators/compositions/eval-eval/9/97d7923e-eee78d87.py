"""Periodic yield selection: unary ranks compete over samples of one lattice.

All roles and the lattice origin are visible in the input. Hollow, colored
rectangles are windows; a same-colored isolated run is their unary key. The
remaining color draws a full-bounding-box 3x3 motif. Each window samples the
infinite repetition registered to that motif's actual position. Rank windows
by sampled occupancy, then use the frame color for the chosen window's hits.
Other hits use the motif color, and all gaps retain the background.
"""
from functools import lru_cache

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.tobytes()


def _components(grid, color):
    pending = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in pending:
                    pending.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    a = np.asarray(cells)
    return (*map(int, a.min(axis=0)), *map(int, a.max(axis=0)))


def _separate(a, b):
    ar, ac, ay, ax = a
    br, bc, by, bx = b
    return ay + 1 < br or by + 1 < ar or ax + 1 < bc or bx + 1 < ac


def _decode(value):
    """Read only input pixels; never consult a seed or a generated profile."""
    g = _grid(value)
    bg = int(g[0, 0])
    _require(all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])),
             "the outer border must be background")
    families, source_colors, objects = {}, [], []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        frames, others = [], []
        for cells in _components(g, color):
            r, c, y, x = box = _bbox(cells)
            h, w = y - r + 1, x - c + 1
            if h >= 4 and w >= 4:
                patch = g[r:y + 1, c:x + 1]
                perimeter = np.ones((h, w), dtype=bool)
                perimeter[1:-1, 1:-1] = False
                _require(len(cells) == 2 * (h + w) - 4
                         and np.all(patch[perimeter] == color)
                         and np.all(patch[~perimeter] == bg), "malformed hollow frame")
                frames.append(box)
            else:
                others.append(cells)
        if frames:
            _require(len(others) == 1, "each frame family needs exactly one unary run")
            cells = others[0]
            r, c, y, x = key_box = _bbox(cells)
            rank = len(cells)
            _require((r == y or c == x) and rank == (y - r + 1) * (x - c + 1),
                     "unary key must be a contiguous straight run")
            _require(2 <= len(frames) <= 5 and 1 <= rank <= min(4, len(frames)),
                     "invalid population or requested ordinal")
            families[color] = {"rank": rank, "key": key_box, "frames": sorted(frames)}
            objects.extend(frames + [key_box])
        else:
            source_colors.append(color)
    _require(1 <= len(families) <= 3 and len(source_colors) == 1,
             "need frame families and one distinct motif color")
    accent = source_colors[0]
    r0, c0, y0, x0 = source_box = _bbox(np.argwhere(g == accent))
    _require(y0 - r0 == x0 - c0 == 2, "motif must have a minimal 3x3 box")
    mask = g[r0:y0 + 1, c0:x0 + 1] == accent
    _require(2 <= int(mask.sum()) <= 8
             and np.all(np.isin(g[r0:y0 + 1, c0:x0 + 1], [bg, accent])),
             "motif must contain accent and background only")
    objects.append(source_box)
    _require(all(_separate(a, b) for i, a in enumerate(objects) for b in objects[i + 1:]),
             "objects must have a background moat")
    info = {"grid": g, "background": bg, "accent": accent,
            "origin": (r0, c0), "mask": mask, "families": families}
    for family in families.values():
        yields = [int(_sample(info, box).sum()) for box in family["frames"]]
        _require(len(set(yields)) == len(yields), "yield ties are outside this family")
        _require(all(0 < count < (y - r - 1) * (x - c - 1)
                     for count, (r, c, y, x) in zip(yields, family["frames"])),
                 "every window must sample both hits and gaps")
        family["yields"] = yields
    return info


def _sample(info, box, projection="registered"):
    r, c, y, x = box
    rr, cc = np.arange(r + 1, y), np.arange(c + 1, x)
    r0, c0 = info["origin"]
    mask = info["mask"]
    if projection == "local":
        r0, c0 = r + 1, c + 1
    elif projection == "canvas":
        r0 = c0 = 0
    elif projection == "shift":
        r0, c0 = r0 - 1, c0 - 1
    elif projection == "flip_rows":
        mask = mask[::-1]
    elif projection == "flip_cols":
        mask = mask[:, ::-1]
    elif projection == "transpose":
        mask = mask.T
    elif projection == "fixed_plus":
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return mask[(rr[:, None] - r0) % 3, (cc[None, :] - c0) % 3]


def _answer(info, ranking="yield", projection="registered"):
    out = info["grid"].copy()
    for cap, family in info["families"].items():
        frames = family["frames"]
        samples = [_sample(info, box, projection) for box in frames]
        scores = [int(m.sum()) for m in samples]
        if ranking == "area":
            scores = [(y - r - 1) * (x - c - 1) for r, c, y, x in frames]
        elif ranking == "height":
            scores = [y - r - 1 for r, c, y, x in frames]
        elif ranking == "width":
            scores = [x - c - 1 for r, c, y, x in frames]
        elif ranking in ("left", "right", "top", "bottom"):
            axis = 1 if ranking in ("left", "right") else 0
            scores = [(-1 if ranking in ("left", "top") else 1) * b[axis] for b in frames]
        ordered = sorted(range(len(frames)), key=lambda k: (-scores[k], frames[k]))
        if ranking == "ascending":
            ordered = sorted(range(len(frames)), key=lambda k: (scores[k], frames[k]))
        chosen = ordered[0 if ranking == "always_longest" else family["rank"] - 1]
        for k, ((r, c, y, x), mask) in enumerate(zip(frames, samples)):
            patch = out[r + 1:y, c + 1:x]
            patch[mask] = cap if k == chosen or ranking == "all" else info["accent"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Project the visible motif, measure each family's yields, select its rank."""
    return _answer(_decode(input_grid))


def _stamped_reference(info):
    """Separate cell-stamping construction for validator/reference comparisons."""
    g, mask = info["grid"], info["mask"]
    r0, c0 = info["origin"]
    hits = np.zeros(g.shape, dtype=bool)
    for dr, dc in np.argwhere(mask):
        for r in range((r0 + int(dr)) % 3, g.shape[0], 3):
            for c in range((c0 + int(dc)) % 3, g.shape[1], 3):
                hits[r, c] = True
    out = g.copy()
    for cap, family in info["families"].items():
        scored = []
        for box in family["frames"]:
            r, c, y, x = box
            scored.append((int(np.count_nonzero(hits[r + 1:y, c + 1:x])), box))
        selected = sorted(scored, reverse=True)[family["rank"] - 1][1]
        for _, box in scored:
            r, c, y, x = box
            for row in range(r + 1, y):
                for col in range(c + 1, x):
                    if hits[row, col]:
                        out[row, col] = cap if box == selected else info["accent"]
    return out


_RANK_SHORTCUTS = ("area", "height", "width", "ascending", "left", "right",
                   "top", "bottom", "always_longest", "all")
_PROJECTION_SHORTCUTS = ("local", "canvas", "shift", "flip_rows", "flip_cols",
                         "transpose", "fixed_plus")


def _surviving_shortcuts(infos):
    correct = [_answer(i) for i in infos]
    surviving = []
    for ranking in _RANK_SHORTCUTS:
        if all(np.array_equal(_answer(i, ranking=ranking), out) for i, out in zip(infos, correct)):
            surviving.append(ranking)
    for projection in _PROJECTION_SHORTCUTS:
        if all(np.array_equal(_answer(i, projection=projection), out) for i, out in zip(infos, correct)):
            surviving.append(projection)
    return surviving


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(out, _stamped_reference(info)), "incorrect projected-yield selection")
            _require(np.any(inp != out), "every example must change")
            writable = np.zeros(inp.shape, dtype=bool)
            for cap, family in info["families"].items():
                order = np.argsort(family["yields"])[::-1]
                chosen = int(order[family["rank"] - 1])
                for index, box in enumerate(family["frames"]):
                    r, c, y, x = box
                    hit = _sample(info, box)
                    writable[r + 1:y, c + 1:x] = hit
                    patch = out[r + 1:y, c + 1:x]
                    _require(np.all(patch[~hit] == info["background"]), "carrier gaps overwritten")
                    _require(np.all(patch[hit] == (cap if index == chosen else info["accent"])),
                             "wrong highlight role")
                _require(int(np.count_nonzero(out == cap) - np.count_nonzero(inp == cap))
                         == family["yields"][chosen], "cap gain must equal selected yield")
            _require(np.array_equal(inp[~writable], out[~writable]), "source/key/frame/background changed")
            infos[split].append(info)
    train = infos["train"]
    ranks = {f["rank"] for i in train for f in i["families"].values()}
    _require({1, 2} <= ranks, "training must show ordinal progression")
    _require(any(f["rank"] == 2 and len(f["frames"]) >= 3 for i in train for f in i["families"].values()),
             "training must show an intermediate selection")
    histories = {}
    for i in train:
        for cap, family in i["families"].items():
            histories.setdefault(cap, set()).add(family["rank"])
    _require(any({1, 2} <= r for r in histories.values()), "need a repeated-cap rank contrast")
    _require(len({i["mask"].tobytes() for i in train}) >= 2, "training needs different motifs")
    _require(not _surviving_shortcuts(train), "a concrete simpler alternative survives training")
    for info in infos["test"]:
        for family in info["families"].values():
            rank = family["rank"]
            _require(rank in ranks or set(range(1, rank)) <= ranks, "unseen ordinal lacks progression")


def _full_box(mask):
    return 2 <= int(mask.sum()) <= 8 and all((mask[0].any(), mask[-1].any(),
                                             mask[:, 0].any(), mask[:, -1].any()))


_MASKS = tuple(m for code in range(1, 511)
              if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                                         dtype=bool).reshape(3, 3)))


@lru_cache(maxsize=1024)
def _profiles(mask_bytes, mode):
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(3, 3)
    sizes = [(h, w) for h in range(2, 7) for w in range(2, 7)]
    if mode == "small":
        sizes = [(h, w) for h in range(2, 6) for w in range(2, 6)]
    elif mode == "tall":
        sizes = [(h, w) for h in range(6, 9) for w in range(2, 5)]
    elif mode == "wide":
        sizes = [(h, w) for h in range(2, 5) for w in range(6, 9)]
    profiles = []
    for h, w in sizes:
        for pr in range(3):
            for pc in range(3):
                count = int(mask[(np.arange(h)[:, None] + pr) % 3,
                                 (np.arange(w)[None, :] + pc) % 3].sum())
                if 0 < count < h * w:
                    profiles.append((h, w, pr, pc, count))
    return tuple(profiles)


def _choose_profiles(rng, mask, mode, population, rank, contrast):
    choices = _profiles(mask.tobytes(), mode)
    for _ in range(200):
        picks, used = [], set()
        for _ in range(population):
            eligible = [p for p in choices if p[4] not in used]
            if not eligible:
                break
            # First balance yields; otherwise the many high-area profiles dominate.
            value = int(rng.choice(sorted({p[4] for p in eligible})))
            eligible = [p for p in eligible if p[4] == value]
            picked = eligible[int(rng.integers(len(eligible)))]
            picks.append(picked)
            used.add(value)
        if len(picks) != population:
            continue
        if contrast:
            chosen = sorted(range(population), key=lambda k: -picks[k][4])[rank - 1]
            area_order = sorted(range(population), key=lambda k: (-picks[k][0] * picks[k][1], k))
            # Require a strict inversion involving the selected window, not an area tie.
            selected = picks[chosen]
            inversion = any((p[4] - selected[4]) * (p[0] * p[1] - selected[0] * selected[1]) < 0
                            for p in picks)
            if chosen == area_order[rank - 1] or not inversion:
                continue
        return picks
    raise RuntimeError("could not find distinct-yield profiles with teaching contrast")


def _pack(rng, mask, profiles, ranks):
    """Irregular free placement, constrained by each window's lattice phase."""
    for _ in range(60):
        occupied = np.zeros((30, 30), dtype=bool)
        boxes = {}
        source = (int(rng.integers(1, 27)), int(rng.integers(1, 27)))
        r0, c0 = source
        occupied[r0 - 1:r0 + 4, c0 - 1:c0 + 4] = True
        key_h, key_w = max(ranks), 2 * len(ranks) - 1
        requests = [("key", key_h, key_w, None)]
        for family, picks in enumerate(profiles):
            for index, (h, w, pr, pc, _) in enumerate(picks):
                requests.append(((family, index), h + 2, w + 2, ((r0 + pr - 1) % 3, (c0 + pc - 1) % 3)))
        requests.sort(key=lambda item: -(item[1] * item[2]))
        failed = False
        for name, h, w, phase in requests:
            places = []
            for r in range(1, 30 - h):
                for c in range(1, 30 - w):
                    if phase is not None and (r % 3, c % 3) != phase:
                        continue
                    if not occupied[r:r + h, c:c + w].any():
                        places.append((r, c))
            if not places:
                failed = True
                break
            r, c = places[int(rng.integers(len(places)))]
            occupied[r - 1:r + h + 1, c - 1:c + w + 1] = True
            boxes[name] = (r, c, r + h - 1, c + w - 1)
        if not failed:
            return source, boxes
    raise RuntimeError("free placement exhausted")


def _make_scene(rng, colors, mask, populations, ranks, mode, anchor):
    for _ in range(30):
        try:
            profiles = [_choose_profiles(rng, mask, mode, n, rank, anchor and f == 0)
                        for f, (n, rank) in enumerate(zip(populations, ranks))]
            source, boxes = _pack(rng, mask, profiles, ranks)
        except RuntimeError:
            continue
        break
    else:
        raise RuntimeError("bounded scene construction exhausted")
    bg, accent, *caps = colors
    inp = np.full((30, 30), bg, dtype=np.int8)
    r0, c0 = source
    inp[r0:r0 + 3, c0:c0 + 3][mask] = accent
    kr, kc, _, _ = boxes["key"]
    for family, rank in enumerate(ranks):
        inp[kr:kr + rank, kc + 2 * family] = caps[family]
    for family, picks in enumerate(profiles):
        for index in range(len(picks)):
            r, c, y, x = boxes[(family, index)]
            inp[r:y + 1, c] = inp[r:y + 1, x] = caps[family]
            inp[r, c:x + 1] = inp[y, c:x + 1] = caps[family]
    # Generate labels by explicit planned footprints, separately from solve's decoder.
    out = inp.copy()
    for family, picks in enumerate(profiles):
        selected = sorted(range(len(picks)), key=lambda k: -picks[k][4])[ranks[family] - 1]
        for index, (h, w, pr, pc, _) in enumerate(picks):
            r, c, y, x = boxes[(family, index)]
            for dr in range(h):
                for dc in range(w):
                    if mask[(dr + pr) % 3, (dc + pc) % 3]:
                        out[r + 1 + dr, c + 1 + dc] = caps[family] if index == selected else accent
    used_boxes = list(boxes.values()) + [(r0, c0, r0 + 2, c0 + 2)]
    lo_r = min(b[0] for b in used_boxes)
    lo_c = min(b[1] for b in used_boxes)
    hi_r = max(b[2] for b in used_boxes)
    hi_c = max(b[3] for b in used_boxes)
    inp, out = inp[lo_r:hi_r + 1, lo_c:hi_c + 1], out[lo_r:hi_r + 1, lo_c:hi_c + 1]
    vertical = int(rng.integers(2, min(6, 30 - inp.shape[0]) + 1))
    horizontal = int(rng.integers(2, min(6, 30 - inp.shape[1]) + 1))
    top, left = int(rng.integers(1, vertical)), int(rng.integers(1, horizontal))
    padding = ((top, vertical - top), (left, horizontal - left))
    pair = {"input": np.pad(inp, padding, constant_values=bg),
            "output": np.pad(out, padding, constant_values=bg)}
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    return {role: np.ascontiguousarray(np.rot90(np.fliplr(g) if reflect else g, rotation))
            for role, g in pair.items()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    shift = str(rng.choice(["ordinal", "population", "aspect", "density"]))
    dense_train = bool(rng.integers(2))
    mixed_background = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_masks = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            anchor = split == "train" and index < 2
            for attempt in range(100):
                allowed = list(_MASKS)
                if shift == "density":
                    density_range = ((6, 7) if dense_train else (3, 4)) if split == "train" else ((2, 4) if dense_train else (6, 8))
                    allowed = [m for m in allowed if density_range[0] <= int(m.sum()) <= density_range[1]]
                if anchor:
                    allowed = [m for m in allowed if m.tobytes() not in first_masks
                               and not np.array_equal(m, m[::-1])
                               and not np.array_equal(m, m[:, ::-1])
                               and not np.array_equal(m, m.T)]
                density = int(rng.choice(sorted({int(m.sum()) for m in allowed})))
                allowed = [m for m in allowed if int(m.sum()) == density]
                mask = allowed[int(rng.integers(len(allowed)))]
                if anchor:
                    ranks = [1, 2] if index == 0 else [2, 1]
                    populations = [3, int(rng.integers(2, 4))]
                else:
                    f = int(rng.integers(1, 4))
                    rank_limit = 3 if split == "test" and shift == "ordinal" else 2
                    ranks = list(map(int, rng.integers(1, rank_limit + 1, size=f)))
                    if split == "test" and shift == "ordinal":
                        ranks[0] = 3
                    low, high = (4, 5) if split == "test" and shift == "population" else (2, 4)
                    if shift == "population" and split == "train":
                        high = 3
                    populations = [int(rng.integers(max(low, rank), high + 1)) for rank in ranks]
                    while sum(populations) > 8:
                        removable = [k for k, n in enumerate(populations) if n > max(low, ranks[k])]
                        if removable:
                            populations[int(rng.choice(removable))] -= 1
                        else:
                            populations.pop()
                            ranks.pop()
                mode = "small" if shift == "aspect" else "normal"
                if split == "test" and shift == "aspect":
                    mode = str(rng.choice(["tall", "wide"]))
                bg = int(rng.choice([c for c in range(10) if c not in palette[2:5]])) if mixed_background else palette[0]
                caps = palette[2:2 + len(ranks)] if anchor or rng.random() < 0.65 else list(map(int, rng.choice([c for c in range(10) if c != bg], len(ranks), replace=False)))
                accent = int(rng.choice([c for c in range(10) if c != bg and c not in caps]))
                colors = [bg, accent] + caps
                try:
                    pair = _make_scene(rng, colors, mask, populations, ranks, mode, anchor)
                except RuntimeError:
                    continue
                if _key(pair["input"]) in seen:
                    continue
                info = _decode(pair["input"])
                if anchor and index == 1:
                    first_info = _decode(puzzle["train"][0]["input"])
                    if np.array_equal(first_info["mask"], info["mask"]) or _surviving_shortcuts([first_info, info]):
                        continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if anchor:
                    first_masks.add(mask.tobytes())
                break
            else:
                raise RuntimeError("bounded example construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded 42-seed variation audit plus metamorphic and rejection checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, masks, profiles, colors, populations, ranks, families, signatures = (set() for _ in range(8))
    smaller_winners = phases = carrier_centers = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver disagrees")
                shapes.add(pair["input"].shape)
                masks.add(info["mask"].tobytes())
                colors.add((info["background"], info["accent"]))
                carrier_centers += not bool(info["mask"][1, 1])
                families.add(len(info["families"]))
                for f in info["families"].values():
                    populations.add(len(f["frames"]))
                    ranks.add(f["rank"])
                    selected = int(np.argsort(f["yields"])[::-1][f["rank"] - 1])
                    areas = [(y - r - 1) * (x - c - 1) for r, c, y, x in f["frames"]]
                    smaller_winners += areas[selected] < max(areas)
                    for box, count in zip(f["frames"], f["yields"]):
                        r, c, y, x = box
                        pr, pc = (r + 1 - info["origin"][0]) % 3, (c + 1 - info["origin"][1]) % 3
                        profiles.add((y - r - 1, x - c - 1, pr, pc, count))
                        phases += (pr, pc) != (0, 0)
    _require(len(signatures) == 42, "cross-seed uniqueness")
    _require(len(shapes) >= 30 and len(masks) >= 60 and len(profiles) >= 150, "structural variation")
    _require(len(colors) >= 30 and families == {1, 2, 3}, "palette/family variation")
    _require({2, 3, 4, 5} <= populations and {1, 2, 3} <= ranks, "population/ordinal variation")
    _require(smaller_winners >= 42 and phases >= 100 and carrier_centers >= 20, "interaction coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "determinism")
    pair = p["test"][0]
    translated_input = np.pad(pair["input"], ((1, 2), (2, 1)), constant_values=int(pair["input"][0, 0]))
    translated_output = np.pad(pair["output"], ((1, 2), (2, 1)), constant_values=int(pair["input"][0, 0]))
    _require(np.array_equal(solve(translated_input), translated_output), "translation equivariance")
    for rotation in range(4):
        for reflected in (False, True):
            inp = np.rot90(np.fliplr(pair["input"]) if reflected else pair["input"], rotation)
            out = np.rot90(np.fliplr(pair["output"]) if reflected else pair["output"], rotation)
            permutation = np.array([8, 3, 6, 1, 9, 0, 4, 7, 2, 5], dtype=np.int8)
            _require(np.array_equal(solve(permutation[inp]), permutation[out]), "dihedral/color equivariance")
    # Admissible members beyond the current sampler's ordinal/size ranges.
    mask = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    rank_four = _make_scene(np.random.default_rng(314), [0, 8, 2], mask, [5], [4], "normal", False)
    _require(np.array_equal(solve(rank_four["input"]), rank_four["output"]), "fourth of five")
    large = np.zeros((30, 30), dtype=np.int8)
    large[1:4, 15:18][mask] = 8
    large[1:3, 20] = 2
    for r, c, y, x in ((1, 1, 28, 4), (6, 7, 27, 11)):
        large[r:y + 1, c] = large[r:y + 1, x] = 2
        large[r, c:x + 1] = large[y, c:x + 1] = 2
    large_info = _decode(large)
    _require(np.array_equal(solve(large), _stamped_reference(large_info)), "large interiors")
    tied = np.zeros((15, 15), dtype=np.int8)
    tied[1:4, 1:4][mask] = 8
    tied[1, 7] = 2
    for r, c, y, x in ((7, 1, 10, 4), (7, 7, 10, 10)):
        tied[r:y + 1, c] = tied[r:y + 1, x] = 2
        tied[r, c:x + 1] = tied[y, c:x + 1] = 2
    try:
        solve(tied)
    except AssertionError as error:
        _require("yield ties" in str(error), "tie fixture failed for an unrelated reason")
    else:
        raise AssertionError("tied yields accepted")
    for damage in ("hit", "gap", "frame"):
        bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        f = next(iter(info["families"].values()))
        r, c, y, x = box = f["frames"][0]
        if damage == "frame":
            row, col = r, c
        else:
            hit = _sample(info, box)
            dr, dc = np.argwhere(hit if damage == "hit" else ~hit)[0]
            row, col = r + 1 + int(dr), c + 1 + int(dc)
        pair["output"][row, col] = (int(pair["output"][row, col]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, legal counts, pixel invariants, metamorphic checks)")
