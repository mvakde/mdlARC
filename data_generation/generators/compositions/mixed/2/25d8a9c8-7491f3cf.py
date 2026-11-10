"""Complete-row verdicts spliced by a marker-selected digital cut.

No parent imports, files, latent conventions, or seed-dependent solver state.
Generation and self-tests retain one small puzzle at a time and write no files.
"""

from collections import Counter
import hashlib

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _array(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(grid.dtype.kind in "iu" and np.all((0 <= grid) & (grid <= 9)),
             "grid must contain integer colors 0..9")
    return grid


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    moves = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if diagonal:
        moves += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            p, q = stack.pop()
            cells.append((p, q))
            for dr, dc in moves:
                nr, nc = p + dr, q + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _cell_mask(shape, cells):
    mask = np.zeros(shape, dtype=bool)
    for p in cells:
        mask[p] = True
    return mask


def _geometry(cut):
    """Validate a simple digital path and recover its two open regions."""
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    _require(len(cells) >= 3, "cut is too short")
    graph = {}
    for r, c in sorted(cells):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if not (dr or dc) or q not in cells:
                    continue
                # Do not create diagonal shortcuts around orthogonal elbows.
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p, ns in graph.items() if len(ns) == 1]
    _require(len(ends) == 2 and all(1 <= len(ns) <= 2 for ns in graph.values()),
             "cut has a branch, loop, or isolated component")
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) for p in ends), "cut endpoints must meet the boundary")
    _require(all(not border(p) for p in cells if p not in ends),
             "only cut endpoints may touch the boundary")
    path, visited, previous, current = [], set(), None, min(ends)
    while current is not None and current not in visited:
        path.append(current)
        visited.add(current)
        forward = [q for q in graph[current] if q != previous]
        previous, current = current, forward[0] if forward else None
    _require(len(path) == len(cells), "cut contains disconnected material")
    regions = [_cell_mask(cut.shape, group) for group in _components(~cut)]
    _require(len(regions) == 2, "cut must induce exactly two regions")
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
    sites = [[tuple(map(int, p)) for p in np.argwhere(region & ~near)]
             for region in regions]
    _require(all(sites), "both regions need an isolated marker site")
    return regions, sites, path


def _pack(panels, frame):
    h, w = panels[0].shape
    grid = np.full((h + 2, 4 * w + 5), frame, dtype=np.int8)
    for index, panel in enumerate(panels):
        start = 1 + index * (w + 1)
        grid[1:-1, start:start + w] = panel
    return grid


def _decode(value):
    grid = _array(value)
    height, width = grid.shape
    _require((width - 5) % 4 == 0, "four congruent panels required")
    h, w = height - 2, (width - 5) // 4
    _require(3 <= h <= 28 and 3 <= w <= 6, "invalid panel dimensions")
    frame = int(grid[0, 0])
    _require(np.all(grid[[0, -1], :] == frame) and np.all(grid[:, ::w + 1] == frame),
             "broken frame or dividers")
    panels = [grid[1:-1, 1 + i * (w + 1):1 + i * (w + 1) + w] for i in range(4)]
    bg = int(panels[3][0, 0])
    _require(np.all(panels[3] == bg), "destination must be blank")
    positive = set(map(int, np.unique(panels[0]))) - {bg}
    _require(len(positive) == 1, "control must have one non-background color")
    pos = positive.pop()
    _require(len({frame, bg, pos}) == 3, "frame, background and control must differ")
    groups = sorted(_components(panels[0] != bg, diagonal=True), key=len)
    _require(len(groups) == 2 and len(groups[0]) == 1 and len(groups[1]) >= 3,
             "control needs one singleton marker and one cut")
    marker = groups[0][0]
    cut = _cell_mask((h, w), groups[1])
    regions, sites, path = _geometry(cut)
    side = 0 if regions[0][marker] else 1
    _require(marker in sites[side], "marker is not isolated from the cut")
    return {"grid": grid, "panels": panels, "a": panels[1], "b": panels[2],
            "h": h, "w": w, "frame": frame, "bg": bg, "pos": pos,
            "cut": cut, "marker": marker, "regions": regions, "sites": sites,
            "side": side, "marked": regions[side], "other": regions[1 - side],
            "path": path}


def _truth(panel):
    return np.all(panel == panel[:, :1], axis=1)


def _expanded(truth, width):
    return np.repeat(truth[:, None], width, axis=1)


def _splice(a, b, cut, marked):
    """Boolean compositor for construction and named counterfactuals."""
    result = np.where(marked, a, b)
    result[cut] = (a | b)[cut]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct all roles from this input and apply the complete-row rule."""
    d = _decode(input_grid)
    out = d["grid"].copy()
    # Scalar equality and cellwise assembly are separate from construction's
    # vectorized masks. Nothing here reads generation parameters or outputs.
    uniform_a = [len(set(map(int, row))) == 1 for row in d["a"]]
    uniform_b = [len(set(map(int, row))) == 1 for row in d["b"]]
    start = 1 + 3 * (d["w"] + 1)
    for r in range(d["h"]):
        for c in range(d["w"]):
            if d["cut"][r, c]:
                # First positive has precedence, with second positive fallback.
                positive = uniform_a[r] or uniform_b[r]
            elif d["marked"][r, c]:
                positive = uniform_a[r]
            else:
                positive = uniform_b[r]
            out[1 + r, start + c] = d["pos"] if positive else d["bg"]
    return out


def _line(p, q):
    r, c = p
    rr, cc = q
    dr, dc = abs(rr - r), abs(cc - c)
    sr, sc = (1 if r < rr else -1), (1 if c < cc else -1)
    error, points = dc - dr, []
    while True:
        points.append((r, c))
        if (r, c) == (rr, cc):
            return points
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _cut_family(d):
    path = d["path"]
    rows, cols = zip(*path)
    if len(set(rows)) == 1:
        return "horizontal"
    if len(set(cols)) == 1:
        return "vertical"
    if path == _line(path[0], path[-1]) or path[::-1] == _line(path[-1], path[0]):
        return "diagonal"
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return "elbow" if sum(a != b for a, b in zip(steps, steps[1:])) == 1 else "stepped"


def _sample_cut(rng, h, w, family):
    for attempt in range(48):
        cut = np.zeros((h, w), dtype=bool)
        if family == "horizontal":
            cut[int(rng.integers(2, h - 2)), :] = True
        elif family == "vertical":
            cut[:, int(rng.integers(2, w - 2))] = True
        elif family == "diagonal":
            if h == w and rng.random() < 0.35:
                endpoints = (0, 0), (h - 1, w - 1)
            else:
                c0, c1 = map(int, rng.choice(np.arange(1, w - 1), 2, replace=False))
                endpoints = (0, c0), (h - 1, c1)
            for p in _line(*endpoints):
                cut[p] = True
        elif family == "elbow":
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(1, w - 2))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        elif family == "stepped":
            across = bool(rng.random() < 0.3 and w >= 6)
            ch, cw = (w, h) if across else (h, w)
            cut = np.zeros((ch, cw), dtype=bool)
            choices = list(range(2, ch - 2, 2))
            count = int(rng.integers(1, min(3, len(choices)) + 1))
            turns = sorted(map(int, rng.choice(choices, count, replace=False)))
            col, previous = int(rng.integers(1, cw - 1)), 0
            for row in turns:
                cut[previous:row + 1, col] = True
                options = [x for x in range(1, cw - 1) if x != col]
                next_col = int(rng.choice(options))
                cut[row, min(col, next_col):max(col, next_col) + 1] = True
                previous, col = row, next_col
            cut[previous:, col] = True
            if across:
                cut = cut.T.copy()
        else:
            raise ValueError("unknown cut family")
        if rng.random() < 0.5:
            cut = cut[::-1, :].copy()
        if rng.random() < 0.5:
            cut = cut[:, ::-1].copy()
        try:
            _geometry(cut)
        except AssertionError:
            continue
        return cut
    raise RuntimeError(f"could not sample readable {family} cut")


def _shape(rng, family, richness):
    maximum = (10, 16, 28)[richness]
    if family == "horizontal":
        return int(rng.integers(5, maximum + 1)), int(rng.integers(3, 7))
    if family == "vertical":
        return int(rng.integers(3, maximum + 1)), int(rng.integers(5, 7))
    if family == "elbow":
        return int(rng.integers(6, maximum + 1)), int(rng.integers(4, 7))
    minimum = 7 if family == "stepped" else 5
    w = int(rng.integers(5, 7))
    return int(rng.integers(max(minimum, w), maximum + 1)), w


def _random_truth(rng, height, mixed=False):
    for _ in range(32):
        style = int(rng.integers(5))
        if style == 0:
            result = rng.random(height) < float(rng.choice((0.2, 0.4, 0.6, 0.8)))
        elif style == 1:
            start = int(rng.integers(height))
            length = int(rng.integers(1, height + 1))
            result = (np.arange(height) - start) % height < length
        elif style == 2:
            period = int(rng.integers(2, 5))
            result = (np.arange(height) + int(rng.integers(period))) % period == 0
        elif style == 3:
            result = np.zeros(height, dtype=bool)
            result[int(rng.integers(height))] = True
            if rng.random() < 0.5:
                result = ~result
        else:
            result = np.full(height, bool(rng.integers(2)))
        if not mixed or (np.any(result) and np.any(~result)):
            return result
    result = np.zeros(height, dtype=bool)
    result[int(rng.integers(height))] = True
    return result


def _mixed_row(rng, palette, width, style=None, colors=None):
    a, b = map(int, rng.choice(palette, 2, replace=False) if colors is None else colors)
    if style is None:
        style = str(rng.choice(("exception", "alternating", "runs", "balanced", "random")))
    if style in ("first", "middle", "last", "exception"):
        row = np.full(width, a, dtype=np.int8)
        position = (0 if style == "first" else width - 1 if style == "last"
                    else int(rng.integers(1, width - 1)))
        row[position] = b
    elif style == "alternating":
        row = np.where(np.arange(width) % 2 == 0, a, b).astype(np.int8)
    elif style == "runs":
        row = np.where(np.arange(width) < int(rng.integers(1, width)), a, b).astype(np.int8)
    elif style == "balanced":
        row = np.array([a] * (width // 2) + [b] * (width - width // 2), dtype=np.int8)
        rng.shuffle(row)
    else:
        row = np.asarray(rng.choice(palette, width), dtype=np.int8)
        if np.all(row == row[0]):
            row[-1] = b if row[0] == a else a
    return row


def _source(rng, truth, width, bg, pos, teaching=False):
    count = int(rng.integers(2, 11))
    if teaching:
        extras = [x for x in range(10) if x not in (bg, pos)]
        palette = [bg, pos, *map(int, rng.choice(extras, count - 2, replace=False))]
    else:
        palette = list(map(int, rng.choice(10, count, replace=False)))
    negatives = list(map(int, np.flatnonzero(~truth)))
    positives = list(map(int, np.flatnonzero(truth)))
    forced = {}
    if teaching:
        _require(len(negatives) >= 3 and len(positives) >= 2, "teaching rows missing")
        rng.shuffle(negatives)
        forced = dict(zip(negatives[:3], ("first", "middle", "last")))
    panel = np.empty((len(truth), width), dtype=np.int8)
    prior_positive, prior_negative = [], []
    for r in range(len(truth)):
        if truth[r]:
            if teaching and r == positives[0]:
                color = bg
            elif teaching and r == positives[1]:
                color = pos
            else:
                color = int(rng.choice(prior_positive if prior_positive and rng.random() < 0.35
                                       else palette))
            panel[r] = color
            prior_positive.append(color)
        elif r in forced:
            panel[r] = _mixed_row(rng, palette, width, forced[r], (bg, pos))
            prior_negative.append(panel[r].copy())
        elif prior_negative and rng.random() < 0.12:
            panel[r] = prior_negative[int(rng.integers(len(prior_negative)))]
        else:
            panel[r] = _mixed_row(rng, palette, width)
            prior_negative.append(panel[r].copy())
    _require(np.array_equal(_truth(panel), truth), "source constructor changed row truth")
    return panel


def _roles(rng, base, policy):
    if policy == "fixed" or (policy == "mixed" and rng.random() < 0.3):
        return base
    if policy == "partial":
        roles = list(base)
        index = int(rng.integers(3))
        roles[index] = int(rng.choice([x for x in range(10) if x not in roles]))
        return tuple(roles)
    return tuple(map(int, rng.choice(10, 3, replace=False)))


def _make_pair(cut, marker, a, b, roles, truths=None):
    frame, bg, pos = roles
    control = np.full(cut.shape, bg, dtype=np.int8)
    control[cut] = pos
    control[marker] = pos
    regions, _, _ = _geometry(cut)
    marked = regions[0] if regions[0][marker] else regions[1]
    if truths is None:
        truths = _truth(a), _truth(b)
    am, bm = (_expanded(t, cut.shape[1]) for t in truths)
    result = np.where(_splice(am, bm, cut, marked), pos, bg).astype(np.int8)
    inp = _pack([control, a, b, np.full(cut.shape, bg, dtype=np.int8)], frame)
    out = inp.copy()
    out[1:-1, 1 + 3 * (cut.shape[1] + 1):-1] = result
    return {"input": inp, "output": out}


def _mask_pair(d):
    return tuple(map(bool, _truth(d["a"]))), tuple(map(bool, _truth(d["b"])))


def _destination(pair):
    w = (np.asarray(pair["input"]).shape[1] - 5) // 4
    return np.asarray(pair["output"])[1:-1, 1 + 3 * (w + 1):-1]


def _alternative_predictions(d):
    """Named shortcuts, not an independent solver or candidate-rule search."""
    a, b, cut, marked = d["a"], d["b"], d["cut"], d["marked"]
    am, bm = (_expanded(_truth(x), d["w"]) for x in (a, b))
    good = _splice(am, bm, cut, marked)
    predictions = {"source1_only": am, "source2_only": bm,
                   "union_everywhere": am | bm, "intersection_everywhere": am & bm,
                   "marker_blind": _splice(am, bm, cut, d["regions"][0]),
                   "reverse_sides": _splice(am, bm, cut, d["other"])}
    for name, predicate in (
            ("endpoints", lambda row: row[0] == row[-1]),
            ("majority", lambda row: max(Counter(map(int, row)).values()) > len(row) // 2)):
        masks = [_expanded(np.array([predicate(row) for row in source]), d["w"])
                 for source in (a, b)]
        predictions[name] = _splice(*masks, cut, marked)
    column_masks = [np.repeat(np.all(x == x[:1, :], axis=0)[None, :], d["h"], axis=0)
                    for x in (a, b)]
    predictions["columns"] = _splice(*column_masks, cut, marked)
    cropped = good.copy()
    for source, region in ((a, marked), (b, d["other"])):
        for r in range(d["h"]):
            fragment = source[r, region[r]]
            if fragment.size:
                cropped[r, region[r]] = np.all(fragment == fragment[0])
    predictions["crop_before_classification"] = cropped
    for name, seam in (("cut_source1_only", am), ("cut_source2_only", bm),
                       ("cut_always_positive", np.ones(cut.shape, dtype=bool)),
                       ("cut_always_blank", np.zeros(cut.shape, dtype=bool))):
        predictions[name] = good.copy()
        predictions[name][cut] = seam[cut]
    # Raw B-style copying removes A's contribution while retaining B's rule.
    raw = np.where(marked, a, b).copy()
    raw[cut] = np.where(a[cut] != d["bg"], a[cut], b[cut])
    return predictions, raw


def _exception_style(row):
    values, counts = np.unique(row, return_counts=True)
    if len(values) != 2 or sorted(map(int, counts)) != [1, len(row) - 1]:
        return None
    exceptional = values[int(np.argmin(counts))]
    index = int(np.flatnonzero(row == exceptional)[0])
    return "first" if index == 0 else "last" if index == len(row) - 1 else "middle"


def _teaching_missing(pairs):
    defeated, styles, sides, seam_states = set(), set(), set(), set()
    uniform_bg = False
    for pair in pairs:
        d = _decode(pair["input"])
        result = _destination(pair)
        binary = result == d["pos"]
        am, bm = (_expanded(_truth(x), d["w"]) for x in (d["a"], d["b"]))
        predictions, raw = _alternative_predictions(d)
        for name, prediction in predictions.items():
            if not np.array_equal(prediction, binary):
                defeated.add(name)
        if not np.array_equal(raw, result):
            defeated.add("raw_copy")
        if np.any((am != bm) & ~d["cut"]):
            sides.add(d["side"])
        seam_states.update(zip(map(bool, am[d["cut"]]), map(bool, bm[d["cut"]])))
        for source, region in ((d["a"], d["marked"]), (d["b"], d["other"])):
            for r, row in enumerate(source):
                if not np.any(region[r]):
                    continue
                if np.all(row == d["bg"]):
                    uniform_bg |= bool(np.all(result[r, region[r]] == d["pos"]))
                style = _exception_style(row)
                if style and np.all(result[r, region[r]] == d["bg"]):
                    styles.add(style)
    required = {"source1_only", "source2_only", "union_everywhere", "intersection_everywhere",
                "marker_blind", "reverse_sides", "endpoints", "majority", "columns",
                "crop_before_classification", "cut_source1_only", "cut_source2_only",
                "cut_always_positive", "cut_always_blank", "raw_copy"}
    missing = sorted(required - defeated)
    if styles != {"first", "middle", "last"}:
        missing.append("selected first/interior/last exceptions")
    if sides != {0, 1}:
        missing.append("effective marker selections on both sides")
    if seam_states != {(False, False), (False, True), (True, False), (True, True)}:
        missing.append("four seam verdict combinations")
    if not uniform_bg:
        missing.append("selected uniform-background positive")
    return missing


_FAMILIES = ("horizontal", "vertical", "diagonal", "elbow", "stepped")
_REGIMES = ("straight_to_elbow", "simple_to_steps", "orthogonal_to_diagonal", "extreme_source")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(_REGIMES))
    richness = int(rng.integers(3))
    policy = str(rng.choice(("fixed", "partial", "mixed")))
    base = tuple(map(int, rng.choice(10, 3, replace=False)))
    initial_family = ("vertical" if regime == "orthogonal_to_diagonal"
                      else str(rng.choice(("vertical", "diagonal"))))
    # A bounded complementary pair proves marker causality even at counts 2/1.
    for _ in range(32):
        h, w = int(rng.integers(6, 11)), int(rng.integers(5, 7))
        cut = _sample_cut(rng, h, w, initial_family)
        _, sites, _ = _geometry(cut)
        core = np.array(((1, 0), (0, 1), (0, 0), (1, 1), (0, 1), (1, 0)), dtype=bool)
        extra = rng.random((h - 6, 2)) < 0.5
        truths = np.concatenate((core, extra), axis=0)
        rng.shuffle(truths, axis=0)
        a, b = (_source(rng, truths[:, i], w, base[1], base[2], teaching=True) for i in range(2))
        side = int(rng.integers(2))
        markers = [sites[s][int(rng.integers(len(sites[s])))] for s in (side, 1 - side)]
        twins = [_make_pair(cut, marker, a, b, base, (truths[:, 0], truths[:, 1]))
                 for marker in markers]
        if not _teaching_missing(twins):
            break
    else:
        raise RuntimeError("could not construct teaching contrasts")
    puzzle = {"train": twins, "test": []}
    seen = {_key(pair["input"]) for pair in twins}
    train_masks = {_mask_pair(_decode(pair["input"])) for pair in twins}
    train_families = [initial_family]
    used_test_answers = set()
    for split, count in (("train", nt - 2), ("test", nq)):
        for index in range(count):
            if split == "train":
                allowed = (_FAMILIES[:3] if regime == "straight_to_elbow" else
                           _FAMILIES[:4] if regime == "simple_to_steps" else
                           _FAMILIES[:2] if regime == "orthogonal_to_diagonal" else _FAMILIES)
                family = str(rng.choice(allowed))
            else:
                family = ("elbow" if regime == "straight_to_elbow" else
                          "stepped" if regime == "simple_to_steps" else
                          "diagonal" if regime == "orthogonal_to_diagonal" else
                          str(rng.choice(train_families)))
            for _ in range(96):
                h, w = _shape(rng, family, richness)
                cut = _sample_cut(rng, h, w, family)
                _, sites, _ = _geometry(cut)
                side = int(rng.integers(2))
                marker = sites[side][int(rng.integers(len(sites[side])))]
                roles = _roles(rng, base, policy)
                truths = [_random_truth(rng, h, mixed=regime == "extreme_source") for _ in range(2)]
                if split == "test" and regime == "extreme_source":
                    truths[int(rng.integers(2))] = np.full(h, bool(rng.integers(2)))
                a, b = (_source(rng, truths[i], w, roles[1], roles[2]) for i in range(2))
                pair = _make_pair(cut, marker, a, b, roles, truths)
                key = _key(pair["input"])
                masks = tuple(tuple(map(bool, truth)) for truth in truths)
                result = _destination(pair) == roles[2]
                answer_key = _key(result)
                if key in seen or not np.any(result):
                    continue
                if split == "test" and (masks in train_masks or answer_key in used_test_answers):
                    continue
                break
            else:
                raise RuntimeError("bounded fresh-example construction exhausted")
            puzzle[split].append(pair)
            seen.add(key)
            if split == "train":
                train_masks.add(masks)
                train_families.append(family)
            else:
                used_test_answers.add(answer_key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _shift_valid(train, test):
    tf = {_cut_family(d) for d in train}
    qf = {_cut_family(d) for d in test}
    straight = {"horizontal", "vertical", "diagonal"}
    if (tf <= straight and qf == {"elbow"}) or (
            tf <= straight | {"elbow"} and qf == {"stepped"}) or (
            tf <= {"horizontal", "vertical"} and qf == {"diagonal"}):
        return True
    def mixed(d):
        return all(np.any(_truth(x)) and np.any(~_truth(x)) for x in (d["a"], d["b"]))
    def one_extreme(d):
        return sum(bool(np.all(t) or not np.any(t)) for t in (_truth(d["a"]), _truth(d["b"]))) == 1
    return qf <= tf and all(mixed(d) for d in train) and all(one_extreme(d) for d in test)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, test_answers = set(), set()
    decoded = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = (_array(pair[name]) for name in ("input", "output"))
            _require(inp.shape == out.shape, "input/output shape mismatch")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            decoded[split].append(d)
            _require(np.array_equal(solve(inp), out), "wrong result or changed preserved cells")
            result = _destination(pair)
            _require(set(map(int, result.flat)) <= {d["bg"], d["pos"]}, "non-binary destination")
            _require(np.any(result == d["pos"]), "generated result must expose a positive")
            if split == "test":
                answer_key = _key(result == d["pos"])
                _require(answer_key not in test_answers, "duplicate normalized test answer")
                test_answers.add(answer_key)
    missing = _teaching_missing(puzzle["train"])
    _require(not missing, f"insufficient training evidence: {missing}")
    training_masks = {_mask_pair(d) for d in decoded["train"]}
    _require(all(_mask_pair(d) not in training_masks for d in decoded["test"]),
             "test reuses a complete training row-verdict pair")
    _require(_shift_valid(decoded["train"], decoded["test"]), "missing documented test shift")


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _expect_rejection(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError("malformed case was accepted")


def self_test():
    """Bounded streaming checks over 42 seeds; returns compact aggregate facts."""
    shapes, panel_shapes, palettes, role_tuples, cuts, masks, answers = (set() for _ in range(7))
    fingerprints, family_sides = set(), set()
    family_counts = Counter()
    pairs = extreme_sources = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                d = _decode(pair["input"])
                digest.update(pair["input"].tobytes())
                shapes.add(pair["input"].shape)
                panel_shapes.add((d["h"], d["w"]))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                role_tuples.add((d["frame"], d["bg"], d["pos"]))
                cuts.add((d["cut"].shape, d["cut"].tobytes()))
                masks.add(_mask_pair(d))
                answers.add(_key(_destination(pair) == d["pos"]))
                family = _cut_family(d)
                family_counts[split + "/" + family] += 1
                family_sides.add((family, d["side"]))
                extreme_sources += sum(bool(np.all(t) or not np.any(t))
                                       for t in (_truth(d["a"]), _truth(d["b"])))
        fingerprints.add(digest.hexdigest())
        first, twin = puzzle["train"][:2]
        d = _decode(first["input"])
        result, other = _destination(first), _destination(twin)
        _require(not np.array_equal(result, other), "marker reversal has no effect")
        _require(np.array_equal(result[d["cut"]], other[d["cut"]]), "marker changed seam")
        am, bm = (_expanded(_truth(x), d["w"]) for x in (d["a"], d["b"]))
        _require(np.array_equal(other[d["marked"]] == d["pos"], bm[d["marked"]]),
                 "marker reversal failed to select second mask")
        _require(np.array_equal(other[d["other"]] == d["pos"], am[d["other"]]),
                 "marker reversal failed to select first mask")
        # Permuting all ten colors must commute with solving, including P=0.
        permutation = np.random.default_rng(seed + 10000).permutation(10)
        _require(np.array_equal(solve(permutation[first["input"]]), permutation[first["output"]]),
                 "color-role permutation changed the rule")
        # Source-only recoloring and row shuffling preserve equality classes.
        sources = [(x + 1) % 10 for x in (d["a"], d["b"])]
        local_rng = np.random.default_rng(seed + 20000)
        for source in sources:
            for row in source:
                local_rng.shuffle(row)
        changed = _pack([d["panels"][0], *sources, d["panels"][3]], d["frame"])
        _require(np.array_equal(_destination({"input": changed, "output": solve(changed)}), result),
                 "equality-preserving source change affected destination")
        for axis in (0, 1):
            inp = _pack([np.flip(p, axis) for p in d["panels"]], d["frame"])
            expected = _pack([np.flip(p, axis) for p in (*d["panels"][:3], result)], d["frame"])
            _require(np.array_equal(solve(inp), expected), "aligned reflection failed")
        if seed in (0, 41):
            _require(_same_puzzle(puzzle, generate_puzzle(seed)), "seed replay failed")

    _require(len(fingerprints) == 42, "seed diversity collapsed")
    _require(len(shapes) >= 25 and len(cuts) >= 100, "geometry diversity collapsed")
    _require(len(palettes) >= 25 and len(role_tuples) >= 35, "palette diversity collapsed")
    _require(len(masks) >= 100 and len(answers) >= 100, "row-verdict diversity collapsed")
    _require({f for f, _ in family_sides} == set(_FAMILIES), "cut family missing")
    _require(len(family_sides) == 10 and extreme_sources >= 10, "side/extreme-source diversity missing")
    legal_counts = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(314159 + nt * 10 + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts ignored")
            legal_counts += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (None, 1), (4, None), (True, 1), (4, False), (4.0, 1), (4, 1.0)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_same_puzzle(generate_puzzle(741, 10, 4), generate_puzzle(741, 10, 4)),
             "explicit replay failed")
    for seed in (-1, True, 2.5):
        _expect_rejection(lambda seed=seed: generate_puzzle(seed))

    p = generate_puzzle(0, 2, 1)
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupt["train"][0]["output"][0, 0] = (int(corrupt["train"][0]["output"][0, 0]) + 1) % 10
    _expect_rejection(lambda: validate_generated_puzzle(corrupt))
    wrong = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    start = 1 + 3 * ((_decode(wrong["test"][0]["input"])["w"]) + 1)
    wrong["test"][0]["output"][1, start] = (int(wrong["test"][0]["output"][1, start]) + 1) % 10
    _expect_rejection(lambda: validate_generated_puzzle(wrong))
    d = _decode(p["train"][0]["input"])
    control = d["panels"][0].copy()
    control[d["marker"]] = d["bg"]
    touching = next((int(r), int(c)) for r, c in np.argwhere(~d["cut"])
                    if np.any(d["cut"][max(0, r - 1):r + 2, max(0, c - 1):c + 2]))
    control[touching] = d["pos"]
    _expect_rejection(lambda: solve(_pack([control, *d["panels"][1:]], d["frame"])))
    for style in ("branch", "loop", "extra_region"):
        cut = np.zeros((9, 6), dtype=bool)
        if style == "branch":
            cut[:, 2] = True
            cut[4, 2:5] = True
        elif style == "loop":
            cut[2:7, 1] = cut[2:7, 4] = True
            cut[2, 1:5] = cut[6, 1:5] = True
        else:
            cut[:, 1] = cut[:, 4] = True
        _expect_rejection(lambda cut=cut: _geometry(cut))
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "canvas_shapes": len(shapes), "panel_shapes": len(panel_shapes),
            "palettes": len(palettes), "role_tuples": len(role_tuples), "cuts": len(cuts),
            "source_mask_pairs": len(masks), "destination_patterns": len(answers),
            "family_side_combinations": len(family_sides), "extreme_sources": extreme_sources,
            "families_by_split": dict(sorted(family_counts.items())),
            "legal_count_combinations": legal_counts, "independent_solver": False}
