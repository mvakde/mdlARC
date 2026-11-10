"""Calibrated color probes. Only NumPy and the Python standard library.

The constructor paints from an unoccluded map; solve reconstructs that map from
the input. Neither imports parent code, reads evidence, nor uses latent state.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 252054037, 779289107, 227081432, 501523187, 182139347,
              95027271, 253714956, 575043565, 124715034, 703279410, 652442755,
              732609771, 682457727, 616408863, 686272169, 90983825, 989790142,
              462575499, 830400118, 246162963, 749862049, 320082091, 184637173,
              322322458, 78388342, 787446461, 790664255, 795744141, 186559067,
              769524664, 56565042, 265867115, 161519148, 301441957, 33261626,
              620210732, 604144063, 125849934, 849637977, 222713590, 348012523)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(grid):
    """Maximal four-connected equal-color territories, including all colors."""
    h, w = grid.shape
    ids = np.full((h, w), -1, dtype=np.int16)
    colors, cells = [], []
    for r in range(h):
        for c in range(w):
            if ids[r, c] >= 0:
                continue
            label, color = len(colors), int(grid[r, c])
            colors.append(color)
            ids[r, c] = label
            stack, points = [(r, c)], []
            while stack:
                y, x = stack.pop()
                points.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < h and 0 <= xx < w and ids[yy, xx] < 0
                            and grid[yy, xx] == color):
                        ids[yy, xx] = label
                        stack.append((yy, xx))
            cells.append(points)
    return ids, colors, cells


def _paint(info, distances=None, *, color_wide=False):
    """Routing on the ORIGINAL map; returns None for impossible alternatives."""
    clean, ids = info["clean"], info["ids"]
    rows, x, payloads = info["rows"], info["x"], info["payloads"]
    if distances is None:
        distances = [info["reference"]] * len(rows)
    assignments = {}
    for r, distance, payload in zip(rows, distances, payloads):
        col = x + int(distance)
        if not 0 <= col < clean.shape[1]:
            return None
        recipient = int(clean[r, col]) if color_wide else int(ids[r, col])
        if recipient in assignments and assignments[recipient] != payload:
            return None
        assignments[recipient] = payload
    out = clean.copy()
    for recipient, payload in assignments.items():
        out[(clean if color_wide else ids) == recipient] = payload
    return out


def _parse_right(grid, ink):
    """Read one candidate in coordinates in which the probes point right."""
    h, w = grid.shape
    yy, xx = np.nonzero(grid == ink)
    if not len(xx):
        return None
    x = int(xx.min())
    if x < 1 or x + 2 >= w:
        return None
    rows = sorted(set(int(y) for y, col in zip(yy, xx) if col > x))
    if len(rows) not in (3, 5, 7) or any(b-a < 4 for a, b in zip(rows, rows[1:])):
        return None
    expected = np.zeros(grid.shape, dtype=bool)
    clean = grid.copy()
    lengths, hosts, payloads, explicit, writes = [], [], [], [], []
    for r in rows:
        if r < 1 or r+1 >= h:
            return None
        cols = np.flatnonzero(grid[r, x+1:] == ink) + x+1
        if not len(cols) or not np.array_equal(cols, np.arange(x+1, int(cols[-1])+1)):
            return None
        length = int(cols[-1]) - x + 1
        if grid[r-1, x] != ink or grid[r+1, x] != ink:
            return None
        behind = grid[r-1:r+2, x-1]
        if np.any(behind != behind[0]) or behind[0] == ink:
            return None
        host = int(behind[0])
        top, bottom = grid[r-1, x+1:x+length], grid[r+1, x+1:x+length]
        if np.any(top != bottom) or np.any(top == ink):
            return None
        expected[r, x+1:x+length] = True
        expected[r-1, x] = expected[r+1, x] = True
        is_explicit = bool(grid[r, x] != ink)
        if not is_explicit:
            expected[r, x] = True
        payload = int(grid[r, x]) if is_explicit else host
        clean[r-1:r+2, x] = host
        clean[r, x+1:x+length] = top
        lengths.append(length)
        hosts.append(host)
        payloads.append(payload)
        explicit.append(is_explicit)
        writes.append([(r-1, x), (r+1, x)] + [(r, col) for col in range(x, x+length)])
    if not np.array_equal(expected, grid == ink) or np.any(clean == ink):
        return None
    reference = lengths[len(rows)//2]
    if x+reference >= w:
        return None
    ids, colors, cells = _components(clean)
    sources = [int(ids[r, x]) for r in rows]
    targets = [int(ids[r, x+reference]) for r in rows]
    info = dict(clean=clean, ids=ids, colors=colors, cells=cells, ink=ink, x=x,
                rows=rows, lengths=lengths, hosts=hosts, payloads=payloads,
                explicit=explicit, reference=reference, sources=sources,
                targets=targets, writes=writes)
    out = _paint(info)
    if out is None:
        return None
    info["painted"] = out
    return info


def _decode(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for k in range(4):
        turned = np.rot90(grid, k)
        for value in np.unique(turned):
            info = _parse_right(turned, int(value))
            if info is not None:
                info["rotation"] = k
                candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("input must have one complete, unambiguous calibrated-probe interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the complete answer from this input alone."""
    info = _decode(input_grid)
    return np.rot90(info["painted"], -info["rotation"]).copy()


def _different(candidate, answer):
    return candidate is None or not np.array_equal(candidate, answer)


def _fixed_range_fits(info):
    return {distance for distance in range(2, info["clean"].shape[1]-info["x"])
            if not _different(_paint(info, [distance]*len(info["rows"])), info["painted"])}


def _features(info):
    n, ref = len(info["rows"]), info["reference"]
    spans, rows = info["lengths"], info["rows"]
    distances = [abs(2*r - rows[0] - rows[-1]) for r in rows]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    alternatives = {"median": [sorted(spans)[n//2]] * n,
                    "first": [spans[0]] * n, "last": [spans[-1]] * n,
                    "shortest": [min(spans)] * n, "longest": [max(spans)] * n,
                    "raw": spans, "tip": [ref-1] * n}
    if len(nearest) == 1:
        alternatives["geometric"] = [spans[nearest[0]]] * n
    first_boundary = []
    for r, source in zip(rows, info["sources"]):
        options = [c-info["x"] for c in range(info["x"]+1, info["clean"].shape[1])
                   if info["ids"][r, c] != source]
        first_boundary.append(options[0] if options else ref)
    alternatives["boundary"] = first_boundary
    rejected = {name for name, ds in alternatives.items()
                if _different(_paint(info, ds), info["painted"])}
    path = info["clean"].copy()
    for r, payload in zip(rows, info["payloads"]):
        path[r, info["x"]+1:info["x"]+ref+1] = payload
    if _different(path, info["painted"]):
        rejected.add("path")
    final = [int(info["painted"][points[0]]) for points in info["cells"]]
    locality = any(info["colors"][i] == info["colors"][j] and final[i] != final[j]
                   for i in range(len(final)) for j in range(i))
    chain = any(info["targets"][i] == info["sources"][j]
                and info["targets"][j] != info["targets"][i]
                and not info["explicit"][j]
                and info["payloads"][i] != info["hosts"][j]
                for i in range(n) for j in range(n) if i != j)
    nonrectangular = sum(len(points) != (max(r for r,c in points)-min(r for r,c in points)+1)
                         * (max(c for r,c in points)-min(c for r,c in points)+1)
                         for points in info["cells"])
    return dict(count=n, reference=ref, gaps=tuple(b-a for a,b in zip(rows, rows[1:])),
                lengths=tuple(spans), shape=info["clean"].shape, ink=info["ink"],
                direction=info["rotation"], components=len(final), nonrectangular=nonrectangular,
                explicit=sum(info["explicit"]), novel=any(p not in info["colors"] for p in info["payloads"]),
                locality=locality, chain=chain, rejected=rejected,
                color_wide=_different(_paint(info, color_wide=True), info["painted"]),
                changed=sum(a != b for a,b in zip(info["colors"], final)),
                mixed=min(spans) < ref < max(spans))


def _terrain(rng, h, w, rows, x, ref, palette, role):
    """Independent unoccluded scene construction; only probe corridors are flat."""
    n, sample = len(rows), x+ref
    boundaries = [0] + [(a+b)//2 for a,b in zip(rows, rows[1:])] + [h]
    clean = np.empty((h, w), dtype=np.int8)
    for i, r in enumerate(rows):
        source = palette[0] if role == "range" else int(rng.choice(palette))
        if role == "transfer":
            source = palette[1] if i == 1 else palette[0]
        # A sample need not sit on a boundary. Only one range-witness lane
        # deliberately places a boundary there, to teach the L versus L-1 case.
        target_start = (sample if role == "range" and i == 0
                        else max(x+3, sample-int(rng.integers(0, 4))))
        source_end = int(rng.integers(x+2, target_start))
        cuts = [source_end]
        if source_end < target_start-1 and rng.random() < .65:
            cuts.append(int(rng.integers(source_end+1, target_start)))
        cuts += [target_start, sample+int(rng.integers(1, 5))]
        cuts = sorted(set(cuts))
        if cuts[-1] < w-3 and rng.random() < .5:
            cuts.append(int(rng.integers(cuts[-1]+1, w-1)))
        fills = [source]
        for _ in cuts:
            fills.append(int(rng.choice([c for c in palette if c != fills[-1]])))
        target_index = cuts.index(target_start)+1
        fills[target_index] = int(rng.choice([c for c in palette if c != source]))
        if role == "transfer":
            # Three sources X,Y,X; destinations Y,Z,W. Y wraps back to source 1.
            fills[target_index] = palette[i+1]
            cuts = [source_end, target_start, sample+int(rng.integers(1, 4)), w-2]
            fills = [source, palette[4], palette[i+1], palette[4],
                     palette[2] if i == 0 else palette[4]]
        for y in range(boundaries[i], boundaries[i+1]):
            row_cuts = list(cuts)
            if abs(y-r) > 1 and role != "transfer":
                for j, cut in enumerate(row_cuts):
                    lo = (row_cuts[j-1]+1) if j else x+2
                    hi = (cuts[j+1]-1) if j+1 < len(cuts) else w-1
                    row_cuts[j] = int(np.clip(cut+rng.integers(-1, 2), lo, hi))
            start = 0
            for end, color in zip(row_cuts+[w], fills):
                clean[y, start:end] = color
                start = end
    if role == "transfer":
        clean[rows[0]+2:rows[1]-1, :] = palette[1]
    elif role != "range":
        for i in range(n-1):
            if rng.random() < .35:
                clean[rows[i]+2:rows[i+1]-1, :] = clean[rows[i+1], x]
    return clean


def _build_pair(rng, params, role, used_refs, test, count, rotation):
    if role == "range":
        count = 5
        gaps = [4, 4, 4, 4]
        gaps[int(rng.choice((0, 3)))] = 10
    else:
        gaps = [4] * (count-1)
        budget = int(rng.integers(0, 25-sum(gaps)))
        for _ in range(budget):
            gaps[int(rng.integers(count-1))] += 1
    span = sum(gaps)
    h = int(rng.integers(span+3, 31))
    first = int(rng.integers(1, h-span-1))
    rows = [first] + [first+int(v) for v in np.cumsum(gaps)]
    w = int(rng.integers(23 if test else 18, 31))
    x = int(rng.integers(1, min(5, w-13)))
    capacity = w-x-1
    choices = [r for r in range(5, min(18 if test else 14, w-x-7)+1)
               if r not in used_refs and (role != "transfer" or all(abs(r-old) >= 4 for old in used_refs))]
    if not choices:
        return None
    ref = int(rng.choice(choices))
    others = [i for i in range(count) if i != count//2]
    lengths = [ref]*count
    mode = str(rng.choice(("mixed", "short", "long", "single")))
    if role == "range":
        for i in others:
            lengths[i] = int(rng.integers(2, ref))
        lengths[int(rng.choice(others))] = int(rng.integers(ref+1, capacity+1))
    else:
        for i in others:
            if mode == "single" or rng.random() < .20:
                continue
            lengths[i] = (int(rng.integers(2, ref)) if mode == "short" or (mode == "mixed" and rng.random() < .5)
                          else int(rng.integers(ref+1, capacity+1)))
        if mode == "single":
            i = int(rng.choice(others))
            lengths[i] = int(rng.choice([v for v in range(2, capacity+1) if v != ref]))
        if role == "transfer" or mode == "mixed":
            lengths[others[0]], lengths[others[-1]] = int(rng.integers(2, ref)), int(rng.integers(ref+1, capacity+1))
    ink = int(rng.integers(10)) if params["vary_ink"] else params["ink"]
    available = [c for c in range(10) if c != ink]
    rng.shuffle(available)
    novel = available[-1]
    palette = available[:int(rng.integers(5, 9))]
    clean = _terrain(rng, h, w, rows, x, ref, palette, role)
    ids, colors, cells = _components(clean)
    sources = [int(ids[r, x]) for r in rows]
    targets = [int(ids[r, x+ref]) for r in rows]
    hosts = [int(clean[r, x]) for r in rows]
    if role == "range":
        payloads = list(hosts)
    elif role == "transfer":
        payloads = [hosts[0], hosts[1], novel]
    else:
        chosen = {}
        for target in set(targets):
            candidates = [hosts[i] for i, t in enumerate(targets) if t == target]
            choices = candidates if rng.random() < .65 else [c for c in available if c != colors[target]]
            chosen[target] = int(rng.choice(choices))
        payloads = [chosen[t] for t in targets]
    assignments = {}
    for target, payload in zip(targets, payloads):
        if target in assignments and assignments[target] != payload:
            return None
        assignments[target] = payload
    out, inp = clean.copy(), clean.copy()
    for target, payload in assignments.items():
        out[ids == target] = payload
    for r, length, host, payload in zip(rows, lengths, hosts, payloads):
        inp[r-1:r+2, x] = ink
        inp[r, x:x+length] = ink
        if payload != host:
            inp[r, x] = payload
    pair = dict(input=np.rot90(inp, rotation).copy(), output=np.rot90(out, rotation).copy())
    try:
        info = _decode(pair["input"])
    except ValueError:
        return None
    if not np.array_equal(np.rot90(info["clean"], -info["rotation"]), np.rot90(clean, rotation)):
        return None
    if not np.array_equal(solve(pair["input"]), pair["output"]):
        return None
    f = _features(info)
    if not f["changed"] or "raw" not in f["rejected"]:
        return None
    required = {"median", "geometric", "first", "last", "shortest", "longest", "raw", "tip", "boundary", "path"}
    if role == "range" and not required <= f["rejected"]:
        return None
    if role == "transfer" and not (f["chain"] and f["novel"] and f["locality"] and f["color_wide"]):
        return None
    return pair, info


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(name + " is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    params = dict(ink=int(rng.integers(10)), vary_ink=bool(rng.integers(2)),
                  base=int(rng.integers(4)), shift=int(rng.integers(2)))
    puzzle, seen, train_refs = {"train": [], "test": []}, set(), set()
    constant_fits = set(range(2, 30))
    for split, total in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(total):
            role = ("range" if index == 0 else "transfer" if index == 1 else "free") if split == "train" else "free"
            test = split == "test"
            count = 3 if role == "transfer" else 5 if role == "range" else int(rng.choice((3, 5, 7) if test or index > 2 else (3, 5)))
            rotation = (params["base"] + (index % 2 if not test else 2+index % 2)) % 4
            if params["shift"] == 0 and test:
                count = int(rng.choice((5, 7)))
                rotation = (params["base"] + index % 2) % 4
            for _ in range(180):
                result = _build_pair(rng, params, role, train_refs if test or index == 1 else set(), test, count, rotation)
                if result is not None and _key(result[0]["input"]) not in seen:
                    pair, info = result
                    fits = _fixed_range_fits(info) if not test else set()
                    if role == "transfer" and constant_fits & fits:
                        continue
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    if not test:
                        train_refs.add(info["reference"])
                        constant_fits &= fits
                    break
            else:
                raise RuntimeError("bounded probe-map construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, features = set(), {"train": [], "test": []}
    constant_fits = set(range(2, 30))
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "canvas dimensions changed")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            expected = np.rot90(info["painted"], -info["rotation"])
            _need(np.array_equal(expected, out), "incorrect calibrated transfer")
            canonical = np.rot90(out, info["rotation"])
            assignments = dict(zip(info["targets"], info["payloads"]))
            for i, points in enumerate(info["cells"]):
                wanted = assignments.get(i, info["colors"][i])
                _need(all(int(canonical[p]) == wanted for p in points), "territory not uniformly painted")
            _need(not np.array_equal(info["clean"], info["painted"]), "no meaningful color transfer")
            f = _features(info)
            _need("raw" in f["rejected"], "raw endpoints give the answer")
            features[split].append(f)
            if split == "train":
                constant_fits &= _fixed_range_fits(info)
    train = features["train"]
    required = {"median", "geometric", "first", "last", "shortest", "longest", "raw", "tip", "boundary", "path"}
    _need(any(required <= f["rejected"] and f["mixed"] for f in train), "missing range-selection demonstration")
    _need(any(f["chain"] and f["novel"] and f["locality"] and f["color_wide"] for f in train), "missing transfer demonstration")
    refs = {f["reference"] for f in train}
    _need(len(refs) >= 2, "constant training calibration")
    _need(not constant_fits, "a constant range explains all demonstrations")
    _need(all(f["reference"] not in refs for f in features["test"]), "test lacks novel calibration shift")


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed structural, semantic, intervention and diversity checks."""
    rng = np.random.default_rng(718233)
    # Hand-constructed inputs outside the sampler's minimum sizes/ranges:
    # edge bars, two-cell calibration, consistent duplicate recipients, and
    # explicit self-targeting. Expected arrays are direct stripe recolorings.
    for left_end, token in ((3, None), (5, 7)):
        clean = np.full((11, 11), 4, dtype=np.int8)
        clean[:, :left_end] = 2
        clean[:, left_end:left_end+2] = 3
        inp = clean.copy()
        for r, length in zip((1, 5, 9), (2, 2, 5)):
            inp[r-1:r+2, 1] = 9
            inp[r, 1:1+length] = 9
            if token is not None:
                inp[r, 1] = token
        expected = clean.copy()
        expected[clean == int(clean[5, 3])] = 2 if token is None else token
        _need(np.array_equal(solve(inp), expected), "small-range/edge/self-target fixture failed")
        # Changing one of three consistent payloads must be rejected, not ordered.
        conflicting = inp.copy()
        conflicting[5, 1] = 8
        try:
            solve(conflicting)
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting recipient colors accepted")
    signatures, shapes, partitions, ranges, inks, counts = set(), set(), set(), set(), set(), set()
    directions = {"train": set(), "test": set()}
    combined = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _need(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "seed is not deterministic")
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info, f = _decode(inp), _features(_decode(inp))
                shapes.add(inp.shape)
                partitions.add(_key(info["ids"]))
                ranges.add(f["reference"])
                inks.add(f["ink"])
                counts.add(f["count"])
                directions[split].add(f["direction"])
                combined += bool(f["chain"] and f["locality"] and f["nonrectangular"])
                _need(np.array_equal(solve(inp), out), "input-only solve failed")
            # Palette roles and handed coordinates are semantic, not constants.
            pair = puzzle[split][0]
            permutation = rng.permutation(10).astype(np.int8)
            _need(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "palette equivariance failed")
            for k in range(4):
                for mirror in (False, True):
                    inp, out = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                    if mirror:
                        inp, out = np.fliplr(inp), np.fliplr(out)
                    _need(np.array_equal(solve(inp), out), "D4 equivariance failed")
        # Shortening the calibrated reference by one uses the same valid grammar.
        witness = next(p for p in puzzle["train"]
                       if (lambda z: len(z["rows"]) == 5 and not any(z["explicit"])
                           and len(set(z["hosts"])) == 1
                           and {"median", "geometric", "tip"} <= _features(z)["rejected"])(
                               _decode(p["input"])))
        info = _decode(witness["input"])
        altered = np.rot90(witness["input"], info["rotation"]).copy()
        r, c = info["rows"][2], info["x"]+info["reference"]-1
        altered[r, c] = info["clean"][r, c]
        changed = solve(altered)
        _need(not np.array_equal(changed, info["painted"]), "central shaft does not control range")
        # A novel explicit payload changes the recipient but no other component.
        transfer = next(p for p in puzzle["train"]
                        if (lambda f: f["novel"] and f["chain"] and f["count"] == 3 and f["explicit"] == 1)(
                            _features(_decode(p["input"]))))
        info = _decode(transfer["input"])
        index = next(i for i, explicit in enumerate(info["explicit"]) if explicit)
        altered = np.rot90(transfer["input"], info["rotation"]).copy()
        old = info["payloads"][index]
        replacement = next(c for c in range(10) if c not in (old, info["ink"]))
        altered[info["rows"][index], info["x"]] = replacement
        changed = solve(altered)
        mask = info["ids"] == info["targets"][index]
        _need(np.all(changed[mask] == replacement) and np.array_equal(changed[~mask], info["painted"][~mask]), "literal payload intervention failed")
    _need(len(signatures) == 42 and len(partitions) > 150, "collapsed scene structures")
    _need(len(shapes) >= 35 and len(ranges) >= 10, "collapsed geometry or ranges")
    _need(inks == set(range(10)) and counts == {3, 5, 7}, "collapsed roles or counts")
    _need(all(d == set(range(4)) for d in directions.values()), "missing cardinal directions")
    _need(combined >= 42, "missing interacting topology")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(55220+a*7+b, a, b)
            _need(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "ignored explicit counts")
            validate_generated_puzzle(puzzle)
    for a,b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (3.5,1), (None,1), (4,None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0)
    broken["test"][0]["output"][0,0] = (int(broken["test"][0]["output"][0,0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                component_partitions=len(partitions), reference_ranges=sorted(ranges),
                ink_colors=sorted(inks), probe_counts=sorted(counts), interacting_pairs=combined,
                independent_solver=False, blind_prediction=False)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
