"""Growing sluice gates: infer erased strokes, then inspect global enclosure.

Only NumPy and the standard library are used. No file I/O, saved examples, parent
imports, global random state, or generation metadata is needed by solve().
"""
from collections import deque
import hashlib

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(((a >= 0) & (a <= 9)).all()), "invalid grid color")
    return a


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
                 name + " must be an integer")
        _require(lo <= value <= hi, name + " outside legal range")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells, queue = [first], [first]
        for r, c in queue:
            for dr, dc in N4:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    queue.append(q)
        result.append(cells)
    return result


def _exterior(body):
    """Orthogonal flood from an explicit outside padding cell."""
    padded = np.pad(body, 1)
    seen = np.zeros(padded.shape, bool)
    seen[0, 0] = True
    queue = deque([(0, 0)])
    h, w = padded.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return seen[1:-1, 1:-1]


def _dilate(body):
    h, w = body.shape
    p = np.pad(body, 1)
    result = np.zeros_like(body)
    for dr, dc in N8:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _zones(body):
    """Solver path: global exterior, cavity ownership, then exterior dilation."""
    exterior = _exterior(body)
    holes = ~body & ~exterior
    hollow = np.zeros_like(body)
    labels = np.full(body.shape, -1, int)
    components = _components(body)
    for i, cells in enumerate(components):
        rr, cc = np.asarray(cells).T
        labels[rr, cc] = i
    for cells in _components(holes):
        owners = set()
        for r, c in cells:
            for dr, dc in N4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < body.shape[0] and 0 <= cc < body.shape[1] and labels[rr, cc] >= 0:
                    owners.add(int(labels[rr, cc]))
        _require(len(owners) == 1, "ambiguous cavity ownership")
        hollow |= labels == next(iter(owners))
    outline = _dilate(body) & exterior
    return dict(holes=holes, hollow=hollow, solid=body & ~hollow, outline=outline,
                far=exterior & ~outline, components=len(components))


def _local_label(body, colors):
    """Label path: classify each original component in its own padded box.

    This does not call solve(), the model fitter, or the global zone constructor.
    It cross-checks the global solver's ownership and enclosure interpretation.
    """
    bg, fg, outline, hollow_color, fill = colors
    out = np.full(body.shape, bg, np.int8)
    all_holes = np.zeros_like(body)
    for cells in _components(body):
        rr, cc = np.asarray(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
        local[rr - r0, cc - c0] = True
        holes = ~local & ~_exterior(local)
        out[rr, cc] = hollow_color if holes.any() else fg
        hr, hc = np.nonzero(holes)
        all_holes[hr + r0, hc + c0] = True
        out[hr + r0, hc + c0] = fill
    # Paint eight-neighbor exterior cells explicitly, independently of dilation.
    for r, c in np.argwhere(body):
        for dr, dc in N8:
            rr, cc = int(r) + dr, int(c) + dc
            if (0 <= rr < body.shape[0] and 0 <= cc < body.shape[1]
                    and not body[rr, cc] and not all_holes[rr, cc]):
                out[rr, cc] = outline
    return out


def _corner_cells(bounds):
    r0, r1, c0, c1 = bounds
    return ((r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1))


def _annotate(grid, bounds, colors):
    for (r, c), color in zip(_corner_cells(bounds), colors[2:] * 2):
        grid[r, c] = color


def _read_input(value):
    a = _grid(value)
    values, counts = np.unique(a, return_counts=True)
    _require(len(values) == 5, "input needs background, wall, and three key colors")
    rare = values[counts == 2]
    common = values[counts != 2]
    _require(len(rare) == 3 and len(common) == 2, "malformed key multiplicities")
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) > a.size // 2, "background must occupy more than half the input")
    fg = int(next(c for c in common if c != bg))
    _require(int(np.count_nonzero(a == fg)) > 6, "insufficient construction")
    positions = np.argwhere(np.isin(a, rare))
    r0, c0 = map(int, positions.min(axis=0))
    r1, c1 = map(int, positions.max(axis=0))
    bounds = (r0, r1, c0, c1)
    _require(r1 - r0 >= 2 and c1 - c0 >= 2 and r1 - r0 + c1 - c0 >= 5,
             "corner keys touch or overlap")
    _require(set(map(tuple, positions.tolist())) == set(_corner_cells(bounds)), "invalid corner keys")
    roles = tuple(int(a[r, c]) for r, c in _corner_cells(bounds)[:3])
    _require(len(set(roles)) == 3 and set(roles) == set(map(int, rare)), "key roles collide")
    for (r, c), color in zip(_corner_cells(bounds), roles * 2):
        _require(a[r, c] == color, "opposite keys disagree")
    region = a[r0:r1 + 1, c0:c1 + 1]
    _require(bool(np.isin(region, (bg,) + roles).all()), "window is not erased")
    missing = np.zeros(a.shape, bool)
    missing[r0:r1 + 1, c0:c1 + 1] = True
    return a == fg, missing, bounds, (bg, fg) + roles


def _canonical_models(observed, missing):
    """Fit the visible stroke grammar; return all compatible restored masks.

    Off-tooth columns identify the two-row main rail and the one-row catch rail.
    A length constraint table then fits all allowed linear/interleaved laws.
    No generation-side shape painter is called here.
    """
    h, w = observed.shape
    rows, cols = np.nonzero(observed)
    if not len(rows):
        return
    x0, x1 = int(cols.min()), int(cols.max())
    hidden_cols = np.flatnonzero(missing.any(axis=0))
    lo, hi = int(hidden_cols[0]), int(hidden_cols[-1])
    if not x0 < lo <= hi < x1:
        return
    for spacing in (3, 4):
        if (x1 - x0) % spacing:
            continue
        xs = np.arange(x0, x1 + 1, spacing)
        n = len(xs)
        if not 6 <= n <= 8 or np.count_nonzero((xs < lo) | (xs > hi)) < 3:
            continue
        off = np.array([x for x in range(x0, x1 + 1) if x not in set(xs)], int)
        active = np.flatnonzero(np.any(observed[:, off], axis=1))
        if len(active) != 3 or active[1] != active[0] + 1 or active[2] < active[1] + 4:
            continue
        top, bank = int(active[0]), int(active[2])
        valid = ~missing
        main = np.zeros_like(observed)
        main[top:top + 2, x0:x1 + 1] = True
        main[bank, x0:x1 + 1] = True
        # Only observed breaks are drains. Hidden rail portions are continuous.
        drains = [(bank, x) for x in off if valid[bank, x] and not observed[bank, x]]
        if len(drains) > 2:
            continue
        for r, c in drains:
            main[r, c] = False
        check_off = np.zeros_like(observed)
        check_off[:, off] = True
        if np.any((main != observed) & valid & check_off):
            continue
        # No foreground may lie outside the rail's common axial extent.
        if observed[:, :x0].any() or observed[:, x1 + 1:].any():
            continue
        max_length = h - top - 2
        if max_length < 1:
            continue
        allowed = np.zeros((n, max_length + 1), bool)
        for j, x in enumerate(xs):
            for length in range(1, max_length + 1):
                column = main[:, x].copy()
                column[top + 2:top + 2 + length] = True
                allowed[j, length] = not np.any((column != observed[:, x]) & valid[:, x])
        # L_i = b + d*i + a*(i mod 2), with bounded, visible family parameters.
        b, d, alt = np.meshgrid(np.arange(1, max_length + 1), np.arange(-2, 3),
                                np.arange(-6, 7), indexing="ij")
        parameters = np.stack((b.ravel(), d.ravel(), alt.ravel()), axis=1)
        i = np.arange(n)
        lengths = parameters[:, :1] + parameters[:, 1:2] * i + parameters[:, 2:3] * (i % 2)
        ok = ((lengths >= 1) & (lengths <= max_length)).all(axis=1)
        indices = np.flatnonzero(ok)
        if not len(indices):
            continue
        lengths, parameters = lengths[indices], parameters[indices]
        ok = allowed[np.arange(n)[None, :], lengths].all(axis=1)
        for law, lens in zip(parameters[ok], lengths[ok]):
            body = main.copy()
            # Cell predicate rendering, separate from generation's stroke loop.
            rr = np.arange(h)[:, None]
            body[:, xs] |= (rr >= top + 2) & (rr < top + 2 + lens[None, :])
            if np.any((body != observed) & valid):
                continue
            yield body, dict(top=top, bank=bank, spacing=spacing, n=n,
                             lengths=tuple(map(int, lens)), law=tuple(map(int, law)),
                             drains=tuple(drains), xs=tuple(map(int, xs)),
                             hidden_teeth=int(np.count_nonzero((xs >= lo) & (xs <= hi))))


def _recover(input_grid):
    observed, missing, bounds, colors = _read_input(input_grid)
    r0, r1, c0, c1 = bounds
    predictions, seen = [], set()
    chosen = None
    for rotation in range(4):
        obs, erased = np.rot90(observed, rotation), np.rot90(missing, rotation)
        for canonical, info in _canonical_models(obs, erased):
            body = np.rot90(canonical, -rotation).copy()
            key = body.tobytes()
            if key in seen:
                continue
            seen.add(key)
            z = _zones(body)
            bg, fg, outline, hollow, fill = colors
            full = np.full(body.shape, bg, np.int8)
            full[z["solid"]] = fg
            full[z["hollow"]] = hollow
            full[z["holes"]] = fill
            full[z["outline"]] = outline
            predictions.append(full[r0:r1 + 1, c0:c1 + 1].copy())
            if chosen is None:
                chosen = (body, z, dict(info, rotation=rotation), bounds, colors)
    _require(bool(predictions), "visible strokes do not establish a supported continuation")
    _require(all(np.array_equal(predictions[0], p) for p in predictions[1:]),
             "compatible input models disagree on the answer")
    return predictions[0], chosen


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the marked answer entirely from the given input."""
    return _recover(input_grid)[0]


def _rotate_bounds(shape, bounds, rotation):
    marker = np.zeros(shape, bool)
    r0, r1, c0, c1 = bounds
    marker[r0:r1 + 1, c0:c1 + 1] = True
    rr, cc = np.nonzero(np.rot90(marker, rotation))
    return int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())


def _paint_scene(lengths, spacing, top, left, depth, drains, bounds, padding, rotation, colors):
    """Ground truth from explicit rail/tooth strokes, before any erasure."""
    n = len(lengths)
    end = left + (n - 1) * spacing
    bank = top + 1 + depth
    height = max(top + 2 + max(lengths), bank + 1, bounds[1] + 1) + padding[0]
    width = end + 1 + padding[1]
    _require(height <= 30 and width <= 30, "scene exceeds canvas")
    body = np.zeros((height, width), bool)
    body[top:top + 2, left:end + 1] = True
    body[bank, left:end + 1] = True
    for j, length in enumerate(lengths):
        x = left + j * spacing
        for r in range(top + 2, top + 2 + length):
            body[r, x] = True
    for x in drains:
        body[bank, x] = False
    output_field = _local_label(body, colors)
    inp = np.where(body, colors[1], colors[0]).astype(np.int8)
    r0, r1, c0, c1 = bounds
    inp[r0:r1 + 1, c0:c1 + 1] = colors[0]
    out = output_field[r0:r1 + 1, c0:c1 + 1].copy()
    new_bounds = _rotate_bounds(inp.shape, bounds, rotation)
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    _annotate(inp, new_bounds, colors)
    return {"input": inp, "output": out}


def _proposal(rng, colors, kind, rotation, wide, variant=None):
    fallback = variant is not None
    if fallback:
        n, spacing = 7, 3
        top, left = 1 + variant % 3, 1 + (variant // 3) % 3
        base = 2 + variant // 9
    else:
        n = int(rng.integers(6, 9))
        spacing = int(rng.integers(3, 5))
        if (n - 1) * spacing > 26:
            spacing = 3
        top, left = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        base = int(rng.integers(2, 5))
    drains = []
    if kind == "critical":
        # A three-cell-wide open bay retains background beyond its outline,
        # and a two-cell growth step leaves an observable tail below the bank.
        n = 7 if fallback else min(n, 7)
        spacing, delta = 4, 2
        lengths = [base + delta * i for i in range(n)]
        # Exactly three reaching teeth; hide the middle one. A drain lies in
        # its left chamber, outside the erased window. Removing the middle
        # tooth joins the closed right chamber to that drain.
        depth = lengths[-3] + 1
        j = n - 2
        hidden_x = left + j * spacing
        lo, hi = hidden_x - 2, left + (j + 1) * spacing - 1
        drains = [hidden_x - 3]
    else:
        if fallback:
            delta, amplitude = 0, 4
        elif kind == "solid":
            delta, amplitude = 0, int(rng.integers(2, 6))
        else:
            delta = int(rng.integers(-2, 3))
            amplitude = int(rng.integers(-6, 7))
        raw = [delta * i + amplitude * (i % 2) for i in range(n)]
        lengths = [base + r - min(raw) for r in raw]
        depth = max(lengths) + 4 if kind == "solid" else max(4, sorted(lengths)[-3] + 1)
        width_count = 2 if wide else 1
        if kind == "random" and wide and not fallback and n == 8 and rng.random() < .35:
            width_count = 3
        j = 2 if fallback else int(rng.integers(1, n - width_count))
        lo = left + j * spacing - 1
        hi = left + (j + width_count - 1) * spacing + 1
    bank = top + 1 + depth
    bottom = max(bank, top + 1 + max(lengths))
    r0, r1 = top - 1, bottom + 1
    # Partial transverse windows leave the catch rail and its drains visible,
    # making enclosure outside the crop necessary for some returned cells.
    if kind == "random" and not fallback and depth >= 6 and rng.random() < .5:
        r0 = top + 2
        r1 = bank - 1
    bounds = (r0, r1, lo, hi)
    if kind != "critical" and not fallback:
        candidates = [x for x in range(left + 1, left + (n - 1) * spacing)
                      if (x - left) % spacing and not (r0 <= bank <= r1 and lo <= x <= hi)]
        count = min(len(candidates), int(rng.integers(0, 3)))
        if count:
            drains = list(map(int, rng.choice(candidates, count, replace=False)))
    padding = (1, 1) if fallback else (int(rng.integers(0, 4)), int(rng.integers(0, 4)))
    return _paint_scene(lengths, spacing, top, left, depth, drains, bounds, padding, rotation, colors)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = tuple(map(int, rng.permutation(10)[:5]))
    base_rotation = int(rng.integers(4))
    axis_shift = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            kind = "critical" if split == "train" and index == 0 else "solid" if split == "train" and index == 1 else "random"
            rotation = ((base_rotation + 2 * int(rng.integers(2)) + (split == "test")) % 4
                        if axis_shift else int(rng.integers(4)))
            wide = split == "test" or (axis_shift and index >= 2 and rng.random() < .4)
            accepted = None
            # Fixed cap, no unbounded rejection or payload retention.
            for attempt in range(96):
                try:
                    pair = _proposal(rng, colors, kind, rotation, wide,
                                     variant=attempt - 64 if attempt >= 64 else None)
                    predicted, recovered = _recover(pair["input"])
                    if not np.array_equal(predicted, pair["output"]):
                        continue
                    if kind == "critical":
                        body, z, _, bb, _ = recovered
                        observed = _read_input(pair["input"])[0]
                        rr0, rr1, cc0, cc1 = bb
                        without = _local_label(observed, colors)[rr0:rr1 + 1, cc0:cc1 + 1]
                        if not z["holes"].any() or np.array_equal(without, predicted):
                            continue
                    if kind == "solid" and recovered[1]["holes"].any():
                        continue
                    if _key(pair["input"]) in seen:
                        continue
                    accepted = pair
                    break
                except ValueError:
                    continue
            _require(accepted is not None, "bounded construction exhausted")
            seen.add(_key(accepted["input"]))
            puzzle[split].append(accepted)
    return puzzle


def _features(pair, analysis=None):
    answer, (body, z, model, bounds, colors) = _recover(pair["input"]) if analysis is None else analysis
    r0, r1, c0, c1 = bounds
    crop = lambda a: a[r0:r1 + 1, c0:c1 + 1]
    observed = _read_input(pair["input"])[0]
    no_a = _local_label(observed, colors)[r0:r1 + 1, c0:c1 + 1]
    no_b = np.where(crop(body), colors[1], colors[0])
    local_z = _zones(crop(body))
    without_a_z = _zones(observed)
    plugged = np.rot90(body, model["rotation"]).copy()
    for r, c in model["drains"]:
        plugged[r, c] = True
    plugged = np.rot90(plugged, -model["rotation"])
    no_drain = _local_label(plugged, colors)[r0:r1 + 1, c0:c1 + 1]
    # Replace each erased tooth with its nearest preceding visible tooth's
    # length, while keeping both rails and every visible drain correct.
    wrong_lengths = np.rot90(body, model["rotation"]).copy()
    erased = np.rot90(_read_input(pair["input"])[1], model["rotation"])
    missing_columns = erased.any(axis=0)
    for j, x in enumerate(model["xs"]):
        if not missing_columns[x]:
            continue
        previous = [k for k in range(j) if not missing_columns[model["xs"][k]]]
        if not previous:
            continue
        wrong_lengths[model["top"] + 2:, x] = False
        wrong_lengths[model["bank"], x] = True
        length = model["lengths"][previous[-1]]
        wrong_lengths[model["top"] + 2:model["top"] + 2 + length, x] = True
    wrong_lengths = np.rot90(wrong_lengths, -model["rotation"])
    no_growth = _local_label(wrong_lengths, colors)[r0:r1 + 1, c0:c1 + 1]
    return dict(shape=tuple(np.asarray(pair["input"]).shape), output_shape=answer.shape,
                rotation=model["rotation"], n=model["n"], spacing=model["spacing"],
                growth=model["law"][1], alternating=model["law"][2],
                lengths=model["lengths"], drains=len(model["drains"]),
                hidden_teeth=model["hidden_teeth"], holes=len(_components(z["holes"])),
                cavity_area=int(z["holes"].sum()), solid_area=int(z["solid"].sum()),
                hollow_area=int(z["hollow"].sum()), outline_area=int(crop(z["outline"]).sum()),
                no_a_diff=bool(not np.array_equal(no_a, answer)),
                no_b_diff=bool(not np.array_equal(no_b, answer)),
                restoration_creates_hole=bool(z["holes"].any() and not without_a_z["holes"].any()),
                drain_matter=bool(not np.array_equal(no_drain, answer)),
                copy_previous_diff=bool(not np.array_equal(no_growth, answer)),
                topology_after_crop_diff=bool(not np.array_equal(local_z["holes"], crop(z["holes"]))),
                holes_visible=bool(crop(z["holes"]).any()),
                solid_visible=bool(crop(z["solid"]).any()),
                colors=colors, structure=hashlib.sha256(bytes(body.shape) + body.tobytes()).hexdigest())


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features, palettes = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            answer, (body, z, model, bounds, colors) = _recover(inp)
            r0, r1, c0, c1 = bounds
            _require(out.shape == (r1 - r0 + 1, c1 - c0 + 1), "wrong inclusive crop")
            _require(np.array_equal(out, answer), "incorrect solved output")
            local = _local_label(body, colors)[r0:r1 + 1, c0:c1 + 1]
            _require(np.array_equal(out, local), "component-local label disagrees with global solver")
            # Check each disjoint pixel role, including unchanged solids/far background.
            for role, color in (("solid", colors[1]), ("hollow", colors[3]),
                                ("holes", colors[4]), ("outline", colors[2]), ("far", colors[0])):
                zone = z[role][r0:r1 + 1, c0:c1 + 1]
                _require(bool((out[zone] == color).all()), "incorrect zone: " + role)
            _require(set(map(int, np.unique(out))) <= set(colors), "unknown output role")
            _require(model["n"] >= 6 and model["hidden_teeth"] >= 1, "insufficient sequence evidence")
            palettes.add(colors)
            features[split].append(_features(pair, (answer, (body, z, model, bounds, colors))))
    _require(len(palettes) == 1, "generated puzzle changed palette conventions")
    train = features["train"]
    _require(any(f["holes_visible"] and f["hollow_area"] and f["no_a_diff"] and f["no_b_diff"] for f in train),
             "training does not expose both interacting mechanisms")
    _require(any(f["solid_visible"] and f["holes"] == 0 and f["alternating"] for f in train),
             "training lacks an alternating open-comb counterexample")
    _require(any(f["growth"] != 0 for f in train), "training permits constant-tooth repetition")
    _require(any(f["restoration_creates_hole"] and f["drain_matter"] for f in train),
             "training must show reconstruction interacting with a visible drain")
    _require(any(f["growth"] and f["copy_previous_diff"] for f in train),
             "training must distinguish arithmetic growth from copying a tooth")
    _require(all(f["no_b_diff"] for f in train), "topology/outline shortcut is not distinguished")
    return features


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, one puzzle retained at a time; no filesystem output."""
    signatures, shapes, masks, laws, counts, palettes, rotations = set(), set(), set(), set(), set(), set(), set()
    totals = dict(no_a=0, no_b=0, nonlocal_topology=0, drains=0, alternating=0,
                  increasing=0, decreasing=0, multi_hole=0, solids=0, causal_drains=0,
                  restoration_creates_hole=0, copy_previous_diff=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        features = validate_generated_puzzle(p)
        signatures.add(_puzzle_digest(p))
        for split in ("train", "test"):
            for pair, f in zip(p[split], features[split]):
                shapes.add((f["shape"], f["output_shape"]))
                laws.add((f["growth"], f["alternating"]))
                counts.add((f["n"], f["hidden_teeth"], f["holes"]))
                palettes.add(f["colors"])
                rotations.add(f["rotation"])
                masks.add(f["structure"])
                totals["no_a"] += f["no_a_diff"]
                totals["no_b"] += f["no_b_diff"]
                totals["nonlocal_topology"] += f["topology_after_crop_diff"]
                totals["drains"] += f["drains"] > 0
                totals["alternating"] += f["alternating"] != 0
                totals["increasing"] += f["growth"] > 0
                totals["decreasing"] += f["growth"] < 0
                totals["multi_hole"] += f["holes"] > 1
                totals["solids"] += f["holes"] == 0
                totals["causal_drains"] += f["drain_matter"]
                totals["restoration_creates_hole"] += f["restoration_creates_hole"]
                totals["copy_previous_diff"] += f["copy_previous_diff"]
    _require(len(signatures) == 42 and len(shapes) >= 60 and len(masks) >= 150, "insufficient structural variation")
    _require(len(laws) >= 15 and len(counts) >= 15 and len(palettes) >= 35 and len(rotations) == 4,
             "missing law, count, palette, or orientation variation")
    _require(all(v >= 8 for v in totals.values()), "missing family-specific variation: " + str(totals))
    # All legal explicit counts, including the largest 14-pair request.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (0, 760966207, 774692896):
        _require(_puzzle_digest(generate_puzzle(seed)) == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
    # A finite pool larger than the 13 preceding inputs protects every count
    # request if random proposals are rejected. Test the actual fallback pool.
    for kind, wide in (("critical", False), ("solid", False), ("random", False), ("random", True)):
        keys = set()
        for variant in range(32):
            pair = _proposal(np.random.default_rng(0), (0, 1, 2, 3, 4), kind, 0, wide, variant)
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "fallback prediction mismatch")
            keys.add(_key(pair["input"]))
        _require(len(keys) == 32, "fallback pool is not distinct")
    # Color-role equivariance is exact, including arbitrary nonblack backgrounds.
    p = generate_puzzle(9)
    permutation = np.array([7, 4, 9, 0, 8, 6, 1, 5, 3, 2], np.int8)
    for pair in p["train"] + p["test"]:
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "literal colors leaked into semantics")
    # Corrupt each represented semantic output zone and require rejection.
    p = generate_puzzle(0, 2, 1)
    for role in ("solid", "hollow", "holes", "outline", "far"):
        q = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        changed = False
        for pair in q["train"]:
            _, (_, z, _, bb, _) = _recover(pair["input"])
            r0, r1, c0, c1 = bb
            cells = np.argwhere(z[role][r0:r1 + 1, c0:c1 + 1])
            if len(cells):
                r, c = cells[0]
                pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
                changed = True
                break
        _require(changed, "regression zone absent: " + role)
        try:
            validate_generated_puzzle(q)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted corrupted " + role)
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), shape_combinations=len(shapes),
                structural_masks=len(masks), laws=len(laws), count_combinations=len(counts),
                palettes=len(palettes), totals=totals, independent_solver=False)
