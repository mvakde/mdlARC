"""Missing-edge joins: recover symmetric plates, assemble, and mark shared halos.

Only standard-library Python and NumPy are used. The decoder sees the input
alone; construction masks are used only to render labels and challenge it.
Color 7 is the fixed ordinary halo color. The visible divider supplies the
shared-halo color. No parent generators or stored examples are consulted.
"""
from itertools import combinations

import numpy as np


HALO = 7
AUDIT_SEEDS = (
    0, 17906018, 417226307, 470420018, 296985616, 550603531,
    480781167, 12904786, 531485965, 128771846, 589462521, 497048492,
    445449339, 269763150, 473555062, 397867958, 299376203, 694597185,
    161681751, 104306689, 993036382, 784629715, 253546586, 355345085,
    515923762, 935964198, 192917266, 524146869, 55415004, 540134760,
    269795717, 864673578, 265885989, 380063888, 685723134, 113420241,
    581056667, 312408810, 626906167, 122023136, 843166514, 706256571,
)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int64, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _orbit(r, c, h, w):
    return tuple(sorted({(r, c), (h - 1 - r, c),
                         (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _orbit_key(r, c, h, w):
    return min(r, h - 1 - r), min(c, w - 1 - c)


def _symmetrize(mask):
    return mask | mask[::-1] | mask[:, ::-1] | mask[::-1, ::-1]


def _recover(panel, field, solid):
    """Strict reflection-orbit majority, including explicit ambiguity rejection."""
    h, w = panel.shape
    base = np.zeros((h, w), dtype=bool)
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            points = _orbit(r, c, h, w)
            count = sum(panel[y, x] == solid for y, x in points)
            if 2 * count == len(points):
                raise ValueError("tied reflection orbit")
            intended = 2 * count > len(points)
            flips = min(count, len(points) - count)
            if flips > (1 if len(points) == 4 else 0):
                raise ValueError("too much damage in a reflection orbit")
            for y, x in points:
                base[y, x] = intended
    if not base.any() or base.all():
        raise ValueError("a plate must contain both scaffold roles")
    holes = base & (panel == field)
    extras = ~base & (panel == solid)
    return base, holes, extras


def _partitions(a):
    """Infer complete one-cell separator lines rather than a layout catalog."""
    h, w = a.shape
    candidates = []
    for divider in map(int, np.unique(a)):
        if divider == HALO:
            continue
        rows = np.flatnonzero(np.all(a == divider, axis=1)).tolist()
        cols = np.flatnonzero(np.all(a == divider, axis=0)).tolist()
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        hh = [v - u - 1 for u, v in zip(rr, rr[1:])]
        ww = [v - u - 1 for u, v in zip(cc, cc[1:])]
        if (min(hh + ww) < 4 or len(set(hh)) != 1
                or len(set(ww)) != 1 or not 2 <= len(hh) * len(ww) <= 6):
            continue
        panels = [a[y + 1:rr[i + 1], x + 1:cc[j + 1]].copy()
                  for i, y in enumerate(rr[:-1])
                  for j, x in enumerate(cc[:-1])]
        if any(np.any(p == divider) or np.any(p == HALO) for p in panels):
            continue
        corners = {int(p[y, x]) for p in panels
                   for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        field = next(iter(corners))
        other = set().union(*(set(map(int, np.unique(p))) for p in panels)) - {field}
        if len(other) != 1:
            continue
        candidates.append(dict(panels=panels, field=field,
                               solid=next(iter(other)), divider=divider,
                               layout=(len(hh), len(ww))))
    return candidates


def _chain_orders(signatures):
    """Find up to two complete directed chains; local matches alone do not suffice."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, unused):
        if len(answers) == 2:
            return
        if not unused:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = signatures[order[-1]][1]
        if not outgoing:
            return
        for i in unused:
            if signatures[i][0] == outgoing:
                visit(order + [i], tuple(j for j in unused if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _decode(a):
    answers = []
    for partition in _partitions(a):
        try:
            recovered = [_recover(p, partition["field"], partition["solid"])
                         for p in partition["panels"]]
        except ValueError:
            continue
        for axis in (1, 0):
            holes = [rec[1] if axis == 1 else rec[1].T for rec in recovered]
            bases = [rec[0] if axis == 1 else rec[0].T for rec in recovered]
            if any(b[0].any() or b[-1].any() for b in bases):
                continue  # The non-chain outer edges are intended field.
            if any(m[0].any() or m[-1].any() for m in holes):
                continue
            signatures = [(tuple(map(int, np.flatnonzero(m[:, 0]))),
                           tuple(map(int, np.flatnonzero(m[:, -1])))) for m in holes]
            orders = _chain_orders(signatures)
            if len(orders) == 1:
                if len(partition["panels"]) * partition["panels"][0].shape[axis] > 30:
                    continue
                result = dict(partition)
                result.update(axis=axis, recovered=recovered,
                              signatures=signatures, order=orders[0])
                answers.append(result)
            elif len(orders) > 1:
                raise ValueError("interchangeable plates give multiple complete chains")
    if len(answers) != 1:
        raise ValueError("input must determine one partition, axis, and complete order")
    return answers[0]


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for y in range(3):
        for x in range(3):
            result |= p[y:y + h, x:x + w]
    return result


def _assembled(decoded, order=None):
    order = decoded["order"] if order is None else order
    axis = decoded["axis"]
    bases = [decoded["recovered"][i][0] for i in order]
    holes = [decoded["recovered"][i][1] for i in order]
    base, missing = np.concatenate(bases, axis=axis), np.concatenate(holes, axis=axis)
    coverage = np.zeros_like(base, dtype=np.int16)
    for slot, mask in enumerate(holes):
        placed = np.zeros_like(base)
        length = mask.shape[axis]
        slices = [slice(None), slice(None)]
        slices[axis] = slice(slot * length, (slot + 1) * length)
        placed[tuple(slices)] = mask
        coverage += _dilate(placed)  # One vote per source plate, not per hole.
    output = np.where(base, decoded["solid"], decoded["field"])
    output[(coverage == 1) & ~missing] = HALO
    output[(coverage >= 2) & ~missing] = decoded["divider"]
    output[missing] = decoded["field"]
    return output.astype(np.int64), base, missing, coverage


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from input pixels, with no training, seed, or generation state."""
    return _assembled(_decode(_grid(input_grid)))[0]


def _kind(signature):
    if len(signature) == 1:
        return "single"
    return "thick" if signature[-1] - signature[0] + 1 == len(signature) else "separated"


def _row_classes(signature, h):
    return {min(r, h - 1 - r) for r in signature}


def _make_links(n, h, rich, rng, allow_repeats=True, require_revisit=False):
    rows = [r for r in range(1, h - 1) if r != h - 1 - r]
    pool = [(r,) for r in rows]
    if rich:
        pool += [s for s in combinations(rows, 2) if len(_row_classes(s, h)) == 2]
    if require_revisit:
        assert n == 4 and not rich and allow_repeats
        a = _choose(rng, pool)
        b = _choose(rng, [s for s in pool if not (_row_classes(s, h) & _row_classes(a, h))])
        return [a, b, a]  # The excursion must precede the final terminal.
    required = {"single", "thick", "separated"} if rich else {"single"}
    for _ in range(300):
        kinds = sorted(required) + [_kind(_choose(rng, pool))
                                    for _ in range(n - 1 - len(required))]
        rng.shuffle(kinds)
        links = []
        for kind in kinds:
            options = [s for s in pool if _kind(s) == kind and
                       (not links or not (_row_classes(s, h) & _row_classes(links[-1], h)))]
            if not options:
                break
            links.append(_choose(rng, options))
        if len(links) != n - 1:
            continue
        if not allow_repeats and len(set(links)) != len(links):
            continue
        sigs = list(zip([()] + links, links + [()]))
        if len(_chain_orders(sigs)) == 1:
            return links
    raise RuntimeError("could not sample a uniquely ordered sparse edge-key chain")


def _layouts(n, h, w):
    return [(r, n // r) for r in range(1, n + 1) if n % r == 0
            and r * (h + 1) - 1 <= 30 and (n // r) * (w + 1) - 1 <= 30]


def _configuration(n, witness, rng):
    options = []
    for h in range(10 if witness == 2 else 8, 15):
        for w in range(6 if witness else 4, min(12, 30 // n) + 1):
            layouts = _layouts(n, h, w)
            if layouts:
                options.append((h, w, layouts))
    return _choose(rng, options)


def _make_base(h, w, incoming, outgoing, style, rng):
    base = np.zeros((h, w), dtype=bool)
    if style == 0:
        thickness = int(rng.integers(2, max(3, h - 3)))
        top = (h - thickness) // 2
        base[top:top + thickness] = True
    elif style == 1:
        # Orthogonal arms of independently varying thickness.
        t = int(rng.integers(1, max(2, h // 2 - 1)))
        base[t:h - t] = True
        l = int(rng.integers(1, max(2, w // 2)))
        base[1:h - 1, l:w - l] = True
    elif style == 2:
        # A cavity, with optional central island, preserves reflected geometry.
        base[1:h - 1] = True
        t = int(rng.integers(2, max(3, h // 2)))
        l = int(rng.integers(1, max(2, w // 2)))
        base[t:h - t, l:w - l] = False
        if h - 2 * t >= 4 and w - 2 * l >= 3 and rng.random() < .5:
            base[t + 1:h - t - 1, l + 1:w - l - 1] = True
    else:
        # Unions of reflected rectangles can make separated blocks and steps.
        for _ in range(int(rng.integers(1, 5))):
            y = int(rng.integers(1, h - 2))
            v = int(rng.integers(y + 1, h))
            x = int(rng.integers(0, w - 1))
            u = int(rng.integers(x + 1, w + 1))
            base[y:v, x:u] = True
    # Each port is supported by a long intended band, not a tiny solid speck.
    for r in incoming + outgoing:
        lo = max(1, r - int(rng.integers(2)))
        hi = min(h - 1, r + 1 + int(rng.integers(2)))
        base[lo:hi] = True
    base = _symmetrize(base)
    base[[0, -1]] = False
    return base


_MOTIFS = (((0, 0),), ((0, 0), (1, 0)), ((0, 0), (0, 1)),
           ((0, 0), (1, 1)), ((0, 0), (1, 0), (1, 1)),
           ((0, 0), (2, 0)))


def _interior_holes(h, w, used, witness, rng):
    selected = []

    def accept(points):
        keys = [_orbit_key(r, c, h, w) for r, c in points]
        if (len(set(keys)) != len(keys) or any(k in used for k in keys)
                or any(len(_orbit(r, c, h, w)) != 4 for r, c in points)):
            return False
        selected.extend(points)
        used.update(keys)
        return True

    if witness:
        points = [(2, 1), (3, 1)] + ([(3, 2)] if witness == 2 else [])
        if rng.random() < .5:
            points = [(h - 1 - r, c) for r, c in points]
        if rng.random() < .5:
            points = [(r, w - 1 - c) for r, c in points]
        assert accept(points)
    wanted = int(rng.integers(1, 5))
    for _ in range(35):
        if len(selected) >= wanted:
            break
        motif = _choose(rng, _MOTIFS)
        y, x = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        points = [(y + dy, x + dx) for dy, dx in motif]
        if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in points):
            continue
        accept(points)
    return selected


def _make_plate(h, w, incoming, outgoing, colors, style, witness, rng):
    field, solid, _ = colors
    holes = np.zeros((h, w), dtype=bool)
    for r in incoming:
        holes[r, 0] = True
    for r in outgoing:
        holes[r, -1] = True
    used = {_orbit_key(r, c, h, w) for r, c in np.argwhere(holes)}
    assert len(used) == int(holes.sum())
    interior = _interior_holes(h, w, used, witness, rng)
    base = _make_base(h, w, incoming, outgoing, style, rng)
    # Surround each interior defect with a supported rectangular patch.
    for r, c in interior:
        base[max(1, r - 1):min(h - 1, r + 2), max(0, c - 1):min(w, c + 2)] = True
        holes[r, c] = True
    base = _symmetrize(base)
    extras = np.zeros_like(base)
    candidates = [(r, c) for r, c in np.argwhere(~base)
                  if len(_orbit(int(r), int(c), h, w)) == 4
                  and not (r in (0, h - 1) and c in (0, w - 1))
                  and _orbit_key(r, c, h, w) not in used]
    rng.shuffle(candidates)
    # Teaching witnesses must visibly erase a speck outside all local halos.
    if witness:
        far = [(r, c) for r, c in candidates if not _dilate(holes)[r, c]]
        if not far:
            raise ValueError("resample a plate with a visible far speck")
        r, c = far[0]
        extras[r, c] = True
        used.add(_orbit_key(r, c, h, w))
    wanted = int(rng.integers(0 if not witness else 1, 5))
    for r, c in candidates:
        if int(extras.sum()) >= wanted:
            break
        key = _orbit_key(r, c, h, w)
        if key not in used:
            extras[r, c] = True
            used.add(key)
    panel = np.where(base, solid, field).astype(np.int64)
    panel[holes], panel[extras] = field, solid
    return panel, base, holes, extras


def _layout_grid(panels, permutation, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    result = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int64)
    for slot, i in enumerate(permutation):
        y, x = divmod(slot, nc)
        result[y * (h + 1):y * (h + 1) + h,
               x * (w + 1):x * (w + 1) + w] = panels[i]
    return result


def _constructed_output(records, field, solid, divider):
    """Labels from planned masks, using explicit neighbor sets and source IDs."""
    h, w = records[0][1].shape
    output = np.concatenate([np.where(rec[1], solid, field) for rec in records], axis=1)
    missing, owners = set(), {}
    for i, rec in enumerate(records):
        for r, c in np.argwhere(rec[2]):
            r, c = int(r), int(c) + i * w
            missing.add((r, c))
            for y in range(max(0, r - 1), min(h, r + 2)):
                for x in range(max(0, c - 1), min(output.shape[1], c + 2)):
                    owners.setdefault((y, x), set()).add(i)
    for point, sources in owners.items():
        output[point] = divider if len(sources) > 1 else HALO
    for point in missing:
        output[point] = field
    return output.astype(np.int64)


def _generate_pair(n, witness, rich, axis, colors, rng,
                   allow_repeats=True, require_revisit=False):
    for _ in range(80):
        h, w, layouts = _configuration(n, witness, rng)
        links = _make_links(n, h, rich, rng, allow_repeats, require_revisit)
        try:
            records = [_make_plate(h, w, () if i == 0 else links[i - 1],
                                   () if i == n - 1 else links[i], colors,
                                   int(rng.integers(4)), witness if i == 0 else 0, rng)
                       for i in range(n)]
        except ValueError:
            continue
        # Check all construction masks against the input-only recovery path.
        for panel, base, holes, extras in records:
            decoded = _recover(panel, colors[0], colors[1])
            assert all(np.array_equal(a, b) for a, b in zip(decoded, (base, holes, extras)))
        panels = [rec[0] for rec in records]
        output = _constructed_output(records, *colors)
        if axis == 0:
            panels = [p.T.copy() for p in panels]
            output = output.T.copy()
            layouts = [(c, r) for r, c in layouts]
        permutation = list(map(int, rng.permutation(n)))
        layout = _choose(rng, layouts)
        strip = (1, n) if axis == 1 else (n, 1)
        if permutation == list(range(n)) and (witness or layout == strip):
            permutation[0], permutation[-1] = permutation[-1], permutation[0]
        inp = _layout_grid(panels, permutation, layout, colors[2])
        pair = {"input": inp, "output": output}
        assert np.array_equal(solve(inp), output)
        return pair
    raise RuntimeError("could not create the required training evidence")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = [c for c in range(10) if c != HALO]
    fixed_colors = tuple(map(int, rng.choice(palette, 3, replace=False)))
    local_palette = bool(rng.random() < .65)
    revisits = bool(rng.integers(2))
    axes = list(map(int, rng.permutation([0, 1])))
    plans = [(4, 1, False, axes[0]), (4, 2, True, axes[1])]
    plans += [(int(rng.integers(2, 5)), 0, bool(rng.integers(2)), int(rng.integers(2)))
              for _ in range(num_train - 2)]
    # Two-panel pairs have one link, three-panel pairs two: no unsupported rich
    # category recipe is imposed on them. Both anchors already teach all kinds.
    plans = [(n, witness, rich and n >= 4, axis) for n, witness, rich, axis in plans]
    rng.shuffle(plans)
    plans += [(int(rng.integers(5, 7)), 0, bool(rng.integers(2)), int(rng.integers(2)))
              for _ in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for index, (n, witness, rich, axis) in enumerate(plans):
        for _ in range(40):
            colors = (tuple(map(int, rng.choice(palette, 3, replace=False)))
                      if local_palette else fixed_colors)
            pair = _generate_pair(n, witness, rich, axis, colors, rng,
                                  revisits, revisits and witness == 1)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if index < num_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not sample distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(decoded):
    output, base, missing, coverage = _assembled(decoded)
    n, axis = len(decoded["panels"]), decoded["axis"]
    h, w = decoded["panels"][0].shape
    far_extras, solo_overlap = 0, False
    for slot, index in enumerate(decoded["order"]):
        _, holes, extras = decoded["recovered"][index]
        far_extras += int((extras & ~_dilate(holes)).sum())
        votes = np.zeros_like(holes, dtype=np.int16)
        for r, c in np.argwhere(holes):
            votes[max(0, r - 1):r + 2, max(0, c - 1):c + 2] += 1
        length = holes.shape[axis]
        slices = [slice(None), slice(None)]
        slices[axis] = slice(slot * length, (slot + 1) * length)
        solo_overlap |= bool(((votes >= 2) & ~holes
                              & (coverage[tuple(slices)] == 1)).any())
    return dict(n=n, axis=axis, tile_shape=(h, w), layout=decoded["layout"],
                order=decoded["order"], colors=(decoded["field"], decoded["solid"], decoded["divider"]),
                holes=int(missing.sum()), extras=sum(int(r[2].sum()) for r in decoded["recovered"]),
                far_extras=far_extras, solo_overlap=solo_overlap,
                shared_cells=int(((coverage >= 2) & ~missing).sum()),
                ordinary_cells=int(((coverage == 1) & ~missing).sum()),
                kinds={_kind(b) for a, b in decoded["signatures"] if b},
                repeated=len({b for a, b in decoded["signatures"] if b}) < n - 1,
                base_key=_key(base), output_shape=output.shape)


def validate_generated_puzzle(puzzle):
    """Check schema, visible recovery, joint semantics, and training sufficiency."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            decoded = _decode(inp)
            prediction, base, missing, coverage = _assembled(decoded)
            assert np.array_equal(prediction, out), "wrong assembled output"
            assert np.all(out[missing] == decoded["field"]), "hole centers must survive every halo"
            shared = (coverage >= 2) & ~missing
            solo = (coverage == 1) & ~missing
            assert shared.any() and solo.any()
            assert np.array_equal(out == decoded["divider"], shared), "divider marks only shared halos"
            assert np.array_equal(out == HALO, solo), "ordinary halos have one source plate"
            unmarked = (coverage == 0) & ~missing
            clean = np.where(base, decoded["solid"], decoded["field"])
            assert np.array_equal(out[unmarked], clean[unmarked])
            features[split].append(_features(decoded))
    train, test = features["train"], features["test"]
    assert any(f["far_extras"] for f in train), "show observable speck erasure"
    assert any(f["solo_overlap"] for f in train), "show merging within a plate"
    assert {"single", "thick", "separated"} <= set().union(*(f["kinds"] for f in train))
    assert {f["axis"] for f in train} == {0, 1}, "teach both axes"
    assert max(f["n"] for f in train) < min(f["n"] for f in test), "longer-chain test shift"
    assert not any(f["repeated"] for f in test) or any(f["repeated"] for f in train), "teach revisits"
    cardinality_witness = False
    for pair in puzzle["train"]:
        d = _decode(_grid(pair["input"]))
        counts = [(tuple([len(a)]) if a else (), tuple([len(b)]) if b else ())
                  for a, b in d["signatures"]]
        cardinality_witness |= len(_chain_orders(counts)) > 1
    assert cardinality_witness, "positions must distinguish a size-only matching shortcut"
    for pair in puzzle["train"]:
        decoded = _decode(_grid(pair["input"]))
        if decoded["order"] != tuple(range(len(decoded["panels"]))):
            assert not np.array_equal(_assembled(decoded, tuple(range(len(decoded["panels"]))))[0],
                                      _grid(pair["output"])), "reading-order shortcut must fail"
            break
    else:
        raise AssertionError("training must demonstrate geometric reordering")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural audit plus concrete negative and symmetry checks."""
    signatures, records, shapes, bases, palettes = set(), [], set(), set(), set()
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                decoded = _decode(pair["input"])
                item = _features(decoded)
                item["split"] = split
                records.append(item)
                local.append(item)
                shapes.add(pair["input"].shape)
                bases.add(item["base_key"])
                palettes.add(item["colors"])
                # No-A ablation: intact raw-panel assembly, with the true order
                # generously supplied, still misses cleanup and both halos.
                raw = np.concatenate([decoded["panels"][i] for i in decoded["order"]],
                                     axis=decoded["axis"])
                assert not np.array_equal(raw, pair["output"])
                # All-field edge matching falsely treats true margins as keys;
                # it has no terminal because every edge includes field corners.
                p = [p if decoded["axis"] == 1 else p.T for p in decoded["panels"]]
                white = [(tuple(np.flatnonzero(a[:, 0] == decoded["field"])),
                          tuple(np.flatnonzero(a[:, -1] == decoded["field"]))) for a in p]
                assert not _chain_orders(white)
                positive = [(tuple(np.flatnonzero(a[:, 0] == decoded["solid"])),
                             tuple(np.flatnonzero(a[:, -1] == decoded["solid"]))) for a in p]
                assert not _chain_orders(positive)
                # A local per-tile halo cannot yield the divider-colored overlap.
                local_marked = []
                for i in decoded["order"]:
                    b, holes, _ = decoded["recovered"][i]
                    tile = np.where(b, decoded["solid"], decoded["field"])
                    tile[_dilate(holes) & ~holes] = HALO
                    local_marked.append(tile)
                assert not np.array_equal(np.concatenate(local_marked, axis=decoded["axis"]),
                                          pair["output"])
        assert len({f["tile_shape"] for f in local}) >= 2
        if seed == 0:
            assert _puzzle_key(generate_puzzle(seed)) == _puzzle_key(puzzle)
            for pair in puzzle["train"] + puzzle["test"]:
                for transform in (lambda a: a.T.copy(), lambda a: a[::-1].copy(),
                                  lambda a: a[:, ::-1].copy()):
                    inp = transform(pair["input"])
                    expected = transform(pair["output"])
                    # Reflection of a chain reverses its *visible* terminal
                    # directions along that axis, so transformed output agrees.
                    assert np.array_equal(solve(inp), expected)
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(bases) >= 160 and len(palettes) >= 25
    assert {r["n"] for r in records} == {2, 3, 4, 5, 6}
    assert len({r["tile_shape"] for r in records}) >= 30
    assert len({r["holes"] for r in records}) >= 12
    assert len({r["extras"] for r in records}) >= 8
    assert any(r["repeated"] for r in records)
    assert {(2, 2), (2, 3), (3, 2)} <= {r["layout"] for r in records}
    assert max(max(r["output_shape"]) for r in records) == 30
    assert all({0, 1} == {r["axis"] for r in records if r["split"] == split}
               for split in ("train", "test"))
    # Equal-cardinality signatures are not interchangeable. Repeated signatures
    # are legal only when the entire directed chain is unique.
    a, b = (1,), (2,)
    assert len(_chain_orders([((), a), (a, b), (b, a), (a, ())])) == 1
    assert len(_chain_orders([((), a), (a, b), (b, a), (a, b), (b, ())])) == 2
    # A 2:2 orbit is ambiguous; undamaged two-cell axis orbits cannot be flipped.
    tied = np.zeros((8, 6), dtype=int)
    tied[2, 1] = tied[2, 4] = 1
    try:
        _recover(tied, 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("tied orbit accepted")
    axis_tie = np.zeros((8, 5), dtype=int)
    axis_tie[2, 2] = 1
    try:
        _recover(axis_tie, 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged two-cell axis orbit accepted")
    # A held-output mutation must fail semantic validation.
    damaged = generate_puzzle(0, 2, 1)
    d = _decode(damaged["test"][0]["input"])
    _, _, holes, _ = _assembled(d)
    y, x = np.argwhere(holes)[0]
    damaged["test"][0]["output"][y, x] = HALO
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("recolored missing center accepted")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(42, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, reconstruction, assembly, halo semantics, and variation")
