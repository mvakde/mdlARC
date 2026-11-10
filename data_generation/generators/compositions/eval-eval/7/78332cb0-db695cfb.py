"""Beam relay: latent perpendicular rays jointly choose an all-panel assembly.

All semantics are recovered from the input. Generation metadata is never passed
to solve. Only standard-library modules and NumPy are used.
"""
from itertools import permutations
import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _points(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _span(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert dr and abs(dr) == abs(dc)
    sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
    return tuple((a[0] + k * sr, a[1] + k * sc)
                 for k in range(abs(dr) + 1))


def _ray(point, slope, shape):
    """Slope is change in row per positive unit of column (both are +/-1)."""
    r, c = point
    h, w = shape
    return tuple((r + slope * (x - c), x) for x in range(w)
                 if 0 <= r + slope * (x - c) < h)


def _beam(point, slope, shape):
    cells = _ray(point, slope, shape)
    h, w = shape
    return dict(point=point, slope=slope, cells=cells,
                H=(tuple(r for r, c in cells if c == 0),
                   tuple(r for r, c in cells if c == w - 1)),
                V=(tuple(c for r, c in cells if r == 0),
                   tuple(c for r, c in cells if r == h - 1)))


def _partition(a):
    found = []
    h, w = a.shape
    for divider in map(int, np.unique(a)):
        rows = list(map(int, np.flatnonzero(np.all(a == divider, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(a == divider, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        heights = [y - x - 1 for x, y in zip(rr, rr[1:])]
        widths = [y - x - 1 for x, y in zip(cc, cc[1:])]
        if (min(heights + widths) < 4 or len(set(heights)) != 1
                or len(set(widths)) != 1):
            continue
        n = len(heights) * len(widths)
        if not 2 <= n <= 6:
            continue
        panels = [a[rr[i] + 1:rr[i + 1], cc[j] + 1:cc[j + 1]].copy()
                  for i in range(len(heights)) for j in range(len(widths))]
        if any(np.any(p == divider) for p in panels):
            continue
        backgrounds = []
        for p in panels:
            values, counts = np.unique(p, return_counts=True)
            if counts.max() * 2 <= p.size:
                break
            backgrounds.append(int(values[np.argmax(counts)]))
        if len(backgrounds) != n or len(set(backgrounds)) != 1:
            continue
        bg = backgrounds[0]
        if len(set(map(int, np.unique(a))) - {bg, divider}) != 2:
            continue
        found.append(dict(panels=panels, background=bg, divider=divider,
                          layout=(len(heights), len(widths)), shape=panels[0].shape))
    assert len(found) == 1, "one regular divider partition is required"
    return found[0]


def _chain(candidates, axis, limit=2):
    """Find source choices AND order together, stopping after ambiguity is proven."""
    n, answers = len(candidates), []

    def visit(path, remaining, outgoing):
        if len(answers) >= limit:
            return
        if not remaining:
            if not outgoing:
                answers.append(tuple(path))
            return
        if not outgoing:
            return
        for i in remaining:
            for j, beam in enumerate(candidates[i]):
                incoming, next_out = beam[axis]
                if incoming == outgoing:
                    visit(path + [(i, j)], tuple(k for k in remaining if k != i), next_out)

    for i, choices in enumerate(candidates):
        for j, beam in enumerate(choices):
            incoming, outgoing = beam[axis]
            if not incoming and outgoing:
                visit([(i, j)], tuple(k for k in range(n) if k != i), outgoing)
            if len(answers) >= limit:
                return answers
    return answers


def _roles(parsed, primary, secondary):
    spans, choices, inactive = [], [], []
    for p in parsed["panels"]:
        ends = sorted(_points(p, primary))
        assert len(ends) == 2
        span = _span(*ends)
        assert len(span) >= 3
        source_dots = _points(p, secondary)
        eligible = sorted(source_dots & set(span[1:-1]))
        assert eligible, "every panel needs an eligible source"
        assert all(0 < r < p.shape[0] - 1 and 0 < c < p.shape[1] - 1
                   for r, c in eligible), "eligible sources start inside panels"
        slope = 1 if (ends[1][0] - ends[0][0]) * (ends[1][1] - ends[0][1]) > 0 else -1
        spans.append(span)
        choices.append([_beam(point, -slope, p.shape) for point in eligible])
        inactive.append(source_dots - set(span))
    return spans, choices, inactive


def _decode(input_grid):
    a = _grid(input_grid)
    parsed = _partition(a)
    markers = sorted(set(map(int, np.unique(a))) - {parsed["background"], parsed["divider"]})
    answers = []
    for primary, secondary in (markers, markers[::-1]):
        try:
            spans, candidates, inactive = _roles(parsed, primary, secondary)
        except AssertionError:
            continue
        for axis in ("H", "V"):
            for path in _chain(candidates, axis):
                answers.append((primary, secondary, spans, candidates, inactive, axis, path))
    assert len(answers) == 1, "role, axis, order and source selection must be jointly unique"
    primary, secondary, spans, candidates, inactive, axis, path = answers[0]
    canonical = parsed["shape"] if axis == "H" else parsed["shape"][::-1]
    assert canonical[0] > canonical[1] >= 4
    assert canonical[0] <= 24
    selected = [candidates[i][j] for i, j in path]
    assert all(len(b["cells"]) >= 3 for b in selected)
    transverse = parsed["shape"][0 if axis == "H" else 1]
    assert all(all(0 < pos < transverse - 1 for pos in b[axis][1]) for b in selected[:-1])
    parsed.update(primary=primary, secondary=secondary, spans=spans,
                  candidates=candidates, inactive=inactive, axis=axis, path=path)
    return parsed


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    assert nr * nc == len(panels)
    a = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    assert all(d <= 30 for d in a.shape)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        a[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panels[i]
    return a


def _completed(parsed, all_sources=False, no_geometry=False, all_dots=False):
    panels = [p.copy() for p in parsed["panels"]]
    if no_geometry:
        return panels
    for i, p in enumerate(panels):
        for point in parsed["spans"][i]:
            if p[point] != parsed["secondary"]:
                p[point] = parsed["primary"]
    for i, j in parsed["path"]:
        beams = parsed["candidates"][i] if all_sources else [parsed["candidates"][i][j]]
        if all_dots:
            beams = beams + [_beam(point, beams[0]["slope"], panels[i].shape)
                             for point in sorted(parsed["inactive"][i])]
        for beam in beams:
            for point in beam["cells"]:
                # Hypothetical all-dot rays must still preserve original endpoints.
                if panels[i][point] != parsed["primary"] or not all_dots:
                    panels[i][point] = parsed["secondary"]
    return panels


def _render(parsed, panels=None):
    panels = _completed(parsed) if panels is None else panels
    n = len(panels)
    order = [i for i, j in parsed["path"]]
    shape = (1, n) if parsed["axis"] == "H" else (n, 1)
    output = _layout(panels, order, shape, parsed["divider"])
    h, w = panels[0].shape
    for k, (i, j) in enumerate(parsed["path"][:-1]):
        for pos in parsed["candidates"][i][j][parsed["axis"]][1]:
            if parsed["axis"] == "H":
                output[pos, (k + 1) * (w + 1) - 1] = parsed["secondary"]
            else:
                output[(k + 1) * (h + 1) - 1, pos] = parsed["secondary"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every required role and choice from this input only."""
    return _render(_decode(input_grid))


def _geometric_paths(n, h, w):
    """Uncolored horizontal beam paths, not completed parent motifs."""
    d, result = w - 1, []

    def finish(path, row):
        if len(path) == n - 1:
            for slope in (-1, 1):
                if not 0 <= row + slope * d < h:
                    beam = _beam((row, 0), slope, (h, w))
                    if len(beam["cells"]) >= 3:
                        result.append(path + [beam])
            return
        for slope in (-1, 1):
            next_row = row + slope * d
            if not 0 < next_row < h - 1:
                continue
            beam = _beam((row, 0), slope, (h, w))
            if any(b["H"] == beam["H"] for b in path):
                continue
            finish(path + [beam], next_row)

    for row in range(1, h - 1):
        for slope in (-1, 1):
            if not 0 <= row - slope * d < h:
                beam = _beam((row, d), slope, (h, w))
                if len(beam["cells"]) >= 3:
                    finish([beam], row)
    return result


def _primary_options(beam, h, w, minimum):
    options = []
    s = -beam["slope"]
    for r, c in beam["cells"]:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        # Endpoints straddle the selected source; primary extent is independent
        # of how far the perpendicular beam travels.
        left = [(r - s * k, c - k) for k in range(1, c + 1) if 0 <= r - s * k < h]
        right = [(r + s * k, c + k) for k in range(1, w - c) if 0 <= r + s * k < h]
        for a in left:
            for b in right:
                span = _span(a, b)
                if len(span) >= minimum:
                    options.append(((r, c), a, b, span))
    return options


def _make_panels(rng, beams, shape, palette, required, cap, inert_witness, conservative=False):
    h, w = shape
    bg, divider, primary, secondary = palette
    eligible_forced = [i for i, b in enumerate(beams)
                       if _primary_options(b, h, w, required + 2)] if required > 1 else []
    if required > 1 and not eligible_forced:
        return None
    forced = _choose(rng, eligible_forced) if eligible_forced else -1
    panels, spans, chosen = [], [], []
    for i, beam in enumerate(beams):
        minimum = required + 2 if i == forced else 3
        options = _primary_options(beam, h, w, minimum)
        if inert_witness and i == 0:
            options = [x for x in options if any(
                len(_ray(q, beam["slope"], shape)) >= 2
                for q in set(_ray(x[0], -beam["slope"], shape)) - set(x[3]))]
        if not options:
            return None
        lengths = sorted({len(x[3]) for x in options})
        style = int(rng.integers(3))
        length = (lengths[-1] if conservative else lengths[0] if style == 0
                  else lengths[-1] if style == 1 else _choose(rng, lengths))
        source, a, b, span = _choose(rng, [x for x in options if len(x[3]) == length])
        p = np.full(shape, bg, dtype=np.int8)
        p[a] = p[b] = primary
        p[source] = secondary
        other = [q for q in span[1:-1] if q != source
                 and 0 < q[0] < h - 1 and 0 < q[1] < w - 1]
        minimum_count = required if i == forced else 1
        # A competing span may be fully marked; another panel still has to
        # demonstrate actual primary fill. Narrow six-panel relays can therefore
        # teach two choices instead of being excluded by an artificial size floor.
        visible_gaps = 2 if i == forced else 3
        maximum_count = min(cap, len(other) + 1, max(1, len(span) - visible_gaps))
        if maximum_count < minimum_count:
            return None
        count = minimum_count if conservative else int(rng.integers(minimum_count, maximum_count + 1))
        rng.shuffle(other)
        for q in other[:count - 1]:
            p[q] = secondary
        free = [(r, c) for r in range(h) for c in range(w)
                if (r, c) not in span]
        off_count = 0 if conservative else int(rng.choice((0, 0, 1, 2, 3)))
        if inert_witness and i == 0:
            off_count = max(off_count, 1)
            # A collinear dot beyond the finite endpoints distinguishes finite
            # gating from an infinite primary line, as well as from firing all dots.
            primary_line = set(_ray(source, -beam["slope"], shape))
            outside = [q for q in free if q in primary_line
                       and len(_ray(q, beam["slope"], shape)) >= 2]
            if not outside:
                return None
            q = _choose(rng, outside)
            p[q] = secondary
            free.remove(q)
            off_count -= 1
        rng.shuffle(free)
        for q in free[:off_count]:
            p[q] = secondary
        panels.append(p)
        spans.append(span)
        chosen.append(_beam(source, beam["slope"], shape))
    return panels, spans, chosen


def _config(rng, plan):
    minimum_n = 4 if plan["required"] == 2 else 3 if plan["required"] > 2 else 2
    ns = [n for n in plan["counts"] if n >= minimum_n]
    if plan["required"] == 2:
        # A simple walk returning to an end contact uses an even number of tiles.
        ns = [n for n in ns if n % 2 == 0]
    n = _choose(rng, ns)
    w_min = max(4, plan["required"] + 2)
    w_max = min(14, 31 // n - 1)
    if w_min > w_max:
        return None
    w = int(rng.integers(w_min, w_max + 1))
    h_max = 14 if plan["array"] else 24
    if w + 1 > h_max:
        return None
    # Choose count first and search shuffled heights for a compatible geometry;
    # rejection of impossible sizes must not silently suppress odd chain counts.
    for h in map(int, rng.permutation(np.arange(w + 1, h_max + 1))):
        paths = _geometric_paths(n, h, w)
        if plan["required"] == 2:
            paths = [p for p in paths if len({b["slope"] for b in p}) > 1
                     and len({b["H"][1] for b in p[:-1]}) < n - 1]
        if paths:
            return n, h, w, _choose(rng, paths)
    return None


def _assemble_pair(rng, plan, palette, config, conservative=False):
    n, h, w, beams = config
    made = _make_panels(rng, beams, (h, w), palette, plan["required"],
                        plan["cap"], plan["inert"], conservative)
    if made is None:
        return None
    panels, spans, chosen = made
    axis = plan["axis"]
    if axis == "V":
        panels = [p.T.copy() for p in panels]
        spans = [tuple((c, r) for r, c in span) for span in spans]
        chosen = [_beam((b["point"][1], b["point"][0]), b["slope"], (w, h)) for b in chosen]
    ph, pw = panels[0].shape
    layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
               and nr * (ph + 1) - 1 <= 30 and (n // nr) * (pw + 1) - 1 <= 30]
    if plan["array"]:
        layouts = [s for s in layouts if min(s) > 1]
    elif plan["strip_only"]:
        layouts = [s for s in layouts if 1 in s]
    if not layouts:
        return None
    layout = _choose(rng, layouts)
    order = list(map(int, rng.permutation(n)))
    if plan["required"] > 1 and (order == list(range(n)) or order == list(range(n))[::-1]):
        order = list(range(1, n)) + [0]
    elif order == list(range(n)):
        order.reverse()
    bg, divider, primary, secondary = palette
    inp = _layout(panels, order, layout, divider)
    construction = dict(panels=panels, spans=spans, primary=primary,
                        secondary=secondary, divider=divider, axis=axis,
                        candidates=[[b] for b in chosen],
                        path=tuple((i, 0) for i in range(n)))
    out = _render(construction)
    try:
        parsed = _decode(inp)
        if not np.array_equal(_render(parsed), out):
            return None
        features = _features(parsed)
    except AssertionError:
        return None
    if not features["primary_fill"]:
        return None
    if plan["required"] > 1 and not features["suppression"]:
        return None
    if plan["required"] == 2 and not (features["turns"] and features["revisit"]):
        return None
    if plan["inert"] and not features["inert_difference"]:
        return None
    return {"input": inp, "output": out}


def _fallback(rng, plan, palette, seen):
    """Finite conservative construction preserving counts/competition/layout shift."""
    if plan["array"]:
        n, w, h, links = 4, 5, 11, [3, 7, 3]
    elif plan["required"] >= 3:
        n, w, h, links = 4, 6, 12, [3, 8, 3]
    elif plan["required"] == 2:
        n, w, h, links = 4, 5, 11, [3, 7, 3]
    elif min(plan["counts"]) >= 5:
        n, w = 5, 4
        links = [2 + 3 * k for k in range(n - 1)]
        h = links[-1] + 3
    elif min(plan["counts"]) >= 4:
        n, w, h, links = 4, 5, 11, [3, 7, 3]
    else:
        n, w, h, links = 2, 5, 7, [3]
    d = w - 1
    start_slope = 1 if links[0] < d else -1
    beams = [_beam((links[0], d), start_slope, (h, w))]
    beams += [_beam((a, 0), 1 if b > a else -1, (h, w)) for a, b in zip(links, links[1:])]
    end_slope = 1 if links[-1] + d >= h else -1
    beams.append(_beam((links[-1], 0), end_slope, (h, w)))
    assert n in plan["counts"]
    primary, secondary = palette[2:]
    backgrounds = [c for c in range(10) if c not in (primary, secondary)]
    # Different role assignments provide >14 distinct fallback inputs, even if
    # all random permutations happen to coincide. No semantic role is relaxed.
    for bg, divider in permutations(backgrounds, 2):
        pair = _assemble_pair(rng, plan, (bg, divider, primary, secondary),
                              (n, h, w, beams), conservative=True)
        if pair is not None and _key(pair["input"]) not in seen:
            return pair
    raise AssertionError("constructive fallback failed")


def _check_counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)[:4]))
    mode = int(rng.integers(4))
    count_bands = ((tuple(range(2, 5)), (5, 6)),
                   (tuple(range(2, 5)), (3, 4)),
                   (tuple(range(2, 6)), (4, 6)),
                   (tuple(range(4, 7)), (2, 3)))
    witness_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    axes = list(rng.permutation(("H", "V")))
    variable_background = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            train = split == "train"
            plan = dict(counts=count_bands[mode][0 if train else 1],
                        required=2 if train and index == witness_slots[0] else 3 if not train and mode == 1 else 1,
                        cap=2 if train else 3, inert=train and index == witness_slots[1],
                        axis=axes[witness_slots.index(index)] if train and index in witness_slots else _choose(rng, ("H", "V")),
                        array=not train and mode == 2, strip_only=train and mode == 2)
            pair_palette = palette
            if variable_background:
                spare = [c for c in range(10) if c not in palette[2:]]
                bg, divider = map(int, rng.choice(spare, 2, replace=False))
                pair_palette = (bg, divider) + palette[2:]
            for _ in range(160):
                config = _config(rng, plan)
                if config is None:
                    continue
                pair = _assemble_pair(rng, plan, pair_palette, config)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                pair = _fallback(rng, plan, pair_palette, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(parsed):
    panels = parsed["panels"]
    completed = _completed(parsed)
    all_sources = _completed(parsed, all_sources=True)
    all_dots = _completed(parsed, all_sources=True, all_dots=True)
    primary_fill = sum(np.count_nonzero((q == parsed["primary"]) & (p == parsed["background"]))
                       for p, q in zip(panels, completed))
    suppression = sum(np.count_nonzero(a != b) for a, b in zip(completed, all_sources))
    inert_difference = sum(np.count_nonzero(a != b) for a, b in zip(all_sources, all_dots))
    selected_ranks, lengths, slopes, contacts = [], [], [], []
    beyond_endpoints = 0
    for i, j in parsed["path"]:
        selected_ranks.append((j, len(parsed["candidates"][i])))
        lengths.append(len(parsed["spans"][i]))
        b = parsed["candidates"][i][j]
        slopes.append(b["slope"])
        contacts.append(b[parsed["axis"]][1])
        primary_line = set(_ray(b["point"], -b["slope"], panels[i].shape))
        beyond_endpoints += len(parsed["inactive"][i] & primary_line)
    order = tuple(i for i, j in parsed["path"])
    n = len(panels)
    return dict(n=n, panel_shape=parsed["shape"], layout=parsed["layout"],
                axis=parsed["axis"], roles=(parsed["primary"], parsed["secondary"]),
                background=parsed["background"], divider=parsed["divider"],
                primary_fill=int(primary_fill), suppression=int(suppression),
                inert_difference=int(inert_difference),
                max_candidates=max(len(c) for c in parsed["candidates"]),
                candidates=tuple(len(c) for c in parsed["candidates"]),
                inactive=sum(map(len, parsed["inactive"])),
                beyond_endpoints=beyond_endpoints,
                ranks=tuple(selected_ranks), lengths=tuple(lengths), slopes=tuple(slopes),
                turns=len(set(slopes)) > 1,
                revisit=len(set(contacts[:-1])) < len(contacts) - 1,
                nontrivial_order=order not in (tuple(range(n)), tuple(range(n))[::-1]),
                order=order)


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    parsed = _decode(inp)
    assert np.array_equal(_render(parsed), out), "wrong beam relay output"
    n, axis = len(parsed["panels"]), parsed["axis"]
    ph, pw = parsed["shape"]
    completed = _completed(parsed)
    for k, (i, j) in enumerate(parsed["path"]):
        actual = out[:, k * (pw + 1):k * (pw + 1) + pw] if axis == "H" else out[k * (ph + 1):k * (ph + 1) + ph, :]
        raw = parsed["panels"][i]
        assert np.array_equal(actual, completed[i])
        visible = raw != parsed["background"]
        assert np.array_equal(actual[visible], raw[visible]), "preserve every original marker"
        primary_added = _points(actual, parsed["primary"]) - _points(raw, parsed["primary"])
        secondary_added = _points(actual, parsed["secondary"]) - _points(raw, parsed["secondary"])
        assert primary_added <= set(parsed["spans"][i])
        assert secondary_added <= set(parsed["candidates"][i][j]["cells"])
        if k < n - 1:
            seam = out[:, (k + 1) * (pw + 1) - 1] if axis == "H" else out[(k + 1) * (ph + 1) - 1, :]
            gate = parsed["candidates"][i][j][axis][1]
            assert tuple(map(int, np.flatnonzero(seam == parsed["secondary"]))) == gate
            assert np.all(np.isin(seam, (parsed["divider"], parsed["secondary"])))
    assert not np.array_equal(_render(parsed, _completed(parsed, no_geometry=True)), out), "B must change the answer"
    return parsed, _features(parsed)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            parsed, record = _check_pair(pair)
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert record["primary_fill"] > 0, "show finite segment completion"
            records[split].append(record)
    train, test = records["train"], records["test"]
    assert len({r["roles"] for r in train + test}) == 1
    assert {r["axis"] for r in train} == {"H", "V"}
    assert any(r["suppression"] > 0 for r in train), "A must suppress an otherwise eligible ray"
    assert any(r["inert_difference"] > 0 for r in train), "B's finite span must gate a dot"
    assert any(r["beyond_endpoints"] > 0 and r["inert_difference"] > 0 for r in train)
    assert any(r["n"] >= 3 and r["nontrivial_order"] for r in train)
    assert any(r["turns"] and r["revisit"] and r["suppression"] for r in train), "teach returning relays"
    grows = min(r["n"] for r in test) > max(r["n"] for r in train)
    shrinks = max(r["n"] for r in test) < min(r["n"] for r in train)
    competes = min(r["max_candidates"] for r in test) > max(r["max_candidates"] for r in train)
    arrays = all(1 in r["layout"] for r in train) and all(min(r["layout"]) > 1 for r in test)
    assert grows or shrinks or competes or arrays, "an observable train/test shift is required"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17))
    shapes, geometries, records = set(), set(), []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                parsed, record = _check_pair(pair)
                records.append(record)
                shapes.add(pair["input"].shape)
                roles = (parsed["background"], parsed["divider"], parsed["primary"], parsed["secondary"])
                normal = np.zeros_like(pair["input"])
                for i, c in enumerate(roles):
                    normal[pair["input"] == c] = i
                geometries.add(_key(normal))
                assert np.array_equal(solve(pair["input"].T), pair["output"].T)
    assert {r["n"] for r in records} == set(range(2, 7))
    assert {r["axis"] for r in records} == {"H", "V"}
    assert len(shapes) >= 35 and len(geometries) >= 160
    assert len({r["panel_shape"] for r in records}) >= 30
    assert len({r["roles"] for r in records}) >= 20
    assert {1, 2, 3} <= {r["max_candidates"] for r in records}
    assert any(r["turns"] for r in records) and any(r["revisit"] for r in records)
    assert any(j == 0 and m > 1 for r in records for j, m in r["ranks"])
    assert any(j == m - 1 and m > 1 for r in records for j, m in r["ranks"])
    assert any(0 < j < m - 1 for r in records for j, m in r["ranks"])
    assert len({length for r in records for length in r["lengths"]}) >= 6
    # Input-local palette invariance, including the dominant-background role.
    for pair in puzzles[0]["train"]:
        mapping = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(5000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = puzzles[0]["train"][0]
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
                 for s in ("train", "test")}
    corrupted["train"][0]["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Two indistinguishable middle beams are freely interchangeable: reject them.
    a = _beam((3, 0), 1, (12, 6))
    b = _beam((8, 0), -1, (12, 6))
    start = _beam((3, 5), 1, (12, 6))
    end = _beam((3, 0), -1, (12, 6))
    assert len(_chain([[start], [a], [b], [a], [b], [end]], "H")) == 2
    # Reject a full ambiguous input, not just its abstract port graph.
    shape = (8, 4)
    start = _beam((2, 3), 1, shape)
    a, b = _beam((2, 0), 1, shape), _beam((5, 0), -1, shape)
    end = _beam((2, 0), -1, shape)
    made = _make_panels(np.random.default_rng(81), [start, a, b, a, b, end],
                        shape, (0, 5, 1, 2), 1, 1, False, conservative=True)
    ambiguous = _layout(made[0], list(range(6)), (1, 6), 5)
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous input accepted")
    # Direct finite-span and opposite-slope arithmetic, separate from path search.
    assert _span((1, 5), (5, 1)) == ((1, 5), (2, 4), (3, 3), (4, 2), (5, 1))
    assert _ray((3, 3), 1, (7, 7)) == tuple((i, i) for i in range(7))
    # Exercise the finite emergency construction directly, including duplicate
    # handling. Its geometry is conservative; the general sampler is unrestricted.
    seen = set()
    for index in range(14):
        plan = dict(counts=(4, 5), required=2, cap=2, inert=True,
                    axis="H" if index % 2 == 0 else "V", array=False, strip_only=False)
        pair = _fallback(np.random.default_rng(100 + index), plan, (0, 5, 1, 2), seen)
        seen.add(_key(pair["input"]))
        _, feature = _check_pair(pair)
        assert feature["suppression"] and feature["inert_difference"] and feature["revisit"]
    for axis in ("H", "V"):
        for counts, required, array in (((3, 4), 3, False), ((4, 6), 1, True),
                                         ((5, 6), 1, False), ((4, 5, 6), 1, False),
                                         ((2, 3), 1, False)):
            plan = dict(counts=counts, required=required, cap=max(1, required),
                        inert=True, axis=axis, array=array, strip_only=False)
            pair = _fallback(np.random.default_rng(714), plan, (0, 5, 1, 2), set())
            _, feature = _check_pair(pair)
            assert feature["primary_fill"] and feature["beyond_endpoints"] and feature["inert_difference"]
            assert feature["n"] in counts
            if array:
                assert min(feature["layout"]) > 1
            if required > 1:
                assert feature["max_candidates"] >= required and feature["suppression"]


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, geometry/activation invariants and count checks)")
