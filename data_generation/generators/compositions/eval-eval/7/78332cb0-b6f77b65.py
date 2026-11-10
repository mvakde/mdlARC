"""Heal a fractured scaffold, remove commanded rods, and settle rigid beams.

The geometric edge signatures order equal panels. Dividers are cuts, not walls:
same-colored collinear fragments on a restored seam are one original rod.
All inputs encode their own background, divider, commands and unique ordering.
Only standard-library modules and NumPy are used; parents are never imported.
"""

from collections import Counter
from itertools import combinations
import numpy as np


# Rod: (top row, left column, height, width, color), in scene coordinates.
def _cells(rod):
    r, c, h, w, _ = rod
    return {(r + y, c + x) for y in range(h) for x in range(w)}


def _shift(rod, distance):
    return (rod[0] + distance, *rod[1:])


def _paint(height, width, background, rods):
    result = np.full((height, width), background, dtype=np.int8)
    occupied = set()
    for rod in rods:
        cells = _cells(rod)
        assert not cells & occupied, "rods must never overlap"
        assert all(0 <= r < height and 0 <= c < width for r, c in cells)
        occupied.update(cells)
        r, c, h, w, color = rod
        result[r:r + h, c:c + w] = color
    return result


def _pieces(scene, background):
    """Decode maximal monochrome components; the input forbids bent components.

    Rods in a legal healed input are fully visible. Touching rods have different
    colors. Thus no concealed endpoints or junction conventions are required.
    A single occupied cell is a rigid support block and needs no orientation.
    """
    pending = {tuple(map(int, p)) for p in np.argwhere(scene != background)}
    rods = []
    while pending:
        start = min(pending)
        pending.remove(start)
        color = int(scene[start])
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in pending and int(scene[q]) == color:
                    pending.remove(q)
                    component.add(q)
                    stack.append(q)
        rr, cc = zip(*component)
        r, c = min(rr), min(cc)
        h, w = max(rr) - r + 1, max(cc) - c + 1
        if min(h, w) != 1 or h * w != len(component):
            raise ValueError("a healed input must contain straight monochrome rods")
        rods.append((r, c, h, w, color))
    return rods


def _settle(rods, height):
    """Synchronous gravity, including stacks that move together."""
    result = list(rods)
    for _ in range(height * (len(result) + 1)):
        footprints = [_cells(p) for p in result]
        targets = [{(r + 1, c) for r, c in s} for s in footprints]
        moving = {i for i, p in enumerate(result) if p[0] + p[2] < height}
        while True:
            blocked = {i for i in moving if any(targets[i] & footprints[j]
                       for j in range(len(result)) if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, 1) if i in moving else p for i, p in enumerate(result)]
    raise AssertionError("gravity did not converge")


def _settle_distances(rods, height):
    """Construction-side calculation: move lowest pieces by maximal distances.

    This is a cross-check of our specified non-overlap physics, not an
    independent reference for either parent's unresolved source mechanics.
    """
    result = list(rods)
    for _ in range(height * (len(result) + 1)):
        changed = False
        for i in sorted(range(len(result)), key=lambda j: result[j][0] + result[j][2], reverse=True):
            obstacles = set().union(*(_cells(p) for j, p in enumerate(result) if j != i))
            distance = 0
            for d in range(1, height - result[i][0] - result[i][2] + 1):
                if _cells(_shift(result[i], d)) & obstacles:
                    break
                distance = d
            if distance:
                result[i] = _shift(result[i], distance)
                changed = True
        if not changed:
            return result
    raise AssertionError("event gravity did not converge")


def _orders(signatures):
    """Return up to two all-panel directed chains; repeated signatures are legal."""
    starts = [i for i, (left, right) in enumerate(signatures) if not left and right]
    ends = [i for i, (left, right) in enumerate(signatures) if left and not right]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(path, unused):
        if len(answers) == 2:
            return
        if not unused:
            if path[-1] == ends[0]:
                answers.append(tuple(path))
            return
        outgoing = signatures[path[-1]][1]
        if not outgoing:
            return
        for i in unused:
            if signatures[i][0] == outgoing:
                visit(path + [i], tuple(j for j in unused if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _signature(panel, background):
    return (tuple(map(int, np.flatnonzero(panel[:, 0] != background))),
            tuple(map(int, np.flatnonzero(panel[:, -1] != background))))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _decode(input_grid):
    grid = _grid(input_grid)
    if grid.shape[0] < 5:
        raise ValueError("missing control band or scaffold")
    background = int(grid[1, 0])
    if not np.all(grid[1] == background):
        raise ValueError("row 1 must be the blank background row")
    commands = tuple(map(int, grid[0, grid[0] != background]))
    if not 1 <= len(commands) <= 3 or len(set(commands)) != len(commands):
        raise ValueError("one to three distinct swatches required")
    region = grid[2:]
    candidates = []
    for divider in np.unique(region):
        divider = int(divider)
        if divider == background or divider in commands:
            continue
        row_cuts = list(map(int, np.flatnonzero(np.all(region == divider, axis=1))))
        col_cuts = list(map(int, np.flatnonzero(np.all(region == divider, axis=0))))
        if not row_cuts and not col_cuts:
            continue
        rows, cols = [-1] + row_cuts + [region.shape[0]], [-1] + col_cuts + [region.shape[1]]
        heights = [b - a - 1 for a, b in zip(rows, rows[1:])]
        widths = [b - a - 1 for a, b in zip(cols, cols[1:])]
        if (min(heights) < 4 or min(widths) < 3 or len(set(heights)) != 1
                or len(set(widths)) != 1):
            continue
        panels = [region[a + 1:b, c + 1:d].copy()
                  for a, b in zip(rows, rows[1:]) for c, d in zip(cols, cols[1:])]
        if not 2 <= len(panels) <= 6 or any(np.any(p == divider) for p in panels):
            continue
        signatures = [_signature(p, background) for p in panels]
        chains = _orders(signatures)
        if len(chains) != 1:
            continue
        order = chains[0]
        healed = np.concatenate([panels[i] for i in order], axis=1)
        if healed.shape[1] > 30 or healed.shape[0] + 2 > 30:
            continue
        try:
            rods = _pieces(healed, background)
        except ValueError:
            continue
        candidates.append(dict(background=background, commands=commands, divider=divider,
                               panels=panels, signatures=signatures, order=order,
                               layout=(len(heights), len(widths)), healed=healed, rods=rods,
                               band=grid[0].copy()))
    if len(candidates) != 1:
        raise ValueError("input must encode one unique panel chain and partition")
    return candidates[0]


def _with_band(scene, background, band):
    output = np.full((scene.shape[0] + 2, scene.shape[1]), background, dtype=np.int8)
    occupied = np.flatnonzero(band != background)
    if len(occupied) and occupied[-1] >= scene.shape[1]:
        raise ValueError("swatch falls outside restored width")
    output[0, occupied] = band[occupied]
    output[2:] = scene
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only this input: restore neighbors before erasing any port cells."""
    parsed = _decode(input_grid)
    survivors = [p for p in parsed["rods"] if p[4] not in parsed["commands"]]
    h, w = parsed["healed"].shape
    scene = _paint(h, w, parsed["background"], _settle(survivors, h))
    return _with_band(scene, parsed["background"], parsed["band"])


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _links(rng, n, pool, witness=False):
    if witness:
        links = list(pool)
        rng.shuffle(links)
        return links
    for _ in range(64):
        links = [_choice(rng, pool) for _ in range(n - 1)]
        if n >= 3 and len(set(links)) < 2:
            continue
        if len(_orders(list(zip([()] + links, links + [()])))) == 1:
            return links
    # One visit and at most one forced self-loop at each distinct signature.
    distinct = min(len(pool), n - 1)
    selected = list(pool)
    rng.shuffle(selected)
    selected = selected[:distinct]
    repeats = set(map(int, rng.choice(distinct, n - 1 - distinct, replace=False)))
    links = [s for i, s in enumerate(selected) for _ in range(2 if i in repeats else 1)]
    assert len(_orders(list(zip([()] + links, links + [()])))) == 1
    return links


def _beams(rng, n, width, links):
    """An occupied row across consecutive cuts defines ONE continuous beam."""
    rods = []
    for row in sorted({r for s in links for r in s}):
        active = [j for j, s in enumerate(links) if row in s]
        groups = []
        for j in active:
            if groups and groups[-1][-1] + 1 == j:
                groups[-1].append(j)
            else:
                groups.append([j])
        for group in groups:
            first, last = group[0], group[-1] + 1
            left = first * width + int(rng.integers(1, width - 1))
            right = last * width + int(rng.integers(1, width - 1))
            rods.append((row, left, 1, right - left + 1, -1))
    return rods


def _touch(a, b):
    aa, bb = _cells(a), _cells(b)
    return bool(aa & bb or any((r + dr, c + dc) in bb for r, c in aa
                              for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))


def _add_post(rods, row, col, bottom, color=-1):
    if bottom < row:
        return False
    post = (row, col, bottom - row + 1, 1, color)
    if any(_cells(post) & _cells(p) for p in rods):
        return False
    rods.append(post)
    return True


def _drop_extra(rng, rods, height, n, width):
    tile = int(rng.integers(n))
    if rng.random() < .65:
        length = int(rng.integers(2, width - 1))
        col = tile * width + int(rng.integers(1, width - length))
        probe = (1, col, 1, length, -1)
    else:
        length = int(rng.integers(1, min(5, height - 2) + 1))
        col = tile * width + int(rng.integers(1, width - 1))
        probe = (1, col, length, 1, -1)
    if any(_cells(probe) & _cells(p) for p in rods):
        return
    result = _settle_distances(rods + [probe], height)
    # Original supports must stay fixed; insertion must not change edge ports.
    assert result[:-1] == rods
    rods.append(result[-1])


def _scene(rng, port_size, n, height, width, witness=False):
    # Three separated heights produce three equally sized, geometrically
    # different signatures in either profile. Counting contacts cannot order them.
    level_options = [s for s in combinations(range(1, height - 3), 3)
                     if all(b - a >= 3 for a, b in zip(s, s[1:]))]
    levels = _choice(rng, level_options)
    pool = list(combinations(levels, port_size))
    links = _links(rng, n, pool, witness)
    rods = _beams(rng, n, width, links)
    palette = list(map(int, rng.permutation(10)))
    background, divider = palette[:2]
    colors = palette[2:]
    # All witness beams share a color. There is no color-to-panel-position code.
    beam_colors = colors[:1 if witness else int(rng.integers(1, 4))]
    rods = [(*p[:4], _choice(rng, beam_colors)) for p in rods]
    target_index = max(range(len(rods)), key=lambda i: (rods[i][0], rods[i][3]))
    target = rods[target_index]
    primary = (colors[1], colors[2]) if witness else ()
    if witness:
        candidates = [c for c in range(target[1], target[1] + target[3])
                      if 1 <= c % width <= width - 2]
        first_tile, last_tile = min(candidates) // width, max(candidates) // width
        x1 = _choice(rng, [c for c in candidates if c // width == first_tile])
        x2 = _choice(rng, [c for c in candidates if c // width == last_tile])
        assert first_tile != last_tile
        assert _add_post(rods, target[0] + 1, x1, height - 1, primary[0])
        assert _add_post(rods, target[0] + 1, x2, height - 1, primary[1])
        landing_x = _choice(rng, [c for c in candidates if c not in (x1, x2)])
        landing_length = int(rng.integers(1, height - target[0] - 1))
        assert _add_post(rods, height - landing_length, landing_x, height - 1, colors[3])
    # Lower shelves are processed first; new posts rest on any lower piece.
    # No arbitrary overlaps, no guessed endpoints, no hidden joints.
    beams = [p for p in rods if p[2] == 1 and p[3] > 1]
    for beam in sorted(beams, key=lambda p: p[0], reverse=True):
        if witness and beam == target:
            continue
        candidates = [c for c in range(beam[1], beam[1] + beam[3])
                      if 1 <= c % width <= width - 2]
        rng.shuffle(candidates)
        desired = int(rng.integers(1, min(3, len(candidates)) + 1))
        added = 0
        for col in candidates:
            below = [r for p in rods for r, c in _cells(p) if c == col and r > beam[0]]
            bottom = min(below) - 1 if below else height - 1
            if bottom == beam[0]:
                added += 1
            elif _add_post(rods, beam[0] + 1, col, bottom):
                added += 1
            if added >= desired:
                break
        assert added
    # Independently sampled internal shelves and uprights enrich support DAGs
    # without encoding an additional panel identifier or changing its ports.
    for _ in range(int(rng.integers(0, 5 if witness else 8))):
        _drop_extra(rng, rods, height, n, width)
    support_colors = colors[3:3 + int(rng.integers(2, 6))] if witness else colors[len(beam_colors):7]
    for i, rod in enumerate(rods):
        if rod[4] != -1:
            continue
        forbidden = {p[4] for p in rods[:i] if _touch(rod, p)}
        choices = [c for c in support_colors if c not in forbidden]
        if not choices:
            choices = [c for c in colors if c not in forbidden and c not in primary]
        if not choices:
            raise ValueError("uncolorable contact graph")
        rods[i] = (*rod[:4], _choice(rng, choices))
    if any(a[4] == b[4] and _touch(a, b) for i, a in enumerate(rods) for b in rods[:i]):
        raise ValueError("same-color input rods must be separate")
    scene = _paint(height, n * width, background, rods)
    if Counter(_pieces(scene, background)) != Counter(rods):
        raise ValueError("rod decomposition did not recover construction")
    assert _settle(rods, height) == rods
    actual = [_signature(scene[:, i * width:(i + 1) * width], background) for i in range(n)]
    assert actual == list(zip([()] + links, links + [()]))
    return dict(scene=scene, rods=rods, background=background, divider=divider,
                primary=primary, n=n, width=width, links=links)


def _layout_options(n, height, width):
    return [(rows, n // rows) for rows in range(1, n + 1) if n % rows == 0
            and rows * (height + 1) + 1 <= 30
            and (n // rows) * (width + 1) - 1 <= 30]


def _make_input(rng, data, commands, layout=None, permutation=None, positions=None):
    scene, bg, divider = data["scene"], data["background"], data["divider"]
    n, width = data["n"], data["width"]
    h = scene.shape[0]
    layout = layout or _choice(rng, _layout_options(n, h, width))
    if permutation is None:
        permutation = list(map(int, rng.permutation(n)))
        if permutation == list(range(n)):
            permutation = permutation[1:] + permutation[:1]
    rows, cols = layout
    grid = np.full((rows * (h + 1) + 1, cols * (width + 1) - 1), bg, dtype=np.int8)
    grid[2:] = divider
    for slot, i in enumerate(permutation):
        r, c = divmod(slot, cols)
        grid[2 + r * (h + 1):2 + r * (h + 1) + h,
             c * (width + 1):c * (width + 1) + width] = scene[:, i * width:(i + 1) * width]
    positions = positions if positions is not None else list(map(int, rng.choice(width, len(commands), replace=False)))
    for c, color in zip(positions, commands):
        grid[0, c] = color
    survivors = [p for p in data["rods"] if p[4] not in commands]
    out_scene = _paint(h, n * width, bg, _settle_distances(survivors, h))
    output = _with_band(out_scene, bg, grid[0])
    return {"input": grid, "output": output}


def _configuration(rng, n):
    options = [(h, w) for h in range(11, 21) for w in range(4, min(7, 30 // n) + 1)
               if _layout_options(n, h, w)]
    return _choice(rng, options)


def _commands(rng, data, test=False):
    rods, h = data["rods"], data["scene"].shape[0]
    present = sorted({p[4] for p in rods})
    absent = [c for c in range(10) if c not in present and c not in (data["background"], data["divider"])]
    candidates = [(c,) for c in present + absent[:1]]
    for size in (2, 3):
        options = list(combinations(present + absent[:1], size))
        if len(options) > 12:
            options = [options[int(i)] for i in rng.choice(len(options), 12, replace=False)]
        candidates.extend(options)
    effects = {"collapse": [], "harmless": [], "absent": []}
    for command in candidates:
        survivors = [p for p in rods if p[4] not in command]
        if not survivors:
            continue
        settled = _settle_distances(survivors, h)
        kind = "absent" if not set(command) & set(present) else "collapse" if settled != survivors else "harmless"
        # Test answers retain the composition's physical interaction, rather
        # than only use ports to position a pile of surviving posts.
        if test and (kind != "collapse" or not any(
                p[3] > 1 and p[1] // data["width"] != (p[1] + p[3] - 1) // data["width"]
                for p in survivors)):
            continue
        effects[kind].append(command)
    choices = [kind for kind, values in effects.items() if values]
    if not choices:
        raise ValueError("no command preserves a collapsing cross-panel scene")
    weights = np.array([{"collapse": .75 if test else .6, "harmless": .25, "absent": .1}[k] for k in choices])
    kind = str(rng.choice(choices, p=weights / weights.sum()))
    return _choice(rng, effects[kind])


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    port_size = int(rng.integers(1, 3))
    puzzle = {"train": [], "test": []}
    # Two command interventions on exactly one scene are the teaching core.
    for _ in range(80):
        h, w = _configuration(rng, 4)
        try:
            data = _scene(rng, port_size, 4, h, w, witness=True)
            layout = _choice(rng, _layout_options(4, h, w))
            permutation = list(map(int, rng.permutation(4)))
            if permutation == [0, 1, 2, 3]:
                permutation.reverse()
            positions = list(map(int, rng.choice(w, 2, replace=False)))
            single = _make_input(rng, data, data["primary"][:1], layout, permutation, positions[:1])
            both = _make_input(rng, data, data["primary"], layout, permutation, positions)
            records = [_features(p) for p in (single, both)]
            if (records[0]["moved"] == 0 and records[1]["moved"] > 0
                    and records[1]["short_landing"] and all(f["local_wrong"] for f in records)
                    and all(f["count_ambiguous"] for f in records)):
                puzzle["train"] = [single, both]
                break
        except ValueError:
            continue
    else:
        raise AssertionError("bounded teaching-scene construction failed")
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, remaining in (("train", num_train - 2), ("test", num_test)):
        for _ in range(remaining):
            for attempt in range(80):
                n = int(rng.integers(5, 7)) if split == "test" else int(rng.integers(2, 5))
                h, w = _configuration(rng, n)
                try:
                    data = _scene(rng, port_size, n, h, w)
                    pair = _make_input(rng, data, _commands(rng, data, split == "test"))
                    if split == "test" and not _features(pair)["local_wrong"]:
                        continue
                    if _key(pair["input"]) in seen:
                        continue
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise AssertionError("bounded distinct-scene construction failed")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _local_gravity(parsed):
    parts = []
    for i in parsed["order"]:
        panel = parsed["panels"][i]
        rods = [p for p in _pieces(panel, parsed["background"]) if p[4] not in parsed["commands"]]
        parts.append(_paint(*panel.shape, parsed["background"], _settle(rods, panel.shape[0])))
    return np.concatenate(parts, axis=1)


def _features(pair):
    parsed = _decode(pair["input"])
    rods, h = parsed["rods"], parsed["healed"].shape[0]
    commands = parsed["commands"]
    survivors = [p for p in rods if p[4] not in commands]
    settled = _settle(survivors, h)
    distances = [q[0] - p[0] for p, q in zip(survivors, settled)]
    bg = parsed["background"]
    scene = _paint(*parsed["healed"].shape, bg, settled)
    count_signatures = [(tuple(range(len(a))), tuple(range(len(b)))) for a, b in parsed["signatures"]]
    width = parsed["panels"][0].shape[1]
    crossing = [p for p in rods if p[3] > 1 and p[1] // width != (p[1] + p[3] - 1) // width]
    surviving_crossing = [p for p in survivors if p[3] > 1
                          and p[1] // width != (p[1] + p[3] - 1) // width]
    short_landing = False
    for before, after, distance in zip(survivors, settled, distances):
        if before[3] <= 1 or distance <= 0:
            continue
        below = {(r + 1, c) for r, c in _cells(after)}
        for p, q in zip(survivors, settled):
            if p[2] > 1 or p[3] == 1:
                if p == q and p[0] > before[0] + 1 and below & _cells(q):
                    short_landing = True
    support_edges = [(i, j) for i, p in enumerate(rods) for j, q in enumerate(rods)
                     if i != j and _cells(_shift(p, 1)) & _cells(q)]
    ordered_signatures = [parsed["signatures"][i] for i in parsed["order"]]
    joins = [b for a, b in ordered_signatures[:-1]]
    return dict(parsed=parsed, settled=settled, scene=scene, n=len(parsed["panels"]),
                panel_shape=parsed["panels"][0].shape, layout=parsed["layout"],
                port_size=max(map(len, joins)), repeated=len(set(joins)) < len(joins),
                pieces=len(rods), edges=len(support_edges), cross_beams=len(crossing),
                surviving_cross_beams=len(surviving_crossing),
                max_beam=max(p[3] for p in crossing), commands=len(commands),
                deleted=len(rods) - len(survivors), moved=sum(d > 0 for d in distances),
                distances=tuple(sorted(set(distances) - {0})), short_landing=short_landing,
                local_wrong=not np.array_equal(_local_gravity(parsed), scene),
                count_ambiguous=len(_orders(count_signatures)) > 1,
                background=bg, divider=parsed["divider"])


def validate_generated_puzzle(puzzle):
    """Validate actual returned grids, decoding no generation metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _features(pair)
            p = f["parsed"]
            h, w = p["healed"].shape
            assert out.shape == (h + 2, w)
            assert p["divider"] not in np.unique(out)
            assert np.all(out[1] == p["background"])
            assert np.array_equal(out, _with_band(f["scene"], p["background"], p["band"]))
            assert np.array_equal(solve(inp), out)
            assert _settle(p["rods"], h) == p["rods"], "input scaffold must be stable after healing"
            remaining = [rod for rod in p["rods"] if rod[4] not in p["commands"]]
            assert _settle_distances(remaining, h) == f["settled"]
            assert _settle(f["settled"], h) == f["settled"], "output must be stable"
            # Complete lengths, colors and horizontal positions survive. Fully
            # visible rods imply exact scene color-cell conservation on survivors.
            expected_counts = Counter()
            for rod in remaining:
                expected_counts[rod[4]] += rod[2] * rod[3]
            actual_counts = Counter(map(int, out[2:][out[2:] != p["background"]]))
            assert actual_counts == expected_counts
            assert all(q[0] >= rod[0] and q[1:] == rod[1:] for rod, q in zip(remaining, f["settled"]))
            assert f["cross_beams"] > 0
            assert p["order"] != tuple(range(f["n"])), "input panels are deliberately shuffled"
            records[split].append(f)
    train, test = records["train"], records["test"]
    assert any(f["count_ambiguous"] and f["local_wrong"] for f in train)
    assert any(f["moved"] and f["short_landing"] for f in train)
    assert any(f["deleted"] and f["moved"] == 0 for f in train)
    assert all(f["n"] > max(t["n"] for t in train) for f in test)
    assert {f["port_size"] for f in test} <= {f["port_size"] for f in train}
    assert all(f["moved"] and f["surviving_cross_beams"] and f["local_wrong"] for f in test)
    # The mandatory command intervention pair has identical scene, layout and
    # palette, with one command a proper subset of the other.
    contrasts = [(a, b) for i, a in enumerate(train) for j, b in enumerate(train)
                 if _key(np.asarray(puzzle["train"][i]["input"])[2:]) ==
                    _key(np.asarray(puzzle["train"][j]["input"])[2:])
                 and set(a["parsed"]["commands"]) < set(b["parsed"]["commands"])
                 and a["moved"] == 0 and b["moved"] > 0]
    assert contrasts, "training must isolate the redundant-support intervention"


def self_test():
    """Bounded mechanics, omission, API and diversity checks over 42 seeds."""
    # A beam rests on either of two legs, then drops onto a shorter third post.
    rods = [(3, 1, 1, 9, 1), (4, 2, 6, 1, 2), (4, 8, 6, 1, 3), (8, 5, 2, 1, 4)]
    assert _settle(rods, 10) == rods
    assert _settle([p for p in rods if p[4] != 2], 10)[0] == rods[0]
    free = [p for p in rods if p[4] not in (2, 3)]
    assert _settle(free, 10)[0][0] == 7
    assert _settle(free, 10) == _settle_distances(free, 10)
    assert len(_orders([((), (1,)), ((1,), (1,)), ((1,), ())])) == 1
    assert len(_orders([((), (1,)), ((1,), (1,)), ((1,), (1,)), ((1,), ())])) == 2
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, records, masks, shapes = set(), [], set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(p["input"]), _key(p["output"]))
                             for split in ("train", "test") for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                f["split"] = split
                records.append(f)
                shapes.add(np.asarray(pair["input"]).shape)
                masks.add(_key((f["parsed"]["healed"] != f["background"]).astype(np.int8)))
                assert np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"])
        # Both concepts have directly observable answer-changing witnesses.
        assert any(_features(p)["local_wrong"] for p in puzzle["train"])
        assert any(not np.array_equal(_decode(p["input"])["healed"], np.asarray(p["output"])[2:])
                   for p in puzzle["train"])
    assert len(signatures) == 42
    assert len(shapes) >= 15 and len(masks) >= 150
    assert {f["port_size"] for f in records} == {1, 2}
    assert {f["n"] for f in records} == {2, 3, 4, 5, 6}
    assert len({f["pieces"] for f in records}) >= 10
    assert len({f["edges"] for f in records}) >= 10
    assert len({f["max_beam"] for f in records}) >= 12
    assert {f["background"] for f in records} == set(range(10))
    assert {f["divider"] for f in records} == set(range(10))
    assert {f["commands"] for f in records} == {1, 2, 3}
    assert sum(f["repeated"] for f in records if f["split"] == "test") >= 30
    assert sum(len(f["distances"]) > 1 for f in records) >= 5
    assert {(1, 4), (2, 2), (2, 3)} <= {f["layout"] for f in records}
    replay = generate_puzzle(17)
    assert all(np.array_equal(a[k], b[k]) for split in replay
               for a, b in zip(replay[split], puzzles[17][split]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + a * 5 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus legal-count and mechanics checks)")
