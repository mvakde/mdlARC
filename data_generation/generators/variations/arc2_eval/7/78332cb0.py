"""Assemble intact panels by matching complete, color-independent edge ports.

The staged summary incorrectly proposes deleting redundant runs. The original
source preserves every panel, including its straight-through middle panel.
There is no corruption, cleanup, rotation, reflection, or recoloring rule here.
"""

import numpy as np
from collections import Counter
from itertools import combinations, permutations


_AUTO_COUNTS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 1),
    (6, 3),
    (7, 2),
    (8, 4),
    (9, 3),
    (10, 4),
)

# Rejecting an exact source input is stronger than rejecting an exact
# source input/output pair: an exact pair necessarily has an exact input.
_SOURCE_INPUT_ROWS = {
    tuple(text.split("/"))
    for text in (
        (
            "77777674777/77777674447/77777677747/72227677747/"
            "77277677747/66666666666/77717677377/71117673377/"
            "71117673777/71117673777/77777673777"
        ),
        (
            "77777/77887/77887/88877/77777/66666/77777/22277/"
            "77227/77722/77777/66666/77777/79799/79997/79777/77777"
        ),
        (
            "77777677177677977/78887677177679997/77877677177677777/"
            "77877677177677777/77877677177677777"
        ),
        (
            "77777677777/74447644447/44747677744/74447674447/"
            "77777677777/66666666666/77777677777/74447674744/"
            "74747674747/74744644447/77777677777"
        ),
        (
            "77797/77997/79977/79777/77777/66666/72777/72777/"
            "72277/77277/77277/66666/77277/72227/77727/77727/"
            "77727/66666/77777/77797/77997/79977/79777"
        ),
    )
}


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _array_key(pair["input"]),
        _array_key(pair["output"]),
    )


def _check_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)

    selector = abs(int(seed))

    if num_train is None and num_test is None:
        return _AUTO_COUNTS[selector % len(_AUTO_COUNTS)]

    if num_train is None:
        test_count = int(num_test)
        choices = tuple(
            range(max(2, test_count + 1), 11)
        )
        train_count = choices[selector % len(choices)]
        return train_count, test_count

    if num_test is None:
        train_count = int(num_train)
        choices = tuple(
            range(1, min(4, train_count - 1) + 1)
        )
        test_count = choices[selector % len(choices)]
        return train_count, test_count

    train_count = int(num_train)
    test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _assemble_layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    grid = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1),
                   divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r * (h + 1):r * (h + 1) + h,
             c * (w + 1):c * (w + 1) + w] = panels[index]
    return grid


def _extract_layout(grid):
    """Find separators from visible full lines, without a size/layout catalog."""
    candidates = []
    h, w = grid.shape
    for divider in np.unique(grid):
        rows = list(np.flatnonzero(np.all(grid == divider, axis=1)))
        cols = list(np.flatnonzero(np.all(grid == divider, axis=0)))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        heights = [b - a - 1 for a, b in zip(rr, rr[1:])]
        widths = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(heights + widths) < 3 or len(set(heights)) != 1 or len(set(widths)) != 1:
            continue
        panels = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 6 or any(np.any(p == divider) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        background = next(iter(corners))
        if any(len(_components(p != background)) != 1 for p in panels):
            continue
        candidates.append(dict(panels=panels, divider=int(divider),
                               background=corners.pop(), shape=(len(heights), len(widths)),
                               panel_shape=panels[0].shape))
    assert len(candidates) == 1, "input must have one unambiguous equal-panel partition"
    return candidates[0]


def _neighbors(point):
    r, c = point
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    components = []
    while remaining:
        start = remaining.pop()
        component, stack = {start}, [start]
        while stack:
            for neighbor in _neighbors(stack.pop()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _signatures(panel, background, axis):
    panel = panel if axis == "H" else panel.T
    return (tuple(map(int, np.flatnonzero(panel[:, 0] != background))),
            tuple(map(int, np.flatnonzero(panel[:, -1] != background))))


def _chain_orders(signatures):
    """Enumerate at most two complete directed assemblies, including repeats."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1:
        return []
    if any(not a and not b for a, b in signatures):
        return []
    solutions = []

    def visit(order, remaining):
        if len(solutions) >= 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                solutions.append(tuple(order))
            return
        port = signatures[order[-1]][1]
        if not port:
            return
        for i in remaining:
            if signatures[i][0] == port:
                visit(order + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return solutions


def _decode(grid):
    parsed = _extract_layout(grid)
    panels, bg = parsed["panels"], parsed["background"]
    assert all(len(_components(p != bg)) == 1 for p in panels), "one connected motif per panel"
    axes = []
    for axis in ("H", "V"):
        oriented = [p if axis == "H" else p.T for p in panels]
        if all(np.all(p[0] == bg) and np.all(p[-1] == bg) for p in oriented):
            signatures = [_signatures(p, bg, axis) for p in panels]
            orders = _chain_orders(signatures)
            if orders:
                assert len(orders) == 1, "interchangeable panels make the chain ambiguous"
                axes.append((axis, signatures, orders[0]))
    assert len(axes) == 1, "ports must specify exactly one axis and complete chain"
    axis, signatures, order = axes[0]
    parsed.update(axis=axis, signatures=signatures, order=order)
    return parsed


def _predict(input_grid):
    parsed = _decode(input_grid)
    n = len(parsed["panels"])
    shape = (1, n) if parsed["axis"] == "H" else (n, 1)
    return _assemble_layout(parsed["panels"], parsed["order"], shape, parsed["divider"])


def solve(train_pairs, test_input):
    """Input-only geometric prediction; no test labels, seed, or hidden state."""
    for pair in train_pairs:
        assert np.array_equal(_predict(pair["input"]), pair["output"])
    return _predict(test_input)


def _signature_kind(signature):
    if len(signature) == 1:
        return "single"
    return "thick" if signature[-1] - signature[0] + 1 == len(signature) else "separated"


def _signature_pool(h, capability):
    result = [(i,) for i in range(1, h - 1)]
    if capability:
        for size in (2, 3):
            for signature in combinations(range(1, h - 1), size):
                if capability == 2 or _signature_kind(signature) == "thick":
                    result.append(signature)
    return result


def _make_links(n, pool, required, rng, distinct_joins=False):
    def valid(links):
        if not required.issubset({_signature_kind(s) for s in links}):
            return False
        if distinct_joins and len(set(links)) < 2:
            return False
        signatures = list(zip([()] + links, links + [()]))
        return len(_chain_orders(signatures)) == 1

    # Nonadjacent revisits and excursions are allowed only when the full path is unique.
    for _ in range(40):
        links = [_choose(rng, pool) for _ in range(n - 1)]
        if valid(links):
            return links
    # A sequence of distinct signatures, each used once or twice consecutively,
    # is constructive: at most one self-loop at each vertex, and only one exit.
    selected = []
    for kind in sorted(required):
        selected.append(_choose(rng, [s for s in pool if _signature_kind(s) == kind]))
    minimum = max(len(selected), n // 2, 2 if distinct_joins else 1)
    maximum = min(len(pool), n - 1)
    distinct = int(rng.integers(minimum, maximum + 1))
    rest = [s for s in pool if s not in selected]
    rng.shuffle(rest)
    selected.extend(rest[:distinct - len(selected)])
    rng.shuffle(selected)
    doubled = set(map(int, rng.choice(distinct, n - 1 - distinct, replace=False)))
    links = [s for i, s in enumerate(selected) for _ in range(2 if i in doubled else 1)]
    assert valid(links)
    return links


def _path_to_body(start, body, allowed, rng):
    """Random tie-breaking among shortest interior paths; holes remain protected."""
    queue, parent = [start], {start: None}
    endpoint = None
    for point in queue:
        if point in body:
            endpoint = point
            break
        neighbors = list(_neighbors(point))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = point
                queue.append(neighbor)
    assert endpoint is not None
    while endpoint is not None:
        body.add(endpoint)
        endpoint = parent[endpoint]


def _make_panel(h, w, incoming, outgoing, background, colors, richness, rng):
    interior = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    ports = {(r, 0) for r in incoming} | {(r, w - 1) for r in outgoing}
    roots = {(r, 1) for r in incoming} | {(r, w - 2) for r in outgoing}
    style = int(rng.choice(4, p=((.55, .25, .18, .02), (.20, .30, .36, .14),
                                (.08, .22, .43, .27))[richness]))
    allowed = set(interior)
    if style == 0:
        body = {sorted(roots)[int(rng.integers(len(roots)))]}
    elif style == 3 and h >= 5 and w >= 5:
        r0 = int(rng.integers(1, h - 3))
        r1 = int(rng.integers(r0 + 2, h - 1))
        c0 = int(rng.integers(1, w - 3))
        c1 = int(rng.integers(c0 + 2, w - 1))
        hole = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
        allowed -= hole
        body = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)}
    else:
        body = {_choose(rng, sorted(interior))}
    for root in sorted(roots):
        _path_to_body(root, body, allowed, rng)
    if style == 1:
        for _ in range(int(rng.integers(1, 4))):
            _path_to_body(_choose(rng, sorted(allowed)), body, allowed, rng)
    if style in (2, 3):
        target = max(len(body), int(rng.uniform(.20, .90) * len(allowed)))
        while len(body) < target:
            frontier = sorted({q for p in body for q in _neighbors(p) if q in allowed - body})
            if not frontier:
                break
            body.add(_choose(rng, frontier))
    panel = np.full((h, w), background, dtype=np.int8)
    occupied = sorted(body | ports)
    for i, point in enumerate(occupied):
        # Coloring is independent of geometry; optional multicolor motifs retain
        # one connected occupied body, with every literal color copied intact.
        panel[point] = colors[i % len(colors)] if len(colors) > 1 else colors[0]
    return panel


def sample_puzzle_parameters(rng):
    """Curriculum choices, not hidden semantic maps: all roles are input-local."""
    return dict(capability=int(rng.choice(3, p=(.30, .30, .40))),
                orientation=_choose(rng, ("mixed", "mixed", "mixed", "homogeneous", "shift")),
                first_axis=_choose(rng, ("H", "V")),
                source_color_shift=bool(rng.random() < .18),
                multicolor=bool(rng.random() < .25),
                richness=int(rng.integers(3)))


def _pair_plan(rng, params, train_count, test_count):
    plans = []
    for split, count in (("source_train", train_count), ("target_test", test_count)):
        axes = [params["first_axis"]] * count
        if params["orientation"] == "mixed":
            axes = [_choose(rng, ("H", "V")) for _ in range(count)]
            if split == "source_train":
                axes[0:2] = ["H", "V"]
                rng.shuffle(axes)
        elif params["orientation"] == "shift" and split == "target_test":
            axes = ["V" if params["first_axis"] == "H" else "H"] * count
        for axis in axes:
            capability = int(rng.integers(params["capability"] + 1))
            if params["source_color_shift"]:
                color_mode = "distinct" if split == "source_train" else _choose(rng, ("shared", "partial"))
            else:
                modes = ("distinct", "shared", "partial") + (("multi",) if params["multicolor"] else ())
                color_mode = _choose(rng, modes)
            plans.append(dict(regime=split, axis=axis, capability=capability,
                              required=set(), color_mode=color_mode,
                              ordering_witness=False,
                              richness=int(np.clip(params["richness"] + rng.integers(-1, 2), 0, 2))))
    plans[int(rng.integers(train_count))]["ordering_witness"] = True
    # Demonstrate the extensions before tests; witnesses occupy random indices.
    witness = int(rng.integers(train_count))
    if params["capability"]:
        plans[witness]["capability"] = params["capability"]
        plans[witness]["required"] = {"single", "thick"}
        if params["capability"] == 2:
            plans[witness]["required"].add("separated")
    if not params["source_color_shift"]:
        plans[int(rng.integers(train_count))]["color_mode"] = _choose(rng, ("shared", "partial"))
        if params["multicolor"]:
            plans[int(rng.integers(train_count))]["color_mode"] = "multi"
    return plans


def _configuration(rng, plan, previous):
    options = []
    for n in range(2, 7):
        if plan["ordering_witness"] and n < 3:
            continue
        if n - 1 < len(plan["required"]):
            continue
        if previous and n == previous["n"]:
            continue
        for w in range(3, min(10, 31 // n - 1) + 1):
            for h in range(3, 11):
                pool = _signature_pool(h, plan["capability"])
                if len(pool) < max(n // 2, 2 if plan["ordering_witness"] else 1):
                    continue
                if not plan["required"].issubset({_signature_kind(s) for s in pool}):
                    continue
                shape = (h, w) if plan["axis"] == "H" else (w, h)
                if previous and shape == previous["panel_shape"]:
                    continue
                layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
                           and nr * (shape[0] + 1) - 1 <= 30
                           and (n // nr) * (shape[1] + 1) - 1 <= 30]
                if previous:
                    layouts = [s for s in layouts if s != previous["input_layout"]]
                if layouts:
                    options.append((n, h, w, layouts))
    # Sample counts separately so the number of feasible rectangles cannot
    # silently skew the distribution towards two panels.
    counts = sorted({o[0] for o in options})
    weights = np.array([1 + (plan["regime"] == "target_test" and n >= 4) for n in counts], float)
    n = int(rng.choice(counts, p=weights / weights.sum()))
    options = [o for o in options if o[0] == n]
    squares = [o for o in options if o[1] == o[2]]
    if squares and rng.random() < .40:
        options = squares
    return _choose(rng, options)


def _generate_pair(rng, plan, previous=None):
    n, h, w, layouts = _configuration(rng, plan, previous)
    palette = list(map(int, rng.permutation(10)))
    if previous:
        # Role changes are guaranteed within a puzzle, and directly visible.
        bg_choices = [c for c in palette if c != previous["background"]]
        background = _choose(rng, bg_choices)
        divider = _choose(rng, [c for c in palette if c not in (background, previous["divider"])])
    else:
        background, divider = palette[:2]
    palette = [c for c in palette if c not in (background, divider)]
    rng.shuffle(palette)
    mode = plan["color_mode"]
    if mode == "distinct":
        colors = [[c] for c in palette[:n]]
        if plan["ordering_witness"] and (colors == sorted(colors) or colors == sorted(colors, reverse=True)):
            colors = colors[1:] + colors[:1]
    elif mode == "shared":
        colors = [[palette[0]]] * n
    elif mode == "partial":
        k = int(rng.integers(2, n)) if n > 2 else 1
        labels = list(range(k)) + list(map(int, rng.integers(k, size=n - k)))
        rng.shuffle(labels)
        colors = [[palette[k]] for k in labels]
    else:
        colors = [list(map(int, rng.choice(palette, int(rng.integers(2, 5)), replace=False))) for _ in range(n)]
    links = _make_links(n, _signature_pool(h, plan["capability"]), plan["required"], rng,
                        plan["ordering_witness"])
    panels = [_make_panel(h, w, () if i == 0 else links[i - 1],
                          () if i == n - 1 else links[i], background, colors[i],
                          plan["richness"], rng) for i in range(n)]
    if plan["axis"] == "V":
        panels = [p.T.copy() for p in panels]
    output_shape = (1, n) if plan["axis"] == "H" else (n, 1)
    layout = _choose(rng, layouts)
    permutation_kind = int(rng.integers(5))
    order = list(range(n))
    if permutation_kind == 0:
        alternatives = [s for s in layouts if s != output_shape]
        if alternatives:
            layout = _choose(rng, alternatives)
        else:
            order[0], order[-1] = order[-1], order[0]
    elif permutation_kind == 1:
        order[0], order[-1] = order[-1], order[0]
    elif permutation_kind == 2:
        order.reverse()
    elif permutation_kind == 3:
        a, b = map(int, rng.choice(n, 2, replace=False))
        order[a], order[b] = order[b], order[a]
    else:
        rng.shuffle(order)
    if order == list(range(n)) and (layout == output_shape or plan["ordering_witness"]):
        order[0], order[-1] = order[-1], order[0]
    return dict(input=_assemble_layout(panels, order, layout, divider),
                output=_assemble_layout(panels, list(range(n)), output_shape, divider))


def _motif_features(panel, bg):
    mask = panel != bg
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    holes = [part for part in _components(~mask)
             if not any(r in (0, panel.shape[0] - 1) or c in (0, panel.shape[1] - 1) for r, c in part)]
    return dict(area=len(cells), holes=len(holes),
                branch=any(sum(q in cells for q in _neighbors(p)) >= 3 for p in cells),
                thick=bool(np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])))


def _analyze_pair(input_grid, output_grid):
    parsed = _decode(input_grid)
    panels, order, bg = parsed["panels"], parsed["order"], parsed["background"]
    n, axis = len(panels), parsed["axis"]
    shape = (1, n) if axis == "H" else (n, 1)
    expected = _assemble_layout(panels, order, shape, parsed["divider"])
    assert np.array_equal(expected, output_grid), "copy every panel intact in its unique order"
    output = _extract_layout(output_grid)
    assert output["shape"] == shape and output["panel_shape"] == parsed["panel_shape"]
    assert output["divider"] == parsed["divider"] and output["background"] == bg
    assert Counter(_array_key(p) for p in panels) == Counter(_array_key(p) for p in output["panels"])
    ordered = [panels[i] for i in order]
    signatures = [parsed["signatures"][i] for i in order]
    links = [b for a, b in signatures[:-1]]
    colors = [tuple(map(int, np.unique(p[p != bg]))) for p in ordered]
    color_mode = ("multi" if any(len(c) > 1 for c in colors) else
                  "shared" if len(set(colors)) == 1 else
                  "distinct" if len(set(colors)) == n else "partial")
    natural = tuple(range(n))
    endpoints = (n - 1,) + tuple(range(1, n - 1)) + (0,)
    perm = ("identity" if order == natural else "endpoints" if order == endpoints else
            "reverse" if order == natural[::-1] else
            "partial" if sum(a != b for a, b in zip(order, natural)) == 2 else "shuffle")
    motifs = [_motif_features(p, bg) for p in ordered]
    straight = 0
    for p, (a, b) in zip(ordered[1:-1], signatures[1:-1]):
        oriented = p if axis == "H" else p.T
        straight += bool(a == b and len(a) == 1 and np.all(oriented[a[0]] != bg)
                         and np.count_nonzero(oriented != bg) == oriented.shape[1])
    return dict(n=n, panel_shape=parsed["panel_shape"], axis=axis,
                input_layout=parsed["shape"], background=bg, divider=parsed["divider"],
                order=order, permutation=perm, color_mode=color_mode,
                signature_kinds=tuple(sorted({_signature_kind(s) for s in links})),
                max_port=max(map(len, links)), repeated=len(set(links)) < len(links),
                equal_io=sum(bool(a) and a == b for a, b in signatures), straight=straight,
                terminal_areas=(motifs[0]["area"], motifs[-1]["area"]),
                areas=tuple(m["area"] for m in motifs), holes=sum(m["holes"] for m in motifs),
                branches=sum(m["branch"] for m in motifs), thickness=sum(m["thick"] for m in motifs),
                same_axis=parsed["shape"] == shape, changed_panel_cells=0)


def _validate_generated_puzzle_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, metadata = set(), set(), {}
    for split in ("train", "test"):
        metadata[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            key, full_key = _array_key(pair["input"]), _pair_key(pair)
            assert key not in inputs and full_key not in pairs
            inputs.add(key)
            pairs.add(full_key)
            assert _rows_key(pair["input"]) not in _SOURCE_INPUT_ROWS
            assert not np.array_equal(pair["input"], pair["output"])
            metadata[split].append(_analyze_pair(pair["input"], pair["output"]))
    train = metadata["train"]
    all_pairs = train + metadata["test"]
    assert any(p["n"] >= 3 and p["permutation"] != "identity" for p in train), "demonstrate geometric reordering"
    for field in ("n", "panel_shape", "background", "divider", "input_layout"):
        assert len({p[field] for p in all_pairs}) >= 2, "pair-level diversity: " + field
    demonstrated = {kind for p in train for kind in p["signature_kinds"]}
    required = {kind for p in metadata["test"] for kind in p["signature_kinds"]}
    assert required - {"single"} <= demonstrated
    if any(p["color_mode"] == "multi" for p in metadata["test"]):
        assert any(p["color_mode"] == "multi" for p in train)
    # The invariant has no hidden color or orientation convention. Either axis
    # follows from which two opposite sides have ports, including axis shifts.


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def _recolor_fresh(pair, seen):
    """Finite guaranteed duplicate fallback: >14 role assignments, at most 14 pairs."""
    palette = sorted(map(int, np.unique(pair["input"])))
    for assignment in permutations(range(10), len(palette)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[palette] = assignment
        candidate = {role: mapping[grid] for role, grid in pair.items()}
        if (_array_key(candidate["input"]) not in seen
                and _rows_key(candidate["input"]) not in _SOURCE_INPUT_ROWS):
            return candidate
    raise AssertionError("finite palette space exhausted")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(int(seed) % (1 << 64))
    params = sample_puzzle_parameters(rng)
    plans = _pair_plan(rng, params, train_count, test_count)
    puzzle, seen, previous = {"train": [], "test": []}, set(), None
    for index, plan in enumerate(plans):
        pair = _generate_pair(rng, plan, previous)
        if (_array_key(pair["input"]) in seen
                or _rows_key(pair["input"]) in _SOURCE_INPUT_ROWS):
            pair = _recolor_fresh(pair, seen)
        seen.add(_array_key(pair["input"]))
        split = "train" if plan["regime"] == "source_train" else "test"
        puzzle[split].append(pair)
        previous = _analyze_pair(pair["input"], pair["output"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ("train", "test"))


# Exact source regressions, including the untouched straight-through panel.
_SOURCE_REGRESSIONS = (
    ('77777674777/77777674447/77777677747/72227677747/77277677747/66666666666/77717677377/71117673377/71117673777/71117673777/77777673777',
     '77777/77777/77777/72227/77277/66666/77377/73377/73777/73777/73777/66666/74777/74447/77747/77747/77747/66666/77717/71117/71117/71117/77777'),
    ('77777/77887/77887/88877/77777/66666/77777/22277/77227/77722/77777/66666/77777/79799/79997/79777/77777',
     '77777677777677777/79799622277677887/79997677227677887/79777677722688877/77777677777677777'),
    ('77777677177677977/78887677177679997/77877677177677777/77877677177677777/77877677177677777',
     '77777/78887/77877/77877/77877/66666/77177/77177/77177/77177/77177/66666/77977/79997/77777/77777/77777'),
    ('77777677777/74447644447/44747677744/74447674447/77777677777/66666666666/77777677777/74447674744/74747674747/74744644447/77777677777',
     '77777677777677777677777/74447674744644447674447/74747674747677744644747/74744644447674447674447/77777677777677777677777'),
    ('77797/77997/79977/79777/77777/66666/72777/72777/72277/77277/77277/66666/77277/72227/77727/77727/77727/66666/77777/77797/77997/79977/79777',
     '77777/77797/77997/79977/79777/66666/72777/72777/72277/77277/77277/66666/77277/72227/77727/77727/77727/66666/77797/77997/79977/79777/77777'),
)


def _brute_prediction(input_grid):
    """Independent permutation check, deliberately not using the graph search."""
    parsed = _extract_layout(input_grid)
    panels, bg, answers = parsed["panels"], parsed["background"], []
    for axis in (0, 1):
        masks = [(p != bg) if axis == 1 else (p != bg).T for p in panels]
        if any(m[0].any() or m[-1].any() for m in masks):
            continue
        for order in permutations(range(len(panels))):
            sequence = [masks[i] for i in order]
            if sequence[0][:, 0].any() or sequence[-1][:, -1].any():
                continue
            if not all(a[:, -1].any() and np.array_equal(a[:, -1], b[:, 0])
                       for a, b in zip(sequence, sequence[1:])):
                continue
            parts = []
            for j, i in enumerate(order):
                if j:
                    separator_shape = (panels[i].shape[0], 1) if axis == 1 else (1, panels[i].shape[1])
                    parts.append(np.full(separator_shape, parsed["divider"], dtype=np.int8))
                parts.append(panels[i])
            answers.append(np.concatenate(parts, axis=axis))
    assert len(answers) == 1, "brute enumeration did not find exactly one answer"
    return answers[0]


def _source_regressions():
    for inp, out in _SOURCE_REGRESSIONS:
        a, b = (np.array([[int(c) for c in row] for row in text.split("/")], dtype=np.int8)
                for text in (inp, out))
        assert np.array_equal(_brute_prediction(a), b)
        features = _analyze_pair(a, b)
        assert features["changed_panel_cells"] == 0
    inp, out = _SOURCE_REGRESSIONS[2]
    a, b = (np.array([[int(c) for c in row] for row in text.split("/")], dtype=np.int8)
            for text in (inp, out))
    features = _analyze_pair(a, b)
    assert features["straight"] == 1 and features["order"] == (0, 1, 2)
    damaged = b.copy()
    damaged[8, 2] = 7
    try:
        _analyze_pair(a, damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("erasing a source panel cell must be rejected")
    # One self-loop is forced; two distinguishable identical-port loops can swap.
    a, b, c = (1,), (2,), (1, 2)
    assert len(_chain_orders([((), a), (a, a), (a, ())])) == 1
    assert len(_chain_orders([((), a), (a, a), (a, a), (a, ())])) == 2
    assert len(_chain_orders([((), a), (a, b), (b, a), (a, c), (c, a), (a, ())])) == 2
    # Six panels using only three signature sets, with two forced straight-throughs.
    assert len(_chain_orders([((), a), (a, a), (a, b), (b, b), (b, c), (c, ())])) == 1


def self_test() -> None:
    _source_regressions()
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(automatic[17])
    records, shapes, geometry = [], set(), set()
    orientations = set()
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                prediction = _brute_prediction(pair["input"].copy())
                assert np.array_equal(prediction, pair["output"])
                item = _analyze_pair(pair["input"], pair["output"])
                item["split"] = split
                local.append(item)
                shapes.add(pair["input"].shape)
                parsed = _extract_layout(pair["input"])
                for panel in parsed["panels"]:
                    geometry.add(_array_key((panel != parsed["background"]).astype(np.int8)))
        for field in ("n", "panel_shape", "background", "divider", "input_layout"):
            assert len({r[field] for r in local}) >= 2
        train_axes = {r["axis"] for r in local if r["split"] == "train"}
        test_axes = {r["axis"] for r in local if r["split"] == "test"}
        orientations.add("shift" if not test_axes <= train_axes else
                         "homogeneous" if len(train_axes | test_axes) == 1 else "mixed")
        # This prediction interface receives only training labels and a test input.
        train = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
        for pair in puzzle["test"]:
            assert np.array_equal(solve(train, pair["input"].copy()), pair["output"])
        records.extend(local)
    assert {r["n"] for r in records} == set(range(2, 7))
    assert len({r["panel_shape"] for r in records}) >= 30
    assert len(shapes) >= 60 and len(geometry) >= 300
    assert {r["background"] for r in records} == set(range(10))
    assert {r["divider"] for r in records} == set(range(10))
    assert {(2, 2), (2, 3), (3, 2)} <= {r["input_layout"] for r in records}
    assert {"identity", "endpoints", "reverse", "partial", "shuffle"} <= {r["permutation"] for r in records}
    assert orientations == {"shift", "mixed", "homogeneous"}
    for split in ("train", "test"):
        selected = [r for r in records if r["split"] == split]
        assert {"distinct", "shared", "partial", "multi"} <= {r["color_mode"] for r in selected}
        assert {"single", "thick", "separated"} <= {k for r in selected for k in r["signature_kinds"]}
        assert any(r["same_axis"] for r in selected)
        assert any(r["repeated"] for r in selected)
    areas = {a for r in records for a in r["terminal_areas"]}
    assert 2 in areas and len(areas) >= 20 and max(areas) >= 25
    assert sum(r["holes"] for r in records) >= 10
    assert sum(r["branches"] for r in records) >= 50
    assert sum(r["thickness"] for r in records) >= 50
    assert sum(r["straight"] for r in records) >= 10
    assert sum(r["n"] >= 5 and r["panel_shape"][0] != r["panel_shape"][1]
               and r["max_port"] > 1 and max(r["terminal_areas"]) > 2 for r in records) >= 10
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, *counts)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(1000 + i, *counts))
        for split, requested in zip(("train", "test"), counts):
            assert requested is None or len(puzzle[split]) == requested
    illegal = [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3.0, 1)]
    for counts in illegal:
        try:
            generate_puzzle(10, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
