"""Keyed overflow vessels: geometric connections carry conserved liquid.

Only NumPy and the standard library are used. No files, traces, or caches are
written by generation, solving, validation, or the bounded self-test.
"""
from __future__ import annotations

import hashlib
from itertools import permutations

import numpy as np


SINGLE_KEYS = ((1,), (2,), (3,))
DOUBLE_KEYS = ((1, 2), (2, 3), (1, 3))
ALL_KEYS = SINGLE_KEYS + DOUBLE_KEYS


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _arrange(panels, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    a = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, nc)
        a[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panel
    return a


def _partition(a):
    candidates = []
    h, w = a.shape
    for divider in np.unique(a):
        rr = [-1] + list(map(int, np.flatnonzero(np.all(a == divider, axis=1)))) + [h]
        cc = [-1] + list(map(int, np.flatnonzero(np.all(a == divider, axis=0)))) + [w]
        if len(rr) == len(cc) == 2:
            continue
        hs = [b - x - 1 for x, b in zip(rr, rr[1:])]
        ws = [b - x - 1 for x, b in zip(cc, cc[1:])]
        if min(hs) < 9 or min(ws) < 6 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        panels = [a[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 9 or any(np.any(p == divider) for p in panels):
            continue
        candidates.append((panels, (len(hs), len(ws)), int(divider)))
    if len(candidates) != 1:
        raise ValueError("expected one regular divider partition")
    return candidates[0]


def _orders(signatures):
    """Find at most two all-vessel directed chains, including repeated keys."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(path, remaining):
        if len(answers) == 2:
            return
        if not remaining:
            if path[-1] == ends[0]:
                answers.append(tuple(path))
            return
        key = signatures[path[-1]][1]
        if key:
            for i in remaining:
                if signatures[i][0] == key:
                    visit(path + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _terrain(shape, rim, floors, incoming, outgoing):
    """The upper edge keys and the lower open, single-depression vessel."""
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for r in incoming:
        mask[r, :2] = True
    for r in outgoing:
        mask[r, -2:] = True
    mask[rim:, 1] = True
    mask[rim:, -2] = True
    for c, floor in enumerate(floors, 2):
        mask[int(floor):, c] = True
    return mask


def _levels(rim, floors):
    """All admissible complete bottom-up water volumes, including zero."""
    return sorted({0} | {sum(max(0, int(f) - level) for f in floors)
                            for level in range(rim, max(floors))})


def _decode(input_grid):
    a = _grid(input_grid)
    panels, layout, divider = _partition(a)
    corners = {int(p[r, c]) for p in panels for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))}
    if len(corners) != 1:
        raise ValueError("panel corners must identify one background")
    bg = corners.pop()
    walls = [int(p[-1, p.shape[1] // 2]) for p in panels]
    if bg == divider or any(c in (bg, divider) for c in walls):
        raise ValueError("wall, background and divider roles overlap")
    colors = {int(c) for p in panels for c in np.unique(p)}
    liquid = colors - {bg, divider} - set(walls)
    if len(liquid) != 1:
        raise ValueError("expected one liquid color, absent from all vessel walls")
    water = liquid.pop()
    info, signatures = [], []
    for p, wall in zip(panels, walls):
        h, w = p.shape
        if not set(map(int, np.unique(p))) <= {bg, water, wall}:
            raise ValueError("unexpected panel colors")
        if not np.all(p[0] == bg):
            raise ValueError("top margin must be background")
        incoming = tuple(map(int, np.flatnonzero(p[:, 0] == wall)))
        outgoing = tuple(map(int, np.flatnonzero(p[:, -1] == wall)))
        if any(key and key not in ALL_KEYS for key in (incoming, outgoing)):
            raise ValueError("invalid boundary key")
        rim_cells = np.flatnonzero(p[4:, 1] == wall)
        if not len(rim_cells):
            raise ValueError("missing vessel wall")
        rim = int(rim_cells[0]) + 4
        if not 5 <= rim <= h - 3:
            raise ValueError("rim needs headspace and at least two water levels")
        floors = []
        for c in range(2, w - 2):
            cells = np.flatnonzero(p[rim:, c] == wall)
            if not len(cells):
                raise ValueError("missing floor")
            floors.append(int(cells[0]) + rim)
        if min(floors) <= rim or max(floors) >= h:
            raise ValueError("invalid floor heights")
        # Every horizontal bottom-up slice is contiguous: no inaccessible
        # lower pocket can be filled through a higher ridge.
        for level in range(rim, max(floors)):
            cols = [i for i, f in enumerate(floors) if f > level]
            if cols and cols[-1] - cols[0] + 1 != len(cols):
                raise ValueError("floor must form a single depression")
        solid = _terrain(p.shape, rim, floors, incoming, outgoing)
        if not np.array_equal(p == wall, solid):
            raise ValueError("malformed keys or vessel")
        available = np.zeros_like(solid)
        for c, floor in enumerate(floors, 2):
            available[4:floor, c] = True
        if np.any((p == water) & ~available):
            raise ValueError("liquid must be inside its vessel's shaft")
        levels = _levels(rim, floors)
        info.append(dict(panel=p, wall=wall, rim=rim, floors=tuple(floors),
                         capacity=levels[-1], levels=levels, available=available,
                         volume=int(np.count_nonzero(p == water))))
        signatures.append((incoming, outgoing))
    orders = _orders(signatures)
    if len(orders) != 1:
        raise ValueError("keys must define one complete directed chain")
    return dict(grid=a, panels=panels, layout=layout, divider=divider,
                background=bg, water=water, info=info, signatures=signatures,
                order=orders[0])


def _cascade(volumes, capacities, order):
    retained = [0] * len(order)
    flow, carry = [], 0
    for i in order:
        incoming = int(volumes[i]) + carry
        retained[i] = min(incoming, int(capacities[i]))
        carry = incoming - retained[i]
        flow.append(carry)
    return retained, flow


def _filled(item, amount, bg, water):
    if amount not in item["levels"]:
        raise ValueError("partially filled horizontal levels are outside this family")
    p = item["panel"].copy()
    p[p == water] = bg
    if amount:
        for level in range(item["rim"], max(item["floors"])):
            if sum(max(0, f - level) for f in item["floors"]) == amount:
                for c, floor in enumerate(item["floors"], 2):
                    p[level:floor, c] = water
                break
    return p


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, connections, capacities and all water solely from input."""
    d = _decode(input_grid)
    volumes = [x["volume"] for x in d["info"]]
    capacities = [x["capacity"] for x in d["info"]]
    target, flow = _cascade(volumes, capacities, d["order"])
    if flow[-1]:
        raise ValueError("terminal overflow would leave the finite system")
    panels = [_filled(x, v, d["background"], d["water"])
              for x, v in zip(d["info"], target)]
    return _arrange(panels, d["layout"], d["divider"])


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _links(rng, n, mode):
    if mode == "one":
        selected = [SINGLE_KEYS[i] for i in rng.choice(3, 2, replace=False)]
    elif mode == "two":
        selected = list(DOUBLE_KEYS)
    else:
        m = int(rng.integers((n - 1 + 1) // 2, min(6, n - 1) + 1))
        selected = [ALL_KEYS[i] for i in rng.choice(6, m, replace=False)]
    rng.shuffle(selected)
    doubled = set(map(int, rng.choice(len(selected), n - 1 - len(selected), replace=False)))
    links = [key for i, key in enumerate(selected) for _ in range(2 if i in doubled else 1)]
    signatures = list(zip([()] + links, links + [()]))
    assert _orders(signatures) == [tuple(range(n))]
    return links


def _configuration(rng, n):
    options = []
    for nr in range(1, n + 1):
        if n % nr:
            continue
        nc = n // nr
        max_h, max_w = min(14, 31 // nr - 1), min(14, 31 // nc - 1)
        if max_h >= 9 and max_w >= 6:
            options.append((nr, nc, max_h, max_w))
    nr, nc, max_h, max_w = _choose(rng, options)
    return (nr, nc), int(rng.integers(9, max_h + 1)), int(rng.integers(6, max_w + 1))


def _floor(rng, width, rim, height):
    # A flat, stepped, sloping or asymmetric single depression. The deepest
    # position may lie at an edge. All superlevel sets remain intervals.
    m = width - 4
    peak = int(rng.integers(m))
    maximum = int(rng.integers(rim + 2, height))
    if rng.random() < .18:
        return tuple([maximum] * m)
    left = sorted(map(int, rng.integers(rim + 1, maximum + 1, size=peak)))
    right = sorted(map(int, rng.integers(rim + 1, maximum + 1, size=m - peak - 1)), reverse=True)
    return tuple(left + [maximum] + right)


def _make_pair(rng, mode, roles):
    bg, water, divider = roles
    if mode == "one":
        n = 4
    elif mode == "two":
        n = 6
    else:
        n = int(_choose(rng, (8, 9) if mode == "test" else (2, 3, 4, 6)))
    layout, h, w = _configuration(rng, n)
    links = _links(rng, n, mode)
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    color_count = 1 if mode == "one" else int(rng.integers(2, min(7, n) + 1))
    wall_colors = [palette[i % color_count] for i in range(n)]
    rng.shuffle(wall_colors)
    items = []
    for i in range(n):
        rim = int(rng.integers(5, min(7, h - 3) + 1))
        floors = _floor(rng, w, rim, h)
        solid = _terrain((h, w), rim, floors, () if i == 0 else links[i - 1],
                         () if i == n - 1 else links[i])
        p = np.full((h, w), bg, dtype=np.int8)
        p[solid] = wall_colors[i]
        available = np.zeros_like(solid)
        for c, floor in enumerate(floors, 2):
            available[4:floor, c] = True
        levels = _levels(rim, floors)
        items.append(dict(panel=p, wall=wall_colors[i], rim=rim, floors=floors,
                          levels=levels, capacity=levels[-1], available=available))
    capacities = [x["capacity"] for x in items]
    target = [int(_choose(rng, x["levels"])) for x in items]
    if mode == "one":
        length = 1
    elif mode == "two":
        length = 2
    elif mode == "test":
        length = int(rng.integers(3, min(5, n - 2) + 1))
    else:
        length = int(rng.integers(1, min(2, n - 1) + 1))
    start = int(rng.integers(0, n - length))
    receiver = start + length
    for i in range(start, receiver):
        target[i] = capacities[i]
    target[receiver] = int(_choose(rng, [v for v in items[receiver]["levels"] if 0 < v < capacities[receiver]]))
    if receiver + 1 < n:
        target[receiver + 1] = 0  # A dry downstream vessel proves that a partial receiver stops the spill.
    bound = min([target[receiver]] + capacities[start:receiver] +
                [int(x["available"].sum()) - x["capacity"] for x in items[start:receiver]])
    flows = [0] * (n - 1)
    for i in range(start, receiver):
        flows[i] = int(rng.integers(1, bound + 1))
    source = [target[i] + (flows[i] if i < n - 1 else 0) - (flows[i - 1] if i else 0)
              for i in range(n)]
    before, after = [], []
    for item, count, final in zip(items, source, target):
        p = item["panel"].copy()
        eligible = np.argwhere(item["available"])
        assert 0 <= count <= len(eligible)
        # Different source shapes, including scattered, compact and banded.
        if count:
            style = int(rng.integers(3))
            anchor = eligible[int(rng.integers(len(eligible)))]
            dr, dc = np.abs(eligible[:, 0] - anchor[0]), np.abs(eligible[:, 1] - anchor[1])
            score = rng.random(len(eligible))
            if style == 1:
                score += dr + dc
            elif style == 2:
                score += 3 * dr + .1 * dc
            chosen = eligible[np.argsort(score)[:count]]
            p[chosen[:, 0], chosen[:, 1]] = water
        before.append(p)
        after.append(_filled(item, final, bg, water))
    # Reject reading-order routing as an alternative explanation. No unbounded
    # retries: the reverse chain always puts the underfull receiver first.
    order = list(range(n))
    for attempt in range(21):
        if attempt == 20:
            order = list(range(n - 1, -1, -1))
        else:
            rng.shuffle(order)
        wrong, _ = _cascade(source, capacities, order)
        if wrong != target:
            break
    else:
        raise AssertionError("reading-order witness construction failed")
    return {"input": _arrange([before[i] for i in order], layout, divider),
            "output": _arrange([after[i] for i in order], layout, divider)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not lo <= int(value) <= hi:
            raise ValueError("illegal pair count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    # Unique visible role triples guarantee distinct inputs without retries.
    role_pool = list(permutations(range(10), 3))
    role_indices = rng.choice(len(role_pool), int(num_train + num_test), replace=False)
    plans = ["one", "two"] + ["extra"] * (int(num_train) - 2)
    rng.shuffle(plans)
    plans += ["test"] * int(num_test)
    puzzle = {"train": [], "test": []}
    for i, mode in enumerate(plans):
        split = "train" if i < num_train else "test"
        puzzle[split].append(_make_pair(rng, mode, role_pool[int(role_indices[i])]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _key_kind(key):
    return "single" if len(key) == 1 else "thick" if key[-1] - key[0] == 1 else "separated"


def _inspect_pair(inp, out):
    d = _decode(inp)
    assert inp.shape == out.shape and np.any(inp != out)
    bg, water = d["background"], d["water"]
    terrain = (inp != bg) & (inp != water)
    assert np.array_equal(inp[terrain], out[terrain]), "all keys, walls and dividers stay fixed"
    assert np.array_equal(terrain, (out != bg) & (out != water))
    assert np.count_nonzero(inp == water) == np.count_nonzero(out == water)
    final_panels, final_layout, final_divider = _partition(out)
    assert final_layout == d["layout"] and final_divider == d["divider"]
    volumes = [x["volume"] for x in d["info"]]
    capacities = [x["capacity"] for x in d["info"]]
    retained, flow = _cascade(volumes, capacities, d["order"])
    assert not flow[-1] and any(flow), "show bounded cross-vessel transfer"
    actual = [int(np.count_nonzero(p == water)) for p in final_panels]
    assert actual == retained and actual != volumes, "local settling alone must fail"
    wrong, _ = _cascade(volumes, capacities, tuple(range(len(volumes))))
    assert wrong != actual, "spatial reading order must fail"
    for item, amount, panel in zip(d["info"], actual, final_panels):
        assert np.array_equal(panel, _filled(item, amount, bg, water))
        # Explicit support, complete levels and no water above the rim.
        for r, c in np.argwhere(panel == water):
            assert r >= item["rim"] and panel[r + 1, c] != bg
    assert np.array_equal(solve(inp), out)
    longest = run = 0
    for v in flow:
        run = run + 1 if v else 0
        longest = max(longest, run)
    keys = [b for a, b in d["signatures"] if b]
    ordered = [d["info"][i] for i in d["order"]]
    return dict(n=len(volumes), panel_shape=d["panels"][0].shape, layout=d["layout"],
                roles=(bg, water, d["divider"]), longest_spill=longest,
                key_kinds={_key_kind(k) for k in keys}, repeated=len(set(keys)) < len(keys),
                shared_walls=len({x["wall"] for x in ordered}) == 1,
                unequal_match_colors=any(a["wall"] != b["wall"] for a, b in zip(ordered, ordered[1:])),
                nonflat=sum(len(set(x["floors"])) > 1 for x in ordered),
                source_components_hint=sum(x["volume"] > 0 for x in d["info"]),
                capacity_range=(min(capacities), max(capacities)),
                water_volume=int(np.count_nonzero(inp == water)),
                transferred=sum(flow), changed_vessels=sum(a != b for a, b in zip(volumes, actual)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            stats[split].append(_inspect_pair(inp, out))
    train = stats["train"]
    assert any(s["longest_spill"] == 1 and s["shared_walls"] for s in train)
    assert any(s["longest_spill"] == 2 and s["unequal_match_colors"] for s in train)
    assert {"single", "thick", "separated"} <= set().union(*(s["key_kinds"] for s in train))
    assert any(s["repeated"] for s in train)
    assert all(s["longest_spill"] >= 3 and s["n"] >= 8 for s in stats["test"])
    assert min(s["n"] for s in stats["test"]) > max(s["n"] for s in train)


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed seeds; small statistics only, no retained puzzle corpus."""
    puzzle_hashes, geometry, shapes, panel_shapes, layouts = set(), set(), set(), set(), set()
    role_colors = [set(), set(), set()]
    counts, spill_lengths, capacities, volumes = set(), set(), set(), set()
    nonflat = shared = unequal = changed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_hashes.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                s = _inspect_pair(inp, out)
                shapes.add(inp.shape)
                panel_shapes.add(s["panel_shape"])
                layouts.add(s["layout"])
                counts.add(s["n"])
                spill_lengths.add(s["longest_spill"])
                capacities.update(s["capacity_range"])
                volumes.add(s["water_volume"])
                nonflat += s["nonflat"]
                shared += s["shared_walls"]
                unequal += s["unequal_match_colors"]
                changed += s["changed_vessels"]
                for values, color in zip(role_colors, s["roles"]):
                    values.add(color)
                bg, water, divider = s["roles"]
                norm = np.where(inp == bg, 0, np.where(inp == water, 1, np.where(inp == divider, 3, 2))).astype(np.int8)
                geometry.add(hashlib.sha256(bytes(norm.shape) + norm.tobytes()).hexdigest())
        if seed in (0, 17, 41):
            assert _signature(generate_puzzle(seed)) == _signature(puzzle)
            pair = puzzle["train"][0]
            d = _decode(pair["input"])
            # Palette roles are geometric, not literal color lookups.
            mapping = np.arange(9, -1, -1, dtype=np.int8)
            assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
            # Reposition every source within its own shaft, preserving only
            # its count. Nothing about the answer uses the source's shape.
            reshaped = []
            for item in d["info"]:
                panel = item["panel"].copy()
                panel[panel == d["water"]] = d["background"]
                points = np.argwhere(item["available"])[::-1][:item["volume"]]
                if len(points):
                    panel[points[:, 0], points[:, 1]] = d["water"]
                reshaped.append(panel)
            assert np.array_equal(solve(_arrange(reshaped, d["layout"], d["divider"])), pair["output"])
            # Move entire vessels to different input slots: the invisible
            # keyed route stays the same and the output follows those slots.
            final_panels, _, _ = _partition(pair["output"])
            assert np.array_equal(solve(_arrange(d["panels"][::-1], d["layout"], d["divider"])),
                                  _arrange(final_panels[::-1], d["layout"], d["divider"]))
    assert len(puzzle_hashes) == 42 and len(geometry) >= 190
    assert len(shapes) >= 30 and len(panel_shapes) >= 25 and len(layouts) >= 7
    assert counts == {2, 3, 4, 6, 8, 9} and {1, 2, 3, 4, 5} <= spill_lengths
    assert all(len(s) >= 9 for s in role_colors)
    assert len(capacities) >= 20 and len(volumes) >= 60
    assert nonflat >= 300 and shared >= 42 and unequal >= 100 and changed >= 400
    # Ambiguity and full-signature counterexamples, not model predictions.
    a, b = (1,), (2,)
    assert len(_orders([((), a), (a, a), (a, b), (b, ())])) == 1
    assert len(_orders([((), a), (a, a), (a, a), (a, ())])) == 2
    assert len(_orders([((), (1, 2)), ((1, 3), ())])) == 0
    # Hand calculation: excess crosses two keyed vessels; the receiver stops it.
    assert _cascade([7, 4, 1, 0], [5, 4, 6, 3], (0, 1, 2, 3)) == ([5, 4, 3, 0], [2, 2, 0, 0])
    assert _cascade([7, 4, 1, 0], [5, 6, 6, 3], (0, 1, 2, 3)) == ([5, 6, 1, 0], [2, 0, 0, 0])
    for a_count, b_count in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(2026, a_count, b_count)
        assert len(p["train"]) == a_count and len(p["test"]) == b_count
    for a_count, b_count in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                             (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a_count, b_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

