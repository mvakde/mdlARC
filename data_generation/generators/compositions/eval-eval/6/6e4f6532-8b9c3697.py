"""Profile-keyed pistons: color-compassed, contour-matched grooved insertion.

Only standard library and NumPy. No parent imports, files, seed lookup, or
generation state are used by solve. All generated arrays are small int8 grids.
"""
from __future__ import annotations

import hashlib
from itertools import product

import numpy as np


_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_HOST_SEEDS = (0, 392963457, 681054878, 392536752, 721219366, 542117037,
               170138905, 769036219, 372230503, 173460297, 533754488,
               813987883, 281458041, 826858623, 15130856, 934966179,
               395938720, 981683144, 872311649, 972469629, 987981847,
               64342726, 653449440, 157496874, 426336716, 809777535,
               288682548, 586228050, 743560949, 45392010, 748970685,
               471755438, 655232404, 466312840, 433657921, 897765201,
               165954808, 42416382, 644988315, 883429576, 810761286,
               43678040)


def _integer(value, name, lo, hi=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < lo or (hi is not None and int(value) > hi)):
        raise ValueError(f"illegal {name}: {value!r}")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train", 2, 10), _integer(b, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected a 1..30 integer grid using colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _d4(p, t):
    r, c = p
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = c, -r
    return r, c


def _translate(cells, delta):
    return {(r + delta[0], c + delta[1]): v for (r, c), v in cells.items()}


def _pose(cells, anchor, t):
    return {(anchor[0] + _d4((r - anchor[0], c - anchor[1]), t)[0],
             anchor[1] + _d4((r - anchor[0], c - anchor[1]), t)[1]): v
            for (r, c), v in cells.items()}


def _paint(g, cells):
    for p, color in cells.items():
        g[p] = color


def _parse_fork(cells):
    """Recognize current geometry and colors, without construction parameters."""
    found = []
    for outward in _DIRS:
        perp = -outward[1], outward[0]
        kmin = min(r * outward[0] + c * outward[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * outward[0] + c * outward[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin): v
                 for (r, c), v in cells.items()}
        width = max(q for _, q in local) + 1
        if width < 7 or {q for k, q in local if k == 0} != set(range(width)):
            continue
        base_colors = {local[0, q] for q in range(width)}
        if len(base_colors) != 1:
            continue
        base = base_colors.pop()
        lengths = [max((k for k, qq in local if qq == q), default=0)
                   for q in range(width)]
        if not lengths[0] or not lengths[-1]:
            continue
        expected = {(0, q) for q in range(width)}
        expected.update((k, q) for q, length in enumerate(lengths)
                        for k in range(1, length + 1))
        if set(local) != expected:
            continue
        middle = [q for q in range(1, width - 1) if lengths[q]]
        if (not 1 <= len(middle) <= 3
                or middle != list(range(middle[0], middle[-1] + 1))
                or middle[0] + middle[-1] != width - 1
                or middle[0] < 3):
            continue
        left_values = {local[k, 0] for k in range(1, lengths[0] + 1)}
        right_values = {local[k, width - 1] for k in range(1, lengths[-1] + 1)}
        if len(left_values) != 1 or len(right_values) != 1:
            continue
        left, right = next(iter(left_values)), next(iter(right_values))
        if len({base, left, right}) != 3:
            continue
        tips = {(lengths[q], q): local[lengths[q], q] for q in middle}
        anchor_local = lengths[middle[0]], middle[0]
        hub = local[anchor_local]
        rest = {v for p, v in tips.items() if p != anchor_local}
        if len(rest) > 1:
            continue
        reg = next(iter(rest)) if rest else None
        roles = [base, left, right, hub] + ([] if reg is None else [reg])
        if len(set(roles)) != len(roles):
            continue
        if any(local[k, q] != base for q in middle
               for k in range(1, lengths[q])):
            continue
        profile = tuple(lengths[q] - anchor_local[0] for q in middle)
        if min(profile) < 0 or max(profile) > 2:
            continue

        def world(kq):
            k, q = kq[0] + kmin, kq[1] + qmin
            return k * outward[0] + q * perp[0], k * outward[1] + q * perp[1]

        found.append({"cells": cells, "out": outward, "perp": perp,
                      "base": base, "left": left, "right": right,
                      "hub": hub, "reg": reg, "anchor": world(anchor_local),
                      "marks": {world(p): v for p, v in tips.items()},
                      "profile": profile, "span": len(middle), "width": width,
                      "lengths": (lengths[0], anchor_local[0], lengths[-1])})
    return found[0] if len(found) == 1 else None


def _parse(g):
    if np.any(g == 0):
        raise ValueError("black is reserved for trails")
    counts = np.bincount(g.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if sum(counts == counts[background]) != 1:
        raise ValueError("ambiguous background")
    objects = [{p: int(g[p]) for p in comp} for comp in _components(g != background)]
    forks, loose = [], []
    for cells in objects:
        fork = _parse_fork(cells)
        (forks if fork is not None else loose).append(fork if fork is not None else cells)
    if not forks or not loose:
        raise ValueError("need sockets and loose keys")
    first = forks[0]
    bases = {f["base"] for f in forks}
    hubs = {f["hub"] for f in forks}
    regs = {f["reg"] for f in forks if f["reg"] is not None}
    guards = {frozenset((f["left"], f["right"])) for f in forks}
    if len(bases) != 1 or len(hubs) != 1 or len(regs) > 1 or len(guards) != 1:
        raise ValueError("inconsistent socket roles")
    colors = {v for f in forks for v in f["cells"].values()}
    body_colors = {v for cells in loose for v in cells.values()} - colors
    if len(body_colors) != 1:
        raise ValueError("loose-key body role is ambiguous")
    body = next(iter(body_colors))
    labels = {first["base"], first["left"], first["right"]}
    mark_colors = hubs | regs
    keys = []
    for cells in loose:
        anchors = [p for p, v in cells.items() if v == first["hub"]]
        if len(anchors) != 1 or body not in cells.values() or not labels <= set(cells.values()):
            raise ValueError("malformed key")
        if not set(cells.values()) <= labels | mark_colors | {body}:
            raise ValueError("unknown key color")
        keys.append({"cells": cells, "anchor": anchors[0]})
    structure = {p: v for f in forks for p, v in f["cells"].items()}
    return {"background": background, "forks": forks, "keys": keys,
            "body": body, "labels": labels, "mark_colors": mark_colors,
            "structure": structure, "shape": g.shape, "objects": objects}


def _facing(cells, labels):
    core = [p for p, v in cells.items() if v not in labels]
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    result = {color: set() for color in labels}
    for (r, c), v in cells.items():
        if v not in labels:
            continue
        ds = [d for condition, d in ((r < r0, (-1, 0)), (r > r1, (1, 0)),
                                     (c < c0, (0, -1)), (c > c1, (0, 1)))
              if condition]
        if len(ds) != 1:
            return None
        result[v].add(ds[0])
    return result if all(len(ds) == 1 for ds in result.values()) else None


def _relative(cells, anchor):
    return {(r - anchor[0], c - anchor[1]): v for (r, c), v in cells.items()}


def _candidates(key, info, use_labels=True, use_profile=True, check_route=True):
    result, seen = [], set()
    anchor = key["anchor"]
    for fi, fork in enumerate(info["forks"]):
        out, perp = fork["out"], fork["perp"]
        dr, dc = anchor[0] - fork["anchor"][0], anchor[1] - fork["anchor"][1]
        offset = dr * out[0] + dc * out[1]
        if dr * perp[0] + dc * perp[1] or offset <= 1:
            continue
        distance = offset - 1
        desired = {fork["base"]: {(-out[0], -out[1])},
                   fork["left"]: {(-perp[0], -perp[1])},
                   fork["right"]: {perp}}
        target = _relative(fork["marks"], fork["anchor"])
        for t in range(8):
            oriented = _pose(key["cells"], anchor, t)
            if use_labels and _facing(oriented, info["labels"]) != desired:
                continue
            marks = {p: v for p, v in oriented.items() if v in info["mark_colors"]}
            if use_profile and _relative(marks, anchor) != target:
                continue
            final = _translate(oriented, (-distance * out[0], -distance * out[1]))
            signature = fi, tuple(sorted(final.items()))
            if signature in seen:
                continue
            # At most 30 integer translation positions fit a legal grid.
            route = {(r - step * out[0], c - step * out[1])
                     for r, c in oriented for step in range(distance + 1)}
            if check_route and (route & info["structure"].keys() or any(
                    not (0 <= r < info["shape"][0] and 0 <= c < info["shape"][1])
                    for r, c in route)):
                continue
            seen.add(signature)
            result.append({"fi": fi, "t": t, "oriented": oriented,
                           "final": final, "route": route, "distance": distance})
    return result


def _analyse(g):
    info = _parse(g)
    matches, used, swept = [], set(), set()
    for ki, key in enumerate(info["keys"]):
        candidates = _candidates(key, info)
        if len(candidates) > 1:
            raise ValueError("ambiguous key orientation or socket assignment")
        if candidates:
            match = candidates[0]
            if match["fi"] in used or match["route"] & swept:
                raise ValueError("competing keys or intersecting swept regions")
            used.add(match["fi"])
            swept.update(match["route"])
            matches.append(dict(match, ki=ki))
    if not matches:
        raise ValueError("no usable key")
    info["matches"] = matches
    return info


def _render(g, info):
    out = g.copy()
    for key in info["keys"]:
        for p in key["cells"]:
            out[p] = info["background"]
    for match in info["matches"]:
        for p in match["route"] - match["final"].keys():
            out[p] = 0
    for match in info["matches"]:
        _paint(out, match["final"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full answer from this input alone; never mutate the caller."""
    g = _grid(input_grid)
    return _render(g, _analyse(g))


def _key_template(rng, span, profile, roles, compact, singleton_demo):
    _, base, left, right, body, hub, reg = roles
    if singleton_demo:
        wl, wr = 2, 1
    elif compact:
        wl = wr = 1
    else:
        wl, wr = map(int, rng.integers(1, 3, size=2))
    notch = 1 if compact else int(rng.integers(1, 4))
    back = max(profile) + (1 if compact else int(rng.integers(1, 4)))
    core = {(k, q): body for k in range(-notch, back + 1)
            for q in range(-wl, span + wr)
            if q < 0 or q >= span or k >= profile[q]}
    # Perforate thick wings without cutting their outer rail or back cap.
    if not compact:
        for q in (-1 if wl == 2 else None, span if wr == 2 else None):
            if q is not None and back >= 2 and rng.random() < .7:
                core.pop((int(rng.integers(0, back)), q), None)
    for q, height in enumerate(profile):
        core[height, q] = hub if q == 0 else reg
    cells = dict(core)
    # Label patches project beyond exactly one side of the core envelope.
    cells[-notch - 1, -wl] = base
    if wl == 2 and not compact and rng.random() < .4:
        cells[-notch - 1, -1] = base
    for q, color in ((-wl - 1, left), (span + wr, right)):
        start = int(rng.integers(-notch, back + 1))
        cells[start, q] = color
        if start < back and not compact and rng.random() < .4:
            cells[start + 1, q] = color
    return cells, max(wl, wr), notch


def _module(rng, box, spec, roles, allowed, profile_cap):
    """Sample one contact mechanism, including its reserved swept footprint."""
    r0, c0, bh, bw = box
    active, span, pose_class, singleton_demo = spec
    for attempt in range(100):
        compact = attempt >= 72
        local_roles = list(roles)
        if rng.integers(2):
            local_roles[2], local_roles[3] = local_roles[3], local_roles[2]
        _, base, left, right, _, hub, reg = local_roles
        profile = (0,) + tuple(map(int, rng.integers(0, profile_cap + 1, size=span - 1)))
        key_profile = list(profile)
        if not active:
            key_profile[-1] = (key_profile[-1] + 1) % (profile_cap + 1)
        key_cells, wing, notch = _key_template(
            rng, span, key_profile, local_roles, compact, singleton_demo)
        clearance = 1 if compact else int(rng.integers(1, 3))
        margin = wing + 1 + clearance
        center = notch + 2 + (0 if compact else int(rng.integers(0, 3)))
        guards = ((2, 3) if compact else tuple(map(int, rng.integers(1, center + 4, size=2))))
        fork = {(0, q): base for q in range(-margin, span + margin)}
        for q, color, length in ((-margin, left, guards[0]),
                                  (span - 1 + margin, right, guards[1])):
            fork.update({(k, q): color for k in range(1, length + 1)})
        for q, height in enumerate(profile):
            fork.update({(k, q): base for k in range(1, center + height)})
            fork[center + height, q] = hub if q == 0 else reg
        poses = range(1, 4) if pose_class == "rotate" else range(4, 8) if pose_class == "reflect" else range(8)
        t = int(rng.choice(list(poses)))
        scrambled = {_d4(p, t): v for p, v in key_cells.items()}
        fork_end = max(k for k, _ in fork)
        gap = 2 if compact else int(rng.integers(2, 5))
        start = fork_end + gap - min(min(k for k, _ in scrambled), min(k for k, _ in key_cells))
        source = _translate(scrambled, (start, 0))
        final = _translate(key_cells, (center + 1, 0)) if active else {}
        route = ({(k + shift, q) for k, q in key_cells
                  for shift in range(center + 1, start + 1)} if active else set())
        inp_local = {**fork, **source}
        out_local = dict(fork)
        out_local.update({p: 0 for p in route - final.keys()})
        out_local.update(final)
        directions = list(map(int, rng.permutation(allowed)))
        for d in directions:
            outward = _DIRS[d]
            perp = -outward[1], outward[0]

            def convert(cells):
                return {(k * outward[0] + q * perp[0],
                         k * outward[1] + q * perp[1]): v for (k, q), v in cells.items()}

            raw_in, raw_out = convert(inp_local), convert(out_local)
            occupied = raw_in.keys() | raw_out.keys()
            rr, cc = min(r for r, _ in occupied), min(c for _, c in occupied)
            height = max(r for r, _ in occupied) - rr + 1
            width = max(c for _, c in occupied) - cc + 1
            if height <= bh and width <= bw:
                shift = (r0 + int(rng.integers(bh - height + 1)) - rr,
                         c0 + int(rng.integers(bw - width + 1)) - cc)
                return _translate(raw_in, shift), _translate(raw_out, shift)
    return None


def _boxes(rng, n):
    if n == 1:
        h, w = map(int, rng.integers(18, 31, size=2))
        boxes = [(0, 0, h, w)]
    elif n == 2:
        h, w = map(int, rng.integers(24, 31, size=2))
        cut = int(rng.integers(9, h - 9))
        boxes = [(0, 0, cut, w), (cut + 1, 0, h - cut - 1, w)]
    else:
        h, w = map(int, rng.integers(28, 31, size=2))
        if w >= 29 and rng.random() < .35:
            a = int(rng.integers(9, w - 19))
            b = int(rng.integers(a + 10, w - 9))
            boxes = [(0, 0, h, a), (0, a + 1, h, b - a - 1), (0, b + 1, h, w - b - 1)]
        else:
            cut = int(rng.integers(9, 11))
            cross = int(rng.integers(12, w - 12))
            boxes = [(0, 0, cut, w), (cut + 1, 0, h - cut - 1, cross),
                     (cut + 1, cross + 1, h - cut - 1, w - cross - 1)]
    if rng.integers(2):
        boxes = [(c, r, bw, bh) for r, c, bh, bw in boxes]
        h, w = w, h
    if rng.integers(2):
        boxes = [(h - r - bh, c, bh, bw) for r, c, bh, bw in boxes]
    if rng.integers(2):
        boxes = [(r, w - c - bw, bh, bw) for r, c, bh, bw in boxes]
    rng.shuffle(boxes)
    return (h, w), boxes


def _make_pair(rng, specs, roles, allowed, profile_cap):
    for _ in range(40):
        shape, boxes = _boxes(rng, len(specs))
        inp = np.full(shape, roles[0], dtype=np.int8)
        out = inp.copy()
        modules = [_module(rng, box, spec, roles, allowed, profile_cap)
                   for box, spec in zip(boxes, specs)]
        if any(module is None for module in modules):
            continue
        for source, answer in modules:
            _paint(inp, source)
            _paint(out, answer)
        try:
            info = _analyse(inp)
            if len(info["matches"]) != sum(spec[0] for spec in specs):
                continue
            if not np.array_equal(_render(inp, info), out):
                continue
            if specs[0][3] and len(_candidates(info["keys"][0], info, use_labels=False)) < 2:
                continue
        except ValueError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("count", "direction", "profile")))
    held_out = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            train = split == "train"
            cap = 1 if train and shift == "profile" else 2
            max_span = 2 if train and shift == "profile" else 3
            if train and index == 0:
                specs = [(True, 1, "rotate", True)]
            elif train and index == 1:
                specs = [(True, 2, "reflect", False), (False, 2, "any", False)]
            else:
                n = int(rng.integers(1, 3)) if train or shift == "profile" else 3 if shift == "count" else 1
                specs = [(True if j == 0 or not train else bool(rng.random() < .7),
                          3 if not train and shift == "profile" else int(rng.integers(1, max_span + 1)),
                          "any", False) for j in range(n)]
                specs = [(active, span if active else max(2, span), pose, demo)
                         for active, span, pose, demo in specs]
            allowed = ([d for d in range(4) if d != held_out] if train else [held_out]) if shift == "direction" else list(range(4))
            for _ in range(12):
                roles = list(map(int, rng.permutation(np.arange(1, 10))[:7]))
                pair = _make_pair(rng, specs, roles, allowed, cap)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("distinct-input retries exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
    return puzzle


def _inspect_pair(inp, out):
    info = _analyse(inp)
    assert np.array_equal(_render(inp, info), out), "wrong insertion/trail output"
    assert not np.array_equal(inp, out)
    assert np.any(out == 0), "no visible trail"
    assert all(int(out[p]) == v for p, v in info["structure"].items())
    # Separation is checked from actual returned input components.
    occupied = set()
    for obj in info["objects"]:
        halo = {(r + dr, c + dc) for r, c in obj for dr, dc in product((-1, 0, 1), repeat=2)}
        assert not occupied & halo, "touching input objects"
        occupied.update(obj)
    accepted = {m["ki"] for m in info["matches"]}
    mutable = {p for key in info["keys"] for p in key["cells"]}
    for match in info["matches"]:
        source = info["keys"][match["ki"]]
        fork = info["forks"][match["fi"]]
        assert sorted(source["cells"].values()) == sorted(match["final"].values())
        assert len(source["cells"]) == len(match["final"])
        final_marks = {p: v for p, v in match["final"].items() if v in info["mark_colors"]}
        adjacent = _translate(fork["marks"], fork["out"])
        assert final_marks == adjacent, "registration faces must be adjacent"
        # The stop must actually penetrate the open groove. This excludes a
        # bounding-box stop even if it happens to preserve all object colors.
        r0, r1 = min(r for r, _ in match["final"]), max(r for r, _ in match["final"])
        c0, c1 = min(c for _, c in match["final"]), max(c for _, c in match["final"])
        assert any(r0 < r < r1 and c0 < c < c1 for r, c in fork["cells"]), "no interlocking groove"
        assert not match["route"] & info["structure"].keys()
        assert match["distance"] > 0 and match["route"] - match["final"].keys()
        for p in match["route"] - match["final"].keys():
            assert int(out[p]) == 0
        assert all(int(out[p]) == v for p, v in match["final"].items())
        mutable.update(match["route"])
    for ki, key in enumerate(info["keys"]):
        if ki not in accepted:
            assert all(int(out[p]) == info["background"] for p in key["cells"])
    assert all(int(inp[p]) == int(out[p]) for p in np.ndindex(inp.shape) if p not in mutable)
    info["unmatched"] = len(info["keys"]) - len(accepted)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos, test_infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _key(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            info = _inspect_pair(inp, out)
            if split == "train":
                train_infos.append(info)
            else:
                test_infos.append(info)
    first, second = train_infos[:2]
    assert any(len(_candidates(key, first, use_labels=False)) > 1 for key in first["keys"]), "compass not demonstrated"
    assert any(0 < m["t"] < 4 for m in first["matches"]), "rotation not demonstrated"
    assert any(m["t"] >= 4 for m in second["matches"]), "reflection not demonstrated"
    assert second["unmatched"] >= 1, "missing wrong-profile demonstration"
    used = {m["ki"] for m in second["matches"]}
    assert any(_candidates(key, second, use_profile=False, check_route=False)
               for ki, key in enumerate(second["keys"]) if ki not in used), "decoy must pass labels and axis"
    train_forks = [f for info in train_infos for f in info["forks"]]
    test_forks = [f for info in test_infos for f in info["forks"]]
    count_shift = max(len(i["forks"]) for i in test_infos) > max(len(i["forks"]) for i in train_infos)
    direction_shift = {f["out"] for f in test_forks} - {f["out"] for f in train_forks}
    span_shift = max(f["span"] for f in test_forks) > max(f["span"] for f in train_forks)
    assert count_shift or direction_shift or span_shift, "missing structural train/test shift"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and compact aggregates; no files, workers or trace logs."""
    shapes, profiles, geometries, poses, directions = set(), set(), set(), set(), set()
    distances, palettes, structures, signatures, key_shapes = set(), set(), set(), set(), set()
    counts, perforated, near_misses, total_pairs = set(), 0, 0, 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _puzzle_digest(puzzle)
        assert signature not in signatures
        signatures.add(signature)
        assert signature == _puzzle_digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                original = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, original)
                info = _inspect_pair(inp, out)
                shapes.add(inp.shape)
                palettes.add(tuple(sorted(map(int, np.unique(inp)))))
                counts.add(len(info["forks"]))
                near_misses += info["unmatched"]
                total_pairs += 1
                role_map = {}
                normalized = bytes(role_map.setdefault(int(v), len(role_map)) for v in inp.ravel())
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized).hexdigest())
                for fork in info["forks"]:
                    profiles.add(fork["profile"])
                    directions.add(fork["out"])
                    geometries.add((fork["width"], fork["profile"], fork["lengths"]))
                for match in info["matches"]:
                    poses.add(match["t"])
                    distances.add(match["distance"])
                    cells = match["final"]
                    h = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
                    w = max(c for _, c in cells) - min(c for _, c in cells) + 1
                    key_shapes.add((h, w))
                    mask = np.zeros((h + 2, w + 2), dtype=bool)
                    r0, c0 = min(r for r, _ in cells) - 1, min(c for _, c in cells) - 1
                    for r, c in cells:
                        mask[r - r0, c - c0] = True
                    perforated += max(0, len(_components(~mask)) - 1)
    assert len(shapes) >= 35 and len(geometries) >= 80
    assert len(profiles) >= 10 and directions == set(_DIRS)
    assert poses == set(range(8)) and counts == {1, 2, 3}
    assert len(distances) >= 8 and len(palettes) >= 20 and len(key_shapes) >= 20
    assert len(structures) == total_pairs and perforated >= 10 and near_misses >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(681054878, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    info = _analyse(bad["test"][0]["input"])
    p = min(info["matches"][0]["final"])
    bad["test"][0]["output"][p] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted final key accepted")
    return {"seeds": len(_HOST_SEEDS), "pairs": total_pairs,
            "shapes": len(shapes), "geometries": len(geometries),
            "profiles": len(profiles), "poses": sorted(poses),
            "directions": sorted(directions), "key_shapes": len(key_shapes),
            "distances": sorted(distances), "palettes": len(palettes),
            "structures": len(structures), "perforations": perforated,
            "unmatched_keys": near_misses, "independent_solver": False}
