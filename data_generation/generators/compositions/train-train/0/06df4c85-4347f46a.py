"""Stitched windows: only surviving edge rivets can thread opened panels.

All geometry and semantic roles are recovered from the input. No parent module,
stored puzzle, seed-dependent solver convention, or nonstandard package is used.
"""

from collections import defaultdict

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("a grid must be a 1..30 integer rectangle with colors 0..9")
    return a.astype(np.int8, copy=True)


def _geometry(boxes, height, width):
    """Boxes use inclusive bounds; label 0 is outside all panels."""
    labels = np.zeros((height, width), dtype=np.int8)
    inner = np.zeros((height, width), dtype=bool)
    for i, (t, b, l, r) in enumerate(boxes, 1):
        labels[t:b + 1, l:r + 1] = i
        inner[t + 1:b, l + 1:r] = True
    return labels, inner, (labels != 0) & ~inner


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    bg = int(a[0, 0])
    if not all(int(a[r, c]) == bg for r, c in
               ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
        raise ValueError("all four corners must display the surrounding background")
    seen = np.zeros(a.shape, dtype=bool)
    boxes, bases, pins = [], [], {}
    for y, x in np.argwhere(a != bg):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < h and 0 <= cc < w and not seen[rr, cc]
                        and int(a[rr, cc]) != bg):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        t, b = min(r for r, _ in cells), max(r for r, _ in cells)
        l, r = min(c for _, c in cells), max(c for _, c in cells)
        block = a[t:b + 1, l:r + 1]
        if min(block.shape) < 3 or len(cells) != block.size:
            raise ValueError("each nonbackground component must fill a rectangle >=3 by 3")
        colors, counts = np.unique(block, return_counts=True)
        base = int(colors[int(np.argmax(counts))])
        if int(max(counts)) * 2 <= block.size:
            raise ValueError("each panel must have a strict majority carrier color")
        boxes.append((t, b, l, r))
        bases.append(base)
        pins.update({rr * w + cc: int(a[rr, cc]) for rr, cc in cells
                     if int(a[rr, cc]) != base})
    if not boxes:
        raise ValueError("at least one panel is required")
    if set(bases) & set(pins.values()):
        raise ValueError("rivet colors and carrier colors must be disjoint in a scene")
    for i, (t, b, l, r) in enumerate(boxes):
        for tt, bb, ll, rr in boxes[i + 1:]:
            if not (b + 1 < tt or bb + 1 < t or r + 1 < ll or rr + 1 < l):
                raise ValueError("panels need at least one complete background cell of separation")
    labels, inner, edge = _geometry(boxes, h, w)
    border_pins = {p: color for p, color in pins.items() if edge.flat[p]}
    return a, bg, boxes, bases, pins, border_pins, labels, inner, edge


def _intervals(pins, width):
    """All intervals come from the supplied original coordinates, in one pass."""
    rows, columns = defaultdict(list), defaultdict(list)
    for p, color in sorted(pins.items()):
        r, c = divmod(p, width)
        rows[color, r].append(c)
        columns[color, c].append(r)
    for (color, r), cc in sorted(rows.items()):
        if len(cc) >= 2:
            yield color, "H", r, min(cc), max(cc)
    for (color, c), rr in sorted(columns.items()):
        if len(rr) >= 2:
            yield color, "V", c, min(rr), max(rr)


def _claims(pins, inner):
    w = inner.shape[1]
    result = {p: color for p, color in pins.items() if inner.flat[p]}
    for color, direction, fixed, lo, hi in _intervals(pins, w):
        for moving in range(lo, hi + 1):
            p = fixed * w + moving if direction == "H" else moving * w + fixed
            if inner.flat[p]:
                if p in result and result[p] != color:
                    raise ValueError("different thread colors collide inside a window")
                result[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, _, _, _, border_pins, _, inner, _ = _decode(input_grid)
    result = a.copy()
    result[inner] = bg
    for p, color in _claims(border_pins, inner).items():
        result.flat[p] = color
    return result


def _sample_boxes(rng, count, direction, span):
    """Free rejection placement, conditioned on a usable line of sight.

    Conditioning on overlapping projections is necessary for a thread to visit
    several panels. It imposes no equal sizes, shared edges, or regular lattice.
    """
    for _ in range(250):
        if span >= 3:
            h, w = int(rng.integers(16, 31)), int(rng.integers(23, 31))
        elif count == 1:
            h, w = int(rng.integers(10, 26)), int(rng.integers(10, 26))
        else:
            h, w = int(rng.integers(14, 31)), int(rng.integers(14, 31))
        boxes = []
        for _index in range(count):
            for _trial in range(150):
                bh = int(rng.integers(5 if count == 1 else 3, min(19, h - 2) + 1))
                bw = int(rng.integers(5 if count == 1 else 3,
                                      min(8 if span >= 3 else 19, w - 2) + 1))
                t, l = int(rng.integers(1, h - bh)), int(rng.integers(1, w - bw))
                b, r = t + bh - 1, l + bw - 1
                if all(b + 1 < tt or bb + 1 < t or r + 1 < ll or rr + 1 < l
                       for tt, bb, ll, rr in boxes):
                    boxes.append((t, b, l, r))
                    break
            else:
                break
        if len(boxes) != count:
            continue
        labels, inner, _ = _geometry(boxes, h, w)
        if max(len(set(labels[r, inner[r]]) - {0}) for r in range(h)) < span:
            continue
        if direction == "V":
            boxes = [(l, r, t, b) for t, b, l, r in boxes]
            h, w = w, h
        return boxes, h, w
    raise RuntimeError("free panel placement exhausted its bounded attempts")


def _candidates(labels, inner, edge):
    """Candidate straight chords between original boundary coordinates."""
    h, w = inner.shape
    candidates = []
    for direction in ("H", "V"):
        for fixed in range(h if direction == "H" else w):
            border = np.flatnonzero(edge[fixed] if direction == "H" else edge[:, fixed])
            for ia, first in enumerate(border):
                for last in border[ia + 1:]:
                    if last - first < 2:
                        continue
                    pp = (fixed * w + np.arange(first, last + 1) if direction == "H"
                          else np.arange(first, last + 1) * w + fixed)
                    cells = frozenset(int(p) for p in pp if inner.flat[p])
                    if not cells:
                        continue
                    p, q = int(pp[0]), int(pp[-1])
                    visits = len({int(labels.flat[x]) for x in cells})
                    candidates.append((p, q, cells, direction, visits))
    return candidates


def _recursive_differs(pins, inner):
    first = _claims(pins, inner)
    try:
        return _claims(pins | first, inner) != first
    except ValueError:
        return True


def _local_claims(pins, labels, inner):
    result = {}
    for i in range(1, int(labels.max()) + 1):
        local = {p: c for p, c in pins.items() if int(labels.flat[p]) == i}
        result.update(_claims(local, inner & (labels == i)))
    return result


def _unmarked_window(pins, claims, labels):
    by_color = defaultdict(set)
    for p, color in pins.items():
        by_color[color].add(int(labels.flat[p]))
    return any(int(labels.flat[p]) not in by_color[color] for p, color in claims.items())


def _add_decoy(rng, pins, border_pins, inner, labels, boxes, color):
    """Install one or two erased rivets that would imply a spurious run."""
    proper = _claims(border_pins, inner)
    free = [int(p) for p in np.flatnonzero(inner) if int(p) not in proper]
    for index in rng.permutation(len(free)):
        p = free[int(index)]
        trial = dict(border_pins)
        trial[p] = color
        try:
            wrong = _claims(trial, inner)
        except ValueError:
            continue
        if set(wrong) - set(proper) - {p} and _room_for_pin(pins, p, labels, boxes):
            pins[p] = color
            return set(wrong) - set(proper) - {p}
    # A pair of interior rivets can also tempt an incorrect completion. Their
    # intervening cells, as well as the rivets themselves, must remain empty.
    shuffled = [free[int(i)] for i in rng.permutation(len(free))]
    w = inner.shape[1]
    for p in shuffled:
        for q in shuffled:
            pr, pc = divmod(p, w)
            qr, qc = divmod(q, w)
            if not ((pr == qr and abs(pc - qc) >= 2)
                    or (pc == qc and abs(pr - qr) >= 2)):
                continue
            trial = border_pins | {p: color, q: color}
            try:
                wrong = _claims(trial, inner)
            except ValueError:
                continue
            if (set(wrong) - set(proper) - {p, q}
                    and _room_for_pin(pins, p, labels, boxes)
                    and _room_for_pin(pins | {p: color}, q, labels, boxes)):
                pins[p] = pins[q] = color
                return set(wrong) - set(proper) - {p, q}
    return set()


def _room_for_pin(pins, p, labels, boxes):
    i = int(labels.flat[p])
    t, b, l, r = boxes[i - 1]
    count = sum(int(labels.flat[q]) == i for q in pins)
    return (count + (p not in pins)) * 2 < (b - t + 1) * (r - l + 1)


def _make_pair(rng, kind, direction, previous_bg=None):
    # The two core demonstrations are informative even for a 2/1 request.
    if kind == "off_axis":
        count, span, groups = int(rng.integers(1, 3)), 1, int(rng.integers(1, 3))
    elif kind == "global":
        count, span, groups = 3, 3, 2
    elif kind == "test":
        count, span, groups = int(rng.integers(4, 6)), 3, int(rng.integers(2, 4))
    else:
        count = int(rng.integers(1, 5))
        span = int(rng.integers(1, min(3, count) + 1))
        groups = int(rng.integers(1, 4))
    for _attempt in range(60):
        boxes, h, w = _sample_boxes(rng, count, direction, span)
        labels, inner, edge = _geometry(boxes, h, w)
        candidates = _candidates(labels, inner, edge)
        primary = [x for x in candidates if x[3] == direction and x[4] >= span
                   and len(x[2]) >= 2 and
                   (kind != "off_axis" or labels.flat[x[0]] == labels.flat[x[1]])]
        if not primary:
            continue
        base_count = int(rng.integers(1, min(count, 3) + 1))
        palette = [int(x) for x in rng.permutation(10)]
        if previous_bg is not None and palette[0] == previous_bg:
            palette[0], palette[-1] = palette[-1], palette[0]
        bg = palette[0]
        carriers = palette[1:1 + base_count]
        accents = palette[1 + base_count:1 + base_count + groups]
        bases = [int(rng.choice(carriers)) for _ in boxes]
        p, q, _, _, _ = primary[int(rng.integers(len(primary)))]
        border_pins = {p: accents[0], q: accents[0]}
        if kind == "off_axis":
            initial = _claims(border_pins, inner)
            extra = [int(x) for x in np.flatnonzero(edge) if x not in border_pins]
            for index in rng.permutation(len(extra)):
                trial = border_pins | {extra[int(index)]: accents[0]}
                if _claims(trial, inner) == initial and _recursive_differs(trial, inner):
                    border_pins = trial
                    break
            else:
                continue
        # Additional colors have their own chords, never a paint-order tie.
        success = True
        for color in accents[1:]:
            for index in rng.permutation(len(candidates)):
                pp, qq, _, _, _ = candidates[int(index)]
                if pp in border_pins or qq in border_pins:
                    continue
                trial = border_pins | {pp: color, qq: color}
                try:
                    _claims(trial, inner)
                except ValueError:
                    continue
                if (_room_for_pin(border_pins, pp, labels, boxes)
                        and _room_for_pin(border_pins | {pp: color}, qq, labels, boxes)):
                    border_pins = trial
                    break
            else:
                success = False
                break
        if not success:
            continue
        if kind == "global":
            # An unused chord has differently colored ends. Connecting any
            # aligned dots, without checking equality, would falsely fill it.
            endpoint_panels = {int(labels.flat[p]), int(labels.flat[q])}
            for index in rng.permutation(len(candidates)):
                pp, qq, cells, _, _ = candidates[int(index)]
                if (pp in border_pins or qq in border_pins
                        or labels.flat[pp] != labels.flat[qq]
                        or int(labels.flat[pp]) not in endpoint_panels):
                    continue
                c1, c2 = accents[:2] if rng.integers(2) else accents[:2][::-1]
                trial = border_pins | {pp: c1, qq: c2}
                try:
                    claims = _claims(trial, inner)
                except ValueError:
                    continue
                if (cells - set(claims)
                        and _room_for_pin(border_pins, pp, labels, boxes)
                        and _room_for_pin(border_pins | {pp: c1}, qq, labels, boxes)):
                    border_pins = trial
                    break
            else:
                continue
        # Extra same-colored edge rivets broaden collinearity and junctions.
        if kind == "mixed" or (kind == "test" and rng.random() < 0.55):
            for _ in range(int(rng.integers(0, 4))):
                available = [int(x) for x in np.flatnonzero(edge) if x not in border_pins]
                if not available:
                    break
                pp = int(rng.choice(available))
                trial = border_pins | {pp: int(rng.choice(accents))}
                try:
                    _claims(trial, inner)
                except ValueError:
                    continue
                if _room_for_pin(border_pins, pp, labels, boxes):
                    border_pins = trial
        proper = _claims(border_pins, inner)
        if kind == "off_axis" and not _recursive_differs(border_pins, inner):
            continue
        if kind == "global" and not _unmarked_window(border_pins, proper, labels):
            continue
        pins = dict(border_pins)
        reserved = _add_decoy(rng, pins, border_pins, inner, labels, boxes, accents[0])
        if not reserved:
            continue
        for _ in range(int(rng.integers(0, 5))):
            pp = int(rng.choice(np.flatnonzero(inner)))
            if pp not in pins and pp not in reserved and _room_for_pin(pins, pp, labels, boxes):
                pins[pp] = int(rng.choice(accents))
        a = np.full((h, w), bg, dtype=np.int8)
        for (t, b, l, r), base in zip(boxes, bases):
            a[t:b + 1, l:r + 1] = base
        for pp, color in pins.items():
            a.flat[pp] = color
        # Construction uses explicit masks; solve independently decodes input.
        out = a.copy()
        out[inner] = bg
        for pp, color in proper.items():
            out.flat[pp] = color
        return {"input": a, "output": out}
    raise RuntimeError("could not construct informative, nonconflicting rivets")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first_direction = str(rng.choice(("H", "V")))
    puzzle, seen = {"train": [], "test": []}, set()
    last_bg = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            kind = ("test" if split == "test" else
                    "off_axis" if index == 0 else "global" if index == 1 else "mixed")
            direction = (first_direction if kind == "off_axis" else
                         ("V" if first_direction == "H" else "H") if kind == "global"
                         else str(rng.choice(("H", "V"))))
            for _ in range(30):
                pair = _make_pair(rng, kind, direction, last_bg)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
            seen.add(key)
            last_bg = int(pair["input"][0, 0])
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(a):
    a, bg, boxes, bases, pins, border, labels, inner, edge = _decode(a)
    proper = _claims(border, inner)
    visible_directions = set()
    visits = 0
    for color, direction, fixed, lo, hi in _intervals(border, a.shape[1]):
        pp = [fixed * a.shape[1] + x if direction == "H" else x * a.shape[1] + fixed
              for x in range(lo, hi + 1)]
        used = [p for p in pp if inner.flat[p]]
        if used:
            visible_directions.add(direction)
            visits = max(visits, len({int(labels.flat[p]) for p in used}))
    erased = {p for p in pins if inner.flat[p] and p not in proper}
    # Wrong eligibility must imply extra cells beyond retaining erased rivets.
    decoy_contrast = False
    for p in erased:
        try:
            wrong = _claims(border | {p: pins[p]}, inner)
            decoy_contrast |= bool(set(wrong) - set(proper) - {p})
        except ValueError:
            decoy_contrast = True
    try:
        wrong = _claims(pins, inner)
        decoy_contrast |= bool(set(wrong) - set(proper) - set(pins))
    except ValueError:
        decoy_contrast = True
    return {"bg": bg, "boxes": boxes, "bases": bases, "pins": pins, "border": border,
            "labels": labels, "inner": inner, "edge": edge, "claims": proper,
            "directions": visible_directions, "visits": visits, "erased": erased,
            "decoy_contrast": decoy_contrast,
            "recursive_contrast": _recursive_differs(border, inner),
            "global_contrast": proper != _local_claims(border, labels, inner),
            "color_contrast": bool(set(_claims({p: 0 for p in border}, inner)) - set(proper)),
            "unmarked_window": _unmarked_window(border, proper, labels)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _evidence(a)
            inner, proper, bg = info["inner"], info["claims"], info["bg"]
            assert proper and info["erased"] and info["decoy_contrast"]
            assert np.array_equal(a[~inner], b[~inner]), "all frames and exterior must survive"
            # Cellwise mask equation, rather than solving the returned output.
            for p in np.flatnonzero(inner):
                assert int(b.flat[p]) == proper.get(int(p), bg)
            assert np.any(b[inner] == bg), "opening must leave visible empty space"
            assert np.array_equal(b, solve(a))
            assert not np.array_equal(a, b)
            if split == "train":
                train_info.append(info)
            else:
                assert 4 <= len(info["boxes"]) <= 5 and info["visits"] >= 3
    assert set.union(*(x["directions"] for x in train_info)) == {"H", "V"}
    assert any(x["recursive_contrast"] for x in train_info)
    assert any(x["global_contrast"] and x["unmarked_window"] and x["visits"] >= 3
               for x in train_info)
    assert len({x["bg"] for x in train_info}) >= 2
    assert any(len(set(x["border"].values())) >= 2 for x in train_info)
    assert any(x["color_contrast"] for x in train_info)


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity audit plus all legal explicit count pairs."""
    shapes, sizes, backgrounds, carriers, counts, structures, signatures = (
        set() for _ in range(7))
    directions, visits, anchor_counts, erased_counts = set(), set(), set(), set()
    repeated_carriers = black_rivets = global_pairs = recursive_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _evidence(a)
                shapes.add(a.shape)
                backgrounds.add(info["bg"])
                carriers.update(info["bases"])
                counts.add(len(info["boxes"]))
                for t, b, l, r in info["boxes"]:
                    sizes.add((b - t + 1, r - l + 1))
                roles = {}
                structures.add((a.shape, bytes(roles.setdefault(int(c), len(roles))
                                               for c in a.flat)))
                directions.add(tuple(sorted(info["directions"])))
                visits.add(info["visits"])
                anchor_counts.add(len(info["border"]))
                erased_counts.add(len(info["erased"]))
                repeated_carriers += len(info["bases"]) > len(set(info["bases"]))
                black_rivets += 0 in info["pins"].values()
                global_pairs += info["global_contrast"]
                recursive_pairs += info["recursive_contrast"]
                # Axis swapping and arbitrary color bijections cannot change the rule.
                if seed < 3:
                    assert np.array_equal(solve(a.T), pair["output"].T)
                    perm = np.random.default_rng(seed + 101).permutation(10)
                    assert np.array_equal(solve(perm[a]), perm[pair["output"]])
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and len(sizes) >= 70
    assert backgrounds == set(range(10)) and carriers == set(range(10))
    assert {1, 2, 3, 4, 5} <= counts
    assert {("H",), ("V",), ("H", "V")} <= directions
    assert {1, 2, 3} <= visits and len(anchor_counts) >= 5 and len(erased_counts) >= 4
    assert repeated_carriers >= 60 and black_rivets >= 10
    assert global_pairs >= 60 and recursive_pairs >= 42
    # Small hand-built witnesses test the meaning of the rule, beyond sampled
    # output agreement. Coordinates below are zero based.
    probe = np.zeros((11, 11), dtype=np.int8)
    probe[1:10, 1:10] = 7
    for r, c in ((1, 5), (9, 5), (5, 9), (3, 3), (7, 3)):
        probe[r, c] = 2
    answer = solve(probe)
    assert np.all(answer[2:9, 5] == 2)
    assert answer[5, 7] == answer[5, 3] == answer[3, 3] == 0
    assert answer[5, 9] == 2 and answer[1, 1] == 7
    probe = np.zeros((11, 21), dtype=np.int8)
    for (t, b, l, r), color in zip(((1, 7, 1, 5), (3, 9, 8, 12), (2, 8, 15, 19)),
                                  (7, 8, 7)):
        probe[t:b + 1, l:r + 1] = color
    probe[4, 1] = probe[4, 19] = probe[5, 1] = 2
    probe[5, 5] = 4
    answer = solve(probe)
    for left, right in ((2, 5), (9, 12), (16, 19)):
        assert np.all(answer[4, left:right] == 2)
    assert answer[4, 8] == 8 and answer[4, 6] == 0
    assert np.all(answer[5, 2:5] == 0) and answer[5, 5] == 4
    # The solver also handles the domain's smallest panels and complete panels
    # at a noncorner canvas edge; generation normally leaves a clear margin.
    probe = np.zeros((7, 7), dtype=np.int8)
    probe[0:3, 2:5] = 7
    probe[1, 2] = probe[1, 4] = 3
    assert solve(probe)[1, 3] == 3
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(7000 + 10 * ntrain + ntest, ntrain, ntest)
            assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                          (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(42, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _same(generate_puzzle(314159, 5, 2), generate_puzzle(314159, 5, 2))
    assert not _same(generate_puzzle(314159), generate_puzzle(314160))


if __name__ == "__main__":
    self_test()
    print("self_test passed")
