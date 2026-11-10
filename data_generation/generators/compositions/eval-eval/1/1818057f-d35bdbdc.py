"""Cross-clock buses: local plus counts control read-only color-map iteration.

Standard library and NumPy only. Generation and checks never write artifacts.
The input-only solver does not call generation or use a puzzle-wide dictionary.
"""

import hashlib
import numbers

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 442127958, 60281392, 293066279, 70371148, 663239008,
              749069849, 865833401, 412211906, 447718265, 674800636,
              36909784, 912127341, 192684946, 814337443, 792709981,
              640188504, 720626181, 174259298, 107953200, 915641418,
              581247939, 480206196, 459252539, 247240081, 105498121,
              603505709, 262843218, 903788454, 888118791, 285604367,
              673864536, 334568980, 803162175, 360653113, 549103445,
              928099764, 873280764, 12711069, 95447997, 470763386,
              194171232)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " outside legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _components(cells):
    todo = set(cells)
    parts = []
    while todo:
        first = min(todo)
        todo.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in D4:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _cross_centers(cells):
    return {p for p in cells
            if all((p[0] + dr, p[1] + dc) in cells for dr, dc in D4)}


def _ends(cells):
    return {p for p in cells
            if sum((p[0] + dr, p[1] + dc) in cells for dr, dc in D4) == 1}


def _support(center, kind):
    r, c = center
    if kind == "cross":
        return {center} | {(r + dr, c + dc) for dr, dc in D4}
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _neighboring(cells, diagonal=False):
    if diagonal:
        directions = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    else:
        directions = ((0, 0),) + D4
    return {(r + dr, c + dc) for r, c in cells for dr, dc in directions}


def _parse_with_roles(a, background, wire):
    wire_cells = set(zip(*np.where(a == wire)))
    buses = _components(wire_cells)
    assert buses and all(len(b) > 9 for b in buses), "wire components must exceed token size"
    clocks = [_cross_centers(b) for b in buses]
    assert all(clocks), "each bus needs a complete plus"
    ends_by_bus = [_ends(b) for b in buses]
    assert all(ends_by_bus), "each bus needs an endpoint"
    tokens = []
    for cells in _components(zip(*np.where((a != background) & (a != wire)))):
        assert len(cells) in (5, 9), "token support must be complete"
        r = (min(p[0] for p in cells) + max(p[0] for p in cells)) // 2
        c = (min(p[1] for p in cells) + max(p[1] for p in cells)) // 2
        center = r, c
        kind = "cross" if len(cells) == 5 else "square"
        assert cells == _support(center, kind), "token has no unique centered support"
        bodies = {int(a[p]) for p in cells if p != center}
        assert len(bodies) == 1
        contacts = [(i, e) for i, ends in enumerate(ends_by_bus)
                    for dr, dc in D4 if (e := (r + 2 * dr, c + 2 * dc)) in ends]
        assert len(contacts) <= 1, "ambiguous endpoint attachment"
        tokens.append(dict(center=center, cells=cells, kind=kind,
                           body=bodies.pop(), value=int(a[center]),
                           bus=contacts[0][0] if contacts else None))
    assert tokens, "no tokens"
    mapping = {}
    for t in tokens:
        assert t["body"] not in mapping or mapping[t["body"]] == t["value"], "conflicting donors"
        mapping[t["body"]] = t["value"]
    assert any(t["bus"] is not None for t in tokens), "no receivers"
    return dict(background=background, wire=wire, buses=buses, clocks=clocks,
                ends=ends_by_bus, tokens=tokens, mapping=mapping)


def _interpret(a):
    colors, counts = np.unique(a, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert counts.max() > a.size / 2, "background must occupy a strict majority"
    candidates = []
    for wire in colors:
        if wire == background:
            continue
        try:
            candidates.append(_parse_with_roles(a, background, int(wire)))
        except AssertionError:
            pass
    assert len(candidates) == 1, "wire/token interpretation must be unique"
    return candidates[0]


def _walk(value, count, mapping):
    for _ in range(count):
        if value not in mapping:
            return None
        value = mapping[value]
    return value


def _answer(a, parsed, mode="clock"):
    answer = a.copy()
    for t in parsed["tokens"]:
        bus = t["bus"]
        value = None
        if bus is not None:
            count = len(parsed["clocks"][bus])
            if mode == "one_step":
                count = 1
            elif mode == "unchanged":
                count = 0
            value = _walk(t["value"], count, parsed["mapping"])
        if value is None:
            for p in t["cells"]:
                answer[p] = parsed["background"]
        else:
            answer[t["center"]] = value
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from geometry and original color relations in this input."""
    a = _grid(input_grid)
    return _answer(a, _interpret(a))


def _clock_shape(rng, pulses, adjacent, receiver_count, token_kind=None):
    """A bus and attached supports in local coordinates, before free packing."""
    length = int(rng.integers(max(8, pulses + 4), 14))
    line = {(0, c) for c in range(length + 1)}
    available = list(range(2, length - 1))
    if adjacent:
        first = int(rng.integers(2, length - pulses))
        cols = list(range(first, first + pulses))
    else:
        cols = list(map(int, rng.choice(available, pulses, replace=False)))
    for c in cols:
        line.update({(-1, c), (1, c)})
    endpoints, centers = [], []
    for col, side in ((0, -1), (length, 1)):
        extension = int(rng.choice((0, 0, 2, 3, 4)))
        direction = int(rng.choice((-1, 1)))
        if extension:
            line.update((direction * k, col) for k in range(1, extension + 1))
            end, center = (direction * extension, col), (direction * (extension + 2), col)
        else:
            end, center = (0, col), (0, col + 2 * side)
        endpoints.append(end)
        centers.append(center)
    if receiver_count == 1:
        chosen = [int(rng.integers(2))]
    else:
        chosen = [0, 1]
    tokens = [dict(center=centers[i], kind=token_kind or str(rng.choice(("square", "cross"))))
              for i in chosen]
    occupied_tokens = set().union(*(_support(t["center"], t["kind"]) for t in tokens))
    assert not (occupied_tokens & line)
    # Extra spurs and extended arms are allowed only if the visible clock and
    # receiver contacts remain unchanged. No diagnostic traces are retained.
    for _ in range(int(rng.integers(1, 8))):
        p = sorted(line)[int(rng.integers(len(line)))]
        dr, dc = D4[int(rng.integers(4))]
        extra = {(p[0] + k * dr, p[1] + k * dc)
                 for k in range(1, int(rng.integers(2, 5)))}
        proposed = line | extra
        if (not proposed & occupied_tokens and len(_cross_centers(proposed)) == pulses
                and all(end in _ends(proposed) for end in endpoints)):
            line = proposed
    assert len(_cross_centers(line)) == pulses and len(line) > 9
    # Every chosen center must have exactly its intended endpoint contact.
    ends = _ends(line)
    for t in tokens:
        assert sum((t["center"][0] + 2 * dr, t["center"][1] + 2 * dc) in ends
                   for dr, dc in D4) == 1
    return line, tokens


def _orient_group(rng, line, tokens):
    swap, sr, sc = bool(rng.integers(2)), int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))

    def transform(p):
        r, c = p[::-1] if swap else p
        return sr * r, sc * c

    line = {transform(p) for p in line}
    tokens = [dict(t, center=transform(t["center"])) for t in tokens]
    cells = line | set().union(*(_support(t["center"], t["kind"]) for t in tokens))
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    line = {(r - r0, c - c0) for r, c in line}
    tokens = [dict(t, center=(t["center"][0] - r0, t["center"][1] - c0)) for t in tokens]
    cells = {(r - r0, c - c0) for r, c in cells}
    return line, tokens, cells


def _place_groups(rng, pulse_counts, adjacent_first, selected_bodies, token_kind=None):
    # Each attempt holds only one candidate canvas; no retries are saved.
    low = 18 if len(pulse_counts) == 1 else 24
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    occupied, all_line, tokens = set(), set(), []
    for bus, pulses in enumerate(pulse_counts):
        group = _clock_shape(rng, pulses, adjacent_first and bus == 0,
                             len(selected_bodies[bus]), token_kind)
        placed = False
        for _ in range(80):
            line, new_tokens, cells = _orient_group(rng, *group)
            gh, gw = 1 + max(r for r, c in cells), 1 + max(c for r, c in cells)
            if gh > shape[0] or gw > shape[1]:
                continue
            dr = int(rng.integers(shape[0] - gh + 1))
            dc = int(rng.integers(shape[1] - gw + 1))
            moved = {(r + dr, c + dc) for r, c in cells}
            if moved & _neighboring(occupied, diagonal=True):
                continue
            all_line.update((r + dr, c + dc) for r, c in line)
            for body, t in zip(selected_bodies[bus], new_tokens):
                center = t["center"][0] + dr, t["center"][1] + dc
                tokens.append(dict(center=center, kind=t["kind"], body=body,
                                   bus=bus, cells=_support(center, t["kind"])))
            occupied.update(moved)
            placed = True
            break
        if not placed:
            return None
    return shape, all_line, tokens


def _put_donor(rng, shape, line, tokens, body, force_interior, token_kind=None):
    kind = token_kind or str(rng.choice(("square", "cross")))
    forbidden_tokens = _neighboring(set().union(*(t["cells"] for t in tokens)))
    ends = _ends(line)
    candidates = []
    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            center = r, c
            cells = _support(center, kind)
            if cells & (forbidden_tokens | line):
                continue
            if any((r + 2 * dr, c + 2 * dc) in ends for dr, dc in D4):
                continue
            if force_interior and not any((r + 2 * dr, c + 2 * dc) in line - ends
                                          for dr, dc in D4):
                continue
            candidates.append((center, cells))
    if not candidates:
        return False
    center, cells = candidates[int(rng.integers(len(candidates)))]
    tokens.append(dict(center=center, kind=kind, body=body, cells=cells, bus=None))
    return True


def _relations(rng, n, topology):
    mapping = {i: i + 1 for i in range(n - 1)}
    if topology == "chain":
        mapping[n - 1] = n if n < 8 else 0
    elif topology == "cycle":
        mapping[n - 1] = 0
    elif topology == "fixed":
        mapping[5] = 5
        for i in range(6, n):
            mapping[i] = int(rng.integers(1, 6))
    else:  # A chain entering a short cycle, with optional convergent branches.
        mapping[5] = 3
        for i in range(6, n):
            mapping[i] = int(rng.integers(1, 4))
    return mapping


def _make_pair(rng, pulse_counts, witness, palette, topology, seen):
    background, wire = palette[:2]
    identities = palette[2:]
    n = int(rng.integers(6, 8 if witness == "two" else 9))
    mapping = _relations(rng, n, topology)
    morphology = str(rng.choice(("mixed", "square", "cross"), p=(.6, .2, .2)))
    token_kind = None if morphology == "mixed" else morphology
    if witness == "one":
        selected_bodies = [[0, 1]]
    elif witness == "two":
        # Identical visible queries on different clocks expose the exponent
        # within one example. The high query succeeds after one lookup, whose
        # result has no donor, but fails when a second lookup is requested.
        selected_bodies = [[0, n - 2], [0, n - 2]]
        for bodies in selected_bodies:
            rng.shuffle(bodies)
    else:
        selected_bodies = [[0]]
        if rng.random() < .75:
            selected_bodies[0].append(int(rng.choice((1, n - 1))))
        if len(pulse_counts) == 2:
            selected_bodies.append([1])
            if rng.random() < .5:
                selected_bodies[1].append(int(rng.choice((0, n - 1))))
    for _ in range(64):
        try:
            placed = _place_groups(rng, pulse_counts, witness == "two" or rng.random() < .45,
                                   selected_bodies, token_kind)
        except AssertionError:
            continue
        if placed is None:
            continue
        shape, line, tokens = placed
        represented = {t["body"] for t in tokens}
        # Body 2 is never selected, so this donor also supplies the mandatory
        # compatible interior-contact contrast without an unrelated decoration.
        missing = [2] + [b for b in range(n) if b not in represented and b != 2]
        duplicate_count = int(rng.integers(0, min(2, 12 - len(tokens) - len(missing)) + 1))
        missing += list(map(int, rng.integers(0, n, size=duplicate_count)))
        good = True
        for index, body in enumerate(missing):
            if not _put_donor(rng, shape, line, tokens, body,
                              force_interior=bool(witness) and index == 0, token_kind=token_kind):
                good = False
                break
        if not good:
            continue
        a = np.full(shape, background, dtype=np.int8)
        for p in line:
            a[p] = wire
        for t in tokens:
            t["value"] = mapping[t["body"]]
            for p in t["cells"]:
                a[p] = identities[t["body"]]
            a[t["center"]] = identities[t["value"]]
        if np.mean(a == background) <= .5 or _key(a) in seen:
            continue
        # Labels are constructed from sampled relations and explicit attachments,
        # independently of the input parser that solve() and validation use.
        out = a.copy()
        for t in tokens:
            value = None
            if t["bus"] is not None:
                value = t["value"]
                for _ in range(pulse_counts[t["bus"]]):
                    if value not in mapping:
                        value = None
                        break
                    value = mapping[value]
            if value is None:
                for p in t["cells"]:
                    out[p] = background
            else:
                out[t["center"]] = identities[value]
        try:
            parsed = _interpret(a)
            assert sorted(map(len, parsed["clocks"])) == sorted(pulse_counts)
            assert len(parsed["tokens"]) == len(tokens)
            assert np.array_equal(_answer(a, parsed), out)
            if witness:
                f = _features(a, parsed)
                assert f["interior_compatible"] and f["remote_donor"]
                if witness == "one":
                    assert f["retained_donor"]
                if witness == "two":
                    assert f["adjacent"] and f["one_step_diff"] and f["clock_contrast"]
                    assert f["failed"] and f["terminal_missing"]
        except AssertionError:
            continue
        return dict(input=a, output=out)
    raise RuntimeError("bounded packing attempts exhausted")


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    stable_roles = bool(rng.random() < .3)
    initial_palette = list(map(int, rng.permutation(10)))
    topology_preference = str(rng.choice(("chain", "cycle", "fork", "fixed")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            if stable_roles:
                palette = initial_palette[:2] + list(map(int, rng.permutation(initial_palette[2:])))
            else:
                palette = list(map(int, rng.permutation(10)))
                if split == "train" and index == 1 and palette[:2] == first_roles:
                    palette[0], palette[1] = palette[1], palette[0]
            if split == "train" and index == 0:
                first_roles = palette[:2]
            witness = "one" if split == "train" and index == 0 else (
                      "two" if split == "train" and index == 1 else "")
            if witness == "one":
                pulses, topology = [1], "chain"
            elif witness == "two":
                pulses, topology = [2, 1], "chain"
            elif split == "train":
                pulses = [int(rng.integers(1, 3))]
                if rng.random() < .4:
                    pulses.append(3 - pulses[0])
                topology = topology_preference if rng.random() < .65 else str(rng.choice(("chain", "cycle", "fork", "fixed")))
            else:
                pulses = [int(rng.integers(3, 5))]
                if rng.random() < .45:
                    pulses.append(int(rng.choice([v for v in range(1, 5) if v != pulses[0]])))
                topology = topology_preference if rng.random() < .65 else str(rng.choice(("chain", "cycle", "fork", "fixed")))
            pair = _make_pair(rng, pulses, witness, palette, topology, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a, parsed=None):
    parsed = _interpret(a) if parsed is None else parsed
    out = _answer(a, parsed)
    tokens, mapping = parsed["tokens"], parsed["mapping"]
    line = set().union(*parsed["buses"])
    ends = set().union(*parsed["ends"])
    kept = [t for t in tokens if t["bus"] is not None
            and _walk(t["value"], len(parsed["clocks"][t["bus"]]), mapping) is not None]
    interior_compatible, retained_donor, remote_donor = False, False, False
    for t in tokens:
        if t["bus"] is None and _walk(t["value"], 1, mapping) is not None:
            r, c = t["center"]
            interior_compatible |= any((r + 2 * dr, c + 2 * dc) in line - ends for dr, dc in D4)
    for t in kept:
        v = t["value"]
        for _ in range(len(parsed["clocks"][t["bus"]])):
            for donor in tokens:
                if donor["body"] == v:
                    remote_donor |= donor["bus"] is None
                    retained_donor |= (donor in kept and donor is not t
                                       and int(out[donor["center"]]) != donor["value"])
            v = mapping[v]
    centers = set().union(*parsed["clocks"])
    union = centers | {(r + dr, c + dc) for r, c in centers for dr, dc in D4}
    return dict(shape=a.shape, roles=(parsed["background"], parsed["wire"]),
                buses=len(parsed["buses"]), tokens=len(tokens),
                pulses=tuple(sorted(map(len, parsed["clocks"]))),
                survivors=len(kept), failed=sum(t["bus"] is not None for t in tokens) - len(kept),
                overlap=len(union) < 5 * len(centers),
                adjacent=any((r + dr, c + dc) in centers for r, c in centers for dr, dc in D4),
                interior_compatible=interior_compatible, retained_donor=retained_donor,
                remote_donor=remote_donor, repeated=len(tokens) > len(mapping),
                monochrome=any(t["body"] == t["value"] for t in tokens),
                unchanged=any(int(out[t["center"]]) == t["value"] for t in kept),
                terminal_missing=any(int(out[t["center"]]) not in mapping for t in kept),
                clock_contrast=any(t["body"] == u["body"] and t["bus"] != u["bus"]
                                   and len(parsed["clocks"][t["bus"]]) != len(parsed["clocks"][u["bus"]])
                                   and int(out[t["center"]]) != int(out[u["center"]])
                                   for t in kept for u in kept),
                kinds=tuple(sorted({t["kind"] for t in tokens})),
                one_step_diff=not np.array_equal(out, _answer(a, parsed, "one_step")),
                unchanged_diff=not np.array_equal(out, _answer(a, parsed, "unchanged")),
                bends=sum(sum((r + dr, c + dc) in line for dr, dc in D4) >= 2
                          and any((r + dr, c) in line for dr in (-1, 1))
                          and any((r, c + dc) in line for dc in (-1, 1)) for r, c in line),
                edge=any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1) for r, c in line))


def validate_generated_puzzle(puzzle):
    """Validate labels and training witnesses using only the returned grids."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            parsed = _interpret(a)
            assert 1 <= len(parsed["buses"]) <= 2
            assert all(1 <= len(c) <= 4 for c in parsed["clocks"])
            assert 6 <= len(parsed["tokens"]) <= 12
            assert np.array_equal(out, _answer(a, parsed)), "wrong clocked lookup or deletion"
            support = set().union(*(t["cells"] for t in parsed["tokens"]))
            unchanged_mask = np.ones(a.shape, dtype=bool)
            for p in support:
                unchanged_mask[p] = False
            assert np.array_equal(a[unchanged_mask], out[unchanged_mask]), "wire/background changed"
            for t in parsed["tokens"]:
                bus = t["bus"]
                value = None if bus is None else _walk(t["value"], len(parsed["clocks"][bus]), parsed["mapping"])
                if value is None:
                    assert all(out[p] == parsed["background"] for p in t["cells"])
                else:
                    assert out[t["center"]] == value
                    assert all(out[p] == a[p] for p in t["cells"] if p != t["center"])
            f = _features(a, parsed)
            assert 1 <= f["survivors"] < f["tokens"]
            assert not np.array_equal(a, out)
            features[split].append(f)
    train, test = features["train"], features["test"]
    assert any(f["pulses"] == (1,) and f["interior_compatible"] for f in train)
    assert any(f["pulses"] == (1, 2) and f["adjacent"] and f["interior_compatible"]
               and f["clock_contrast"] and f["failed"] and f["terminal_missing"] for f in train)
    for name in ("one_step_diff", "unchanged_diff", "remote_donor", "retained_donor"):
        assert any(f[name] for f in train), "missing training contrast: " + name
    assert max(max(f["pulses"]) for f in train) == 2
    assert all(max(f["pulses"]) in (3, 4) for f in test)


def _expect_failure(fn, exception):
    try:
        fn()
    except exception:
        return
    raise AssertionError("invalid request or corrupted label accepted")


def _intervention_checks(puzzle):
    donor_witness, pulse_witness = False, False
    for pair in puzzle["train"]:
        a = pair["input"]
        parsed = _interpret(a)
        expected = solve(a)
        # Edit a visible function edge, changing every duplicate consistently.
        for body in parsed["mapping"]:
            alternatives = sorted(set(parsed["mapping"]) - {parsed["mapping"][body]})
            for value in alternatives:
                changed = a.copy()
                for t in parsed["tokens"]:
                    if t["body"] == body:
                        changed[t["center"]] = value
                if not np.array_equal(solve(changed), expected):
                    donor_witness = True
                    break
            if donor_witness:
                break
        # Remove one arm, preserving connectivity and a positive but lower count.
        for i, centers in enumerate(parsed["clocks"]):
            if len(centers) < 2:
                continue
            for r, c in sorted(centers):
                for dr, dc in D4:
                    changed = a.copy()
                    changed[r + dr, c + dc] = parsed["background"]
                    try:
                        new = _interpret(changed)
                    except AssertionError:
                        continue
                    if (len(new["buses"]) == len(parsed["buses"])
                            and sum(map(len, new["clocks"])) < sum(map(len, parsed["clocks"]))
                            and {t["center"] for t in new["tokens"] if t["bus"] is not None}
                            == {t["center"] for t in parsed["tokens"] if t["bus"] is not None}
                            and not np.array_equal(solve(changed), expected)):
                        pulse_witness = True
                        break
                if pulse_witness:
                    break
            if pulse_witness:
                break
    assert donor_witness and pulse_witness, "both inherited mechanisms must affect answers"


def self_test():
    """Bounded streaming checks over the 42 supplied seeds and count API."""
    fingerprints, shapes, roles, normalized = set(), set(), set(), set()
    observed = {k: set() for k in ("tokens", "survivors", "pulses", "kinds", "buses")}
    counts = {k: 0 for k in ("overlap", "adjacent", "failed", "repeated", "monochrome",
                              "unchanged", "edge", "retained_donor", "interior_compatible",
                              "clock_contrast", "terminal_missing")}
    varying, stable = 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        role_set = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _features(a)
                shapes.add(f["shape"])
                roles.add(f["roles"])
                role_set.add(f["roles"])
                for name in observed:
                    observed[name].add(f[name])
                for name in counts:
                    counts[name] += bool(f[name])
                colors = {}
                canonical = bytes(colors.setdefault(int(v), len(colors)) for v in a.flat)
                normalized.add((a.shape, hashlib.sha256(canonical).digest()))
                assert np.array_equal(solve(a.tolist()), pair["output"])
        varying += len(role_set) > 1
        stable += len(role_set) == 1
        if seed == 0:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
            _intervention_checks(puzzle)
            bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            a = bad["train"][0]["input"]
            parsed = _interpret(a)
            point = min(parsed["buses"][0])
            bad["train"][0]["output"][point] = parsed["background"]
            _expect_failure(lambda: validate_generated_puzzle(bad), AssertionError)
    assert len(fingerprints) == 42
    assert len(shapes) >= 65 and any(30 in s for s in shapes), ("shapes", len(shapes))
    assert len(roles) >= 30 and min(varying, stable) >= 3, ("palettes", len(roles), varying, stable)
    assert len(normalized) >= 200, ("normalized_inputs", len(normalized))
    assert observed["buses"] == {1, 2}, ("buses", observed["buses"])
    assert len(observed["tokens"]) >= 4 and len(observed["survivors"]) >= 3, ("counts", observed)
    assert {1, 2, 3, 4} == {v for p in observed["pulses"] for v in p}, ("pulses", observed["pulses"])
    assert len(observed["kinds"]) == 3, ("morphology", observed["kinds"])
    assert all(counts[k] >= 3 for k in counts), counts
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(901 + 10 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (4, 1.2), (None, 1), (4, None)):
        _expect_failure(lambda nt=nt, ne=ne: generate_puzzle(0, nt, ne), ValueError)
    for seed in (-1, True, 2.5):
        _expect_failure(lambda seed=seed: generate_puzzle(seed), ValueError)
    # Exact local recognition on every possible 3x3 binary neighborhood.
    for bits in range(512):
        cells = {(r, c) for r in range(3) for c in range(3) if bits & (1 << (3 * r + c))}
        expected = bool(bits & 16) and all(bits & (1 << i) for i in (1, 3, 5, 7))
        assert ((1, 1) in _cross_centers(cells)) == expected
    assert len(_cross_centers({(r, c) for r in range(3) for c in range(4)})) == 2
    assert not _cross_centers({(0, 0), (0, 1), (0, 2), (1, 1)})
    assert _walk(1, 2, {1: 2, 2: 3}) == 3  # Final key need not exist.
    assert _walk(1, 3, {1: 2, 2: 3}) is None
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints),
                distinct_shapes=len(shapes), structural_palettes=len(roles),
                normalized_inputs=len(normalized), stable_puzzles=stable,
                varying_palette_puzzles=varying,
                observed={k: sorted(v) for k, v in observed.items()}, feature_pairs=counts)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
