"""Dispatch ledger: a directed stream of cargo and consuming ordinal tickets.

Only NumPy and the standard library are required. No files are read or written.
See task_spec.md for the visible grammar and composition.md for the rationale.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)),
             f"{name} must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.tobytes()


def _boxes(mask):
    """Components use non-background occupancy, including multicolor cargo."""
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, points = [first], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        r0, c0 = map(int, np.min(points, axis=0))
        r1, c1 = (int(v) + 1 for v in np.max(points, axis=0))
        _require(len(points) == (r1 - r0) * (c1 - c0), "object has a hole")
        result.append((r0, r1, c0, c1))
    return result


def _parse(a, reflected):
    """Parse an origin-at-top-left, fast-axis-horizontal coordinate view."""
    _require(min(a.shape) >= 5, "missing key or payload")
    origin, fast, slow, bg = map(int, (a[0, 0], a[0, 1], a[1, 0], a[-1, -1]))
    _require(len({origin, fast, slow, bg}) == 4, "key roles collide")
    _require(np.all(a[0, 1:] == fast) and np.all(a[1:, 0] == slow), "invalid L key")
    _require(np.all(a[-1, 1:] == bg) and np.all(a[1:, -1] == bg), "missing outer margin")
    mask = a != bg
    mask[0, :] = mask[:, 0] = False
    events, controls, lengths = [], set(), {}
    for r0, r1, c0, c1 in _boxes(mask):
        _require(r0 >= 2 and c0 >= 2 and r1 < a.shape[0] and c1 < a.shape[1],
                 "objects must be separate from the border")
        h, w = r1 - r0, c1 - c0
        b = a[r0:r1, c0:c1]
        cap = int(b[0, 0])
        e = {"cap": cap, "row": r0, "col": c0, "width": w}
        if h == 1:
            _require(1 <= w <= 3 and np.all(b == cap), "ticket must be a unary run")
            e.update(kind="ticket", rank=w)
        else:
            _require(h == 2 and w >= 4, "cargo must be two cells thick and at least four long")
            _require(np.all(b[:, 0] == cap) and np.all(b[:, -1] == cap), "unequal cargo caps")
            _require(np.all(b[:, 1:-1] != cap), "payload overlaps its cap color")
            controls.add(cap)
            lengths.setdefault(cap, []).append(w)
            e.update(kind="cargo", block=(b[::-1] if reflected else b).copy())
        events.append(e)
    _require(1 <= len(controls) <= 3, "need one to three cargo families")
    _require(controls.isdisjoint({origin, fast, slow, bg}), "control and key roles overlap")
    _require(events and any(e["kind"] == "ticket" for e in events), "missing tickets")
    _require(all(e["cap"] in controls for e in events), "ticket has no cargo family")
    _require(all(len(v) == len(set(v)) for v in lengths.values()), "tied family lengths")
    _require(len({n % 2 for v in lengths.values() for n in v}) == 1, "mixed length parity")
    for e in events:
        if e["kind"] == "cargo":
            _require(not (set(map(int, e["block"][:, 1:-1].flat)) & controls),
                     "payload colors must be distinct from all cap identities")
    rows = sorted({e["row"] for e in events})
    _require(all(y - x >= 3 for x, y in zip(rows, rows[1:])), "staggered lane bands")
    events.sort(key=lambda e: (e["row"], e["col"]))
    _require(sum(e["kind"] == "ticket" for e in events) <= 15, "output height exceeds 30")
    return {"events": events, "background": bg, "lanes": len(rows), "controls": controls,
            "origin": origin, "guides": (fast, slow)}


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = []
    # These transforms define coordinate views, not output reflections.
    for rotation in range(4):
        turned = np.rot90(a, -rotation)
        for reflected in (False, True):
            canonical = np.flipud(turned) if reflected else turned
            try:
                d = _parse(canonical, reflected)
            except ValueError:
                continue
            d.update(rotation=rotation, reflected=reflected)
            candidates.append(d)
    _require(len(candidates) == 1, "input needs one unambiguous directed strip axis")
    return candidates[0]


def _dispatch(events, policy="correct"):
    inventory, selected = [], []
    if policy == "global":
        inventory = [e for e in events if e["kind"] == "cargo"]
    sequence = sorted(events, key=lambda e: (e["col"], e["row"])) if policy == "fast_major" else events
    previous_row = None
    for e in sequence:
        if policy == "reset_lane" and e["row"] != previous_row:
            inventory = []
        previous_row = e["row"]
        if e["kind"] == "cargo":
            if policy != "global":
                inventory.append(e)
            continue
        candidates = [c for c in inventory if c["cap"] == e["cap"] or policy == "ignore_caps"]
        if policy == "fifo":
            ordered = candidates
        elif policy == "lifo":
            ordered = candidates[::-1]
        else:
            ordered = sorted(candidates, key=lambda c: c["width"], reverse=policy != "ascending")
        rank = 1 if policy == "always_longest" else e["rank"]
        _require(len(ordered) >= rank, "ticket requests unavailable cargo")
        chosen = ordered[rank - 1]
        selected.append(chosen)
        if policy != "reuse":
            # Dict equality would compare arrays; remove by object identity.
            inventory = [c for c in inventory if c is not chosen]
    return selected


def _stack(blocks, background, width=None):
    _require(bool(blocks), "empty dispatch")
    if width is None:
        width = max(b.shape[1] for b in blocks)
    out = np.full((2 * len(blocks), width), background, dtype=np.int8)
    for i, b in enumerate(blocks):
        _require(width >= b.shape[1] and (width - b.shape[1]) % 2 == 0, "inexact centering")
        left = (width - b.shape[1]) // 2
        out[2 * i:2 * i + 2, left:left + b.shape[1]] = b
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from visible geometry and ticket contents."""
    d = _decode(input_grid)
    return _stack([e["block"] for e in _dispatch(d["events"])], d["background"])


def _block(rng, length, cap, palette, asymmetric=False):
    body = rng.choice(palette, size=(2, length - 2)).astype(np.int8)
    mode = int(rng.integers(4))
    if mode == 0:
        body[:] = rng.choice(palette)
    elif mode == 1:
        body[1] = body[0]
    elif mode == 2:
        body[:] = rng.choice(palette, size=(2, 1))
    if asymmetric:
        # Distinct random colors; no seed-wide or position-specific signature.
        x, y, z = rng.choice(palette, 3, replace=False)
        body[0, 0], body[1, 0], body[0, -1], body[1, -1] = x, y, y, z
    result = np.full((2, length), cap, dtype=np.int8)
    result[:, 1:-1] = body
    return result


def _program(rng, cap, blocks, anchor=None):
    events, inventory = [], []

    def arrive(index):
        e = {"kind": "cargo", "cap": cap, "block": blocks[index]}
        events.append(e)
        inventory.append(index)

    def ticket(rank):
        chosen = sorted(inventory, key=lambda i: blocks[i].shape[1], reverse=True)[rank - 1]
        inventory.remove(chosen)
        events.append({"kind": "ticket", "cap": cap, "rank": rank, "chosen": blocks[chosen]})

    if anchor is not None:
        # Widths are increasing. This is an evidence witness, not the general rule.
        for i in (1, 0, 2):
            arrive(i)
        ticket(anchor)
        ticket(1 if anchor == 2 else 2)
        ticket(1)
        arrive(3)
        ticket(1)
        arrive(4)  # A future, unrequested maximum must not set output width.
    else:
        arrival_order = list(map(int, rng.permutation(len(blocks))))
        requests = int(rng.integers(1, len(blocks) + 1))
        while arrival_order or requests:
            if arrival_order and (not inventory or not requests or rng.random() < .58):
                arrive(arrival_order.pop(0))
            else:
                ticket(int(rng.integers(1, min(3, len(inventory)) + 1)))
                requests -= 1
    return events


def _merge(rng, programs):
    queues = [list(p) for p in programs]
    result = []
    # Random unequal chunks preserve only each family's causal constraints.
    while any(queues):
        live = [i for i, q in enumerate(queues) if q]
        i = int(rng.choice(live))
        count = int(rng.integers(1, min(4, len(queues[i])) + 1))
        result.extend(queues[i][:count])
        del queues[i][:count]
    return result


def _layout(rng, events, bg, markers, configuration, compact=False):
    widths = [e["block"].shape[1] if e["kind"] == "cargo" else e["rank"] for e in events]
    width = 30 if compact else int(rng.integers(max(16, max(widths) + 3), 31))
    capacity = width - 3
    lanes, lane, used = [], [], 0
    for e, w in zip(events, widths):
        gap = 0 if not lane else (1 if compact else int(rng.integers(1, 4)))
        cut = lane and not compact and rng.random() < .10
        if lane and (used + gap + w > capacity or cut):
            lanes.append(lane)
            lane, used, gap = [], 0, 0
        lane.append((e, used + gap, w))
        used += gap + w
    lanes.append(lane)
    if len(lanes) > 9:
        return None
    n = len(lanes)
    extra = int(rng.integers(30 - (3 * n + 2) + 1))
    gaps = 1 + rng.multinomial(extra, np.full(n + 1, 1 / (n + 1)))
    height = 1 + 2 * n + int(gaps.sum())
    grid = np.full((height, width), bg, dtype=np.int8)
    origin, fast, slow = markers
    grid[0, 0], grid[0, 1:], grid[1:, 0] = origin, fast, slow
    rotation, reflected = configuration
    r = 1 + int(gaps[0])
    for i, lane in enumerate(lanes):
        used = lane[-1][1] + lane[-1][2]
        left = int(rng.integers(2, width - used))
        for e, offset, w in lane:
            c = left + offset
            if e["kind"] == "cargo":
                grid[r:r + 2, c:c + w] = e["block"][::-1] if reflected else e["block"]
            else:
                grid[r, c:c + w] = e["cap"]
        r += 2 + int(gaps[i + 1])
    if reflected:
        grid = np.flipud(grid)
    grid = np.ascontiguousarray(np.rot90(grid, rotation))
    # Ground truth uses the planned selections, independently of input decoding.
    selected = [e["chosen"] for e in events if e["kind"] == "ticket"]
    return {"input": grid, "output": _stack(selected, bg)}


def _scene(rng, cap_pool, configuration, anchor=None, background_exclude=None):
    # The final attempt has a proven small packing budget; no unbounded retries.
    for attempt in range(17):
        compact = attempt == 16
        f = (1 if anchor == 2 else int(rng.integers(2, 4))) if anchor else int(rng.integers(1, 4))
        caps = list(cap_pool[:f]) if anchor else list(map(int, rng.choice(cap_pool, f, replace=False)))
        allowed = [c for c in range(10) if c not in caps and c != background_exclude]
        bg = int(rng.choice(allowed))
        markers = list(map(int, rng.choice([c for c in range(10) if c not in caps and c != bg], 3, replace=False)))
        pool = [c for c in range(10) if c not in caps and c != bg]
        palette = list(map(int, rng.choice(pool, int(rng.integers(3, len(pool) + 1)), replace=False)))
        parity = int(rng.integers(2))
        base = 4 + parity
        ceiling = (22 if f == 1 else 16 if f == 2 else 14) + parity
        programs = []
        anchor_lengths = None
        for family, cap in enumerate(caps):
            n = 5 if anchor and family == 0 else (2 if compact else int(rng.integers(2, 5)))
            choices = np.arange(base, ceiling + 1, 2)
            if compact:
                lengths = list(map(int, choices[:n]))
            else:
                lengths = sorted(map(int, rng.choice(choices, n, replace=False)))
            if anchor and family == 0:
                anchor_lengths = lengths
            if anchor and family > 0:
                # An available competitor of another family defeats grouping by payload or proximity.
                n = 2
                lengths = [anchor_lengths[0], anchor_lengths[3]]
            blocks = [_block(rng, length, cap, palette, asymmetric=bool(anchor)) for length in lengths]
            if anchor and family > 0:
                programs.append([{"kind": "cargo", "cap": cap, "block": b} for b in blocks] +
                                [{"kind": "ticket", "cap": cap, "rank": 1, "chosen": blocks[-1]}])
            else:
                programs.append(_program(rng, cap, blocks, anchor if family == 0 else None))
        if anchor == 1:
            # Interleave the opening arrivals, then show a cap-sensitive first request.
            prefix = _merge(rng, [programs[0][:3]] + [p[:2] for p in programs[1:]])
            events = prefix + programs[0][3:4] + _merge(rng, [programs[0][4:]] + [p[2:] for p in programs[1:]])
        else:
            events = _merge(rng, programs)
        pair = _layout(rng, events, bg, markers, configuration, compact)
        if pair is not None:
            return pair
    raise AssertionError("bounded compact packing failed")


def _alternative(d, policy):
    events, bg = d["events"], d["background"]
    if policy == "cargo_scan":
        return _stack([e["block"] for e in events if e["kind"] == "cargo"], bg)
    selected = _dispatch(events, "correct" if policy == "all_width" else policy)
    width = max(e["width"] for e in events if e["kind"] == "cargo") if policy == "all_width" else None
    return _stack([e["block"] for e in selected], bg, width)


_ALTERNATIVES = ("global", "reuse", "ascending", "fifo", "lifo", "always_longest",
                 "ignore_caps", "reset_lane", "fast_major", "cargo_scan", "all_width")


def _survivors(train):
    surviving = []
    decoded = [(_decode(p["input"]), np.asarray(p["output"])) for p in train]
    for policy in _ALTERNATIVES:
        for d, expected in decoded:
            try:
                same = np.array_equal(_alternative(d, policy), expected)
            except ValueError:
                same = False
            if not same:
                break
        else:
            surviving.append(policy)
    return surviving


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    cap_pool = list(map(int, rng.choice(10, 3, replace=False)))
    first = (int(rng.integers(4)), bool(rng.integers(2)))
    # Opposite origin corner, perpendicular cargo axis, opposite handedness.
    second = ((first[0] + (3 if first[1] else 1)) % 4, not first[1])
    taught = (first, second)
    held = [(r, f) for r in range(4) for f in (False, True) if (r, f) not in taught]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            configuration = taught[i % 2] if split == "train" else held[int(rng.integers(len(held)))]
            anchor = (2 if i == 0 else 1) if split == "train" and i < 2 else None
            exclude = _decode(puzzle["train"][0]["input"])["background"] if split == "train" and i == 1 else None
            for _ in range(24):
                pair = _scene(rng, cap_pool, configuration, anchor, exclude)
                if _signature(pair["input"]) in seen:
                    continue
                if split == "train" and i == 1 and _survivors(puzzle["train"] + [pair]):
                    continue
                seen.add(_signature(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("bounded evidence/uniqueness construction failed")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the visible grammar, causal invariants, exact labels and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for p in puzzle[split]:
            _require(isinstance(p, dict) and set(p) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(p["input"]), _grid(p["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            d = _decode(inp)
            selected = _dispatch(d["events"])
            _require(len({id(e) for e in selected}) == len(selected), "cargo dispatched twice")
            _require(out.shape == (2 * len(selected), max(e["width"] for e in selected)), "wrong output extent")
            _require(np.array_equal(out, _stack([e["block"] for e in selected], d["background"])),
                     "incorrect cargo, orientation, ticket order or padding")
            if split == "train":
                train.append(d)
    _require(not _survivors(puzzle["train"]), "training supports a documented shortcut")
    ranks = {e["rank"] for d in train for e in d["events"] if e["kind"] == "ticket"}
    _require({1, 2} <= ranks, "missing ordinal progression")
    _require(any(len(d["controls"]) > 1 for d in train), "missing cap-grouping evidence")
    _require(len({d["rotation"] % 2 for d in train}) == 2, "training must teach both physical axes")
    _require(any(any(not np.array_equal(e["block"], e["block"][::-1]) and
                         not np.array_equal(e["block"], e["block"][:, ::-1])
                         for e in _dispatch(d["events"])) for d in train), "missing rotation witness")


def self_test():
    """Bounded 42-seed semantic/diversity checks plus explicit negative cases."""
    shapes, out_shapes, layouts, colors, configs, families = set(), set(), set(), set(), set(), set()
    cargo_counts, request_counts, ranks, parities, pattern_types = set(), set(), set(), set(), set()
    hashes, count_ranks = set(), set()
    within = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        local_counts = set()
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                d = _decode(a)
                events = d["events"]
                cargo = [e for e in events if e["kind"] == "cargo"]
                tickets = [e for e in events if e["kind"] == "ticket"]
                digest.update(a.tobytes())
                assert np.array_equal(solve(a.copy()), pair["output"])
                # Whole-input rotations leave the normalized dispatch unchanged.
                if seed < 4:
                    for turn in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(a, turn)), pair["output"])
                    mapping = np.array([7, 9, 1, 0, 4, 3, 2, 8, 6, 5], dtype=np.int8)
                    assert np.array_equal(solve(mapping[a]), mapping[pair["output"]])
                shapes.add(a.shape)
                out_shapes.add(pair["output"].shape)
                colors.add((d["background"], tuple(sorted(d["controls"]))))
                configs.add((d["rotation"], d["reflected"]))
                families.add(len(d["controls"]))
                cargo_counts.add(len(cargo))
                request_counts.add(len(tickets))
                parities.add(cargo[0]["width"] % 2)
                local_counts.add((len(cargo), len(tickets)))
                layouts.add(tuple((e["kind"], e["row"], e["col"], e["width"]) for e in events))
                inventory = []
                for e in events:
                    if e["kind"] == "cargo":
                        inventory.append(e)
                        b = e["block"][:, 1:-1]
                        pattern_types.add("solid" if np.all(b == b.flat[0]) else
                                          "repeat_rows" if np.array_equal(b[0], b[1]) else "other")
                    else:
                        ranks.add(e["rank"])
                        candidates = sorted([x for x in inventory if x["cap"] == e["cap"]],
                                            key=lambda x: x["width"], reverse=True)
                        count_ranks.add((len(candidates), e["rank"]))
                        chosen = candidates[e["rank"] - 1]
                        inventory = [x for x in inventory if x is not chosen]
        hashes.add(digest.digest())
        within.append(len(local_counts))
    assert len(hashes) == 42 and len(layouts) >= 180
    assert len(shapes) >= 60 and len(out_shapes) >= 30 and len(colors) >= 60
    assert len(configs) == 8 and families == {1, 2, 3} and parities == {0, 1}
    assert len(cargo_counts) >= 7 and len(request_counts) >= 6 and ranks == {1, 2, 3}
    assert {(3, 2), (2, 2), (1, 1), (3, 3)} <= count_ranks
    assert pattern_types == {"solid", "repeat_rows", "other"} and min(within) >= 2
    # A hand-built contract case outside the sampled curricula: five arrivals
    # before any request, then a ticket-only lane. Expected indices are explicit.
    direct = np.zeros((21, 19), dtype=np.int8)
    direct[0, 0], direct[0, 1:], direct[1:, 0] = 9, 1, 2
    blocks = []
    for i, length in enumerate((4, 6, 8, 10, 12)):
        b = np.full((2, length), 3, dtype=np.int8)
        b[:, 1:-1] = 4
        b[0, 1], b[1, -2] = 5, 6
        blocks.append(b)
        direct[2 + 3 * i:4 + 3 * i, 2:2 + length] = b
    col = 2
    for rank in (3, 3, 3, 1, 1):
        direct[17, col:col + rank] = 3
        col += rank + 1
    chosen = [blocks[i] for i in (2, 1, 0, 4, 3)]
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(direct, rotation)), _stack(chosen, 0))
        reflected = np.rot90(np.flipud(direct), rotation)
        assert np.array_equal(solve(reflected), _stack([b[::-1] for b in chosen], 0))
    malformed = []
    hole = direct.copy()
    hole[2, 3] = 0
    malformed.append(hole)
    unknown = direct.copy()
    unknown[17, 2:5] = 8
    malformed.append(unknown)
    mixed = direct.copy()
    mixed[2:4, 5], mixed[2:4, 6] = 4, 3
    malformed.append(mixed)
    tied = direct.copy()
    tied[5:7, 2:8] = 0
    tied[5:7, 2:6] = blocks[0]
    malformed.append(tied)
    for a in malformed:
        try:
            solve(a)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed input accepted")
    # Every legal explicit combination; illegal types, ranges and equal counts.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(190 + n * 5 + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    again = generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], again[s]) for k in x)
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    # Semantically malformed inputs are rejected without consulting any label.
    d = _decode(p["train"][0]["input"])
    unavailable = [{"kind": "ticket", "cap": next(iter(d["controls"])), "rank": 1, "row": 0, "col": 0}]
    try:
        _dispatch(unavailable)
    except ValueError:
        pass
    else:
        raise AssertionError("unavailable request accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes), "output_shapes": len(out_shapes),
            "layouts": len(layouts), "configurations": len(configs), "cargo_counts": sorted(cargo_counts),
            "request_counts": sorted(request_counts), "count_rank_cases": sorted(count_ranks)}
