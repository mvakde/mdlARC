"""Ranked collision banks: causal object competition controls a branching signal.

Only the standard library and NumPy are used. No parent modules, files, random
state, examples, or generation parameters are consulted by the input-only solve.
Generation/self_test keep grids in memory and never write artifacts or traces.
"""
from __future__ import annotations

from collections import defaultdict, namedtuple
import hashlib
import json

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


class _Bar(namedtuple("Bar", "row left right cap payload")):
    __slots__ = ()

    @property
    def length(self):
        return self.right - self.left + 1


def _runs(mask):
    start = None
    for i, occupied in enumerate(mask):
        if occupied and start is None:
            start = i
        if start is not None and (not occupied or i == len(mask) - 1):
            yield start, i if occupied else i - 1
            start = None


def _parse(a, bg, guide):
    h, w = a.shape
    entries = list(map(int, np.flatnonzero(a[0] == guide)))
    _require(entries and np.all(np.diff(entries) >= 2), "missing/separation of prefixes")
    guide_mask = np.zeros(a.shape, dtype=bool)
    lengths = []
    for c in entries:
        rows = np.flatnonzero(a[:, c] == guide)
        _require(2 <= len(rows) < h and np.array_equal(rows, np.arange(len(rows))),
                 "guide must be a boundary prefix")
        guide_mask[:len(rows), c] = True
        lengths.append(len(rows))
    _require(np.array_equal(guide_mask, a == guide), "guide occurs away from prefixes")
    ranks = {}
    for left, right in _runs((a[0] != bg) & (a[0] != guide)):
        cap = int(a[0, left])
        _require(np.all(a[0, left:right + 1] == cap), "mixed key run")
        _require(cap not in ranks and 1 <= right - left + 1 <= 3, "invalid unary key")
        _require((left == 0 or a[0, left - 1] == bg)
                 and (right == w - 1 or a[0, right + 1] == bg), "keys need separation")
        ranks[cap] = right - left + 1
    _require(1 <= len(ranks) <= 3, "one to three keyed identities required")
    bars = defaultdict(list)
    payloads, used = set(), set()
    for r in range(1, h):
        for left, right in _runs((a[r] != bg) & (a[r] != guide)):
            cap = int(a[r, left])
            _require(right - left >= 2 and cap in ranks and a[r, right] == cap,
                     "bar requires two matching keyed caps")
            payload = int(a[r, left + 1])
            _require(payload not in {bg, guide, *ranks}
                     and np.all(a[r, left + 1:right] == payload), "invalid uniform payload")
            _require(r >= max(3, max(lengths)), "bars must follow the entry prefixes")
            bars[r].append(_Bar(r, left, right, cap, payload))
            payloads.add(payload)
            used.add(cap)
    _require(bars and used == set(ranks) and 1 <= len(payloads) <= 2,
             "missing keyed bars or invalid payload palette")
    for row_bars in bars.values():
        groups = defaultdict(list)
        for bar in row_bars:
            groups[bar.cap].append(bar.length)
        _require(all(len(v) == len(set(v)) for v in groups.values()), "tied same-bank lengths")
    return {"background": bg, "guide": guide, "ranks": ranks, "bars": dict(bars),
            "entries": set(entries), "stub_lengths": tuple(lengths)}


def _decode(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be unique modal color")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(a, -rotation)
        for guide in map(int, colors):
            if guide == bg:
                continue
            try:
                meta = _parse(canonical, bg, guide)
            except AssertionError:
                continue
            candidates.append((rotation, canonical, meta))
    _require(len(candidates) == 1, "roles, entry edge and bar axis must be unambiguous")
    return candidates[0]


_SHORTCUTS = ("global_pool", "all_hits", "longest", "shortest", "left", "right",
              "ray_candidates", "no_split", "pass_losers", "clamp_missing", "one_family")


def _simulate(a, meta, mode="correct"):
    """Immutable-input, simultaneous layer simulation; alternatives are QA only."""
    out = a.copy()
    bg, guide, ranks = meta["background"], meta["guide"], meta["ranks"]
    h, w = a.shape
    active = set(meta["entries"])
    stats = {"contact_banks": 0, "winners": 0, "descendant_contacts": 0,
             "clipped": 0, "missing": 0, "multi_hits": 0, "simultaneous_families": 0,
             "merges": 0, "unreached": 0, "first_pool": 0, "winner_ranks": set(),
             "winner_lengths": [], "contacts": [], "selected": set()}
    for r in range(1, h):
        row_bars = meta["bars"].get(r, [])
        incident = {b: {c for c in active if b.left <= c <= b.right} for b in row_bars}
        incident = {b: cs for b, cs in incident.items() if cs}
        stopped = set().union(*incident.values()) if incident else set()
        following = active - stopped
        for c in following:
            if a[r, c] == bg:
                out[r, c] = guide
        groups = defaultdict(list)
        for b, cs in incident.items():
            groups[b.cap].append(b)
            stats["multi_hits"] += len(cs) > 1
            stats["descendant_contacts"] += bool(cs - meta["entries"])
        stats["unreached"] += len(row_bars) - len(incident)
        if incident:
            if not stats["contact_banks"]:
                stats["first_pool"] = max(map(len, groups.values()))
            stats["contact_banks"] += 1
            stats["contacts"].append((r, tuple(sorted((cap, len(bs)) for cap, bs in groups.items()))))
        chosen = []
        for cap, hit_bars in sorted(groups.items()):
            rank = ranks[cap]
            if len(hit_bars) < rank:
                stats["missing"] += 1
            candidates = [b for b in row_bars if b.cap == cap] if mode == "global_pool" else hit_bars
            if mode == "all_hits":
                chosen.extend(candidates)
                continue
            if mode == "ray_candidates":
                candidates = [b for b in hit_bars for _ in incident[b]]
            if mode in ("left", "right"):
                ordered = sorted(candidates, key=lambda b: b.left, reverse=mode == "right")
            else:
                ordered = sorted(candidates, key=lambda b: b.length, reverse=True)
            if mode == "longest":
                rank = 1
            elif mode == "shortest":
                rank = len(ordered)
            elif mode == "clamp_missing":
                rank = min(rank, len(ordered))
            if rank <= len(ordered):
                chosen.append(ordered[rank - 1])
        if mode == "one_family" and chosen:
            chosen = [min(chosen, key=lambda b: b.cap)]
        chosen = set(chosen)
        stats["simultaneous_families"] += len({b.cap for b in chosen}) > 1
        if mode == "pass_losers":
            for b, cs in incident.items():
                if b not in chosen:
                    following.update(cs)
        for b in chosen:
            stats["winners"] += 1
            stats["winner_lengths"].append(b.length)
            stats["winner_ranks"].add(ranks[b.cap])
            stats["selected"].add(b)
            out[r, b.left + 1:b.right] = b.cap
            if mode == "no_split":
                following.update(incident.get(b, set()))
                continue
            stats["clipped"] += b.left == 0 or b.right == w - 1
            left, right = max(0, b.left - 1), min(w, b.right + 2)
            upstream = out[r - 1, left:right]
            upstream[a[r - 1, left:right] == bg] = guide
            for c in (b.left - 1, b.right + 1):
                if 0 <= c < w:
                    stats["merges"] += c in following
                    following.add(c)
                    if a[r, c] == bg:
                        out[r, c] = guide
        active = following
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive everything from the input's stubs, unary keys, caps and geometry."""
    rotation, a, meta = _decode(input_grid)
    out, _ = _simulate(a, meta)
    return np.ascontiguousarray(np.rot90(out, rotation))


def _advance(active, bars, ranks, width):
    """Construction-only planner; final labels are decoded afresh by solve."""
    following = set(active)
    groups = defaultdict(list)
    for b in bars:
        hits = {c for c in active if b.left <= c <= b.right}
        if hits:
            following.difference_update(hits)
            groups[b.cap].append(b)
    for cap, group in groups.items():
        if len(group) >= ranks[cap]:
            b = sorted(group, key=lambda b: b.length, reverse=True)[ranks[cap] - 1]
            following.update(c for c in (b.left - 1, b.right + 1) if 0 <= c < width)
    return following


def _pack(rng, width, count):
    for _ in range(24):
        lengths = list(map(int, rng.choice(np.arange(3, min(10, width) + 1), count, replace=False)))
        if sum(lengths) + count - 1 <= width:
            break
    else:
        lengths = list(map(int, rng.permutation(np.arange(3, 3 + count))))
    return lengths


def _intervals(rng, width, lengths):
    spare = width - sum(lengths) - len(lengths) + 1
    extras = rng.multinomial(spare, np.full(len(lengths) + 1, 1 / (len(lengths) + 1)))
    x, result = int(extras[0]), []
    for i, n in enumerate(lengths):
        result.append((x, x + n - 1))
        x += n + 1 + int(extras[i + 1])
    return result


def _random_bank(rng, row, width, active, caps, ranks, payloads, forced_cap):
    result = []
    count = int(rng.integers(1, 4))
    for i in range(count):
        options = []
        for length in range(3, width + 1):
            for left in range(width - length + 1):
                right = left + length - 1
                if any(not (right + 1 < b.left or left > b.right + 1) for b in result):
                    continue
                touched = any(left <= c <= right for c in active)
                if i == 0 and active and not touched:
                    continue
                options.append((left, right, touched))
        if not options:
            break
        # Prefer short local switches, sometimes use long, clipped crossbars.
        short = [p for p in options if p[1] - p[0] < 9]
        pool = short if short and rng.random() < .83 else options
        if i and active and rng.random() < .65:
            touched = [p for p in pool if p[2]]
            if touched:
                pool = touched
        left, right, _ = pool[int(rng.integers(len(pool)))]
        cap = forced_cap if i == 0 else int(rng.choice(caps))
        available = [c for c in caps if all(b.cap != c or b.length != right - left + 1 for b in result)]
        if cap not in available:
            if not available:
                continue
            cap = int(rng.choice(available))
        result.append(_Bar(row, left, right, cap, int(rng.choice(payloads))))
    return sorted(result, key=lambda b: b.left)


def _put_keys(rng, width, entries, caps, ranks, bg, guide):
    edge = np.full(width, bg, dtype=np.int8)
    edge[list(entries)] = guide
    order = list(caps)
    rng.shuffle(order)
    order.sort(key=lambda c: ranks[c], reverse=True)
    for cap in order:
        n = ranks[cap]
        options = [x for x in range(width - n + 1)
                   if np.all(edge[max(0, x - 1):min(width, x + n + 1)] == bg)]
        if not options:
            return None
        left = int(rng.choice(options))
        edge[left:left + n] = cap
    return edge


def _construct(rng, palette, anchor, target, max_caps, max_rank):
    cap_count = 2 if anchor else int(rng.integers(1, max_caps + 1))
    if target:
        cap_count = max(2, cap_count)
    bg, guide = palette[:2]
    caps, payloads = palette[2:2 + cap_count], palette[5:7]
    ranks = {c: int(rng.integers(1, max_rank + 1)) for c in caps}
    if anchor:
        ranks[caps[0]], ranks[caps[1]] = anchor, 1
    elif len(caps) > 1:
        ranks[caps[1]] = 1
    contacted = 3 if anchor else (4 if target else int(rng.integers(1, 4)))
    total = 4 if anchor or target else min(4, contacted + int(rng.integers(2)))
    minimum_width = sum(range(3, 3 + total)) + total - 1
    width = int(rng.integers(max(8, minimum_width + int(bool(anchor or target))), 31))
    lengths = _pack(rng, width, total)
    absent = int(np.argmax(lengths)) if anchor else None
    hit_indices = [i for i in range(total) if i != absent] if anchor else list(map(int, rng.choice(total, contacted, replace=False)))
    if anchor:
        # Put the winner centrally. Rank one then excludes both spatial
        # ordinals; both anchors retain two clear in-bounds outgoing flanks.
        chosen = sorted(hit_indices, key=lambda i: lengths[i], reverse=True)[anchor - 1]
        middle = sorted(hit_indices)[1]
        lengths[chosen], lengths[middle] = lengths[middle], lengths[chosen]
    intervals = _intervals(rng, width, lengths)
    entries = set()
    longest_hit = max(hit_indices, key=lambda i: lengths[i])
    for i in hit_indices:
        left, right = intervals[i]
        multi = (anchor == 2 and i == longest_hit) or (not anchor and rng.random() < .25)
        if multi:
            entries.update((left, right))
        else:
            entries.add(int(rng.integers(left, right + 1)))
    stub_lengths = {c: int(rng.integers(2, 5)) for c in sorted(entries)}
    first = max(3, max(stub_lengths.values()) + int(rng.integers(0, 3)))
    edge = _put_keys(rng, width, entries, caps, ranks, bg, guide)
    if edge is None:
        return None
    banks = [[_Bar(first, l, r, caps[0], int(rng.choice(payloads))) for l, r in intervals]]
    active = _advance(entries, banks[0], ranks, width)
    row = first
    if anchor == 1:
        # Two contacted cap families emit into one shared flank; the next
        # clipped bar demonstrates aggregation, merging and boundary loss.
        lo, hi = min(active), max(active)
        row += 4
        bank = [_Bar(row, lo, hi - 3, caps[0], int(rng.choice(payloads))),
                _Bar(row, hi - 1, hi + 1, caps[1], int(rng.choice(payloads)))]
        banks.append(bank)
        active = _advance(active, bank, ranks, width)
        row += int(rng.integers(2, 4))
        bank = [_Bar(row, 0, max(active), caps[1], int(rng.choice(payloads)))]
        banks.append(bank)
        active = _advance(active, bank, ranks, width)
    elif anchor == 2:
        row += 4
        bank = [_Bar(row, min(active), max(active), caps[1], int(rng.choice(payloads)))]
        banks.append(bank)
        active = _advance(active, bank, ranks, width)
        row += int(rng.integers(2, 4))
        # Exactly one contacted candidate with rank two must absorb all rays.
        lo, hi = min(active), max(active)
        while hi - lo < 2:
            if lo > 0:
                lo -= 1
            elif hi + 1 < width:
                hi += 1
        bank = [_Bar(row, lo, hi, caps[0], int(rng.choice(payloads)))]
        banks.append(bank)
        active = _advance(active, bank, ranks, width)
    else:
        additions = int(rng.integers(3, 6)) if target else int(rng.integers(1, 5))
        for j in range(additions):
            row += int(rng.integers(2, 5))
            if row > 28:
                break
            forced = caps[j + 1] if j < len(caps) - 1 else caps[int(rng.integers(len(caps)))]
            bank = _random_bank(rng, row, width, active, caps, ranks, payloads, forced)
            banks.append(bank)
            active = _advance(active, bank, ranks, width)
    height = min(30, row + int(rng.integers(2, 6)))
    a = np.full((height, width), bg, dtype=np.int8)
    a[0] = edge
    for c, n in stub_lengths.items():
        a[:n, c] = guide
    for bank in banks:
        for b in bank:
            a[b.row, b.left:b.right + 1] = b.payload
            a[b.row, b.left] = a[b.row, b.right] = b.cap
    return a


def _analysis(pair):
    rotation, a, meta = _decode(pair["input"])
    expected, stats = _simulate(a, meta)
    o = np.rot90(_grid(pair["output"]), -rotation)
    _require(np.array_equal(o, expected), "incorrect ranked collision output")
    return rotation, a, meta, stats


def _surviving_shortcuts(analyses):
    return {mode for mode in _SHORTCUTS if all(
        np.array_equal(_simulate(a, meta, mode)[0], _simulate(a, meta)[0])
        for _, a, meta, _ in analyses)}


def _make_pair(rng, palette, anchor, target, max_caps, max_rank, rotation, seen, surviving):
    for _ in range(96):
        a = _construct(rng, palette, anchor, target, max_caps, max_rank)
        if a is None:
            continue
        if rng.random() < .5:
            a = np.fliplr(a)
        actual = np.ascontiguousarray(np.rot90(a, rotation))
        if _signature(actual) in seen:
            continue
        try:
            _, canonical, meta = _decode(actual)
        except AssertionError:
            continue
        o, _ = _simulate(canonical, meta)
        if np.array_equal(o, canonical):
            continue
        if any(not np.array_equal(_simulate(canonical, meta, mode)[0], o) for mode in surviving):
            continue
        # canonical is the same downward geometry including any reflection.
        return {"input": actual, "output": np.ascontiguousarray(np.rot90(o, rotation))}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    max_caps, max_rank = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    mixed_palette, mixed_direction = bool(rng.integers(2)), bool(rng.integers(2))
    orientation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt):
        local = palette if i < 2 or not mixed_palette else list(map(int, rng.permutation(10)))
        rotation = (orientation + i) % 4 if mixed_direction else orientation
        pair = _make_pair(rng, local, i + 1 if i < 2 else 0, False,
                          max_caps, max_rank, rotation, seen, set())
        puzzle["train"].append(pair)
        seen.add(_signature(pair["input"]))
    surviving = _surviving_shortcuts([_analysis(p) for p in puzzle["train"]])
    for i in range(nq):
        local = list(map(int, rng.permutation(10))) if mixed_palette else palette
        rotation = int(rng.integers(4)) if mixed_direction else orientation
        pair = _make_pair(rng, local, 0, True, max_caps, max_rank, rotation, seen, surviving)
        puzzle["test"].append(pair)
        seen.add(_signature(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, o = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == o.shape and np.any(a != o), "shape change or identity pair")
            sig = _signature(a)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            rotation, canonical, meta, stats = _analysis(pair)
            out = np.rot90(o, -rotation)
            bg, guide = meta["background"], meta["guide"]
            _require(np.all(out[canonical == bg][out[canonical == bg] != bg] == guide),
                     "background changes must use guide identity")
            mutable = np.zeros(canonical.shape, dtype=bool)
            for bank in meta["bars"].values():
                for b in bank:
                    if b in stats["selected"]:
                        mutable[b.row, b.left + 1:b.right] = True
                        _require(np.all(out[b.row, b.left:b.right + 1] == b.cap), "winner not solid")
                    else:
                        _require(np.array_equal(out[b.row, b.left:b.right + 1],
                                                canonical[b.row, b.left:b.right + 1]), "loser/unreached bar changed")
            fixed = (canonical != bg) & ~mutable
            _require(np.array_equal(out[fixed], canonical[fixed]), "immutable input object overwritten")
            analyses[split].append((rotation, canonical, meta, stats))
    train = analyses["train"]
    _require({1, 2} <= {n for _, _, m, _ in train for n in m["ranks"].values()}, "missing ordinal progression")
    histories = defaultdict(set)
    for _, _, m, _ in train:
        for cap, rank in m["ranks"].items():
            histories[cap].add(rank)
    _require(any({1, 2} <= v for v in histories.values()), "no repeated identity changes rank")
    for name in ("descendant_contacts", "clipped", "missing", "multi_hits", "simultaneous_families", "merges"):
        _require(any(s[name] for _, _, _, s in train), f"missing training witness: {name}")
    surviving = _surviving_shortcuts(train)
    _require(not surviving, "a designed training counterexample was lost: " + ",".join(sorted(surviving)))
    _require(all(s["first_pool"] <= 3 for _, _, _, s in train), "training first pool too large")
    _require(all(s["first_pool"] == 4 for _, _, _, s in analyses["test"]), "missing four-competitor test shift")
    return True


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _fixture():
    """Small hand-computable case: rank two, a multi-hit, and an inert longest."""
    a = np.zeros((10, 24), dtype=np.int8)
    a[0, 17:19] = 2
    a[:2, [2, 8, 12, 14]] = 1
    for left, right in ((1, 3), (6, 9), (11, 15), (17, 22)):
        a[4, left:right + 1] = 3
        a[4, [left, right]] = 2
    expected = a.copy()
    expected[2:4, [2, 8, 12, 14]] = 1
    expected[4, 7:9] = 2
    expected[3, 5:11] = 1
    expected[4:, [5, 10]] = 1
    return a, expected


def self_test():
    """Stream 42 seeds; no saved traces, sample banks or unbounded retries."""
    a, expected = _fixture()
    before = a.copy()
    _require(np.array_equal(solve(a), expected), "hand-computed local competition regression")
    _require(np.array_equal(a, before), "solve mutated its input")
    _require(np.array_equal(solve(a[:5]), expected[:5]), "final-depth collision regression")
    # A consecutive bank's crosspiece intersects an earlier winning bar.
    # It must preserve that bar's new cap fill as well as the original caps.
    adjacent, adjacent_expected = a.copy(), expected.copy()
    adjacent[0, 21] = adjacent_expected[0, 21] = 4
    adjacent[5, 4:7] = [4, 3, 4]
    adjacent_expected[4, 4] = 1
    adjacent_expected[5, 4:7] = 4
    adjacent_expected[5, [3, 7]] = 1
    adjacent_expected[6:, 5] = 0
    adjacent_expected[6:, [3, 7]] = 1
    _require(np.array_equal(solve(adjacent), adjacent_expected), "interrupted crosspiece regression")
    spanning = np.zeros((6, 7), dtype=np.int8)
    spanning[:2, 3] = 1
    spanning[0, 0] = 2
    spanning[3] = [2, 3, 3, 3, 3, 3, 2]
    spanning_expected = spanning.copy()
    spanning_expected[2] = 1
    spanning_expected[3] = 2
    _require(np.array_equal(solve(spanning), spanning_expected), "both flanks clipped regression")
    for rotation in range(4):
        for reflected in (False, True):
            x, y = (np.fliplr(a), np.fliplr(expected)) if reflected else (a, expected)
            mapping = np.array([8, 3, 6, 0, 9, 5, 4, 2, 1, 7], dtype=np.int8)
            _require(np.array_equal(solve(mapping[np.rot90(x, rotation)]), mapping[np.rot90(y, rotation)]),
                     "rotation, reflection or palette equivariance")
    digests, structures, shapes, directions, colors = set(), set(), set(), set(), [set() for _ in range(4)]
    populations, ranks, bank_counts, entry_counts, lengths, payload_classes = set(), set(), set(), set(), set(), set()
    features = defaultdict(int)
    for seed in range(42):
        p = generate_puzzle(seed)
        digests.add(_puzzle_digest(p))
        if seed < 3:
            _require(_puzzle_digest(p) == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in p[split]:
                rotation, canonical, meta, stats = _analysis(pair)
                directions.add(rotation)
                shapes.add(pair["input"].shape)
                bank_counts.add(len(meta["bars"]))
                entry_counts.add(len(meta["entries"]))
                ranks.update(meta["ranks"].values())
                lengths.update(b.length for bs in meta["bars"].values() for b in bs)
                populations.update(n for _, groups in stats["contacts"] for _, n in groups)
                payloads = {b.payload for bs in meta["bars"].values() for b in bs}
                payload_classes.add(len(payloads))
                for i, role_colors in enumerate(({meta["background"]}, {meta["guide"]}, set(meta["ranks"]), payloads)):
                    colors[i].update(role_colors)
                roles = {meta["background"]: 0, meta["guide"]: 1}
                for cap in sorted(meta["ranks"]):
                    roles[cap] = len(roles)
                for payload in sorted(payloads):
                    roles[payload] = len(roles)
                normalized = np.array([roles[int(c)] for c in canonical.flat], dtype=np.int8).reshape(canonical.shape)
                structures.add(_signature(normalized))
                for feature in ("descendant_contacts", "clipped", "missing", "multi_hits", "simultaneous_families", "merges"):
                    features[feature] += stats[feature] > 0
        if seed == 0:
            bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
            pair = bad["test"][0]
            r, c = np.argwhere(pair["input"] != pair["output"])[0]
            pair["output"][r, c] = pair["input"][r, c]
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("corrupt label accepted")
    _require(len(digests) == 42 and len(structures) >= 150, "structural variation collapsed")
    _require(len(shapes) >= 40 and directions == set(range(4)), "geometry variation collapsed")
    _require(all(v == set(range(10)) for v in colors), "semantic palette roles collapsed")
    _require({1, 2, 3, 4} <= populations and {1, 2, 3} <= ranks, "contact/rank diversity")
    _require(len(bank_counts) >= 4 and len(entry_counts) >= 4 and len(lengths) >= 15, "mechanism diversity")
    _require(payload_classes == {1, 2} and min(features.values()) > 10, "family-feature coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(20261023 + nt * 10 + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit count ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests), "normalized_structures": len(structures),
            "grid_shapes": len(shapes), "directions": sorted(directions), "contact_populations": sorted(populations),
            "ranks": sorted(ranks), "bank_counts": sorted(bank_counts), "entry_counts": sorted(entry_counts),
            "bar_lengths": sorted(lengths), "payload_classes": sorted(payload_classes), "feature_pairs": dict(features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
