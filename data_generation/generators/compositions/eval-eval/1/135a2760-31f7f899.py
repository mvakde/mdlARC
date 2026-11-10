"""Repeated bar inventories: align by length rank, repair, and reassign.

Only NumPy and the standard library are used. No parent code or data is imported.
The output constructor uses clean inventory parameters; solve sees only cells.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


HOST_SEEDS = (
    0, 167439564, 118191105, 342249648, 833580230, 740667118, 217785445,
    676465358, 255974507, 136743767, 517845994, 86522872, 930722560,
    361564360, 513534993, 709966126, 848261595, 553604868, 345399085,
    278952855, 672848923, 937818478, 582102141, 81691149, 820545394,
    386693524, 275921487, 905703319, 691004586, 19882052, 14609441,
    342388996, 192695669, 544480247, 507262096, 70757049, 118596882,
    970906516, 300189496, 535460048, 989720061, 258948164,
)


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _key(value):
    a = np.asarray(value, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _parse(value):
    """Read visible roles, delimiter positions, and width-normalized profiles."""
    grid = _grid(value)
    bg = int(grid[0, 0])
    rows = np.flatnonzero(np.all(grid != bg, axis=1))
    cols = np.flatnonzero(np.all(grid != bg, axis=0))
    if len(rows) + len(cols) != 1:
        raise ValueError("carrier must be the unique full non-background line")
    vertical = bool(len(cols))
    a = grid.T if vertical else grid
    row = int(cols[0] if vertical else rows[0])
    if not 0 < row < a.shape[0] - 1 or a.shape[1] < 3:
        raise ValueError("carrier needs perpendicular margins and endpoint pips")
    line = a[row]
    marker, carrier = int(line[0]), int(line[1])
    if (len({bg, marker, carrier}) != 3 or int(line[-1]) != marker
            or int(line[-2]) != carrier):
        raise ValueError("invalid endpoint role convention")
    pips = np.flatnonzero(line == marker).tolist()
    if not 4 <= len(pips) <= 6:
        raise ValueError("expected three to five packets")
    explained = np.zeros(a.shape, dtype=bool)
    explained[row] = True
    packets = []
    for left, right in zip(pips, pips[1:]):
        if right - left < 4 or line[left + 1] != carrier or line[right - 1] != carrier:
            raise ValueError("a packet needs carrier gaps beside both pips")
        slots = []
        x = left + 1
        while x < right:
            color = int(line[x])
            end = x + 1
            while end < right and line[end] == color:
                end += 1
            if color != carrier:
                if color in (bg, marker) or end - x > 3:
                    raise ValueError("invalid slot run")
                strip = a[:, x:end]
                if not np.all((strip == bg) | (strip == color)):
                    raise ValueError("bar strips must be binary in their own color")
                occupancy = np.mean(strip == color, axis=1)
                slots.append({"x": x, "width": end - x, "color": color,
                              "profile": occupancy, "mass": float(occupancy.sum())})
                explained[:, x:end] = True
            x = end
        if not 2 <= len(slots) <= 4:
            raise ValueError("each packet needs two to four slots")
        packets.append(slots)
    if len({len(p) for p in packets}) != 1:
        raise ValueError("packet inventory counts must agree")
    if np.any(a[~explained] != bg):
        raise ValueError("unexplained cells outside the carrier and slot strips")
    ranked = [sorted(p, key=lambda s: s["mass"]) for p in packets]
    return {"a": a, "background": bg, "carrier": carrier, "marker": marker,
            "vertical": vertical, "row": row, "pips": pips,
            "packets": packets, "ranked": ranked}


def _recover(parsed):
    ranked = parsed["ranked"]
    r, n = len(ranked), len(ranked[0])
    profiles, spans = [], []
    by_length = {}
    for rank in range(n):
        score = np.sum([packet[rank]["profile"] for packet in ranked], axis=0)
        if np.any(np.isclose(score, r / 2, atol=1e-10, rtol=0)):
            raise ValueError("slice consensus must be strict")
        profile = score > r / 2
        occupied = np.flatnonzero(profile)
        if (not len(occupied) or not profile[parsed["row"]]
                or not np.all(np.diff(occupied) == 1)):
            raise ValueError("consensus must be an interval containing the carrier")
        lo, hi = int(occupied[0]) - parsed["row"], int(occupied[-1]) - parsed["row"]
        length = hi - lo + 1
        if spans and length < spans[-1][1] - spans[-1][0] + 1:
            raise ValueError("recovered length ranks must increase")
        if by_length.setdefault(length, (lo, hi)) != (lo, hi):
            raise ValueError("equal lengths must have identical geometry")
        profiles.append(profile)
        spans.append((lo, hi))
    return profiles, spans


def _paint_profiles(parsed, packet_profiles):
    out = np.full_like(parsed["a"], parsed["background"])
    out[parsed["row"]] = parsed["a"][parsed["row"]]
    for packet, profiles in zip(parsed["packets"], packet_profiles):
        for slot, profile in zip(packet, profiles):
            out[np.asarray(profile, dtype=bool), slot["x"]:slot["x"] + slot["width"]] = slot["color"]
    return out.T.copy() if parsed["vertical"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; never consult generation or a seed."""
    parsed = _parse(input_grid)
    profiles, _ = _recover(parsed)
    return _paint_profiles(parsed, [profiles] * len(parsed["packets"]))


def _draw(shape, row, slots, pips, bg, carrier, marker, packet_spans):
    """Constructor path: render clean integer intervals, not solver profiles."""
    out = np.full(shape, bg, dtype=np.int8)
    out[row] = carrier
    out[row, pips] = marker
    for packet, spans in zip(slots, packet_spans):
        for slot, (lo, hi) in zip(packet, spans):
            x, width, color = slot
            out[row + lo:row + hi + 1, x:x + width] = color
    return out


def _choose_inventory(rng, n, cap, witness, long_test):
    if witness == 1:
        # Three ranks, with room for a strict width-versus-length inversion.
        first = int(rng.integers(1, 4))
        lengths = [first, first + 3, first + 6]
    else:
        distinct = n
        if not witness and rng.random() < 0.28:
            distinct = int(rng.integers(2, n)) if n > 2 else 2
        # The gap composition samples every feasible inventory, not a bank.
        spare = cap - (1 + 3 * (distinct - 1))
        allocation = rng.multinomial(spare, [1 / (distinct + 1)] * (distinct + 1))
        unique = [1 + int(allocation[0])]
        for i in range(1, distinct):
            unique.append(unique[-1] + 3 + int(allocation[i]))
        if long_test:
            unique[-1] = cap
        lengths = unique + [int(rng.choice(unique)) for _ in range(n - distinct)]
        lengths.sort()
    intervals = {}
    for length in sorted(set(lengths)):
        mode = int(rng.integers(4))
        if mode == 0:
            low = -(length // 2)
        elif mode == 1:
            low = 0
        elif mode == 2:
            low = 1 - length
        else:
            low = -int(rng.integers(length))
        intervals[length] = (low, low + length - 1)
    if witness == 1:
        # Moved asymmetric templates in both directions are visibly demonstrated.
        intervals[lengths[1]] = (1 - lengths[1], 0)
        intervals[lengths[2]] = (0, lengths[2] - 1)
    return [intervals[length] for length in lengths]


def _orders(rng, n, copies, witness):
    if witness == 1:
        base = rng.permutation(n)
        return [np.roll(base, k).tolist() for k in range(copies)]
    orders = [rng.permutation(n).tolist() for _ in range(copies)]
    if witness:
        orders[0] = list(range(1, n)) + [0]
    elif rng.random() < 0.12:
        orders = [list(range(n)) for _ in range(copies)]
    return orders


def _layout(rng, n, copies, orders, witness, accents):
    # Base layout: one gap at either side of every packet, one pip per boundary.
    colors = [rng.choice(accents, n).astype(int).tolist() for _ in range(copies)]
    for packet in colors:
        if len(set(packet)) == 1:
            packet[-1] = next(c for c in accents if c != packet[0])
    gaps = [[int(colors[k][j] == colors[k][j + 1]) for j in range(n - 1)]
            for k in range(copies)]
    widths = [[1] * n for _ in range(copies)]
    minimum = copies * (n + 2) + copies + 1 + sum(sum(g) for g in gaps)
    if minimum > 30:
        # Rare dense repeated palettes: make touching runs distinguishable.
        for k in range(copies):
            colors[k] = [accents[j % len(accents)] for j in range(n)]
        gaps = [[0] * (n - 1) for _ in range(copies)]
        minimum = copies * (n + 2) + copies + 1
    locked = set()
    if witness == 1:
        short = orders[0].index(n - 2)
        long = orders[0].index(n - 1)
        widths[0][short] = 3
        locked.add((0, long))
        minimum += 2
    budget = int(rng.integers(minimum, 31)) - minimum
    # Allocate extra cells to widths and irregular gaps, subject to 30 cells.
    edges = [[1, 1] for _ in range(copies)]
    for _ in range(budget):
        choices = [("width", k, j) for k in range(copies) for j in range(n)
                   if widths[k][j] < 3 and (k, j) not in locked]
        choices += [("edge", k, j) for k in range(copies) for j in range(2)]
        choices += [("gap", k, j) for k in range(copies) for j in range(n - 1)]
        kind, k, j = choices[int(rng.integers(len(choices)))]
        {"width": widths, "edge": edges, "gap": gaps}[kind][k][j] += 1
    x, slots, pips = 0, [], [0]
    for k in range(copies):
        packet = []
        x += 1 + edges[k][0]
        for j in range(n):
            packet.append((x, widths[k][j], colors[k][j]))
            x += widths[k][j]
            if j < n - 1:
                x += gaps[k][j]
        x += edges[k][1]
        pips.append(x)
        slots.append(packet)
    return x + 1, slots, pips


def _damage(rng, inp, row, slots, orders, spans, witness):
    """At most one width-normalized slice error per slot; no repeated bad slice."""
    bg = int(inp[0, 0])
    used, used_slots = set(), set()
    n, copies = len(spans), len(slots)
    proposals = []
    if witness == 1:
        # Every packet is impaired, but different ranks retain other witnesses.
        proposals = [(k, orders[k].index(k)) for k in range(copies)]
    elif witness:
        proposals = [(0, 0), (1, 1)]
    elif rng.random() >= 0.10:
        count = int(rng.integers(1, min(6, n * copies) + 1))
        proposals = [divmod(int(i), n) for i in rng.choice(n * copies, count, replace=False)]
    for k, j in proposals:
        rank = orders[k][j]
        lo, hi = spans[rank]
        length = hi - lo + 1
        x, width, color = slots[k][j]
        endpoint_rows = [row + lo - 1, row + hi + 1]
        if lo < 0:
            endpoint_rows.append(row + lo)
        if hi > 0:
            endpoint_rows.append(row + hi)
        possibilities = [t for t in endpoint_rows if (length, t) not in used]
        # Witnesses always contain whole-slice endpoint errors: locally plausible
        # solid rectangles that cannot be corrected without the repeated inventory.
        whole_slice = bool(witness or rng.random() < 0.7)
        if not whole_slice:
            possibilities = [t for t in range(row + lo - 1, row + hi + 2)
                             if t != row and (length, t) not in used]
        if not possibilities or (k, j) in used_slots:
            continue
        t = int(rng.choice(possibilities))
        if whole_slice:
            inp[t, x:x + width] = color if inp[t, x] == bg else bg
        else:
            col = x + int(rng.integers(width))
            inp[t, col] = color if inp[t, col] == bg else bg
        used.add((length, t))
        used_slots.add((k, j))


def _make_pair(rng, config, split, vertical, witness=0):
    shift = config["shift"]
    if witness == 1:
        n, copies = 3, 3
    elif witness == 2:
        n, copies = (2, 3) if shift == "inventory" else (4, 3)
    elif shift == "inventory":
        n = int(rng.integers(2, 4)) if split == "train" else 4
        copies = int(rng.integers(3, 6 if n == 2 else 5)) if split == "train" else 3
    elif shift == "copies":
        if split == "train":
            n, copies = int(rng.integers(2, 5)), 3
        else:
            copies = int(rng.integers(4, 6))
            n = 2 if copies == 5 else int(rng.integers(2, 5))
    else:
        n = int(rng.integers(2, 5))
        copies = int(rng.integers(3, 6 if n == 2 else 5))
    long_test = split == "test" and shift == "spans"
    cap = int(rng.integers(11, 15)) if long_test else 10
    spans = _choose_inventory(rng, n, cap, witness, long_test)
    orders = _orders(rng, n, copies, witness)
    if config["fixed_roles"]:
        bg, carrier, marker = config["roles"]
    else:
        bg, carrier, marker = rng.choice(10, 3, replace=False).astype(int).tolist()
    pool = [c for c in range(10) if c not in (bg, carrier, marker)]
    accents = rng.choice(pool, int(rng.integers(2, min(n + 1, 6) + 1)), replace=False).astype(int).tolist()
    width, slots, pips = _layout(rng, n, copies, orders, witness, accents)
    low, high = min(s[0] for s in spans), max(s[1] for s in spans)
    # Two clean margin rows permit a one-slice extension while keeping the
    # outermost background rows intact. Extra margin shifts the carrier freely.
    base_height = high - low + 5
    if base_height > 30:
        raise AssertionError("inventory exceeds perpendicular grid budget")
    extra = int(rng.integers(0, min(6, 30 - base_height) + 1))
    before = int(rng.integers(extra + 1))
    row = 2 - low + before
    shape = (base_height + extra, width)
    packet_spans = [[spans[r] for r in order] for order in orders]
    inp = _draw(shape, row, slots, pips, bg, carrier, marker, packet_spans)
    out = _draw(shape, row, slots, pips, bg, carrier, marker, [spans] * copies)
    _damage(rng, inp, row, slots, orders, spans, witness)
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    config = {"shift": str(rng.choice(["inventory", "copies", "spans"])),
              "fixed_roles": bool(rng.random() < 0.3),
              "roles": rng.choice(10, 3, replace=False).astype(int).tolist()}
    axis = str(rng.choice(["horizontal", "vertical", "mixed"]))
    axes = [axis == "vertical"] * (nt + nq)
    if axis == "mixed":
        axes = [bool(x) for x in rng.integers(0, 2, nt + nq)]
        axes[:2] = [False, True]
    puzzle, seen = {"train": [], "test": []}, set()
    for serial in range(nt + nq):
        split = "train" if serial < nt else "test"
        witness = serial + 1 if serial < 2 else 0
        for _ in range(64):
            pair = _make_pair(rng, config, split, axes[serial], witness)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("finite distinct-input construction exhausted")
        puzzle[split].append(pair)
        seen.add(key)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Relational label checks; does not call solve, _recover, or either painter."""
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise ValueError("a pair must have exactly input and output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape, "canvas dimensions changed"
    a, b = _parse(inp), _parse(out)
    for role in ("background", "carrier", "marker", "vertical", "row", "pips"):
        assert a[role] == b[role], f"changed {role}"
    assert np.array_equal(a["a"][a["row"]], b["a"][b["row"]]), "changed carrier"
    inventory, lengths = [], []
    for packet_index, (src, dst) in enumerate(zip(a["packets"], b["packets"])):
        assert [(s["x"], s["width"], s["color"]) for s in src] == [
            (s["x"], s["width"], s["color"]) for s in dst], "changed slot identity"
        profiles = []
        for slot in dst:
            profile = slot["profile"]
            assert np.all((profile == 0) | (profile == 1)), "output bar is not solid across width"
            occupied = np.flatnonzero(profile)
            assert len(occupied) and np.all(np.diff(occupied) == 1), "output interval has holes"
            assert profile[b["row"]] == 1, "output misses carrier"
            assert occupied[0] >= 2 and occupied[-1] <= b["a"].shape[0] - 3, "clean margins lost"
            profiles.append(profile.astype(bool))
        current_lengths = [int(p.sum()) for p in profiles]
        assert current_lengths == sorted(current_lengths), "output ranks are not increasing"
        if packet_index == 0:
            inventory, lengths = profiles, current_lengths
        else:
            assert all(np.array_equal(p, q) for p, q in zip(inventory, profiles)), "inventories disagree"
    unique_lengths = sorted(set(lengths))
    assert all(y - x >= 3 for x, y in zip(unique_lengths, unique_lengths[1:])), "ranks can cross"
    for i, length in enumerate(lengths):
        assert np.array_equal(inventory[i], inventory[lengths.index(length)]), "unequal-offset tie"
    r, n = len(a["packets"]), len(inventory)
    errors, repaired_original, raw_sorted = [], [], []
    changed_order = area_diff = reversal_diff = False
    for packet, ranked in zip(a["packets"], a["ranked"]):
        true_by_x = {slot["x"]: inventory[rank] for rank, slot in enumerate(ranked)}
        true_original = [true_by_x[slot["x"]] for slot in packet]
        repaired_original.append(true_original)
        raw_sorted.append([slot["profile"] > 0.5 for slot in ranked])
        packet_errors = 0
        for rank, slot in enumerate(ranked):
            profile = slot["profile"]
            error = float(np.abs(profile - inventory[rank]).sum())
            assert error <= 1 + 1e-9, "more than one slice-equivalent error in a bar"
            assert abs(slot["mass"] - lengths[rank]) <= 1 + 1e-9
            packet_errors += int(round(error * slot["width"]))
        errors.append(packet_errors)
        changed_order |= any(not np.array_equal(p, q) for p, q in zip(true_original, inventory))
        reversal_diff |= any(not np.array_equal(p, q) for p, q in zip(true_original[::-1], inventory))
        area_order = sorted(packet, key=lambda slot: true_by_x[slot["x"]].sum() * slot["width"])
        area_diff |= any(not np.array_equal(true_by_x[slot["x"]], profile)
                         for slot, profile in zip(area_order, inventory))
    for length in unique_lengths:
        ranks = [j for j, value in enumerate(lengths) if value == length]
        # This includes all tied observations, so arbitrary ordering of their
        # perturbed masses cannot create a false rank-dependent majority.
        damaged_packets = np.zeros((r, a["a"].shape[0]), dtype=bool)
        for k, packet in enumerate(a["ranked"]):
            for rank in ranks:
                damaged_packets[k] |= np.abs(packet[rank]["profile"] - inventory[rank]) > 1e-9
        assert np.all(damaged_packets.sum(axis=0) <= 1), "repeated slice is majority-corruptible"
    for rank in range(n):
        total = np.sum([packet[rank]["profile"] for packet in a["ranked"]], axis=0)
        assert np.all(np.abs(total - r / 2) > 1e-9), "tied slice consensus"
        assert np.array_equal(total > r / 2, inventory[rank]), "output violates rank consensus"
    omit_a = _paint_profiles(a, raw_sorted)
    omit_b = _paint_profiles(a, repaired_original)
    positional = [np.mean([packet[j]["profile"] for packet in a["packets"]], axis=0) > 0.5
                  for j in range(n)]
    omit_rank_alignment = _paint_profiles(a, [positional] * r)
    spans = []
    for profile in inventory:
        occupied = np.flatnonzero(profile)
        spans.append((int(occupied[0]) - a["row"], int(occupied[-1]) - a["row"]))
    return {"n": n, "copies": r, "lengths": tuple(lengths), "spans": tuple(spans),
            "widths": tuple(s["width"] for p in a["packets"] for s in p),
            "vertical": a["vertical"], "shape": tuple(inp.shape),
            "roles": (a["background"], a["carrier"], a["marker"]),
            "errors": tuple(errors), "no_intact_packet": all(e > 0 for e in errors),
            "sort_changes": changed_order, "area_diff": area_diff,
            "reversal_diff": reversal_diff,
            "omit_a_diff": not np.array_equal(omit_a, out),
            "omit_b_diff": not np.array_equal(omit_b, out),
            "positional_diff": not np.array_equal(omit_rank_alignment, out),
            "touching": any(q["x"] == p["x"] + p["width"]
                            for packet in a["packets"] for p, q in zip(packet, packet[1:])),
            "ties": len(unique_lengths) < n,
            "layout": (a["vertical"], tuple(inp.shape), a["row"], tuple(a["pips"]),
                       tuple(tuple((s["x"], s["width"]) for s in p) for p in a["packets"]),
                       tuple(spans))}


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("train and test must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("invalid example counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            feature = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            features[split].append(feature)
    train = features["train"]
    assert len({f["n"] for f in train}) >= 2, "training lacks inventory-count variation"
    assert sum(f["sort_changes"] for f in train) >= 2, "need two nontrivial demonstrations"
    for feature in ("omit_a_diff", "omit_b_diff", "area_diff", "positional_diff"):
        assert any(f[feature] for f in train), f"training does not rule out {feature}"
    assert any(f["sort_changes"] and f["reversal_diff"] for f in train), "reversal shortcut"
    assert any(f["no_intact_packet"] for f in train), "intact-copy shortcut"
    shown_axes = {f["vertical"] for f in train if f["sort_changes"]}
    assert all(f["vertical"] in shown_axes for f in features["test"]), "undemonstrated test orientation"


def _hand_example(widths, orders, spans, defects=(), vertical=False):
    """Small compositional fixtures for solver and relational mutation checks."""
    copies, n = len(orders), len(spans)
    slots, pips, x = [], [0], 0
    for k in range(copies):
        x += 2
        packet = []
        for j in range(n):
            packet.append((x, widths[k][j], 3 + j))
            x += widths[k][j]
        x += 1
        pips.append(x)
        slots.append(packet)
    low, high = min(s[0] for s in spans), max(s[1] for s in spans)
    row = 2 - low
    shape = (high - low + 5, x + 1)
    inp = _draw(shape, row, slots, pips, 0, 1, 2, [[spans[j] for j in order] for order in orders])
    out = _draw(shape, row, slots, pips, 0, 1, 2, [spans] * copies)
    for packet, index, offset, whole in defects:
        col, width, color = slots[packet][index]
        if whole:
            inp[row + offset, col:col + width] = color if inp[row + offset, col] == 0 else 0
        else:
            inp[row + offset, col] = color if inp[row + offset, col] == 0 else 0
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def self_test():
    """Bounded 42-seed diversity sweep, counts, ablations, and targeted fixtures."""
    puzzles, features = [], []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.array(pair["input"], copy=True)
                assert np.array_equal(solve(before), pair["output"]), "input-only solve disagreement"
                assert np.array_equal(before, pair["input"]), "solve mutated its input"
                f = _check_pair(pair)
                features.append(f)
    assert len(set(puzzles)) == 42
    assert {f["n"] for f in features} == {2, 3, 4}
    assert {f["copies"] for f in features} == {3, 4, 5}
    assert {f["vertical"] for f in features} == {False, True}
    assert {w for f in features for w in f["widths"]} == {1, 2, 3}
    assert {f["ties"] for f in features} == {False, True}
    assert {f["touching"] for f in features} == {False, True}
    assert len({f["shape"] for f in features}) >= 40
    assert len({f["layout"] for f in features}) >= 180
    assert len({f["roles"] for f in features}) >= 30
    assert max(max(f["lengths"]) for f in features) >= 13
    assert min(min(f["lengths"]) for f in features) == 1
    assert _puzzle_key(generate_puzzle(0)) == puzzles[0]
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(73, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None), (None, 1), (4, "1")):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Wide noisy copy must not outweigh two narrow correct copies. All original
    # bars are solid: local rectangle completion cannot reveal the false endpoint.
    weighted = _hand_example([[3, 1], [1, 1], [1, 1]], [[0, 1]] * 3,
                             [(0, 0), (-2, 2)], [(0, 0, 1, True)])
    assert np.array_equal(solve(weighted["input"]), weighted["output"])
    a = _parse(weighted["input"])
    raw_pixel_votes = sum(p[0]["profile"] * p[0]["width"] for p in a["ranked"])
    assert raw_pixel_votes[a["row"] + 1] > sum(p[0]["width"] for p in a["ranked"]) / 2
    _check_pair(weighted)
    # Perturbed identical-geometry ties may swap observed ranks; consensus still
    # recovers the same tied inventory. Include both orientations and unit spans.
    tied = _hand_example([[1, 1, 1]] * 3, [[1, 2, 0], [2, 0, 1], [0, 1, 2]],
                         [(0, 0), (0, 0), (-3, 0)], [(0, 0, 1, True), (1, 0, -3, True)])
    for fixture in (tied, {k: v.T.copy() for k, v in tied.items()}):
        assert np.array_equal(solve(fixture["input"]), fixture["output"])
        _check_pair(fixture)
    # The solver's domain is broader than the sampling cap of fourteen. This
    # near-boundary inventory includes lengths 22 and 25 on a 29-row canvas.
    large = _hand_example([[1, 1]] * 3, [[1, 0], [0, 1], [1, 0]],
                          [(-10, 11), (-12, 12)], [(0, 0, 13, True)])
    fractional = _hand_example([[2, 1], [1, 2], [1, 1]], [[1, 0], [0, 1], [1, 0]],
                               [(-1, 0), (-2, 2)], [(0, 0, 1, False), (1, 0, 1, True)])
    all_tied = _hand_example([[1, 1]] * 3, [[0, 1], [1, 0], [0, 1]],
                             [(-1, 1), (-1, 1)], [(0, 0, 2, True)])
    for fixture in (large, fractional, all_tied):
        assert np.array_equal(solve(fixture["input"]), fixture["output"])
        _check_pair(fixture)
    # Complementary endpoint defects: no complete packet is correct.
    no_perfect = _hand_example([[1, 1, 1]] * 3, [[1, 2, 0], [2, 0, 1], [0, 1, 2]],
                              [(0, 1), (-4, 0), (0, 7)],
                              [(0, 2, 2, True), (1, 2, -4, True), (2, 2, 8, True)])
    assert _check_pair(no_perfect)["no_intact_packet"]
    assert np.array_equal(solve(no_perfect["input"]), no_perfect["output"])
    # Relational checker rejects ablated labels, ordering mistakes, and moved pips.
    rejected = Counter()
    for pair in generate_puzzle(0)["train"]:
        a = _parse(pair["input"])
        profiles, _ = _recover(a)
        cleaned_original = []
        for packet, ranked in zip(a["packets"], a["ranked"]):
            truth = {s["x"]: profiles[j] for j, s in enumerate(ranked)}
            cleaned_original.append([truth[s["x"]] for s in packet])
        variants = {
            "omit_a": _paint_profiles(a, [[s["profile"] > 0.5 for s in p] for p in a["ranked"]]),
            "omit_b": _paint_profiles(a, cleaned_original),
            "descending": _paint_profiles(a, [profiles[::-1]] * len(a["packets"])),
        }
        area_profiles = []
        for packet, ranked in zip(a["packets"], a["ranked"]):
            truth = {s["x"]: profiles[j] for j, s in enumerate(ranked)}
            order = sorted(packet, key=lambda s: truth[s["x"]].sum() * s["width"])
            area_profiles.append([truth[s["x"]] for s in order])
        variants["area"] = _paint_profiles(a, area_profiles)
        positional = [np.mean([p[j]["profile"] for p in a["packets"]], axis=0) > 0.5
                      for j in range(len(profiles))]
        variants["positional"] = _paint_profiles(a, [positional] * len(a["packets"]))
        centered = []
        for profile in profiles:
            length = int(profile.sum())
            p = np.zeros(a["a"].shape[0], dtype=bool)
            lo = a["row"] - length // 2
            hi = lo + length
            if not (0 <= lo < hi <= len(p)):
                break
            p[lo:hi] = True
            centered.append(p)
        if len(centered) == len(profiles):
            variants["recenter"] = _paint_profiles(a, [centered] * len(a["packets"]))
        changed_pip = np.array(pair["output"], copy=True)
        canonical = changed_pip.T if a["vertical"] else changed_pip
        canonical[a["row"], a["pips"][1]] = a["carrier"]
        variants["changed_pip"] = changed_pip
        for name, bad_output in variants.items():
            if np.array_equal(bad_output, pair["output"]):
                continue
            try:
                _check_pair({"input": pair["input"], "output": bad_output})
            except (AssertionError, ValueError):
                rejected[name] += 1
            else:
                raise AssertionError(f"accepted {name} label mutation")
    assert set(rejected) == {"omit_a", "omit_b", "descending", "area", "positional", "recenter", "changed_pip"}
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(set(puzzles)),
            "shapes": len({f["shape"] for f in features}),
            "structural_layouts": len({f["layout"] for f in features}),
            "role_palettes": len({f["roles"] for f in features}),
            "inventory_counts": sorted({f["n"] for f in features}),
            "packet_counts": sorted({f["copies"] for f in features}),
            "lengths": sorted({length for f in features for length in f["lengths"]}),
            "mutation_rejections": dict(rejected)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
