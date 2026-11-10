"""A periodic diagonal ribbon elects the color allowed through a voting gate.

Only standard-library modules and NumPy are used.  The solver reads the gate,
the periodic profiles, and the election from the supplied input alone.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a two-dimensional integer ARC grid")
    return a.astype(np.int8, copy=True)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _mode(values):
    counts = Counter(int(c) for c in values)
    if not counts:
        return None
    high = max(counts.values())
    winners = [c for c, n in counts.items() if n == high]
    return winners[0] if len(winners) == 1 else None


def _base(width, row, direction):
    return row if direction == 1 else width - 1 - row


def _recover(source, background, width):
    """Recover the shortest fully observed profile cycle, in either direction."""
    depth = len(source)
    points = np.argwhere(source != background)
    candidates = []
    for direction in (1, -1):
        if any(abs(int(c) - _base(width, int(r), direction)) > 2 for r, c in points):
            continue
        for period in range(1, 5):
            profiles = []
            valid = True
            for phase in range(period):
                rows = list(range(phase, depth, period))
                profile = {}
                for offset in range(-2, 3):
                    seen = [int(source[r, _base(width, r, direction) + offset])
                            for r in rows
                            if 0 <= _base(width, r, direction) + offset < width]
                    # Repeated evidence for absence is as important as for ink.
                    if len(seen) < 2 or len(set(seen)) != 1:
                        valid = False
                        break
                    if seen[0] != background:
                        profile[offset] = seen[0]
                if not valid or not profile:
                    valid = False
                    break
                profiles.append(profile)
            if not valid:
                continue
            envelope = sorted({o for p in profiles for o in p})
            if (0 not in envelope
                    or envelope != list(range(envelope[0], envelope[-1] + 1))):
                continue
            reconstructed = np.full_like(source, background)
            for row in range(depth):
                for offset, color in profiles[row % period].items():
                    col = _base(width, row, direction) + offset
                    if 0 <= col < width:
                        reconstructed[row, col] = color
            if np.array_equal(reconstructed, source):
                candidates.append((direction, profiles))
                break
    if len(candidates) != 1:
        raise ValueError("the diagonal cycle is not uniquely recoverable")
    return candidates[0]


def _analyze(input_grid):
    grid = _grid(input_grid)
    decompositions = []
    for rotation in range(4):
        canonical = np.rot90(grid, rotation)
        height, width = canonical.shape
        for gate in range(1, height - 1):
            marker = int(canonical[gate, 0])
            background = int(canonical[-1, 0])
            if marker == background or not np.all(canonical[gate] == marker):
                continue
            if not np.all(canonical[gate + 1:] == background):
                continue
            source = canonical[:gate]
            if np.any(source == marker):
                continue
            try:
                direction, profiles = _recover(source, background, width)
            except ValueError:
                continue
            motif = [c for p in profiles for c in p.values()]
            winner = _mode(motif)
            if len(set(motif)) < 2 or winner is None:
                continue
            decompositions.append(dict(
                canonical=canonical.copy(), rotation=rotation, gate=gate,
                background=background, marker=marker, direction=direction,
                profiles=profiles, period=len(profiles), winner=winner,
                votes=Counter(motif), width=width, height=height,
                visible_winner=_mode(source[source != background].flat)))
    if len(decompositions) != 1:
        raise ValueError("expected one source/gate/blank decomposition")
    return decompositions[0]


def _continue(canonical, gate, direction, profiles, winner, phase_delta=0,
              keep_all=False):
    result = canonical.copy()
    height, width = result.shape
    for row in range(gate + 1, height):
        for offset, color in profiles[(row + phase_delta) % len(profiles)].items():
            col = _base(width, row, direction) + offset
            if 0 <= col < width and (keep_all or color == winner):
                result[row, col] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Continue only the complete-cycle plurality, without resetting at the gate."""
    info = _analyze(input_grid)
    result = _continue(info["canonical"], info["gate"], info["direction"],
                       info["profiles"], info["winner"])
    return np.rot90(result, -info["rotation"]).copy()


_ENVELOPES = {
    1: [(0,)],
    2: [(-1, 0), (0, 1)],
    3: [(-2, -1, 0), (-1, 0, 1), (0, 1, 2)],
    4: [(-2, -1, 0, 1), (-1, 0, 1, 2)],
    5: [(-2, -1, 0, 1, 2)],
}


def _supports(rng, period, width, conflict):
    envelopes = _ENVELOPES[width]
    if conflict:
        envelopes = [e for e in envelopes if min(e) < 0]
    envelope = envelopes[int(rng.integers(len(envelopes)))]
    supports = []
    for _ in range(period):
        count = int(rng.integers(1 if not conflict else 2, width + 1))
        supports.append(sorted(int(x) for x in rng.choice(envelope, count, replace=False)))
    # Cover the entire narrow envelope, while allowing gaps in individual phases.
    covered = set().union(*(set(s) for s in supports))
    for offset in set(envelope) - covered:
        supports[int(rng.integers(period))].append(offset)
    return [sorted(s) for s in supports]


def _near_counts(rng, total):
    """Close pluralities below half: the election cannot mean majority only."""
    choices = []
    for k in range(2, min(5, total)):
        for winner in range(2, total):
            if winner * 2 >= total:
                continue
            for margin in (1, 2):
                second = winner - margin
                rest = total - winner - second
                if second < 1:
                    continue
                if k == 2 and rest == 0:
                    choices.append((k, winner, second, rest))
                elif k > 2 and k - 2 <= rest <= (k - 2) * second:
                    choices.append((k, winner, second, rest))
    if not choices:
        return None
    k, winner, second, rest = choices[int(rng.integers(len(choices)))]
    tail = [1] * (k - 2)
    for _ in range(rest - len(tail)):
        eligible = [i for i, n in enumerate(tail) if n < second]
        tail[eligible[int(rng.integers(len(eligible)))]] += 1
    return [winner, second] + tail


def _sample_profiles(rng, period, depth, conflict):
    width_options = (3, 4, 5) if conflict else ((1, 2, 3, 4, 5)
                                               if period >= 3 else (2, 3, 4, 5))
    width = int(rng.choice(width_options))
    supports = _supports(rng, period, width, conflict)
    slots = [(p, o) for p, support in enumerate(supports) for o in support]
    if len(slots) < 3:
        return None
    profiles = [{} for _ in range(period)]
    if conflict:
        counts = _near_counts(rng, len(slots))
        if counts is None:
            return None
        # A motif vote counts a slot once.  The visible image weights that same
        # slot by how often it escaped clipping and occurred before the gate.
        # Put the motif winner preferentially in the underrepresented slots.
        weights = [sum(r + o >= 0 for r in range(p, depth, period)) for p, o in slots]
        order = sorted(range(len(slots)), key=lambda i: weights[i] + float(rng.random()) * .9)
        winner_slots = order[:counts[0]]
        rest = order[counts[0]:]
        # Put its closest rival into the most represented remaining slots.
        rival_slots = rest[-counts[1]:]
        remaining = rest[:-counts[1]]
        assignments = [None] * len(slots)
        for i in winner_slots:
            assignments[i] = 0
        for i in rival_slots:
            assignments[i] = 1
        tail = [c for c in range(2, len(counts)) for _ in range(counts[c])]
        rng.shuffle(tail)
        for i, color in zip(remaining, tail):
            assignments[i] = color
    else:
        k = int(rng.integers(2, min(5, len(slots)) + 1))
        assignments = rng.integers(0, k, len(slots)).tolist()
        for i, color in zip(rng.permutation(len(slots)), range(k)):
            assignments[int(i)] = color
    for (phase, offset), color in zip(slots, assignments):
        profiles[phase][offset] = int(color)
    # Familiar rhythms with repeated phases remain available, not just ABCD.
    if not conflict and period == 3 and rng.random() < .24:
        profiles[1] = dict(profiles[0])
    if not conflict and period == 4 and rng.random() < .24:
        profiles[2] = dict(profiles[0])
    return profiles


def _build_pair(rng, period, rotation, reflected, target, conflict=False,
                gap=False, avoid_winner=None, avoid_gate_phase=None):
    for _ in range(1200):
        depth = int(rng.integers(2 * period + 2, 2 * period + 7))
        if avoid_gate_phase is not None and (depth + 1) % period == avoid_gate_phase:
            continue
        if gap and (depth + 1) % period == 0:
            continue
        blank = int(rng.integers(8, 13)) if target else int(rng.integers(max(4, period + 1), 8))
        height = depth + 1 + blank
        width = height + int(rng.integers(0, min(4, 30 - height) + 1))
        profiles = _sample_profiles(rng, period, depth, conflict)
        if profiles is None:
            continue
        motif = [c for p in profiles for c in p.values()]
        winner = _mode(motif)
        if len(set(motif)) < 2 or winner is None:
            continue
        if gap and not any(winner not in p.values() for p in profiles):
            continue
        visible = [c for r in range(depth) for o, c in profiles[r % period].items()
                   if 0 <= r + o < width]
        if conflict and (_mode(visible) is None or _mode(visible) == winner):
            continue
        color_order = rng.permutation(10).tolist()
        background, marker = color_order[:2]
        color_map = {c: color_order[c + 2] for c in set(motif)}
        actual_winner = color_map[winner]
        if actual_winner == avoid_winner:
            continue
        colored = [{o: color_map[c] for o, c in p.items()} for p in profiles]
        inp = np.full((height, width), background, dtype=np.int8)
        inp[depth] = marker
        for row in range(depth):
            for offset, color in colored[row % period].items():
                col = row + offset
                if 0 <= col < width:
                    inp[row, col] = color
        # Generation draws from the sampled motif; solve separately recovers it.
        out = inp.copy()
        for row in range(depth + 1, height):
            for offset, color in colored[row % period].items():
                col = row + offset
                if 0 <= col < width and color == actual_winner:
                    out[row, col] = color
        if reflected:
            inp, out = np.fliplr(inp), np.fliplr(out)
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        try:
            info = _analyze(inp)
        except ValueError:
            continue
        if info["period"] != period or np.count_nonzero(inp != out) < 2:
            continue
        # The second demonstration must distinguish resetting or skipping phase.
        if gap:
            reset = _continue(info["canonical"], info["gate"], info["direction"],
                              info["profiles"], info["winner"], -(info["gate"] + 1))
            skipped = _continue(info["canonical"], info["gate"], info["direction"],
                                info["profiles"], info["winner"], -1)
            correct = np.rot90(out, info["rotation"])
            if np.array_equal(reset, correct) or np.array_equal(skipped, correct):
                continue
        return {"input": inp, "output": out}, info
    raise RuntimeError("bounded sampling could not construct the requested witness")


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    first_direction = bool(rng.integers(2))
    periods = rng.choice([2, 3, 4], 2, replace=False).tolist()
    puzzle = {"train": [], "test": []}
    used = set()
    infos = []
    for index in range(train_count + test_count):
        target = index >= train_count
        if target:
            anchor = infos[int(rng.integers(len(infos)))]
            period = anchor["period"]
            reflected = bool(rng.integers(2))
            avoid_phase = ((anchor["gate"] + 1) % period) if period > 1 else None
        else:
            period = periods[index] if index < 2 else int(rng.integers(1, 5))
            reflected = first_direction if index == 0 else (
                not first_direction if index == 1 else bool(rng.integers(2)))
            avoid_phase = None
        for _ in range(24):
            pair, info = _build_pair(
                rng, period, rotation, reflected, target,
                conflict=index == 0, gap=index == 1,
                avoid_winner=infos[0]["winner"] if index == 1 else None,
                avoid_gate_phase=avoid_phase)
            key = _signature(pair["input"])
            if key not in used:
                used.add(key)
                break
        else:
            raise RuntimeError("could not obtain distinct inputs")
        puzzle["test" if target else "train"].append(pair)
        if not target:
            infos.append(info)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(inp)
            canonical = info["canonical"]
            got = np.rot90(out, info["rotation"])
            gate, winner = info["gate"], info["winner"]
            _require(np.array_equal(got[:gate + 1], canonical[:gate + 1]), "source or gate changed")
            _require(np.count_nonzero(inp != out) >= 2, "no substantial continuation")
            _require(set(int(c) for c in got[gate + 1:].flat) <= {info["background"], winner},
                     "a losing color crossed the gate")
            expected = _continue(canonical, gate, info["direction"], info["profiles"], winner)
            _require(np.array_equal(got, expected), "wrong modal continuation or phase")
            _require(np.array_equal(solve(inp), out), "solver disagreement")
            unfiltered = _continue(canonical, gate, info["direction"], info["profiles"],
                                   winner, keep_all=True)
            _require(not np.array_equal(unfiltered, got), "ignoring the election gives the answer")
            # Filling the whole lower diagonal with the winner ignores its stencil.
            solid = canonical.copy()
            for r in range(gate + 1, info["height"]):
                solid[r, _base(info["width"], r, info["direction"])] = winner
            info["geometry_witness"] = not np.array_equal(solid, got)
            reset = _continue(canonical, gate, info["direction"], info["profiles"],
                              winner, -(gate + 1))
            skipped = _continue(canonical, gate, info["direction"], info["profiles"], winner, -1)
            info["phase_witness"] = (not np.array_equal(reset, got)
                                     and not np.array_equal(skipped, got))
            information[split].append(info)
    train, test = information["train"], information["test"]
    _require(any(i["visible_winner"] is not None and i["visible_winner"] != i["winner"]
                 for i in train), "no visible-count counterexample")
    _require(any(max(i["votes"].values()) * 2 < sum(i["votes"].values()) for i in train),
             "training never demonstrates a plurality below half")
    _require(any(any(i["winner"] not in p.values() for p in i["profiles"]) for i in train),
             "no silent winning phase in training")
    _require(any(i["geometry_witness"] and i["phase_witness"] for i in train),
             "training does not expose geometry and uninterrupted phase")
    _require(len({i["winner"] for i in train}) >= 2, "only one winning color")
    _require({i["direction"] for i in test} <= {i["direction"] for i in train},
             "unseen diagonal direction")
    _require({i["period"] for i in test} <= {i["period"] for i in train}, "unseen test period")
    _require(min(i["height"] - i["gate"] - 1 for i in test)
             > max(i["height"] - i["gate"] - 1 for i in train), "missing continuation-length shift")


def _puzzle_signature(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    """Bounded 42-seed sweep, semantic witnesses, variation, counts and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "duplicate puzzles")
    sizes, periods, widths, rotations, winners, backgrounds, geometries = (set() for _ in range(7))
    gaps = clipped = varied_support = repeated_phase = margin_one = submajority = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _analyze(pair["input"])
                profiles = i["profiles"]
                envelope = {o for p in profiles for o in p}
                sizes.add(pair["input"].shape)
                periods.add(i["period"])
                widths.add(len(envelope))
                rotations.add(i["rotation"])
                winners.add(i["winner"])
                backgrounds.add(i["background"])
                geometries.add((i["period"], tuple(tuple(sorted(p)) for p in profiles)))
                gaps += any(i["winner"] not in p.values() for p in profiles)
                varied_support += len({tuple(sorted(p)) for p in profiles}) > 1
                repeated_phase += len({tuple(sorted(p.items())) for p in profiles}) < len(profiles)
                clipped += any(not 0 <= _base(i["width"], r, i["direction"]) + o < i["width"]
                               for r in range(i["gate"]) for o in profiles[r % i["period"]])
                counts = sorted(i["votes"].values(), reverse=True)
                margin_one += counts[0] - counts[1] == 1
                submajority += counts[0] * 2 < sum(counts)
    _require(len(sizes) >= 30 and periods == {1, 2, 3, 4}, "size/period diversity")
    _require(widths == {1, 2, 3, 4, 5} and len(geometries) >= 80, "profile diversity")
    _require(len(rotations) == 4 and len(winners) == len(backgrounds) == 10, "role diversity")
    _require(gaps >= 60 and clipped >= 80 and varied_support >= 100, "structural coverage")
    _require(repeated_phase >= 5 and margin_one >= 60 and submajority >= 5, "rhythm/vote coverage")
    _require(_puzzle_signature(generate_puzzle(71023, 7, 3))
             == _puzzle_signature(generate_puzzle(71023, 7, 3)), "nondeterminism")
    # All explicit count combinations, including the two-demonstration boundary.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9182 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A counterfactual that removes either inherited mechanism must be rejected.
    p = generate_puzzle(0, 2, 1)
    i = _analyze(p["train"][0]["input"])
    broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]]
              for s in ("train", "test")}
    all_colors = _continue(i["canonical"], i["gate"], i["direction"],
                           i["profiles"], i["winner"], keep_all=True)
    broken["train"][0]["output"] = np.rot90(all_colors, -i["rotation"]).copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted omission of the election")
    visible = i["visible_winner"]
    raw_vote = _continue(i["canonical"], i["gate"], i["direction"], i["profiles"], visible)
    broken["train"][0]["output"] = np.rot90(raw_vote, -i["rotation"]).copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted visible counts without cycle recovery")
    j = _analyze(p["train"][1]["input"])
    broken["train"][0]["output"] = p["train"][0]["output"].copy()
    skipped = _continue(j["canonical"], j["gate"], j["direction"], j["profiles"], j["winner"], -1)
    broken["train"][1]["output"] = np.rot90(skipped, -j["rotation"]).copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted omission of the gate's phase")
    permutation = np.random.default_rng(8701).permutation(10)
    for pair in p["train"]:
        for rotation in range(4):
            inp = np.rot90(permutation[pair["input"]], rotation)
            out = np.rot90(permutation[pair["output"]], rotation)
            _require(np.array_equal(solve(inp), out), "rotation/color covariance failed")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus count and counterfactual checks)")
