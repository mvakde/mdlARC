"""Reflection-restored, locator-selected panels for the required saved proposal.

The relocation and destination crop are implemented literally. Their composition
equals the source payload; this unavoidable shortcut is documented, not claimed
to be defeated. A unique reflection-fixed locator remains visible.
"""
from functools import lru_cache
import hashlib
import json

import numpy as np


GROUPS = (("H",), ("V",), ("D",), ("A",), ("H", "V"),
          ("D", "A"), ("H", "V", "D"))
DIVIDERS = np.zeros((11, 11), dtype=bool)
DIVIDERS[[3, 7], :] = True
DIVIDERS[:, [3, 7]] = True


def _reflect(cell, name):
    r, c = cell
    return {"H": (2-r, c), "V": (r, 2-c),
            "D": (c, r), "A": (2-c, 2-r)}[name]


@lru_cache(maxsize=7)
def _orbits(group):
    unseen = {(r, c) for r in range(3) for c in range(3)}
    result = []
    while unseen:
        orbit = {min(unseen)}
        while True:
            expanded = orbit | {_reflect(p, g) for p in orbit for g in group}
            if expanded == orbit:
                break
            orbit = expanded
        unseen -= orbit
        result.append(tuple(sorted(orbit)))
    return tuple(result)


def _panel(position):
    r, c = position
    return slice(4*r, 4*r+3), slice(4*c, 4*c+3)


def _blank(background, divider):
    grid = np.full((11, 11), background, dtype=np.int8)
    grid[DIVIDERS] = divider
    return grid


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _overwrite(grid):
    candidates = []
    for color in np.unique(grid):
        rr, cc = np.nonzero(grid == color)
        r, s = int(rr.min()), int(rr.max())+1
        c, d = int(cc.min()), int(cc.max())+1
        if 2 <= len(rr) <= 22 and (s-r)*(d-c) == len(rr):
            candidates.append((int(color), (r, s, c, d)))
    if len(candidates) != 1:
        raise ValueError("Occluder must be the sole rectangular color of area 2..22")
    return candidates[0]


def _place_and_crop(payload, address, background, divider):
    """Execute the proposal's clearing, relocation, and destination crop."""
    relocated = _blank(background, divider)
    relocated[_panel(address)] = payload
    return relocated[_panel(address)].copy(), relocated


def _completion(grid, occluder, divider, group):
    full = grid.copy()
    full[DIVIDERS] = divider
    support = []
    for pr in range(3):
        for pc in range(3):
            panel = grid[_panel((pr, pc))]
            repaired = full[_panel((pr, pc))]
            for orbit in _orbits(group):
                values = [int(panel[p]) for p in orbit if panel[p] != occluder]
                if not values or len(set(values)) != 1:
                    return None
                for p in orbit:
                    repaired[p] = values[0]
                if len(orbit) > 1 and len(values) >= 2:
                    support.append(((pr, pc), values[0]))
    if len(support) < 12 or len({v for _, v in support}) < 2:
        return None
    counts = np.bincount(full[~DIVIDERS].astype(int), minlength=10)
    singletons = np.flatnonzero(counts == 1)
    modes = np.flatnonzero(counts == counts.max())
    if len(singletons) != 1 or len(modes) != 1:
        return None
    background, locator = int(modes[0]), int(singletons[0])
    if len({p for p, v in support if v != background}) < 3:
        return None
    rr, cc = map(int, np.argwhere(full == locator)[0])
    source, address = (rr//4, cc//4), (rr % 4, cc % 4)
    payload = full[_panel(source)].copy()
    hidden = grid[_panel(source)] == occluder
    if (not np.any(hidden & (payload != background))
            or not np.any(payload == background)
            or not np.any((payload != background) & (payload != locator))):
        return None
    # A globally unique color must occupy a singleton reflection orbit.
    if any(_reflect(address, g) != address for g in group):
        return None
    if grid[rr, cc] != locator:
        return None
    answer, relocated = _place_and_crop(payload, address, background, divider)
    return {"full": full, "payload": payload, "output": answer,
            "relocated": relocated, "source": source, "address": address,
            "background": background, "divider": divider, "locator": locator,
            "support_orbits": len(support)}


def _infer(input_grid):
    grid = _grid(input_grid)
    if grid.shape != (11, 11):
        raise ValueError("This saved composition uses an 11-by-11 lattice")
    occluder, box = _overwrite(grid)
    visible_dividers = grid[DIVIDERS & (grid != occluder)]
    if not len(visible_dividers) or len(np.unique(visible_dividers)) != 1:
        raise ValueError("Divider lattice is inconsistent")
    divider = int(visible_dividers[0])
    if np.any(grid[~DIVIDERS] == divider):
        raise ValueError("Divider color occurs inside a panel")
    candidates = []
    for group in GROUPS:
        record = _completion(grid, occluder, divider, group)
        if record is not None:
            candidates.append((group, record))
    if not candidates:
        raise ValueError("No sufficiently witnessed, recoverable reflection model")
    record = candidates[0][1]
    if any(not np.array_equal(record["full"], r["full"]) for _, r in candidates):
        raise ValueError("Supported reflection completions disagree")
    r, s, c, d = box
    covered = {(rr//4, cc//4) for rr in range(r, s) for cc in range(c, d)
               if not DIVIDERS[rr, cc]}
    record.update({"models": tuple(g for g, _ in candidates), "box": box,
                   "occluder": occluder, "covered_panels": frozenset(covered),
                   "masked_divider_cells": int(np.count_nonzero(grid[DIVIDERS] == occluder))})
    return record


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer reflection, roles, locator, and payload using only this input."""
    return _infer(input_grid)["output"].copy()


@lru_cache(maxsize=7)
def _boxes(group):
    """Bounded geometry table: no rectangle may erase an entire panel orbit."""
    global_orbits = [tuple((4*pr+r, 4*pc+c) for r, c in orbit)
                     for pr in range(3) for pc in range(3) for orbit in _orbits(group)]
    result = []
    for r in range(11):
        for s in range(r+1, 12):
            for c in range(11):
                for d in range(c+1, 12):
                    if not 2 <= (s-r)*(d-c) <= 22:
                        continue
                    if any(all(r <= rr < s and c <= cc < d for rr, cc in orbit)
                           for orbit in global_orbits):
                        continue
                    covered = frozenset((rr//4, cc//4) for rr in range(r, s)
                                        for cc in range(c, d) if not DIVIDERS[rr, cc])
                    if sum(not DIVIDERS[rr, cc] for rr in range(r, s) for cc in range(c, d)) < 2:
                        continue
                    result.append(((r, s, c, d), covered,
                                   bool(np.any(DIVIDERS[r:s, c:d]))))
    return tuple(result)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _draw_pair(rng, group, source, is_test, rich, seen):
    orbits = _orbits(group)
    fixed = [orbit[0] for orbit in orbits if len(orbit) == 1]
    choices = [(box, covered) for box, covered, crosses in _boxes(group)
               if source in covered and ((len(covered) >= 2) if is_test
                                         else (covered == {source} and not crosses))]
    if not choices:
        raise RuntimeError("No compatible occluder geometry")
    for _ in range(160):
        colors = [int(c) for c in rng.permutation(10)]
        background, divider, locator, occluder = colors[:4]
        palette = colors[4:4+int(rng.integers(1, 6))]
        full = _blank(background, divider)
        address = fixed[int(rng.integers(len(fixed)))]
        for pr in range(3):
            for pc in range(3):
                panel = full[_panel((pr, pc))]
                density = float(rng.choice([0.0, .18, .35, .60, .82]))
                for orbit in orbits:
                    value = int(rng.choice(palette)) if rng.random() < density else background
                    for p in orbit:
                        panel[p] = value
        # Construct the payload from whole orbits, with preserved internal background.
        payload = full[_panel(source)]
        payload[:] = background
        available = [o for o in orbits if address not in o]
        shuffled = [available[int(i)] for i in rng.permutation(len(available))]
        maximum = len(available)-1  # at least one background orbit survives
        number = maximum if rich else int(rng.integers(1, maximum+1))
        for orbit in shuffled[:number]:
            value = int(rng.choice(palette))
            for p in orbit:
                payload[p] = value
        payload[address] = locator
        # Incidental singleton auxiliaries cannot be learned by an input-only solver.
        # They can only lie on fixed points; removing them preserves every reflection.
        counts = np.bincount(full[~DIVIDERS].astype(int), minlength=10)
        for color in np.flatnonzero(counts == 1):
            if color != locator:
                full[full == color] = background
        counts = np.bincount(full[~DIVIDERS].astype(int), minlength=10)
        if counts[background] <= max(counts[c] for c in range(10) if c != background):
            continue
        order = rng.permutation(len(choices))
        for index in order[:32]:
            box, _ = choices[int(index)]
            r, s, c, d = box
            inp = full.copy()
            inp[r:s, c:d] = occluder
            if _key(inp) in seen:
                continue
            try:
                record = _infer(inp)
            except ValueError:
                continue
            if group not in record["models"] or not np.array_equal(record["full"], full):
                continue
            # Labels come from the unoccluded construction, independently of inference.
            output, _ = _place_and_crop(full[_panel(source)], address, background, divider)
            return {"input": inp, "output": output}
    raise RuntimeError("Bounded sampling exhausted without a supported example")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    group = GROUPS[int(rng.integers(len(GROUPS)))]
    positions = [(r, c) for r in range(3) for c in range(3)]
    order = [positions[int(i)] for i in rng.permutation(9)]
    sources = order[:2] + [order[int(rng.integers(8))] for _ in range(num_train-2)]
    tests = [order[8]] + [positions[int(rng.integers(9))] for _ in range(num_test-1)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, sources_here in (("train", sources), ("test", tests)):
        for index, source in enumerate(sources_here):
            pair = _draw_pair(rng, group, source, split == "test", index % 2 == 0, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert output.shape == (3, 3)
            assert _key(grid) not in seen, "Duplicate input"
            seen.add(_key(grid))
            record = _infer(grid)
            assert np.array_equal(output, record["output"]), "Wrong restored payload"
            assert record["occluder"] not in output and record["divider"] not in output
            assert np.count_nonzero(output == record["locator"]) == 1
            assert not np.array_equal(grid[_panel(record["source"])], output), "Reflection must matter"
            # Counterfactual: filling the hidden payload with background must be wrong.
            naive = grid[_panel(record["source"])].copy()
            naive[naive == record["occluder"]] = record["background"]
            assert not np.array_equal(naive, output), "Hidden nonbackground content required"
            if split == "train":
                assert record["covered_panels"] == {record["source"]}
                assert record["masked_divider_cells"] == 0
            else:
                assert len(record["covered_panels"]) >= 2
            records[split].append(record)
    train_sources = {r["source"] for r in records["train"]}
    assert len(train_sources) >= 2
    assert any(r["source"] not in train_sources for r in records["test"])
    common = set(records["train"][0]["models"])
    for record in records["train"] + records["test"]:
        common &= set(record["models"])
    assert common, "All examples must visibly support a common local reflection rule"
    return True


def _signature(puzzle):
    return tuple(tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def _normalized(grid):
    roles = {}
    return bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def _expect_failure(function):
    try:
        function()
    except (AssertionError, ValueError):
        return
    raise AssertionError("Invalid case was accepted")


def self_test():
    """42 seeds, all legal counts, semantic mutations; no files or traces saved."""
    signatures, structures, boxes, groups = set(), set(), set(), set()
    sources, addresses, roles, occupancies = set(), set(), set(), set()
    pairs = address_invisible = nonfirst_selection = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        if seed in (0, 17, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                before = grid.copy()
                assert np.array_equal(solve(grid), pair["output"])
                assert np.array_equal(grid, before), "solve mutated its input"
                rec = _infer(grid)
                pairs += 1
                structures.add(_normalized(grid))
                groups.update(rec["models"])
                r, s, c, d = rec["box"]
                boxes.add((s-r, d-c, len(rec["covered_panels"])))
                sources.add(rec["source"])
                addresses.add(rec["address"])
                roles.add(tuple(rec[k] for k in ("background", "divider", "locator", "occluder")))
                occupancies.add(int(np.count_nonzero(rec["payload"] != rec["background"])))
                nonfirst_selection += rec["source"] != (0, 0) and not np.array_equal(
                    rec["full"][:3, :3], pair["output"])
                # Explicitly verify, rather than hide, the proposal's fatal shortcut.
                wrong = ((rec["address"][0]+1) % 3, rec["address"][1])
                crop, canvas = _place_and_crop(rec["payload"], wrong, rec["background"], rec["divider"])
                assert np.array_equal(crop, pair["output"])
                assert not np.array_equal(canvas, rec["relocated"])
                address_invisible += 1
    assert len(signatures) == 42 and len(structures) >= 180
    assert set(GROUPS) <= groups and len(boxes) >= 12
    assert len(sources) == 9 and len(addresses) == 9
    assert len(roles) >= 100 and len(occupancies) >= 4 and nonfirst_selection >= 100
    legal_count_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(9000+10*nt+nq, nt, nq))
            legal_count_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        _expect_failure(lambda: generate_puzzle(0, nt, nq))
    puzzle = generate_puzzle(73, 2, 1)
    changed = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    changed["test"][0]["output"][0, 0] = (int(changed["test"][0]["output"][0, 0])+1) % 10
    _expect_failure(lambda: validate_generated_puzzle(changed))
    rec = _infer(puzzle["test"][0]["input"])
    rr, cc = map(int, np.argwhere(rec["full"] == rec["locator"])[0])
    # A valid solid rectangle erases the fixed-point locator orbit completely.
    # Check the orbit failure itself, not just an irregular-mask rejection.
    orbit_erased = rec["full"].copy()
    left = cc if cc % 4 < 2 else cc-1
    orbit_erased[rr:rr+1, left:left+2] = rec["occluder"]
    assert _overwrite(orbit_erased)[0] == rec["occluder"]
    for group in rec["models"]:
        assert _completion(orbit_erased, rec["occluder"], rec["divider"], group) is None
    _expect_failure(lambda: solve(orbit_erased))
    # Contradict one visible reflection witness; no completion may silently repair it.
    corrupted = puzzle["test"][0]["input"].copy()
    group = rec["models"][0]
    witnesses = [(4*pr+o[0][0], 4*pc+o[0][1])
                 for pr in range(3) for pc in range(3) for o in _orbits(group)
                 if len(o) > 1 and all(corrupted[4*pr+r, 4*pc+c] != rec["occluder"]
                                       for r, c in o)]
    cell = witnesses[0]
    replacement = next(int(v) for v in rec["full"][~DIVIDERS]
                       if v != corrupted[cell] and v != rec["locator"])
    corrupted[cell] = replacement
    assert _completion(corrupted, rec["occluder"], rec["divider"], group) is None
    _expect_failure(lambda: solve(corrupted))
    transformed = {s: [{k: np.rot90(v).copy() for k, v in p.items()}
                       for p in puzzle[s]] for s in puzzle}
    validate_generated_puzzle(transformed)
    rng = np.random.default_rng(817)
    for split in transformed:
        for pair in transformed[split]:
            permutation = rng.permutation(10).astype(np.int8)
            for key in pair:
                pair[key] = permutation[pair[key]]
    validate_generated_puzzle(transformed)
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "normalized_inputs": len(structures), "reflection_groups": len(groups),
            "mask_geometry_classes": len(boxes), "sources": len(sources),
            "addresses": len(addresses), "role_assignments": len(roles),
            "payload_occupancies": sorted(occupancies), "legal_count_calls": legal_count_calls,
            "address_irrelevance_confirmed_pairs": address_invisible,
            "hidden_locator_supported": False, "independent_semantic_verification": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
