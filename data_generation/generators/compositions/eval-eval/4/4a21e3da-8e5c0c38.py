"""Mirror winches: symmetry completes an instruction and protects a channel.

Only NumPy and the standard library are used. No files, parent generators,
examples, seeds or generation state enter solve(). Experiments are in scratch/.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g


def _support(mask):
    """Doubled axis coordinate, supported cells and unique-winner margin."""
    scores = np.zeros(2 * mask.shape[1] - 1, dtype=int)
    for row in mask.astype(np.int16):
        scores += np.convolve(row, row)
    winners = np.flatnonzero(scores == scores.max())
    assert len(winners) == 1, "tied reflection axes are outside this family"
    axis = int(winners[0])
    reflection = axis - np.arange(mask.shape[1])
    valid = (reflection >= 0) & (reflection < mask.shape[1])
    keep = np.zeros_like(mask, dtype=bool)
    keep[:, valid] = mask[:, valid] & mask[:, reflection[valid]]
    competitors = np.delete(scores, axis)
    return axis, keep, int(scores[axis] - competitors.max())


def _canonical(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > g.size // 2, "background must be a strict majority"
    rim = np.zeros(g.shape, dtype=bool)
    rim[0] = rim[-1] = True
    rim[:, 0] = rim[:, -1] = True
    marks = np.argwhere(rim & (g != bg))
    assert len(marks) == 1, "exactly one observed edge marker"
    r, c = map(int, marks[0])
    assert (r in (0, g.shape[0]-1)) != (c in (0, g.shape[1]-1))
    turn = 0 if r == 0 else 1 if c == g.shape[1]-1 else 2 if r == g.shape[0]-1 else 3
    marker = int(g[r, c])
    assert np.count_nonzero(g == marker) == 1
    colors = set(map(int, np.unique(g)))
    assert len(colors) == 3
    payload = (colors - {bg, marker}).pop()
    work = np.rot90(g, turn).copy()
    m = int(np.flatnonzero(work[0] == marker)[0])
    mask = work == payload
    assert not np.any(mask[0]) and not np.any(mask[-1])
    assert not np.any(mask[:, 0]) and not np.any(mask[:, -1])
    return work, turn, bg, payload, marker, m, mask


def _analyze(input_grid):
    work, turn, bg, payload, marker, m, raw = _canonical(input_grid)
    axis, core, margin = _support(raw)
    other = axis - m
    left, right = sorted((m, other))
    assert 0 < left < right < work.shape[1]-1
    assert right-left >= 2
    assert np.any(core[:, left]) and np.any(core[:, right])
    assert np.any(core[:, :left]) and np.any(core[:, right+1:])
    assert np.any(core[:, left+1:right])
    return {"work": work, "turn": turn, "bg": bg, "payload": payload,
            "marker": marker, "m": m, "axis": axis, "core": core,
            "raw": raw, "left": left, "right": right, "margin": margin}


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _render(core, left, right, bg, payload, marker, *, nearest=False, fragments=False):
    """Canonical geometry, also used for explicit wrong-rule comparisons."""
    h, w = core.shape
    out = np.full((h, w), bg, dtype=np.int8)
    rr, cc = np.nonzero(core)
    for r, c in zip(rr.tolist(), cc.tolist()):
        if left <= c <= right:
            out[r, c] = payload
    for col in (left, right):
        hits = np.flatnonzero(core[:, col])
        assert len(hits)
        stop = int(hits[0] if nearest else hits[-1])
        for row in range(stop+1):
            if out[row, col] == bg:
                out[row, col] = marker
    for wing, is_left in ((core & (np.arange(w)[None, :] < left), True),
                           (core & (np.arange(w)[None, :] > right), False)):
        parts = _components(wing) if fragments else [set(map(tuple, np.argwhere(wing).tolist()))]
        for part in parts:
            assert part
            rows, cols = zip(*part)
            dr = -min(rows)
            dc = -min(cols) if is_left else w-1-max(cols)
            for r, c in part:
                if not fragments:
                    assert out[r+dr, c+dc] == bg, "corner collision"
                out[r+dr, c+dc] = payload
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire result from the supplied grid, without latent state."""
    a = _analyze(input_grid)
    out = _render(a["core"], a["left"], a["right"], a["bg"], a["payload"], a["marker"])
    return np.rot90(out, -a["turn"]).copy()


def _single_rail_mutant(a):
    """Clean, then apply A's original one-marker cut; omits the new channel."""
    core, m = a["core"], a["m"]
    h, w = core.shape
    out = np.full_like(a["work"], a["bg"])
    out[:, m][core[:, m]] = a["payload"]
    for r in range(int(np.flatnonzero(core[:, m])[-1])+1):
        if out[r, m] == a["bg"]:
            out[r, m] = a["marker"]
    for mask, is_left in ((core[:, :m], True), (core[:, m+1:], False)):
        rows, cols = np.nonzero(mask)
        dc = -int(cols.min()) if is_left else w-1-int(cols.max())
        for r, c in zip(rows, cols):
            out[int(r-rows.min()), int(c+dc)] = a["payload"]
    return out


def _make_core(rng, h, w, left, right, witness):
    """Random mirrored orbits; constraints add informative geometric witnesses."""
    mask = np.zeros((h, w), dtype=bool)
    mid = (w-1)//2
    style = int(rng.integers(5))
    track = int(rng.integers(mid+1))

    def orbit(r, c, value=True):
        mask[r, c] = mask[r, w-1-c] = value

    for r in range(h):
        if style == 0:  # irregular, variably filled row profiles
            track = int(np.clip(track + rng.integers(-2, 3), 0, mid))
            for c in range(track, mid+1):
                if c == track or rng.random() < .72:
                    orbit(r, c)
        elif style == 1:  # hollow outline, possibly stepped and thickened
            x = 0 if rng.random() < .65 else int(rng.integers(min(3, mid+1)))
            cols = range(x, mid+1) if r in (0, h-1) else range(x, min(mid+1, x+int(rng.integers(1, 3))))
            for c in cols:
                orbit(r, c)
        elif style == 2:  # a thin spine and independently sized crossbars
            orbit(r, mid)
            if rng.random() < .5:
                for c in range(int(rng.integers(mid+1)), mid+1):
                    orbit(r, c)
        elif style == 3:  # supported islands, gaps and holes
            for c in range(mid+1):
                if rng.random() < .28:
                    orbit(r, c)
        else:  # a wandering mirrored branch, with sparse side branches
            track = int(np.clip(track + rng.integers(-1, 2), 0, mid))
            orbit(r, track)
            if rng.random() < .55:
                orbit(r, int(rng.integers(mid+1)))
        if not mask[r].any():
            orbit(r, int(rng.integers(mid+1)))

    orbit(int(rng.integers(h)), 0)  # attain the full declared width
    orbit(int(rng.integers(h)), mid)  # material strictly inside the channel
    if witness == 0:
        # Two separated hits, then a supported deeper row off the rails.
        orbit(1, left)
        orbit(h-3, left)
        orbit(2, left, False)
        mask[h-2:, left] = mask[h-2:, right] = False
        orbit(h-1, 0)
        orbit(h-1, mid)
        # An in-channel, off-axis empty orbit becomes an asymmetric addition.
        orbit(0, left+1, False)
        orbit(0, 0)
    else:
        orbit(int(rng.integers(h)), left)
    if witness == 1:
        split = int(rng.integers(2, h-2))
        mask[split, :left] = mask[split, right+1:] = False
        orbit(split-1, 0)
        orbit(split+1, 0)
        # A genuinely isolated supported pair must survive the cleanup.
        orbit(0, 0)
        orbit(0, 1, False)
        orbit(1, 0, False)
        orbit(h-1, mid)
        # Restore a rail hit if isolation happened to remove the sole one.
        if not np.any(mask[:, left]):
            orbit(h-2, left)
    # Every transverse slice retains some support, without a pair-in-row quota.
    for r in range(h):
        if not mask[r].any():
            orbit(r, mid)
    return mask


def _add_noise(rng, core, axis, forced, amount):
    raw = core.copy()
    expected = core.copy()

    def accept(p):
        r, c = p
        if raw[r, c] or 2*c == axis or not 0 <= axis-c < core.shape[1]:
            return False
        if raw[r, axis-c]:
            return False
        raw[r, c] = True
        try:
            got, keep, _ = _support(raw)
            valid = got == axis and np.array_equal(keep, expected)
        except AssertionError:
            valid = False
        if not valid:
            raw[r, c] = False
        return valid

    for p in forced:
        assert accept(p), "informative addition changed the intended axis"
    candidates = [(r, c) for r in range(1, core.shape[0]-1)
                  for c in range(1, core.shape[1]-1)
                  if not raw[r, c] and 0 <= axis-c < core.shape[1]]
    rng.shuffle(candidates)
    # Relative preference for local additions; detached additions remain possible.
    rs, cs = np.nonzero(core)
    local = [p for p in candidates if rs.min()-1 <= p[0] <= rs.max()+1
             and cs.min()-1 <= p[1] <= cs.max()+1]
    distant = [p for p in candidates if p not in set(local)]
    candidates = local + distant if rng.random() < .7 else candidates
    added = len(forced)
    for p in candidates[:128]:
        if added >= amount:
            break
        added += int(accept(p))
    assert added >= len(forced)
    return raw


def _draw_pair(rng, parity, gap, turn, roles, observed_left, witness):
    widths = list(range(gap+3, min(24, gap+15)+1, 2))
    w = int(rng.choice(widths))
    assert (w-1) % 2 == parity
    h = int(rng.integers(6, 18))
    left, right = (w-1-gap)//2, (w-1+gap)//2
    local = _make_core(rng, h, w, left, right, witness)
    height = int(rng.integers(max(10, h+2), 31))
    width = int(rng.integers(max(10, w+2), 31))
    # Allow dense geometry while keeping a visibly identifiable background.
    while height*width <= 2*(int(local.sum())+15):
        if height < 30 and (height <= width or width == 30):
            height += 1
        else:
            width += 1
        assert height <= 30 and width <= 30
    top = int(rng.integers(1, height-h))
    offset = int(rng.integers(1, width-w))
    core = np.zeros((height, width), dtype=bool)
    core[top:top+h, offset:offset+w] = local
    axis = 2*offset+w-1
    left, right = left+offset, right+offset
    marker_col = left if observed_left else right
    forced = []
    if witness == 0:
        forced = [(top+h-1, marker_col), (top, left+1)]
    amount = int(rng.integers(1, min(14, max(3, int(core.sum())//4))+1))
    if witness < 2:
        amount = max(amount, 2)
    elif rng.random() < .2:
        amount = 0
    raw = _add_noise(rng, core, axis, forced, max(amount, len(forced)))
    bg, payload, marker = roles
    inp = np.full(core.shape, bg, dtype=np.int8)
    inp[raw] = payload
    inp[0, marker_col] = marker
    # Labels are constructed from the pristine core, not by calling solve().
    out = _render(core, left, right, bg, payload, marker)
    return {"input": np.rot90(inp, -turn).copy(), "output": np.rot90(out, -turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    widening = bool(rng.integers(2))
    palette_varies = rng.random() < .75
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    directions = list(map(int, rng.choice(4, min(num_train, int(rng.integers(1, 5))), replace=False)))
    train_directions = directions + list(map(int, rng.choice(directions, num_train-len(directions))))
    rng.shuffle(train_directions)
    initial_parity = int(rng.integers(2))
    initial_left = bool(rng.integers(2))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else 2
            parity = (initial_parity+index) % 2 if witness < 2 else int(rng.integers(2))
            wide = widening == (split == "test")
            choices = ((6, 8) if parity == 0 else (5, 7, 9)) if wide else ((2, 4) if parity == 0 else (3,))
            gap = int(rng.choice(choices))
            if witness == 0 and not wide and parity == 0:
                gap = 4  # an off-axis interior-strip deletion witness
            turn = train_directions[index] if split == "train" else int(rng.choice(directions))
            observed_left = initial_left if witness == 0 else not initial_left if witness == 1 else bool(rng.integers(2))
            if palette_varies:
                roles = tuple(map(int, rng.choice(10, 3, replace=False)))
            for _ in range(24):
                pair = _draw_pair(rng, parity, gap, turn, roles, observed_left, witness)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(result)
    return result


def pair_features(input_grid):
    a = _analyze(input_grid)
    core, raw, left, right = a["core"], a["raw"], a["left"], a["right"]
    noise = raw & ~core
    rail_rows = np.flatnonzero(core[:, left])
    wing = core.copy()
    wing[:, left:] = False
    rs, cs = np.nonzero(raw)
    band_noise = noise[:, left+1:right]
    components = _components(wing)
    return {"shape": list(np.asarray(input_grid).shape), "direction": a["turn"],
            "roles": [a["bg"], a["payload"], a["marker"]], "axis2": a["axis"],
            "parity": a["axis"] % 2, "gap": right-left,
            "observed_left": a["m"] == left, "core_cells": int(core.sum()),
            "noise": int(noise.sum()), "strip_noise": int(band_noise.sum()),
            "gap_in_rail": bool(np.any(np.diff(rail_rows) > 1)),
            "farther_rail_noise": bool(np.any(noise[int(rail_rows[-1])+1:, [left, right]])),
            "short_endpoint": int(rail_rows[-1]) < int(np.nonzero(core)[0].max()),
            "wing_components": len(components), "wing_cells": int(wing.sum()),
            "supported_island": any(len(c) == 1 for c in _components(core)),
            "off_center": a["axis"] != core.shape[1]-1,
            "noisy_box_wrong": int(cs.min()+cs.max()) != a["axis"],
            "support_margin": a["margin"]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    mutants = {"single": 0, "no_a": 0, "no_b": 0, "nearest": 0, "fragments": 0}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            a = _analyze(inp)
            expected = solve(inp)
            assert np.array_equal(out, expected), "wrong mirror-winch output"
            core, left, right = a["core"], a["left"], a["right"]
            actual = np.rot90(out, a["turn"])
            assert np.count_nonzero(actual == a["payload"]) == core.sum(), "supported material not conserved"
            assert np.array_equal(actual[:, left+1:right] == a["payload"], core[:, left+1:right])
            for c in (left, right):
                endpoint = int(np.flatnonzero(core[:, c])[-1])
                assert np.all(actual[:endpoint+1, c] == np.where(core[:endpoint+1, c], a["payload"], a["marker"]))
                assert not np.any(actual[endpoint+1:, c] == a["marker"])
            assert not np.array_equal(inp, out)
            features[split].append(pair_features(inp))
            if split == "train":
                args = (left, right, a["bg"], a["payload"], a["marker"])
                clean_only = np.full_like(a["work"], a["bg"])
                clean_only[core] = a["payload"]
                clean_only[0, a["m"]] = a["marker"]
                mutants["single"] += not np.array_equal(_single_rail_mutant(a), actual)
                mutants["no_a"] += not np.array_equal(clean_only, actual)
                mutants["no_b"] += not np.array_equal(_render(a["raw"], *args), actual)
                mutants["nearest"] += not np.array_equal(_render(core, *args, nearest=True), actual)
                mutants["fragments"] += not np.array_equal(_render(core, *args, fragments=True), actual)
    first, second = features["train"][:2]
    assert first["gap_in_rail"] and first["short_endpoint"] and first["farther_rail_noise"]
    assert first["strip_noise"] > 0
    assert second["wing_components"] > 1 and second["supported_island"]
    assert first["parity"] != second["parity"]
    assert first["observed_left"] != second["observed_left"]
    assert first["noise"] > 0 and second["noise"] > 0
    assert all(mutants.values()), "training did not distinguish a named shortcut"
    train, test = features["train"], features["test"]
    assert {f["direction"] for f in test} <= {f["direction"] for f in train}
    tr_gaps, te_gaps = [f["gap"] for f in train], [f["gap"] for f in test]
    assert max(tr_gaps) < min(te_gaps) or max(te_gaps) < min(tr_gaps)


def self_test():
    """Bounded: 42 default puzzles, small boundary calls, no persistent payloads."""
    shapes, roles, directions, gaps, parities, hashes, structures = (set() for _ in range(7))
    extremes = {"widen": 0, "narrow": 0, "box": 0, "island": 0, "clean": 0}
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        tr_gaps, te_gaps = [], []
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                f = pair_features(inp)
                shapes.add(tuple(f["shape"]))
                roles.add(tuple(f["roles"]))
                directions.add(f["direction"])
                gaps.add(f["gap"])
                parities.add(f["parity"])
                (tr_gaps if split == "train" else te_gaps).append(f["gap"])
                extremes["box"] += int(f["noisy_box_wrong"])
                extremes["island"] += int(f["supported_island"])
                extremes["clean"] += int(f["noise"] == 0)
                a = _analyze(inp)
                rows, cols = np.nonzero(a["raw"])
                crop = a["raw"][rows.min():rows.max()+1, cols.min():cols.max()+1]
                structures.add(hashlib.sha256(bytes(crop.shape)+crop.tobytes()).hexdigest())
        hashes.add(digest.digest())
        extremes["widen"] += max(tr_gaps) < min(te_gaps)
        extremes["narrow"] += max(te_gaps) < min(tr_gaps)
        # Input-only solver must respect arbitrary literal palettes and rotations.
        for pair in p["test"]:
            for k in range(4):
                perm = np.roll(np.arange(10), seed % 10)
                rotated = np.rot90(pair["input"], k)
                expected = np.rot90(pair["output"], k)
                assert np.array_equal(solve(perm[rotated]), perm[expected])
        if seed < 2:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                       for a, b in zip(p[s], q[s]) for k in ("input", "output"))
    assert len(hashes) == 42 and len(shapes) >= 60 and len(structures) >= 170
    assert len(roles) >= 80 and directions == set(range(4)) and parities == {0, 1}
    assert gaps == set(range(2, 10)) and all(extremes.values())
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    damaged = generate_puzzle(91, 2, 1)
    a = _analyze(damaged["test"][0]["input"])
    out = damaged["test"][0]["output"]
    where = np.argwhere(out == a["payload"])[0]
    out[tuple(where)] = a["bg"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted loss of supported material")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, symmetry/geometry checks")
