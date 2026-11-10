"""Cross-change echoes: compare motif existence against repeated observations.

Only the standard library and NumPy are used. No files are read or written by
generation, solving, validation, or self_test. All conventions are input-visible.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and a.dtype != np.bool_,
             "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _centers(mask):
    result = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        result[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                              & mask[2:, 1:-1] & mask[1:-1, :-2]
                              & mask[1:-1, 2:])
    return result


def _expand(centers):
    result = centers.copy()
    result[1:, :] |= centers[:-1, :]
    result[:-1, :] |= centers[1:, :]
    result[:, 1:] |= centers[:, :-1]
    result[:, :-1] |= centers[:, 1:]
    return result


def _direct_centers(mask):
    """Coordinate scanner used for validation; no production mask expansion."""
    h, w = mask.shape
    return {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
            if all(mask[r + dr, c + dc] for dr, dc in _CROSS)}


def _footprints(points):
    return {(r + dr, c + dc) for r, c in points for dr, dc in _CROSS}


def _render(mask, carrier, ink):
    return np.where(mask, ink, carrier).astype(np.int8)


def _assemble(rows, frame):
    ph, pw = rows[0][0].shape
    grid = np.full((len(rows) * (ph + 1) + 1,
                    len(rows[0]) * (pw + 1) + 1), frame, dtype=np.int8)
    for b, panels in enumerate(rows):
        for c, panel in enumerate(panels):
            r0, c0 = 1 + b * (ph + 1), 1 + c * (pw + 1)
            grid[r0:r0 + ph, c0:c0 + pw] = panel
    return grid


def _parse(input_grid):
    grid = _grid(input_grid)
    frame = int(grid[0, 0])
    rs = np.flatnonzero(np.all(grid == frame, axis=1))
    cs = np.flatnonzero(np.all(grid == frame, axis=0))
    _require(len(rs) >= 2 and len(cs) >= 4, "need a framed panel lattice")
    _require(rs[0] == cs[0] == 0 and rs[-1] == grid.shape[0] - 1
             and cs[-1] == grid.shape[1] - 1, "outer frame must be complete")
    _require(bool(np.all(np.diff(rs) == rs[1]))
             and bool(np.all(np.diff(cs) == cs[1])), "lattice must be regular")
    ph, pw = int(rs[1] - 1), int(cs[1] - 1)
    _require(ph >= 5 and pw >= 5, "panels need a carrier margin and cross space")
    rows = [[grid[r + 1:s, c + 1:d].copy()
             for c, d in zip(cs, cs[1:])] for r, s in zip(rs, rs[1:])]
    counts = Counter(_key(p) for row in rows for p in row)
    maximum = max(counts.values())
    winners = [k for k, n in counts.items() if n == maximum]
    _require(len(winners) == 1, "reference bitmap must have a unique majority")
    master = next(p for row in rows for p in row if _key(p) == winners[0])
    colors = set(map(int, np.unique(grid)))
    carrier = int(master[0, 0])
    _require(len(colors) == 3 and carrier != frame, "need three visible color roles")
    ink = next(iter(colors - {frame, carrier}))
    _require(set(map(int, np.unique(master))) == {carrier, ink},
             "reference must contain both interior roles")
    selected, columns = [], []
    for row in rows:
        for panel in row:
            _require(bool(np.all((panel == carrier) | (panel == ink))),
                     "input panel interiors cannot contain the frame color")
            _require(bool(np.all(panel[[0, -1], :] == carrier))
                     and bool(np.all(panel[:, [0, -1]] == carrier)),
                     "all panel perimeters must be carrier")
        exceptional = [c for c, p in enumerate(row) if not np.array_equal(p, master)]
        _require(len(exceptional) == 1, "every band needs one visible exception")
        columns.append(exceptional[0])
        selected.append(row[exceptional[0]])
    return dict(grid=grid, frame=frame, carrier=carrier, ink=ink,
                ph=ph, pw=pw, rows=rows, master=master,
                selected=selected, columns=columns)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, reference, exceptions, and motif events from input."""
    info = _parse(input_grid)
    reference_centers = _centers(info["master"] == info["ink"])
    answers = []
    for variant in info["selected"]:
        events = reference_centers ^ _centers(variant == info["ink"])
        answer = variant.copy()
        answer[_expand(events)] = info["frame"]
        answers.append([answer])
    return _assemble(answers, info["frame"])


def _points(r, c):
    return [(r + dr, c + dc) for dr, dc in _CROSS]


def _plant(mask, r, c):
    for p in _points(r, c):
        mask[p] = True


def _event_counts(master, variant):
    before, after = _centers(master), _centers(variant)
    return int(np.count_nonzero(after & ~before)), int(np.count_nonzero(before & ~after))


def _orient(rng, masks, transpose=False):
    # Apply the same geometric transform to the reference and every observation.
    flip_r, flip_c = rng.random() < .5, rng.random() < .5
    swap = transpose and rng.random() < .5
    result = []
    for mask in masks:
        if flip_r:
            mask = mask[::-1, :]
        if flip_c:
            mask = mask[:, ::-1]
        if swap:
            mask = mask.T
        result.append(mask.copy())
    return result


def _diagnostic_witness(rng):
    """Three bands: balanced loss, balanced gain, and a harmless balanced edit."""
    ph, pw = map(int, rng.integers(7, 9, size=2))
    master = np.zeros((ph, pw), dtype=bool)
    a, b = (2, 2), (ph - 3, pw - 3)
    _plant(master, *a)
    _plant(master, *b)
    missing = _points(*b)[int(rng.integers(1, 5))]
    master[missing] = False
    inert, other = (ph - 2, 1), (1, pw - 2)
    master[inert] = True
    # Decorations are shared by all copies, without making components the unit.
    if rng.random() < .7:
        master[1, 1] = True
    if rng.random() < .5:
        master[2, 4] = True
    lost, gained, neutral = master.copy(), master.copy(), master.copy()
    lost[_points(*a)[int(rng.integers(5))]] = False
    lost[other] = True
    gained[missing] = True
    gained[inert] = False
    neutral[inert], neutral[other] = False, True
    masks = _orient(rng, [master, lost, gained, neutral], transpose=True)
    master, variants = masks[0], masks[1:]
    assert _event_counts(master, variants[0])[1] >= 1
    assert _event_counts(master, variants[1])[0] >= 1
    assert _event_counts(master, variants[2]) == (0, 0)
    # Band order carries no edit schedule.
    variants = [variants[int(i)] for i in rng.permutation(3)]
    return master, variants, 3


def _overlap_witness(rng):
    """One loss shares cells with a surviving cross; another kills both."""
    ph, pw = int(rng.integers(5, 14)), int(rng.integers(6, 9))
    master = np.zeros((ph, pw), dtype=bool)
    r, c = int(rng.integers(1, ph - 3)), int(rng.integers(1, pw - 4))
    master[r:r + 3, c:c + 4] = True
    one, two = master.copy(), master.copy()
    one[r, c + 2] = False
    two[r + 1, c + 1] = False
    variants = [one, two]
    max_bands = 29 // (ph + 1)
    if max_bands >= 3 and rng.random() < .5:
        extra = master.copy()
        extra[r, c] = False  # A block corner participates in no plus.
        variants.append(extra)
    masks = _orient(rng, [master] + variants)
    variants = [masks[1:][int(i)] for i in rng.permutation(len(variants))]
    return masks[0], variants, int(rng.integers(3, 29 // (pw + 1) + 1))


def _sample_master(rng, ph, pw):
    """Several compatible morphologies; every final cross is recomputed."""
    h, w = ph - 2, pw - 2
    core = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(6))
    if style == 0:
        core = rng.random((h, w)) < rng.uniform(.25, .73)
        if rng.random() < .3:
            core |= core[:, ::-1]
    elif style == 1:
        for _ in range(int(rng.integers(1, 6))):
            a, b = int(rng.integers(2, min(7, h) + 1)), int(rng.integers(2, w + 1))
            r, c = int(rng.integers(h - a + 1)), int(rng.integers(w - b + 1))
            core[r:r + a, c:c + b] = True
    elif style == 2:
        # Irregular partial arrays of local pluses and missing-arm crosses.
        rgap, cgap = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        for r in range(1, h - 1, rgap):
            for c in range(1, w - 1, cgap):
                if rng.random() < .85:
                    _plant(core, r, c)
                    if rng.random() < .35:
                        core[_points(r, c)[int(rng.integers(5))]] = False
    elif style == 3:
        # Hollow frames with independently positioned crossing strokes.
        a = int(rng.integers(3, h + 1))
        b = int(rng.integers(3, w + 1))
        r, c = int(rng.integers(h - a + 1)), int(rng.integers(w - b + 1))
        core[r:r + a, c] = core[r:r + a, c + b - 1] = True
        core[r, c:c + b] = core[r + a - 1, c:c + b] = True
        core[int(rng.integers(h)), :] = True
        core[:, int(rng.integers(w))] = True
    elif style == 4:
        # A connected branching walk, with occasional wider junctions.
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(h * w // 2, 2 * h * w + 1))):
            core[r, c] = True
            dr, dc = _CROSS[int(rng.integers(1, 5))]
            r, c = min(h - 1, max(0, r + dr)), min(w - 1, max(0, c + dc))
    else:
        rr, cc = np.indices((h, w))
        period = int(rng.integers(2, 5))
        core = (rr % period == int(rng.integers(period))) | (cc % 3 == int(rng.integers(3)))
        core ^= rng.random((h, w)) < .12
    # Stamps may be detached or embedded, and may overlap existing structure.
    for _ in range(int(rng.integers(1, 5))):
        r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        _plant(core, r, c)
        if rng.random() < .45:
            core[_points(r, c)[int(rng.integers(5))]] = False
    mask = np.zeros((ph, pw), dtype=bool)
    mask[1:-1, 1:-1] = core
    return mask


def _almost(mask):
    result = []
    for r in range(2, mask.shape[0] - 2):
        for c in range(2, mask.shape[1] - 2):
            missing = [p for p in _points(r, c) if not mask[p]]
            if len(missing) == 1:
                result.append(((r, c), missing[0]))
    return result


def _propose_variant(rng, master, mixed=False):
    ph, pw = master.shape
    eligible = [(r, c) for r in range(1, ph - 1) for c in range(1, pw - 1)]
    cap = max(2, min(12, len(eligible) // 5))
    centers = _centers(master)
    if mixed:
        coverage = centers.astype(np.int16)
        coverage[1:] += centers[:-1]
        coverage[:-1] += centers[1:]
        coverage[:, 1:] += centers[:, :-1]
        coverage[:, :-1] += centers[:, 1:]
        shared = [tuple(map(int, p)) for p in np.argwhere(coverage >= 2)]
        almost = _almost(master)
        if not shared or not almost:
            return None
    for _ in range(32):
        variant = master.copy()
        protected = set()
        if mixed:
            gain_center, add = almost[int(rng.integers(len(almost)))]
            removable = [p for p in shared if p not in _points(*gain_center)]
            if not removable:
                continue
            remove = removable[int(rng.integers(len(removable)))]
            variant[add], variant[remove] = True, False
            protected.update(_points(*gain_center))
            protected.add(remove)
        elif rng.random() < .5 and centers.any():
            p = tuple(map(int, np.argwhere(centers)[int(rng.integers(centers.sum()))]))
            variant[_points(*p)[int(rng.integers(5))]] = False
        extra = int(rng.integers(0 if mixed else 1, cap - 1 if mixed else cap + 1))
        candidates = [p for p in eligible if p not in protected]
        if extra:
            for i in rng.choice(len(candidates), min(extra, len(candidates)), replace=False):
                p = candidates[int(i)]
                variant[p] = ~variant[p]
        if np.array_equal(master, variant):
            continue
        gains, losses = _event_counts(master, variant)
        if not mixed or (gains >= 1 and losses >= 2):
            return variant
    return None


def _mixed_fallback(rng, ph, pw):
    """Finite constructive witness: shared-contact loss plus a remote gain."""
    master = _sample_master(rng, ph, pw)
    master[1:4, :] = False
    master[1:4, 1:5] = True
    master[4:8, :] = False
    _plant(master, 6, 2)
    master[5, 2] = False
    # Restoring margins after the protected local construction is explicit.
    master[[0, -1], :] = False
    master[:, [0, -1]] = False
    return _orient(rng, [master])[0]


def _random_case(rng, mixed=False):
    if mixed:
        ph, pw = int(rng.integers(9, 29)), int(rng.integers(6, 9))
    else:
        ph, pw = int(rng.integers(5, 29)), int(rng.integers(5, 9))
    bands = int(rng.integers(1, 29 // (ph + 1) + 1))
    copies = int(rng.integers(3, 29 // (pw + 1) + 1))
    for _ in range(24):
        master = _sample_master(rng, ph, pw)
        variants = [_propose_variant(rng, master, mixed) for _ in range(bands)]
        if all(v is not None for v in variants):
            return master, variants, copies
    # The fallback is geometric, not a different transformation rule.
    if mixed:
        master = _mixed_fallback(rng, ph, pw)
        variants = [_propose_variant(rng, master, True) for _ in range(bands)]
        if all(v is not None for v in variants):
            return master, variants, copies
        raise AssertionError("constructive mixed witness failed")
    # Any one-cell interior flip is a valid visible exception.
    master = _sample_master(rng, ph, pw)
    variants = []
    for _ in range(bands):
        variant = master.copy()
        r, c = int(rng.integers(1, ph - 1)), int(rng.integers(1, pw - 1))
        variant[r, c] = ~variant[r, c]
        variants.append(variant)
    return master, variants, copies


def _make_pair(rng, case, palette, variable_columns=False):
    master, variants, copies = case
    frame, carrier, ink = palette
    reference = _render(master, carrier, ink)
    columns = list(map(int, rng.integers(copies, size=len(variants))))
    if variable_columns and len(set(columns)) == 1:
        columns[-1] = (columns[-1] + int(rng.integers(1, copies))) % copies
    rows, answers = [], []
    before = _centers(master)
    for col, variant in zip(columns, variants):
        observation = _render(variant, carrier, ink)
        row = [reference.copy() for _ in range(copies)]
        row[col] = observation
        answer = observation.copy()
        answer[_expand(before ^ _centers(variant))] = frame
        rows.append(row)
        answers.append([answer])
    return {"input": _assemble(rows, frame), "output": _assemble(answers, frame)}


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        _require(not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, np.integer)) and low <= value <= high,
                 f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate demonstrations with explicit contrast cases and combined tests."""
    nt, ne = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    common_palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    stable_palette = rng.random() < .2
    for index in range(nt + ne):
        split = "train" if index < nt else "test"
        if index == 0:
            case = _diagnostic_witness(rng)
        elif index == 1:
            case = _overlap_witness(rng)
        else:
            case = _random_case(rng, mixed=split == "test")
        palette = (common_palette if stable_palette else
                   tuple(map(int, rng.choice(10, 3, replace=False))))
        pair = _make_pair(rng, case, palette, variable_columns=index == 0)
        # Collision resolution uses at most 720 color-role assignments, with
        # at most 13 previous inputs. Structure and witness properties persist.
        if _key(pair["input"]) in seen:
            found = False
            for frame in range(10):
                for carrier in range(10):
                    for ink in range(10):
                        if len({frame, carrier, ink}) != 3:
                            continue
                        candidate = _make_pair(rng, case, (frame, carrier, ink),
                                               variable_columns=index == 0)
                        if _key(candidate["input"]) not in seen:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                if found:
                    break
            _require(found, "finite palette collision resolution exhausted")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    """Check labels by direct center/coordinate sets; measure visible features."""
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must have exactly input/output")
    info = _parse(pair["input"])
    out = _grid(pair["output"])
    _require(out.shape == (info["grid"].shape[0], info["pw"] + 2),
             "output must preserve bands and compress to one panel")
    reference = info["master"] == info["ink"]
    before = _direct_centers(reference)
    reference_union = _footprints(before)
    expected, raw, all_variant, pixel_only, xor_union = [], [], [], [], []
    features = dict(gains=0, losses=0, neutral=False, ghost=False, stable=False,
                    overlap=False, balanced=False, mixed_bands=0, edits=[],
                    event_counts=[], selected_columns=info["columns"])
    for variant in info["selected"]:
        mask = variant == info["ink"]
        after = _direct_centers(mask)
        gained, lost = after - before, before - after
        points = _footprints(before ^ after)
        answer = variant.copy()
        for p in points:
            answer[p] = info["frame"]
        expected.append([answer])
        raw.append([variant.copy()])
        for target, cells in ((all_variant, _footprints(after)),
                              (pixel_only, set(map(tuple, np.argwhere(reference != mask)))),
                              (xor_union, reference_union ^ _footprints(after))):
            alternative = variant.copy()
            for p in cells:
                alternative[p] = info["frame"]
            target.append([alternative])
        features["gains"] += len(gained)
        features["losses"] += len(lost)
        features["neutral"] |= not gained and not lost
        features["ghost"] |= any(not mask[p] for p in points)
        features["stable"] |= bool(_footprints(before & after) - points)
        features["overlap"] |= points != reference_union ^ _footprints(after)
        added = int(np.count_nonzero(mask & ~reference))
        removed = int(np.count_nonzero(reference & ~mask))
        features["balanced"] |= added == removed > 0
        features["mixed_bands"] += int(bool(gained) and len(lost) >= 2)
        features["edits"].append(added + removed)
        features["event_counts"].append(len(gained) + len(lost))
    expected = _assemble(expected, info["frame"])
    _require(np.array_equal(out, expected), "output violates differential cross stamping")
    features.update(ignores_a=not np.array_equal(out, _assemble(pixel_only, info["frame"])),
                    ignores_b=not np.array_equal(out, _assemble(all_variant, info["frame"])),
                    raw_differs=not np.array_equal(out, _assemble(raw, info["frame"])),
                    xor_differs=not np.array_equal(out, _assemble(xor_union, info["frame"])),
                    geometry=(info["ph"], info["pw"], len(info["rows"]), len(info["rows"][0])),
                    palette=(info["frame"], info["carrier"], info["ink"]),
                    reference_key=_key(reference), input_key=_key(info["grid"]))
    return features


def validate_generated_puzzle(puzzle) -> None:
    """Validate returned grids, differential labels, and demonstration evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, measurements = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            features = _inspect_pair(pair)
            _require(features["input_key"] not in seen, "duplicate input")
            seen.add(features["input_key"])
            measurements[split].append(features)
    training = measurements["train"]
    _require(sum(f["gains"] for f in training) > 0, "training must show a gained plus")
    _require(sum(f["losses"] for f in training) > 0, "training must show a lost plus")
    for feature in ("neutral", "ghost", "stable", "overlap", "balanced",
                    "ignores_a", "ignores_b", "raw_differs", "xor_differs"):
        _require(any(f[feature] for f in training), f"missing training evidence: {feature}")
    _require(any(len(set(f["selected_columns"])) > 1 for f in training),
             "training must distinguish a fixed-column shortcut")
    for f in measurements["test"]:
        _require(f["mixed_bands"] == f["geometry"][2],
                 "generated tests combine gains and multiple losses in every band")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds, all legal count pairs, local cases, and bad-label checks."""
    signatures, geometries, palettes, references, shapes = set(), set(), set(), set(), set()
    edit_counts, motif_events, colors = set(), set(), [set(), set(), set()]
    features_count = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _inspect_pair(pair)
                geometries.add(f["geometry"])
                palettes.add(f["palette"])
                references.add(f["reference_key"])
                shapes.add(np.asarray(pair["input"]).shape)
                edit_counts.update(f["edits"])
                motif_events.update(f["event_counts"])
                for i, color in enumerate(f["palette"]):
                    colors[i].add(color)
                for name in ("neutral", "ghost", "stable", "overlap", "balanced"):
                    features_count[name] += int(f[name])
    assert len(signatures) == 42
    assert min(map(len, colors)) == 10
    assert len(geometries) >= 35 and len(shapes) >= 25
    assert len(references) >= 75 and len(palettes) >= 60
    assert len(edit_counts) >= 6 and len(motif_events) >= 6
    assert min(features_count.values()) >= 42
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    legal_counts = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7100 + 10 * nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
            legal_counts += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        expected = _direct_centers(mask)
        assert set(map(tuple, np.argwhere(_centers(mask)))) == expected
        assert set(map(tuple, np.argwhere(_expand(_centers(mask))))) == _footprints(expected)
    # Two adjacent centers share two footprint cells. Removing the upper
    # arm of the right center stamps five cells; pixelwise mask XOR stamps three.
    master = np.pad(np.ones((3, 4), dtype=bool), 1)
    variant = master.copy()
    variant[1, 3] = False
    a, b = _centers(master), _centers(variant)
    assert int(a.sum()) == 2 and int(b.sum()) == 1
    assert int(_expand(a ^ b).sum()) == 5
    assert int(np.count_nonzero(_expand(a) ^ _expand(b))) == 3
    assert _expand(a ^ b)[1, 3] and not variant[1, 3]
    both_lost = master.copy()
    both_lost[2, 2] = False
    assert not _centers(both_lost).any()
    assert int(_expand(a ^ _centers(both_lost)).sum()) == 8
    # solve also accepts valid family members outside the mixed-test sampler:
    # an entire neutral example selects the exception without any new stamps.
    neutral_master = np.zeros((5, 5), dtype=bool)
    _plant(neutral_master, 2, 2)
    neutral_variant = neutral_master.copy()
    neutral_variant[1, 1] = True
    ref = _render(neutral_master, 8, 0)
    var = _render(neutral_variant, 8, 0)
    neutral_input = _assemble([[ref, var, ref]], 4)
    assert np.array_equal(solve(neutral_input), _assemble([[var]], 4))
    # Mutations must fail on content, selection, and geometry, not just shape.
    base = generate_puzzle(0, 2, 1)
    for defect in ("label", "missing_exception", "frame", "raw_selection"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()}
                   for pair in base[s]] for s in base}
        pair = bad["train"][0]
        if defect == "label":
            pair["output"][1, 1] = (int(pair["output"][1, 1]) + 1) % 10
        elif defect == "frame":
            pair["input"][0, 1] = (int(pair["input"][0, 1]) + 1) % 10
        else:
            info = _parse(pair["input"])
            if defect == "raw_selection":
                pair["output"] = _assemble([[p] for p in info["selected"]], info["frame"])
            else:
                c0 = 1 + info["columns"][0] * (info["pw"] + 1)
                pair["input"][1:1 + info["ph"], c0:c0 + info["pw"]] = info["master"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted " + defect)
    return dict(seeds=42, default_pairs=210, legal_count_combinations=legal_counts,
                distinct_puzzles=len(signatures), geometries=len(geometries),
                input_shapes=len(shapes), palettes=len(palettes), references=len(references),
                edit_counts=sorted(edit_counts), motif_event_counts=sorted(motif_events),
                features=dict(features_count), semantic_scope="generator-owned checks only")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
