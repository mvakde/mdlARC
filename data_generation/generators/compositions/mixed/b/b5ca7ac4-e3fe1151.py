"""Complete class-specific 3x3 inventories, then stably pack their 5x5 frames.

Fixed, input-only conventions: frame 8 moves left; frame 2 moves right.
Generation constructs complete panels before corrupting them. Solving uses
pointwise histogram maxima; validation enumerates completions and uses unit steps.
No files, training labels, random state, or generation metadata are read by solve.
"""
from __future__ import annotations

import hashlib

import numpy as np


_FRAMES = (8, 2)
_SEEDS = (0, 638199425, 322184194, 940792820, 798090057, 856823644,
          161813236, 114168082, 933047251, 518997919, 49005869,
          391784190, 428560218, 856365433, 822435038, 471639607,
          395116038, 94271242, 13814009, 667920409, 205700715,
          166560539, 651173416, 231895880, 786484380, 97246636,
          332461574, 452351533, 125785841, 14394745, 305856051,
          161246366, 558444435, 796991718, 375982357, 589287394,
          284945814, 49769814, 796871740, 425792463, 113149975,
          918735320)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer)
             and np.all((a >= 0) & (a <= 9)), "grid colors must be integers 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _hist(a):
    return np.bincount(a.ravel().astype(np.int64), minlength=10)


def _rows_overlap(a, b):
    return abs(a["r"] - b["r"]) < 5


def _overlap(a, b):
    return _rows_overlap(a, b) and abs(a["c"] - b["c"]) < 5


def _decode(grid, holes):
    """Raster decoding is unique because every 5x5 footprint is non-background.

    At the first uncovered non-background pixel, the top-left corner of its
    panel has been reached. Fixed size makes touching panels harmless.
    """
    a = _grid(grid)
    counts = _hist(a)
    bg = int(counts.argmax())
    _require(counts[bg] * 2 > a.size and bg not in _FRAMES,
             "a neutral background must occupy strictly over half the grid")
    covered = a == bg
    objects = []
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if covered[r, c]:
                continue
            f = int(a[r, c])
            _require(f in _FRAMES and r + 5 <= h and c + 5 <= w,
                     "incomplete or unrecognized 5x5 frame")
            patch = a[r:r+5, c:c+5]
            _require(not np.any(covered[r:r+5, c:c+5]), "overlapping panel footprints")
            _require(np.all(patch[0] == f) and np.all(patch[-1] == f)
                     and np.all(patch[:, 0] == f) and np.all(patch[:, -1] == f),
                     "frame border is not complete")
            p = patch[1:4, 1:4]
            _require(np.count_nonzero(p == f) == holes, "wrong placeholder count")
            _require(not np.any(p == bg) and not np.any(p == (2 if f == 8 else 8)),
                     "background and the other frame color cannot be payload")
            objects.append(dict(r=r, c=c, f=f, patch=patch.copy()))
            covered[r:r+5, c:c+5] = True
    _require({o["f"] for o in objects} == set(_FRAMES), "both frame classes required")
    _require(all(sum(o["f"] == f for o in objects) >= 2 for f in _FRAMES),
             "each class needs at least two panels")
    return bg, objects


def _observed(o):
    counts = _hist(o["patch"][1:4, 1:4])
    counts[o["f"]] = 0
    return counts


def _targets(objects):
    result = {}
    for f in _FRAMES:
        obs = [_observed(o) for o in objects if o["f"] == f]
        target = np.maximum.reduce(obs)
        _require(target.sum() == 9, "class inventory is ambiguous or inconsistent")
        for counts in obs:
            deficit = target - counts
            _require(np.all(deficit >= 0) and deficit.sum() == 1
                     and np.count_nonzero(deficit) == 1, "invalid single-color deficit")
        result[f] = target
    _require(not np.array_equal(result[8], result[2]), "class inventories must differ")
    return result


def _repair(objects):
    targets = _targets(objects)
    repaired = []
    for o in objects:
        p = o["patch"].copy()
        deficit = targets[o["f"]] - _observed(o)
        color = int(np.flatnonzero(deficit)[0])
        interior = p[1:4, 1:4]
        interior[interior == o["f"]] = color
        repaired.append(dict(o, patch=p))
    return repaired


def _pack(objects, width):
    """Keep every overlapping predecessor ahead, even across overlap chains."""
    settled = [dict(o) for o in objects]
    for f in _FRAMES:
        indices = [i for i, o in enumerate(objects) if o["f"] == f]
        indices.sort(key=lambda i: (objects[i]["c"] if f == 8 else -objects[i]["c"],
                                    objects[i]["r"]))
        earlier = []
        for i in indices:
            blockers = [p for p in earlier if _rows_overlap(p, objects[i])]
            col = (max((p["c"] + 5 for p in blockers), default=0) if f == 8
                   else min((p["c"] for p in blockers), default=width) - 5)
            _require(0 <= col <= width - 5, "class chain exceeds canvas width")
            settled[i]["c"] = col
            earlier.append(settled[i])
    _require(not any(_overlap(a, b) for i, a in enumerate(settled)
                     for b in settled[i+1:]), "final directional groups overlap")
    return settled


def _render(shape, bg, objects):
    a = np.full(shape, bg, dtype=np.int8)
    for o in objects:
        a[o["r"]:o["r"]+5, o["c"]:o["c"]+5] = o["patch"]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover inventories from this input, repair, and translate complete panels."""
    a = _grid(input_grid)
    bg, objects = _decode(a, holes=1)
    repaired = _repair(objects)  # Required operational dependency: repair first.
    return _render(a.shape, bg, _pack(repaired, a.shape[1]))


def _enumerate_targets(objects, bg):
    """Validator: enumerate all one-cell completions of the first class member."""
    targets = {}
    for f in _FRAMES:
        observations = [_observed(o) for o in objects if o["f"] == f]
        candidates = []
        for color in range(10):
            if color in (*_FRAMES, bg):
                continue
            target = observations[0].copy()
            target[color] += 1
            if all(np.all(target >= q) and (target-q).sum() == 1
                   and np.count_nonzero(target-q) == 1 for q in observations):
                candidates.append(target)
        _require(len(candidates) == 1, "enumerated inventory completion is not unique")
        targets[f] = candidates[0]
    _require(not np.array_equal(targets[8], targets[2]), "class inventories must differ")
    return targets


def _unit_settle(objects, width):
    """Validator: slide by individual cells; a same-class blocker cannot be jumped."""
    moved = [dict(o) for o in objects]
    for f, delta in ((8, -1), (2, 1)):
        group = [o for o in moved if o["f"] == f]
        for _ in range(width * len(group) + 1):
            changed = False
            for o in group:
                candidate = dict(o, c=o["c"] + delta)
                if (0 <= candidate["c"] <= width - 5
                        and all(p is o or not _overlap(candidate, p) for p in group)):
                    o["c"] += delta
                    changed = True
            if not changed:
                break
        else:
            raise ValueError("unit-step settlement did not converge")
    _require(not any(_overlap(a, b) for i, a in enumerate(moved)
                     for b in moved[i+1:]), "unit-step final groups collide")
    return moved


def _edge_first_columns(objects, width):
    """Competing shortcut, used only to detect an overtaking witness."""
    columns = [None] * len(objects)
    for f in _FRAMES:
        indices = [i for i, o in enumerate(objects) if o["f"] == f]
        indices.sort(key=lambda i: (objects[i]["c"] if f == 8 else -objects[i]["c"],
                                    objects[i]["r"]))
        settled = []
        for i in indices:
            for c in (range(width-4) if f == 8 else range(width-5, -1, -1)):
                q = dict(objects[i], c=c)
                if all(not _overlap(q, p) for p in settled):
                    columns[i] = c
                    settled.append(q)
                    break
    return columns


def _spatial_witness(objects, targets):
    """Some visible peer at a hole's coordinates gives the wrong payload value."""
    witnessed = set()
    for o in objects:
        p = o["patch"][1:4, 1:4]
        pos = int(np.flatnonzero(p.ravel() == o["f"])[0])
        truth = int(np.flatnonzero(targets[o["f"]] - _observed(o))[0])
        peer_values = [int(q["patch"][1:4, 1:4].ravel()[pos]) for q in objects
                       if q is not o and q["f"] == o["f"]]
        visible = [v for v in peer_values if v != o["f"]]
        if visible and all(v != truth for v in visible):
            witnessed.add(o["f"])
    return witnessed


def _details(inp, out):
    bg, objects = _decode(inp, 1)
    out_bg, out_objects = _decode(out, 0)
    _require(inp.shape == out.shape and bg == out_bg, "canvas/background changed")
    _require(len(objects) == len(out_objects), "panel count changed")
    targets = _enumerate_targets(objects, bg)
    completed = []
    missing, positions, patterns = {}, {}, {}
    for f in _FRAMES:
        missing[f], positions[f], patterns[f] = [], [], []
    for o in objects:
        f = o["f"]
        deficit = targets[f] - _observed(o)
        color = int(np.flatnonzero(deficit)[0])
        p = o["patch"].copy()
        interior = p[1:4, 1:4]
        positions[f].append(int(np.flatnonzero(interior.ravel() == f)[0]))
        interior[interior == f] = color
        patterns[f].append(interior.tobytes())
        missing[f].append(color)
        completed.append(dict(o, patch=p))
    settled = _unit_settle(completed, inp.shape[1])
    _require(np.array_equal(out, _render(inp.shape, bg, settled)),
             "output violates inventory repair or stable settlement")
    for f in _FRAMES:
        _require(len(set(missing[f])) >= 2, "class lacks complementary omissions")
        _require(len(set(patterns[f])) >= 2, "class must vary spatial arrangements")
        _require(np.count_nonzero(inp == f) - np.count_nonzero(out == f)
                 == len(missing[f]), "placeholder accounting failed")
        _require(all(np.array_equal(_hist(o["patch"][1:4, 1:4]), targets[f])
                     for o in out_objects if o["f"] == f), "unequal repaired inventories")
    no_a = _render(inp.shape, bg, completed)
    no_b = _render(inp.shape, bg, _unit_settle(objects, inp.shape[1]))
    _require(not np.array_equal(no_a, out), "packing makes no difference")
    _require(np.count_nonzero(no_b != out) == len(objects), "repair ablation failed")
    moves = [a["c"] != b["c"] for a, b in zip(objects, settled)]
    return dict(bg=bg, objects=objects, settled=settled, targets=targets,
                missing=missing, positions=positions, spatial=_spatial_witness(objects, targets),
                moving_frames={o["f"] for o, m in zip(objects, moves) if m},
                stationary=moves.count(False),
                witness=_edge_first_columns(objects, inp.shape[1]) != [o["c"] for o in settled])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            records[split].append(_details(inp, out))
    train = records["train"]
    _require(set.union(*(r["moving_frames"] for r in train)) == set(_FRAMES),
             "training must demonstrate both directions")
    _require(any(r["witness"] for r in train), "training lacks stable-order witness")
    _require(set.union(*(r["spatial"] for r in train)) == set(_FRAMES),
             "training does not distinguish copying positions from inventory completion")
    # The first two demonstrations exchange complete inventories across frame roles.
    _require(all(np.array_equal(train[0]["targets"][f], train[1]["targets"][10-f])
                 for f in _FRAMES), "training must dissociate inventory from direction")
    _require(len(records["test"][0]["objects"]) > max(len(r["objects"]) for r in train),
             "first test must combine the rule at a larger panel count")


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _random_inventory(rng, bg, excluded=()):
    allowed = [c for c in range(10) if c not in (*_FRAMES, bg)]
    for _ in range(256):
        k = int(rng.integers(2, 7))
        colors = rng.choice(allowed, k, replace=False)
        counts = np.ones(k, dtype=np.int64) + rng.multinomial(9-k, np.full(k, 1/k))
        hist = np.zeros(10, dtype=np.int64)
        hist[colors] = counts
        if tuple(hist) not in excluded:
            return hist
    raise RuntimeError("bounded inventory sampling exhausted")


def _teach_layout(rng):
    w = int(rng.integers(22, 31))
    h = int(rng.integers(max(10, 200//w + 1), 31))
    row = int(rng.integers(h-4))
    gaps = rng.multinomial(w-20, np.full(5, 0.2))
    c = int(gaps[0])
    objects = []
    for i, f in enumerate((2, 8, 2, 8)):
        objects.append(dict(r=row, c=c, f=f))
        c += 5 + int(gaps[i+1])
    return (h, w), objects


def _chain_layout(rng):
    step = int(rng.integers(3, 5))
    h = int(rng.integers(2*step+15, 31))
    w = int(rng.integers(20, 31))
    base = int(rng.integers(h-2*step-14))
    f = _choose(rng, _FRAMES)
    gaps = rng.multinomial(w-15, np.full(4, 0.25))
    c = int(gaps[0])
    objects = []
    for i in range(3):
        objects.append(dict(r=base+i*step, c=c if f == 8 else w-5-c, f=f))
        c += 5 + int(gaps[i+1])
    # Disjoint bands directly demonstrate reusing the same destination edge.
    for row in (h-10, h-5):
        objects.append(dict(r=row, c=int(rng.integers(w-4)), f=10-f))
    if rng.integers(2):
        objects = [dict(o, r=h-5-o["r"]) for o in objects]
    return (h, w), objects


def _free_layout(rng, count):
    left = int(rng.integers(2, count-1))
    frames = [8]*left + [2]*(count-left)
    for _ in range(60):
        w = int(rng.integers(20, 31))
        h = int(rng.integers(max(15, 50*count//w+1), 31))
        order = list(frames)
        rng.shuffle(order)
        objects = []
        for f in order:
            for _ in range(70):
                row, col = int(rng.integers(h-4)), int(rng.integers(w-4))
                if objects and rng.random() < 0.6:
                    p = _choose(rng, objects)
                    row = p["r"] + _choose(rng, (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5))
                    if rng.random() < 0.35:
                        col = p["c"] + _choose(rng, (-5, 5))
                if rng.random() < 0.2:
                    col = _choose(rng, (0, w-5))
                q = dict(r=row, c=col, f=f)
                if not (0 <= row <= h-5 and 0 <= col <= w-5):
                    continue
                if any(_overlap(q, p) for p in objects):
                    continue
                try:
                    _pack(objects+[q], w)
                except ValueError:
                    continue
                objects.append(q)
                break
            else:
                break
        if len(objects) == count:
            settled = _pack(objects, w)
            if any(a["c"] != b["c"] for a, b in zip(objects, settled)):
                return (h, w), objects
    # Finite constructive fallback: at most ten panels in three disjoint bands.
    rng.shuffle(frames)
    objects = [dict(r=10*(i//4), c=3+5*(i % 4), f=f) for i, f in enumerate(frames)]
    return (30, 30), objects


def _appearance(rng, layout, targets, require_spatial=False):
    for _ in range(96):
        complete = [dict(o) for o in layout]
        damaged = [dict(o) for o in layout]
        for f in _FRAMES:
            indices = [i for i, o in enumerate(layout) if o["f"] == f]
            support = np.flatnonzero(targets[f])
            first = rng.choice(support, 2, replace=False)
            missing = [int(x) for x in first]
            if len(indices) <= len(support) and rng.random() < 0.4:
                missing = [int(x) for x in rng.choice(support, len(indices), replace=False)]
            else:
                missing += [int(rng.choice(support)) for _ in range(len(indices)-2)]
            rng.shuffle(missing)
            for i, color in zip(indices, missing):
                values = np.repeat(np.arange(10, dtype=np.int8), targets[f])
                rng.shuffle(values)
                patch = np.full((5, 5), f, dtype=np.int8)
                patch[1:4, 1:4] = values.reshape(3, 3)
                bad = patch.copy()
                pos = int(rng.choice(np.flatnonzero(values == color)))
                bad[1+pos//3, 1+pos % 3] = f
                complete[i]["patch"], damaged[i]["patch"] = patch, bad
        if any(len({o["patch"][1:4, 1:4].tobytes() for o in complete if o["f"] == f}) < 2
               for f in _FRAMES):
            continue
        if require_spatial and _spatial_witness(damaged, targets) != set(_FRAMES):
            continue
        return damaged, complete
    raise RuntimeError("bounded appearance sampling exhausted")


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             f"{name} must be an integer")
    value = int(value)
    _require(low <= value <= high, f"{name} must lie in {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Bounded generation; each requested example is retained only once in memory."""
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be a nonnegative integer")
    seed = int(seed)
    _require(seed >= 0, "seed must be a nonnegative integer")
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen, histories = {"train": [], "test": []}, set(), []
    neutral = [c for c in range(10) if c not in _FRAMES]
    bg0 = _choose(rng, neutral)
    colors = rng.choice([c for c in neutral if c != bg0], 2, replace=False)
    m = int(rng.integers(2, 8))
    first = {f: np.zeros(10, dtype=np.int64) for f in _FRAMES}
    first[8][colors], first[2][colors] = (m, 9-m), (9-m, m)
    train_max = 0
    for index in range(a+b):
        split = "train" if index < a else "test"
        for _ in range(32):
            if index == 0:
                bg, targets = bg0, first
                shape, layout = _teach_layout(rng)
            elif index == 1:
                bg = _choose(rng, [c for c in neutral if c not in colors])
                targets = {8: first[2], 2: first[8]}
                shape, layout = _chain_layout(rng)
            else:
                bg = _choose(rng, neutral)
                targets = {}
                for f in _FRAMES:
                    prior = [h for h in histories if h[bg] == 0]
                    excluded = [tuple(h) for h in targets.values()]
                    if split == "test":
                        excluded += [tuple(h) for h in histories]
                    reusable = [h for h in prior if tuple(h) not in excluded]
                    targets[f] = (_choose(rng, reusable).copy()
                                  if split == "train" and reusable and rng.random() < 0.25
                                  else _random_inventory(rng, bg, excluded))
                count = (train_max+1 if index == a else
                         int(rng.integers(4, 9 if split == "train" else 11)))
                shape, layout = _free_layout(rng, count)
            damaged, complete = _appearance(rng, layout, targets, require_spatial=index == 0)
            inp = _render(shape, bg, damaged)
            out = _render(shape, bg, _pack(complete, shape[1]))
            if _key(inp) not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-input sampling exhausted")
        seen.add(_key(inp))
        histories.extend(h.copy() for h in targets.values())
        if split == "train":
            train_max = max(train_max, len(layout))
        puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for s in ("train", "test"):
        for p in puzzle[s]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streaming seeds, both ablations, semantic variation, and negative checks.

    No corpus files or execution traces are written; puzzles are processed in a stream.
    Returned summary is optional diagnostic data, not generation state.
    """
    fingerprints, shapes, counts, profiles, class_counts, overlaps, bgs = (set() for _ in range(7))
    stationary = 0
    witness_frames = set()
    pair_count = 0
    for seed in _SEEDS:
        p = generate_puzzle(seed)
        fingerprint = _fingerprint(p)
        _require(fingerprint == _fingerprint(generate_puzzle(seed)), "nondeterministic seed")
        fingerprints.add(fingerprint)
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                original = inp.copy()
                _require(np.array_equal(solve(inp), out), "input-only solve disagrees")
                _require(np.array_equal(inp, original), "solve mutated its input")
                d = _details(inp, out)
                objects, settled = d["objects"], d["settled"]
                pair_count += 1
                shapes.add(inp.shape)
                counts.add(len(objects))
                class_counts.add(tuple(sum(o["f"] == f for o in objects) for f in _FRAMES))
                bgs.add(d["bg"])
                stationary += d["stationary"]
                for f in _FRAMES:
                    profiles.add(tuple(sorted(int(x) for x in d["targets"][f] if x)))
                old_columns = _edge_first_columns(objects, inp.shape[1])
                witness_frames.update(o["f"] for o, q, old in zip(objects, settled, old_columns)
                                      if q["c"] != old)
                for i, o in enumerate(objects):
                    for q in objects[i+1:]:
                        overlap = 5-abs(o["r"]-q["r"])
                        if overlap > 0 and o["f"] == q["f"]:
                            overlaps.add(overlap)
                # Interior repair commutes with geometry; this is the saved idea's
                # acknowledged limitation, not a claim of causal geometry changes.
                before = _render(inp.shape, d["bg"], _pack(_repair(objects), inp.shape[1]))
                after = _render(inp.shape, d["bg"], _repair(_pack(objects, inp.shape[1])))
                _require(np.array_equal(before, after), "unexpected repair/packing interaction")
    _require(len(fingerprints) == 42 and len(shapes) >= 30, "seed/shape diversity missing")
    _require(len(counts) >= 5 and len(class_counts) >= 8 and len(profiles) >= 8,
             "structural or multiset diversity missing")
    _require(overlaps == {1, 2, 3, 4, 5} and stationary > 0 and len(bgs) == 8,
             "overlap, stationary, or neutral-palette coverage missing")
    _require(witness_frames == set(_FRAMES), "stable chains must occur in both directions")
    # Exercise valid members beyond the generator's count/support sampling range:
    # 17 panels on 30x30 (425 occupied cells), seven-color nine-cell inventories.
    rng = np.random.default_rng(2323)
    palette = np.array((1, 3, 4, 5, 6, 7, 9))
    targets = {f: np.zeros(10, dtype=np.int64) for f in _FRAMES}
    for f, extra_color in ((8, 1), (2, 3)):
        targets[f][palette] = 1
        targets[f][extra_color] += 2
    layout = [dict(r=5*(i//6), c=5*(i % 6), f=_FRAMES[i % 2]) for i in range(17)]
    damaged, complete = _appearance(rng, layout, targets)
    big_input = _render((30, 30), 0, damaged)
    big_output = _render((30, 30), 0, _pack(complete, 30))
    _require(np.array_equal(solve(big_input), big_output), "larger family member unsupported")
    _details(big_input, big_output)
    # Exercise all 30 legal explicit count combinations, including minimum demos.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            q = generate_puzzle(173+a*10+b, a, b)
            _require((len(q["train"]), len(q["test"])) == (a, b), "counts changed")
    for a, b in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    a = broken["test"][0]["output"]
    a[0, 0] = (int(a[0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted a damaged output")
    broken = {s: [dict(pair) for pair in p[s]] for s in p}
    inp = p["train"][1]["input"]
    bg, objects = _decode(inp, 1)
    repaired = _repair(objects)
    old_columns = _edge_first_columns(objects, inp.shape[1])
    old = [dict(o, c=c) for o, c in zip(repaired, old_columns)]
    broken["train"][1]["output"] = _render(inp.shape, bg, old)
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted same-class overtaking")
    bg, objects = _decode(p["train"][0]["input"], 1)
    # Make every class member omit the same value: the input is now ambiguous.
    repaired = _repair(objects)
    for o in repaired:
        interior = o["patch"][1:4, 1:4]
        target_color = int(np.flatnonzero(_hist(interior))[0])
        pos = int(np.flatnonzero(interior.ravel() == target_color)[0])
        interior[pos//3, pos % 3] = o["f"]
    try:
        solve(_render(p["train"][0]["input"].shape, bg, repaired))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous inventory was accepted")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(fingerprints),
                canvas_shapes=len(shapes), panel_counts=sorted(counts),
                class_count_profiles=len(class_counts), multiset_profiles=len(profiles),
                same_class_overlap_widths=sorted(overlaps), stationary_panels=stationary,
                backgrounds=sorted(bgs), legal_count_combinations=30,
                stable_chain_frame_colors=sorted(witness_frames),
                extended_member_panels=17, extended_member_inventory_colors=7,
                ablations="both checked on every pair", independent_solver=False)
