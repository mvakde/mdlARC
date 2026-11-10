"""Framed payloads slide stably toward the edge selected by their frame.

Only the axis and the two frame instructions are semantic puzzle parameters.
Geometry and appearance are visible, pair-local properties.  Generation uses
ordered boundary constraints; validation uses a separate unit-step simulation.
"""
import hashlib

import numpy as np


# Exact 22x22 source input/output content hashes, with grids encoded as int8.
_SOURCE_DIGESTS = frozenset((
    "09efb48be6b4b45baae97c3ee8e4af2cdd9dfd4bf0acc14ec5e61fd5344b235d",
    "593bca9a395778ed5260242e9ab225a9a519818bdf9bc14768fbf45b52219f79",
    "ac9642878e45bc77a21b85cfb78bf847650bc660abd16b46edb3c2dd81db2f83",
    "2e9b19a5e28309b947b17ad4793ac81c805090837904b036125a7de3843598fb",
))


def _normalize_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _choose_counts(rng, num_train, num_test):
    num_train = _normalize_count(num_train, "num_train")
    num_test = _normalize_count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        legal = [(a, b) for b in range(1, 5) for a in range(b + 1, 11)]
        return legal[int(rng.integers(len(legal)))]
    if num_train is None:
        return int(rng.integers(num_test + 1, 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _source_copy(pair):
    return (pair["input"].shape == pair["output"].shape == (22, 22)
            and hashlib.sha256(pair["input"].astype(np.int8).tobytes()
                               + pair["output"].astype(np.int8).tobytes()).hexdigest()
            in _SOURCE_DIGESTS)


def _band_overlap(a, b):
    return max(a["r"], b["r"]) < min(a["r"] + a["h"], b["r"] + b["h"])


def _overlap(a, b):
    return (_band_overlap(a, b)
            and max(a["c"], b["c"]) < min(a["c"] + a["w"], b["c"] + b["w"]))


def _compact_objects(objects, width):
    """Stable ordering: every overlapping predecessor remains ahead."""
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["d"] == direction]
        indices.sort(key=lambda i: (objects[i]["c"] if direction == 0
                                   else -objects[i]["c"] - objects[i]["w"],
                                   objects[i]["r"]))
        settled = []
        for i in indices:
            o = result[i]
            blockers = [p for p in settled if _band_overlap(o, p)]
            if direction == 0:
                o["c"] = max((p["c"] + p["w"] for p in blockers), default=0)
            else:
                o["c"] = min((p["c"] for p in blockers), default=width) - o["w"]
            if not 0 <= o["c"] <= width - o["w"]:
                raise ValueError("directional chain exceeds the canvas")
            settled.append(o)
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]):
        raise ValueError("opposite packed groups overlap")
    return result


def _render(shape, background, objects):
    grid = np.full(shape, background, dtype=np.int8)
    for o in objects:
        grid[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = o["patch"]
    return grid


def _decode(grid):
    """Read visible complete frames, including adjacent and mixed-size frames.

    Local interiors exclude their own frame and local background. Thus the
    first return to frame color along the interior's first row/column uniquely
    determines each rectangle. Background occupies strictly over half the grid.
    No component identity, fixed object size, or generation metadata is used.
    """
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(counts.argmax())
    assert counts[background] * 2 > grid.size, "background is not identifiable"
    covered = grid == background
    objects = []
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            if covered[r, c]:
                continue
            frame = int(grid[r, c])
            assert r + 2 < height and c + 2 < width, "incomplete frame"
            assert grid[r + 1, c + 1] not in (frame, background), "erased interior"
            bottom, right = r + 2, c + 2
            while bottom < height and grid[bottom, c + 1] != frame:
                bottom += 1
            while right < width and grid[r + 1, right] != frame:
                right += 1
            assert bottom < height and right < width, "open frame"
            patch = grid[r:bottom + 1, c:right + 1]
            assert np.all(patch[0] == frame) and np.all(patch[-1] == frame)
            assert np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)
            interior = patch[1:-1, 1:-1]
            assert np.all(interior != frame) and np.all(interior != background)
            assert not np.any(covered[r:bottom + 1, c:right + 1]), "overlapping frames"
            covered[r:bottom + 1, c:right + 1] = True
            objects.append(dict(r=r, c=c, h=bottom-r+1, w=right-c+1,
                                frame=frame, patch=patch.copy()))
    assert objects and np.all(covered)
    return background, objects


def _simulate(grid, low_frame, high_frame):
    """Independent physical check: move one cell at a time, ignoring other class.

    Unit steps cannot jump a same-class blocker. Repeated sweeps end exactly
    when each object touches its destination edge or a same-class obstacle.
    """
    background, decoded = _decode(grid)
    assert {o["frame"] for o in decoded} <= {low_frame, high_frame}
    objects = [dict(o) for o in decoded]
    for frame, delta in ((low_frame, -1), (high_frame, 1)):
        group = [o for o in objects if o["frame"] == frame]
        occupied = np.zeros(grid.shape, dtype=bool)
        for o in group:
            occupied[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = True
        for _ in range(grid.shape[1] * max(1, len(group)) + 1):
            moved = False
            for o in group:
                r, c, h, w = o["r"], o["c"], o["h"], o["w"]
                occupied[r:r+h, c:c+w] = False
                next_c = c + delta
                if (0 <= next_c <= grid.shape[1] - w
                        and not np.any(occupied[r:r+h, next_c:next_c+w])):
                    o["c"] = next_c
                    moved = True
                occupied[r:r+h, o["c"]:o["c"]+w] = True
            if not moved:
                break
        else:
            raise AssertionError("unit-step packing did not converge")
    output = np.full_like(grid, background)
    occupied = np.zeros(grid.shape, dtype=bool)
    for o in objects:
        r, c, h, w = (o[k] for k in ("r", "c", "h", "w"))
        assert not np.any(occupied[r:r+h, c:c+w]), "final groups overlap"
        occupied[r:r+h, c:c+w] = True
        output[r:r+h, c:c+w] = o["patch"]
    return output, decoded, objects


def _predict(grid, convention):
    axis, low, high = convention
    canonical = grid if axis == 1 else grid.T
    predicted = _simulate(canonical, low, high)[0]
    return predicted if axis == 1 else predicted.T.copy()


def _infer_convention(train_pairs):
    frames = set()
    for pair in train_pairs:
        frames.update(o["frame"] for o in _decode(pair["input"])[1])
    assert len(frames) == 2, "training must expose both frame identities"
    a, b = sorted(frames)
    candidates = []
    for axis in (0, 1):
        for low, high in ((a, b), (b, a)):
            convention = (axis, low, high)
            try:
                matches = all(np.array_equal(_predict(p["input"], convention), p["output"])
                              for p in train_pairs)
            except AssertionError:
                matches = False
            if matches:
                candidates.append(convention)
    assert len(candidates) == 1, "training does not identify one axis and frame map"
    return candidates[0]


def solve(train_pairs, test_input):
    """Prediction from demonstrations and one input only (no test labels)."""
    return _predict(test_input, _infer_convention(train_pairs))


def _edge_first_columns(objects, width, low):
    """Old competing convention, only for coverage and negative regression tests."""
    columns = [None] * len(objects)
    for frame in sorted({o["frame"] for o in objects}):
        indices = [i for i, o in enumerate(objects) if o["frame"] == frame]
        indices.sort(key=lambda i: objects[i]["c"] if frame == low
                     else -objects[i]["c"] - objects[i]["w"])
        settled = []
        for i in indices:
            o = dict(objects[i])
            positions = range(width-o["w"]+1)
            if frame != low:
                positions = reversed(positions)
            for c in positions:
                o["c"] = c
                if not any(_overlap(o, p) for p in settled):
                    columns[i] = c
                    settled.append(o)
                    break
    return columns


def _has_witness(objects, settled, width, low):
    return _edge_first_columns(objects, width, low) != [o["c"] for o in settled]


def _sample_parameters(rng):
    frames = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    ordinary = [c for c in range(10) if c not in frames]
    novel_payload = _pick(rng, ordinary)
    backgrounds = [c for c in ordinary if c != novel_payload]
    rng.shuffle(backgrounds)
    pool = [c for c in range(10) if c != novel_payload]
    rng.shuffle(pool)
    # Include enough non-frame colors for the shared-payload demonstration.
    palette = list(dict.fromkeys(pool[:int(rng.integers(4, 10))]
                                + [c for c in ordinary if c != novel_payload][:2]))
    parameters = dict(axis=int(rng.integers(2)), frames=frames,
                geometry=_pick(rng, ("common", "pair", "mixed", "mixed")),
                size=_pick(rng, (3, 4, 5, 5, 5, 6)),
                patterned=_pick(rng, (0.0, 0.0, 0.35, 0.75)),
                witness=bool(rng.random() < 0.75),
                backgrounds=backgrounds[:int(rng.integers(1, 6))],
                payloads=palette, novel_payload=novel_payload)
    # Deliberate simple subfamily, on either axis. Other styles still combine
    # geometry, appearance and chain depth independently.
    if rng.random() < 0.22:
        parameters.update(geometry="common", size=5, patterned=0.0, witness=False)
    return parameters


def _dimensions(rng, p):
    if p["geometry"] == "common":
        return p["size"], p["size"]
    h = int(rng.integers(3, 9))
    return h, h if rng.random() < 0.45 else int(rng.integers(3, 9))


def _layout(rng, p, purpose, requested, larger=False):
    """Compose anchors/overlap chains with random admissible rectangles."""
    base_h, base_w = _dimensions(rng, p)
    if purpose == "chain":
        # Common-square puzzles keep that square size, including in witnesses.
        if p["geometry"] != "common":
            base_h, base_w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        length = int(rng.integers(3, min(5, 26 // base_w) + 1))
        step = int(rng.integers((base_h + 1) // 2, base_h))
        min_h, min_w = base_h + (length-1)*step, length*base_w + 1
    elif purpose == "teach":
        min_h, min_w = base_h + 2, 2*base_w + 2
    else:
        length = 0
        min_h, min_w = base_h + 1, base_w + 2
    if larger:
        min_h, min_w = max(min_h, 22), max(min_w, 22)
    height = int(rng.integers(max(6, min_h), 31))
    width = int(rng.integers(max(8, min_w), 31))
    # Include square canvases independently of geometry/palette/complexity.
    if rng.random() < 0.2:
        height = width = max(height, width)
    objects = []
    if purpose == "teach":
        row = int(rng.integers(height-base_h+1))
        right_start = int(rng.integers(base_w+1, width-base_w+1))
        left_start = int(rng.integers(0, right_start-base_w+1))
        objects = [dict(r=row, c=right_start, h=base_h, w=base_w, d=0),
                   dict(r=row, c=left_start, h=base_h, w=base_w, d=1)]
    elif purpose == "chain":
        direction = int(rng.integers(2))
        row = int(rng.integers(height-min_h+1))
        free = width-length*base_w
        gaps = rng.multinomial(free, np.full(length+1, 1/(length+1)))
        c = int(gaps[0])
        reverse_rows = bool(rng.integers(2))
        for i in range(length):
            r = row + (length-1-i if reverse_rows else i)*step
            col = c if direction == 0 else width-c-base_w
            objects.append(dict(r=r, c=col, h=base_h, w=base_w, d=direction))
            c += base_w + int(gaps[i+1])
        requested = max(requested, length)
    # A slight canvas enlargement makes all anchors compatible with majority bg.
    area = sum(o["h"]*o["w"] for o in objects)
    if 2*area >= height*width:
        height = max(height, min(30, 2*area//width + 1))
    probability_left = _pick(rng, (0.0, 0.1, 0.25, 0.5, 0.5, 0.75, 0.9, 1.0))
    for _ in range(max(0, requested-len(objects))):
        for attempt in range(75):
            h, w = (_dimensions(rng, p) if p["geometry"] == "mixed"
                    else (base_h, base_w))
            if h > height or w > width or 2*(area+h*w) >= height*width:
                continue
            direction = int(rng.random() >= probability_left)
            r = int(rng.integers(height-h+1))
            c = int(rng.integers(width-w+1))
            if rng.random() < 0.25:
                c = _pick(rng, (0, width-w))
            if objects and rng.random() < 0.55:
                other = _pick(rng, objects)
                r = _pick(rng, (other["r"], other["r"]+other["h"]-1,
                                other["r"]-h+1, other["r"]+other["h"],
                                other["r"]+int(rng.integers(1, other["h"]))))
                if rng.random() < 0.4:
                    c = _pick(rng, (other["c"]+other["w"], other["c"]-w))
            candidate = dict(r=r, c=c, h=h, w=w, d=direction)
            if not (0 <= r <= height-h and 0 <= c <= width-w):
                continue
            if any(_overlap(candidate, o) for o in objects):
                continue
            try:
                _compact_objects(objects + [candidate], width)
            except ValueError:
                continue
            objects.append(candidate)
            area += h*w
            break
        else:
            break
    if not objects:
        # Finite constructive fallback for tiny canvases and large footprints.
        h, w = min(base_h, height-2), min(base_w, width-2)
        if 2*h*w >= height*width:
            height = width = 30
        objects = [dict(r=0, c=1, h=h, w=w, d=int(rng.integers(2)))]
    return (height, width), objects


def _appearance(rng, objects, p, background, palette, shared, novel):
    pair_palette = list(rng.choice(palette, int(rng.integers(1, len(palette)+1)), replace=False))
    common = [c for c in palette if c not in (*p["frames"], background)]
    shared_color = _pick(rng, common)
    shared_interior = None
    for i, o in enumerate(objects):
        frame = p["frames"][o["d"]]
        allowed = [int(c) for c in pair_palette if c not in (frame, background)]
        if not allowed:
            allowed = [int(c) for c in palette if c not in (frame, background)]
        h, w = o["h"], o["w"]
        interior = np.full((h-2, w-2), _pick(rng, allowed), dtype=np.int8)
        if len(allowed) > 1 and rng.random() < p["patterned"]:
            colors = rng.choice(allowed, min(len(allowed), int(rng.integers(2, 5))), replace=False)
            rr, cc = np.indices(interior.shape)
            mode = int(rng.integers(4))
            indices = (rr+cc if mode == 0 else rr if mode == 1 else cc)
            if mode == 3:
                indices = rng.integers(len(colors), size=interior.shape)
            interior = np.asarray(colors[indices % len(colors)], dtype=np.int8)
        if shared and i < 2:
            if shared_interior is None:
                interior[:] = shared_color
                others = [c for c in common if c != shared_color]
                if others and interior.size > 1 and rng.random() < p["patterned"]:
                    interior.flat[::2] = _pick(rng, others)
                shared_interior = interior.copy()
            interior = shared_interior.copy()
        if novel is not None and i == len(objects)-1:
            if p["patterned"] == 0.0 or rng.random() < 0.5:
                interior[:] = novel
            else:
                interior.flat[0] = novel
        o["frame"] = frame
        o["patch"] = np.full((h, w), frame, dtype=np.int8)
        o["patch"][1:-1, 1:-1] = interior


def _make_pair(rng, p, purpose, regime, index, training):
    assert regime in ("source_train", "target_test")
    if regime == "source_train":
        background = (p["backgrounds"][index % len(p["backgrounds"])] if index < 2
                      else _pick(rng, p["backgrounds"]))
        palette, novel = p["payloads"], None
        requested = (int(rng.integers(2, 9)) if purpose == "teach"
                     else int(rng.integers(1, 17)))
        larger = False
    else:
        seen_backgrounds = sorted({_decode(pair["input"])[0] for pair in training})
        seen_payloads = sorted({int(c) for pair in training for o in _decode(pair["input"])[1]
                                for c in np.unique(o["patch"][1:-1, 1:-1])})
        new_backgrounds = [c for c in range(10)
                           if c not in (*p["frames"], *seen_backgrounds, p["novel_payload"])]
        background = _pick(rng, new_backgrounds if rng.random() < 0.45 else seen_backgrounds)
        novel = p["novel_payload"] if rng.random() < 0.5 else None
        palette = sorted(set(seen_payloads) | ({novel} if novel is not None else set()))
        counts = [len(_decode(pair["input"])[1]) for pair in training]
        scale = int(rng.integers(3))
        requested = (int(rng.integers(1, max(counts)+1)) if scale == 0
                     else _pick(rng, counts) if scale == 1
                     else min(16, max(counts)+int(rng.integers(1, 5))))
        larger = bool(rng.random() < 0.35)
    # Avoid erasing the only familiar neutral payload when a test changes bg.
    # Choose the background before filtering so that color is not lost.
    if not any(c not in (*p["frames"], background) for c in palette):
        background = _pick(rng, [c for c in range(10)
                                if c not in (*p["frames"], *palette, p["novel_payload"])])
    palette = [c for c in palette if c != background]
    for attempt in range(40):
        shape, objects = _layout(rng, p, purpose, requested, larger)
        settled = _compact_objects(objects, shape[1])
        # Without a demonstrated witness, keep every split in the shared
        # subfamily on which stable compaction and edge-first repacking agree.
        for o in objects:
            o["frame"] = p["frames"][o["d"]]
        if not p["witness"] and _has_witness(objects, settled, shape[1], p["frames"][0]):
            continue
        if purpose == "chain" and not _has_witness(objects, settled, shape[1], p["frames"][0]):
            continue
        _appearance(rng, objects, p, background, palette, purpose == "teach", novel)
        settled = _compact_objects(objects, shape[1])
        if purpose == "free" and rng.random() < 0.05:
            objects = [dict(o) for o in settled]
        grids = [_render(shape, background, group) for group in (objects, settled)]
        if p["axis"] == 0:
            grids = [grid.T.copy() for grid in grids]
        pair = dict(zip(("input", "output"), grids))
        if not _source_copy(pair):
            return pair
    # Guaranteed fallback retains a requested chain witness, without additions
    # that could mask the distinction. Two teaching anchors are also feasible.
    fallback = "chain" if purpose == "chain" else "teach"
    shape, objects = _layout(rng, p, fallback, 0)
    _appearance(rng, objects, p, background, palette, fallback == "teach", novel)
    grids = [_render(shape, background, group)
             for group in (objects, _compact_objects(objects, shape[1]))]
    if p["axis"] == 0:
        grids = [g.T.copy() for g in grids]
    return dict(zip(("input", "output"), grids))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    p = _sample_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            purpose = ("teach" if split == "train" and index == 0 else "chain"
                       if split == "train" and index == 1 and p["witness"] else "free")
            pair = _make_pair(rng, p, purpose, regime, index, puzzle["train"])
            # Fifteen distinct canvas widths cannot all be used by the at most
            # thirteen earlier inputs. Preserve a puzzle's common square size.
            if _grid_key(pair["input"]) in seen:
                for variant in range(15):
                    size = p["size"] if p["geometry"] == "common" else 5
                    shape = (2*size+2, 2*size+2+variant)
                    objects = [dict(r=1, c=1, h=size, w=size, d=1),
                               dict(r=size+1, c=size+2, h=size, w=size, d=0)]
                    _appearance(rng, objects, p, p["backgrounds"][0], p["payloads"], True, None)
                    grids = [_render(shape, p["backgrounds"][0], group)
                             for group in (objects, _compact_objects(objects, shape[1]))]
                    if p["axis"] == 0:
                        grids = [g.T.copy() for g in grids]
                    pair = dict(zip(("input", "output"), grids))
                    if _grid_key(pair["input"]) not in seen:
                        break
            assert _grid_key(pair["input"]) not in seen
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_puzzle(puzzle):
    """Task measurements reconstructed exclusively from returned grids."""
    convention = _infer_convention(puzzle["train"])
    axis, low, high = convention
    records = []
    train_bgs, train_payloads = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            g = pair["input"] if axis == 1 else pair["input"].T
            background, _ = _decode(g)
            _, objects, settled = _simulate(g, low, high)
            depths = [0]*len(objects)
            edges, overlaps, exact = [], [], 0
            for i, a in enumerate(objects):
                for j in range(i+1, len(objects)):
                    b = objects[j]
                    overlap = min(a["r"]+a["h"], b["r"]+b["h"]) - max(a["r"], b["r"])
                    if overlap > 0:
                        overlaps.append(overlap)
                        exact += a["r"] == b["r"] and a["h"] == b["h"]
                        if a["frame"] == b["frame"]:
                            first, second = (i, j) if a["c"] < b["c"] else (j, i)
                            edges.append((first, second) if a["frame"] == low else (second, first))
            for _ in objects:
                for a, b in edges:
                    depths[b] = max(depths[b], depths[a]+1)
            components = [{i} for i in range(len(objects))]
            for a, b in edges:
                ca = next(c for c in components if a in c)
                cb = next(c for c in components if b in c)
                if ca is not cb:
                    ca.update(cb)
                    components.remove(cb)
            payloads = {int(c) for o in objects for c in np.unique(o["patch"][1:-1, 1:-1])}
            classes = [sum(o["frame"] == f for o in objects) for f in (low, high)]
            layers = [max((depths[i]+1 for i, o in enumerate(objects) if o["frame"] == f), default=0)
                      for f in (low, high)]
            crossings = sum(a["frame"] == low and b["frame"] == high
                            and _band_overlap(a, b) and a["c"] > b["c"]
                            for a in objects for b in objects)
            travels = [abs(a["c"]-b["c"]) for a, b in zip(objects, settled)]
            contacts = sum((a["c"]+a["w"] == b["c"] or b["c"]+b["w"] == a["c"])
                           and _band_overlap(a, b)
                           or (a["r"]+a["h"] == b["r"] or b["r"]+b["h"] == a["r"])
                           and max(a["c"], b["c"]) < min(a["c"]+a["w"], b["c"]+b["w"])
                           for i, a in enumerate(objects) for b in objects[i+1:])
            footprints = sorted({(o["h"], o["w"]) for o in objects})
            same_class_contacts = sum(a["frame"] == b["frame"] and _band_overlap(a, b)
                                      and (a["c"]+a["w"] == b["c"] or b["c"]+b["w"] == a["c"])
                                      for i, a in enumerate(objects) for b in objects[i+1:])
            edge_occupants = [sum(o["frame"] == frame and o["c"] == edge(o) for o in settled)
                              for frame, edge in ((low, lambda o: 0),
                                                  (high, lambda o: g.shape[1]-o["w"]))]
            patterned = sum(len(np.unique(o["patch"][1:-1, 1:-1])) > 1 for o in objects)
            shared = any(a["frame"] != b["frame"] and a["patch"].shape == b["patch"].shape
                         and np.array_equal(a["patch"][1:-1, 1:-1], b["patch"][1:-1, 1:-1])
                         for a in objects for b in objects)
            records.append(dict(split=split, index=index, axis=axis, frames=[low, high],
                                shape=list(pair["input"].shape), background=background,
                                count=len(objects), classes=classes, layers=layers,
                                overlap_edges=len(edges), components=sorted(map(len, components)),
                                overlap_widths=sorted(set(overlaps)), exact_alignments=int(exact),
                                witness=_has_witness(objects, settled, g.shape[1], low),
                                stationary=travels.count(0), travels=travels,
                                edge_starts=sum(o["c"] in (0, g.shape[1]-o["w"]) for o in objects),
                                opposite_edge_starts=sum(o["c"] == (g.shape[1]-o["w"] if o["frame"] == low else 0)
                                                         for o in objects),
                                crossings=int(crossings), contacts=int(contacts),
                                same_class_contacts=int(same_class_contacts),
                                edge_occupants=edge_occupants,
                                footprints=footprints, patterned=int(patterned), shared_payload=shared,
                                payloads=sorted(payloads), frame_as_payload=bool(payloads & {low, high}),
                                new_background=split == "test" and background not in train_bgs,
                                new_payload=split == "test" and bool(payloads-train_payloads)))
            if split == "train":
                train_bgs.add(background)
                train_payloads.update(payloads)
    for record in records:
        record["other_background_as_payload"] = any(
            b["background"] in record["payloads"] for b in records if b is not record)
    return records


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for g in pair.values():
                assert isinstance(g, np.ndarray) and g.ndim == 2
                assert all(1 <= n <= 30 for n in g.shape)
                assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
            assert pair["input"].shape == pair["output"].shape
            ik, ok = _grid_key(pair["input"]), _grid_key(pair["output"])
            assert ik not in inputs and (ik, ok) not in pairs
            inputs.add(ik)
            pairs.add((ik, ok))
            assert not _source_copy(pair)
            assert _decode(pair["input"])[0] == _decode(pair["output"])[0]
    convention = _infer_convention(puzzle["train"])
    axis, low, high = convention
    witnesses = {"train": False, "test": False}
    moved_frames, shared_payload = set(), False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            # Expected output is inferred before comparing with its label.
            g = pair["input"] if axis == 1 else pair["input"].T
            expected, objects, settled = _simulate(g, low, high)
            actual = pair["output"] if axis == 1 else pair["output"].T
            assert np.array_equal(expected, actual), "payload, position, or compaction mismatch"
            witnesses[split] |= _has_witness(objects, settled, g.shape[1], low)
            if split == "train":
                moved_frames.update(a["frame"] for a, b in zip(objects, settled) if a["c"] != b["c"])
                shared_payload |= any(
                    a["frame"] != b["frame"] and a["patch"].shape == b["patch"].shape
                    and np.array_equal(a["patch"][1:-1, 1:-1], b["patch"][1:-1, 1:-1])
                    for a in objects for b in objects)
    assert moved_frames == {low, high}, "both instructions must visibly move in training"
    assert shared_payload, "training must separate frame instruction from payload and size"
    assert not witnesses["test"] or witnesses["train"], "unannounced overtaking convention"


def _fingerprint(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    corpus = [generate_puzzle(10000+i) for i in range(42)]
    assert len({_fingerprint(p) for p in corpus}) == 42
    assert len({len(p["train"]) for p in corpus}) >= 4
    assert len({len(p["test"]) for p in corpus}) >= 3
    for a, b in ((None, None), (7, 3), (2, 1), (10, 1), (10, 4),
                 (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(271828, a, b)
        assert _fingerprint(p) == _fingerprint(generate_puzzle(271828, a, b))
        if a is not None:
            assert len(p["train"]) == a
        if b is not None:
            assert len(p["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2),
                 (3, 4), (4, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts were accepted")
    measurements = [measure_puzzle(p) for p in corpus]
    rows = [r for records in measurements for r in records]
    assert {r["axis"] for r in rows} == {0, 1}
    assert len({tuple(r["frames"]) for r in rows}) >= 25
    assert len({tuple(r["shape"]) for r in rows}) >= 80
    assert min(min(r["shape"]) for r in rows) <= 8
    assert any(r["shape"] == [30, 30] for r in rows)
    assert len({r["count"] for r in rows}) >= 10
    assert any(r["count"] == 1 for r in rows) and any(min(r["classes"]) == 0 for r in rows)
    assert any(r["count"] % 2 for r in rows)
    assert any(min(r["classes"]) and max(r["classes"]) >= 3*min(r["classes"]) for r in rows)
    assert {1, 2, 3, 4} <= {n for r in rows for n in r["layers"]}
    assert len({tuple(r["layers"]) for r in rows}) >= 15
    assert any(0 < r["stationary"] < r["count"] for r in rows)
    assert any(r["stationary"] == r["count"] for r in rows)
    assert any(r["contacts"] for r in rows) and any(not r["contacts"] for r in rows)
    assert any(r["same_class_contacts"] for r in rows)
    assert any(1 in r["edge_occupants"] for r in rows)
    assert any(max(r["edge_occupants"]) >= 3 for r in rows)
    assert any(r["opposite_edge_starts"] for r in rows)
    assert {bool(r["crossings"]) for r in rows} == {False, True}
    assert min(t for r in rows for t in r["travels"] if t) == 1
    assert max(t for r in rows for t in r["travels"]) >= 20
    assert {1, 2, 3, 4} <= {w for r in rows for w in r["overlap_widths"]}
    assert any(r["exact_alignments"] for r in rows)
    assert any(len(r["components"]) > 1 and r["overlap_edges"] for r in rows)
    assert any(len(r["footprints"]) > 1 for r in rows)
    assert any(h != w for r in rows for h, w in r["footprints"])
    assert any(r["patterned"] for r in rows) and any(not r["patterned"] for r in rows)
    assert any(r["frame_as_payload"] for r in rows)
    assert any(r["other_background_as_payload"] for r in rows)
    assert len({len(r["payloads"]) for r in rows}) >= 6
    assert {(r["new_background"], r["new_payload"]) for r in rows if r["split"] == "test"} == {
        (False, False), (False, True), (True, False), (True, True)}
    for feature in ("shape", "count", "layers", "background", "payloads", "footprints"):
        changing = sum(len({str(r[feature]) for r in records}) > 1 for records in measurements)
        assert changing >= (15 if feature == "footprints" else 25), feature
    assert any(r["patterned"] and len(r["footprints"]) > 1 and r["witness"]
               and r["shape"][0] != r["shape"][1] for r in rows)
    assert {rs[0]["axis"] for rs in measurements
            if all(r["footprints"] == [(5, 5)] and not r["patterned"] for r in rs)} == {0, 1}
    assert _infer_convention(generate_puzzle(120, 2, 1)["train"]) == (1, 8, 2)
    for split in ("train", "test"):
        schedules = [[r["background"] for r in rs if r["split"] == split]
                     for rs in measurements]
        assert any(len(set(s)) > 1 for s in schedules)
        assert any(len(set(s)) < len(s) for s in schedules)
    for value in (lambda r: r["count"], lambda r: int(np.prod(r["shape"])),
                  lambda r: max(r["layers"])):
        shifts = set()
        for rs in measurements:
            training = [value(r) for r in rs if r["split"] == "train"]
            for r in rs:
                if r["split"] == "test":
                    shifts.add(-1 if value(r) < min(training) else
                               1 if value(r) > max(training) else 0)
        assert shifts == {-1, 0, 1}
    # A concrete same-class chain catches the old algorithm's 0,5,0 jump.
    objects = [dict(r=r, c=c, h=5, w=5, d=0, frame=6,
                    patch=np.pad(np.full((3, 3), v, dtype=np.int8), 1, constant_values=6))
               for r, c, v in ((8, 2, 2), (4, 7, 9), (0, 12, 2))]
    grid = _render((16, 24), 0, objects)
    expected, decoded, settled = _simulate(grid, 6, 1)
    assert {o["r"]: o["c"] for o in settled} == {8: 0, 4: 5, 0: 10}
    assert _has_witness(decoded, settled, 24, 6)
    assert np.array_equal(expected, _render((16, 24), 0, _compact_objects(objects, 24)))
    # The validator must reject the specific old same-class overtaking output,
    # not merely rely on agreement of the two correct implementations.
    anchors = [dict(objects[0], r=1, c=12),
               dict(objects[0], r=9, c=2, frame=1,
                    patch=np.pad(np.full((3, 3), 2, dtype=np.int8), 1, constant_values=1))]
    teach_input = _render((16, 24), 0, anchors)
    test_input = np.where(grid == 0, 7, grid).astype(np.int8)
    valid = {"train": [{"input": teach_input, "output": _simulate(teach_input, 6, 1)[0]},
                       {"input": grid, "output": expected}],
             "test": [{"input": test_input, "output": _simulate(test_input, 6, 1)[0]}]}
    validate_generated_puzzle(valid)
    old_objects = [dict(o, c=c) for o, c in zip(decoded, _edge_first_columns(decoded, 24, 6))]
    valid["test"][0]["output"] = _render((16, 24), 7, old_objects)
    try:
        validate_generated_puzzle(valid)
    except AssertionError:
        pass
    else:
        raise AssertionError("same-class overtaking was accepted")
    # Combined capacity, rather than capacity of each class alone, matters.
    crowded = [dict(o) for o in objects]
    crowded += [dict(anchors[1], r=0, c=c) for c in (0, 17)]
    try:
        _simulate(_render((14, 22), 0, crowded), 6, 1)
    except AssertionError as exc:
        assert "final groups overlap" in str(exc)
    else:
        raise AssertionError("overlapping final classes were accepted")
    # Corruptions must not be accepted by the visible-grid validator.
    p = corpus[0]
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    if not np.array_equal(p["test"][0]["input"], p["test"][0]["output"]):
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("incorrect compaction accepted")
