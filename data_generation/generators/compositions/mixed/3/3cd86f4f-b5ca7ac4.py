"""Frame-directed stable packing, followed by a bottom-anchored row shear.

Fixed visible semantics: 8 goes left, 2 goes right, all frames are 5x5.
Only standard library and NumPy are used. No evidence files are read at runtime.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np


SIZE = 5
FRAMES = (8, 2)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _band(a, b):
    return abs(a["r"] - b["r"]) < SIZE


def _overlap(a, b):
    return _band(a, b) and abs(a["c"] - b["c"]) < SIZE


def _decode(input_grid):
    """Identify the uniform exposed background and scan complete 5x5 frames.

    Background need not be the majority color. For each possible exposed color,
    the first uncovered nonbackground pixel must be an object's top-left corner.
    A valid interpretation covers every remaining pixel by disjoint full frames.
    Requiring one interpretation makes adjacent/dense scenes well-defined.
    """
    g = _grid(input_grid)
    h, w = g.shape
    if min(h, w) < SIZE or h + w - 1 > 30:
        raise ValueError("5x5 objects and the complete sheared canvas must fit")
    interpretations = []
    for background in sorted(set(map(int, np.unique(g))) - set(FRAMES)):
        covered = g == background
        objects = []
        valid = True
        for r in range(h):
            if not valid:
                break
            for c in range(w):
                if covered[r, c]:
                    continue
                f = int(g[r, c])
                if f not in FRAMES or r + SIZE > h or c + SIZE > w:
                    valid = False
                    break
                p = g[r:r+SIZE, c:c+SIZE]
                inner = p[1:-1, 1:-1]
                if (np.any(covered[r:r+SIZE, c:c+SIZE])
                        or not np.all(p[0] == f) or not np.all(p[-1] == f)
                        or not np.all(p[:, 0] == f) or not np.all(p[:, -1] == f)
                        or np.any(inner == f) or np.any(inner == background)):
                    valid = False
                    break
                covered[r:r+SIZE, c:c+SIZE] = True
                objects.append(dict(r=r, c=c, frame=f, patch=p.copy()))
        if valid and objects and np.all(covered):
            interpretations.append((background, objects))
    if len(interpretations) != 1:
        raise ValueError("scene must have exactly one exposed-background/frame interpretation")
    return interpretations[0]


def _pack(objects, width):
    """Preserve destination-side order for every same-class row overlap."""
    result = [dict(o) for o in objects]
    for frame in FRAMES:
        indices = [i for i, o in enumerate(objects) if o["frame"] == frame]
        indices.sort(key=lambda i: (objects[i]["c"] if frame == 8
                                   else -objects[i]["c"], objects[i]["r"]))
        settled = []
        for i in indices:
            o = result[i]
            blockers = [p for p in settled if _band(o, p)]
            o["c"] = (max((p["c"] + SIZE for p in blockers), default=0) if frame == 8
                      else min((p["c"] for p in blockers), default=width) - SIZE)
            if not 0 <= o["c"] <= width - SIZE:
                raise ValueError("stable directional chain exceeds the canvas")
            settled.append(o)
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i+1:]):
        raise ValueError("independently packed opposite classes overlap")
    return result


def _render(shape, background, objects):
    canvas = np.full(shape, background, dtype=np.int8)
    occupied = np.zeros(shape, dtype=bool)
    for o in objects:
        r, c = o["r"], o["c"]
        if (not 0 <= r <= shape[0] - SIZE or not 0 <= c <= shape[1] - SIZE
                or np.any(occupied[r:r+SIZE, c:c+SIZE])):
            raise ValueError("objects must fit without overlapping")
        canvas[r:r+SIZE, c:c+SIZE] = o["patch"]
        occupied[r:r+SIZE, c:c+SIZE] = True
    return canvas


def _shear(canvas, background):
    h, w = canvas.shape
    out = np.full((h, w+h-1), background, dtype=np.int8)
    for r in range(h):
        start = h - 1 - r
        out[r, start:start+w] = canvas[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention and object from this input alone."""
    g = _grid(input_grid)
    background, objects = _decode(g)
    packed = _render(g.shape, background, _pack(objects, g.shape[1]))
    return _shear(packed, background)


def _edge_first(objects, width):
    """Competing first-fit rule, used only to measure teaching witnesses."""
    result = [dict(o) for o in objects]
    for frame in FRAMES:
        indices = [i for i, o in enumerate(objects) if o["frame"] == frame]
        indices.sort(key=lambda i: (objects[i]["c"] if frame == 8
                                   else -objects[i]["c"], objects[i]["r"]))
        settled = []
        for i in indices:
            positions = range(width-SIZE+1)
            if frame == 2:
                positions = reversed(positions)
            for c in positions:
                result[i]["c"] = c
                if not any(_overlap(result[i], p) for p in settled):
                    settled.append(result[i])
                    break
            else:
                raise ValueError("first-fit class cannot fit")
    return result


def _witness(objects, settled, width):
    return [o["c"] for o in _edge_first(objects, width)] != [o["c"] for o in settled]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _training_layout(rng, purpose, first_height=None):
    if purpose == "directions":
        # An interacting opposite-class pair and an independent lower band.
        h = int(rng.integers(11, 17))
        w = int(rng.integers(11, min(19, 31-h)+1))
        r0 = int(rng.integers(0, h-10))
        r1 = int(rng.integers(r0+SIZE, h-SIZE))  # retain a bottom blank row
        c0 = int(rng.integers(w-2*SIZE+1))
        c1 = int(rng.integers(c0+SIZE, w-SIZE+1))
        repeated_frame = int(_pick(rng, FRAMES))
        objects = [dict(r=r0, c=c0, frame=2), dict(r=r0, c=c1, frame=8),
                   dict(r=r1, c=int(rng.integers(w-SIZE+1)), frame=repeated_frame)]
        if rng.random() < 0.5:
            # Another full pair in the lower independent band.
            objects[2]["c"] = 0 if repeated_frame == 2 else w-SIZE
            objects.append(dict(r=r1, c=w-SIZE if repeated_frame == 2 else 0,
                                frame=8 if repeated_frame == 2 else 2))
        return (h, w), objects
    # A nontransitive row-overlap chain: endpoints are disjoint, but each
    # overlaps the middle. Stable packing uses 0,5,10 rather than 0,5,0.
    step = int(_pick(rng, (3, 4)))
    minimum = SIZE + 2*step
    h = int(_pick(rng, [n for n in range(minimum, 17) if n != first_height]))
    w = int(rng.integers(15, min(19, 31-h)+1))
    gaps = rng.multinomial(w-3*SIZE, [0.25]*4)
    columns = [int(gaps[0])]
    for i in range(2):
        columns.append(columns[-1]+SIZE+int(gaps[i+1]))
    frame = int(_pick(rng, FRAMES))
    rows = [h-SIZE-2*step, h-SIZE-step, h-SIZE]
    if rng.random() < 0.5:
        rows.reverse()
    objects = [dict(r=r, c=c if frame == 8 else w-SIZE-c, frame=frame)
               for r, c in zip(rows, columns)]
    return (h, w), objects


def _free_layout(rng, split, shift):
    if split == "train":
        h = int(rng.integers(10, 17))
        w = int(rng.integers(SIZE, min(19, 31-h)+1))
    elif shift == "wide":
        h = int(rng.integers(SIZE, 10))
        w = int(rng.integers(20, 32-h))
    else:
        # Keep at least one horizontal degree of freedom in held-out scenes.
        h = int(rng.integers(17, 26))
        w = int(rng.integers(SIZE+1, 32-h))
    if split == "test" and shift == "wide" and rng.random() < 0.65:
        # Extrapolate the demonstrated stable chain to four/five layers.
        # Heights below ten make every pair of five-row spans overlap.
        count = int(rng.integers(4, w//SIZE+1))
        if h*w == count*SIZE*SIZE:
            w += 1  # a visible background cell is needed to determine padding
        gaps = rng.multinomial(w-count*SIZE, [1/(count+1)]*(count+1))
        frame = int(_pick(rng, FRAMES))
        objects = []
        c = int(gaps[0])
        for i in range(count):
            objects.append(dict(r=int(rng.integers(h-SIZE+1)),
                                c=c if frame == 8 else w-SIZE-c, frame=frame))
            c += SIZE + int(gaps[i+1])
        return (h, w), objects
    capacity = (h//SIZE)*(w//SIZE)
    target = int(rng.integers(1, capacity+1))
    objects = []
    left_probability = float(_pick(rng, (0.0, 0.15, 0.5, 0.5, 0.85, 1.0)))
    for _ in range(target):
        for attempt in range(60):
            frame = 8 if rng.random() < left_probability else 2
            r = int(rng.integers(h-SIZE+1))
            c = int(rng.integers(w-SIZE+1))
            if rng.random() < 0.25:
                c = int(_pick(rng, (0, w-SIZE)))
            if objects and rng.random() < 0.55:
                other = _pick(rng, objects)
                r = int(_pick(rng, (other["r"], other["r"]+4, other["r"]-4,
                                    other["r"]+SIZE, other["r"]-SIZE)))
                if rng.random() < 0.5:
                    c = int(_pick(rng, (other["c"]+SIZE, other["c"]-SIZE)))
            o = dict(r=r, c=c, frame=frame)
            if not (0 <= r <= h-SIZE and 0 <= c <= w-SIZE):
                continue
            if any(_overlap(o, p) for p in objects) or SIZE*SIZE*(len(objects)+1) >= h*w:
                continue
            try:
                _pack(objects+[o], w)
            except ValueError:
                continue
            objects.append(o)
            break
        else:
            break
    if not objects:
        objects = [dict(r=int(rng.integers(h-SIZE+1)),
                        c=int(rng.integers(w-SIZE+1)), frame=int(_pick(rng, FRAMES)))]
    # Some easy cases are already packed; the shear still acts nontrivially.
    if rng.random() < 0.12:
        objects = _pack(objects, w)
    return (h, w), objects


def _appearance(rng, objects, background, pattern_rate, shared=False):
    pool = [c for c in range(10) if c != background]
    pair_palette = list(map(int, rng.choice(pool, int(rng.integers(2, 8)), replace=False)))
    common = [c for c in pool if c not in FRAMES]
    common_color = int(_pick(rng, common))
    shared_inner = None
    for i, o in enumerate(objects):
        allowed = [c for c in pair_palette if c != o["frame"]]
        if shared and i < 2:
            allowed = common
        inner = np.full((3, 3), _pick(rng, allowed), dtype=np.int8)
        if len(allowed) > 1 and rng.random() < pattern_rate:
            colors = np.array(rng.choice(allowed, min(3, len(allowed)), replace=False), dtype=np.int8)
            rr, cc = np.indices((3, 3))
            mode = int(rng.integers(4))
            idx = rr if mode == 0 else cc if mode == 1 else rr+cc
            if mode == 3:
                idx = rng.integers(len(colors), size=(3, 3))
            inner = colors[idx % len(colors)]
        if shared and i < 2:
            if shared_inner is None:
                inner[1, 1] = common_color
                shared_inner = inner.copy()
            inner = shared_inner.copy()
        o["patch"] = np.full((SIZE, SIZE), o["frame"], dtype=np.int8)
        o["patch"][1:-1, 1:-1] = inner


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(_pick(rng, ("wide", "tall")))
    pattern_rate = float(_pick(rng, (0.0, 0.3, 0.7)))
    backgrounds = list(map(int, rng.choice([c for c in range(10) if c not in FRAMES],
                                         int(rng.integers(1, 5)), replace=False)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            purpose = ("directions" if split == "train" and index == 0 else
                       "chain" if split == "train" and index == 1 else "free")
            for _ in range(120):
                if purpose == "free":
                    shape, objects = _free_layout(rng, split, shift)
                else:
                    first_height = puzzle["train"][0]["input"].shape[0] if index else None
                    shape, objects = _training_layout(rng, purpose, first_height)
                background = int(_pick(rng, backgrounds)) if split == "train" else int(
                    _pick(rng, backgrounds if rng.random() < 0.5 else
                          [c for c in range(10) if c not in FRAMES]))
                _appearance(rng, objects, background, pattern_rate, shared=purpose == "directions")
                inp = _render(shape, background, objects)
                if _key(inp) in seen:
                    continue
                try:
                    # This rejects ambiguous visible interpretations, including
                    # rare dense scenes where an alternate background could fit.
                    _decode(inp)
                    packed = _render(shape, background, _pack(objects, shape[1]))
                    _decode(packed)
                except ValueError:
                    continue
                if split == "test" and np.array_equal(inp, packed):
                    # A held-out answer must actually depend on both parents.
                    # Width-five/fully-settled examples remain useful in train.
                    continue
                pair = {"input": inp, "output": _shear(packed, background)}
                seen.add(_key(inp))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded generation could not construct a distinct valid example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _unshear(inp, out, background):
    h, w = inp.shape
    assert out.shape == (h, h+w-1), "incorrect complete-canvas output dimensions"
    packed = np.empty_like(inp)
    for r in range(h):
        start = h-1-r
        assert np.all(out[r, :start] == background), "incorrect left shear padding"
        assert np.all(out[r, start+w:] == background), "incorrect right shear padding"
        packed[r] = out[r, start:start+w]
    return packed


def _identity(o):
    return o["r"], o["frame"], o["patch"].tobytes()


def _check_settled(source, destination, width):
    """Check identities, motion, order and contact, without calling _pack/solve."""
    before, after = defaultdict(list), defaultdict(list)
    for o in source:
        before[_identity(o)].append(o)
    for o in destination:
        after[_identity(o)].append(o)
    assert before.keys() == after.keys(), "row, frame or payload was changed"
    mapped = {}
    for key, group in before.items():
        assert len(group) == len(after[key]), "an object was lost or duplicated"
        aa = sorted(group, key=lambda o: o["c"])
        bb = sorted(after[key], key=lambda o: o["c"])
        for a, b in zip(aa, bb):
            mapped[id(a)] = b
            assert (b["c"] <= a["c"] if a["frame"] == 8 else b["c"] >= a["c"]), "wrong direction"
    for i, a in enumerate(source):
        for b in source[i+1:]:
            if a["frame"] == b["frame"] and _band(a, b):
                da, db = mapped[id(a)], mapped[id(b)]
                assert ((da["c"]+SIZE <= db["c"]) if a["c"] < b["c"]
                        else (db["c"]+SIZE <= da["c"])), "same-class overtaking"
    for o in destination:
        candidate = dict(o, c=o["c"] + (-1 if o["frame"] == 8 else 1))
        can_step = (0 <= candidate["c"] <= width-SIZE and not any(
            p is not o and p["frame"] == o["frame"] and _overlap(candidate, p)
            for p in destination))
        assert not can_step, "same-class compaction is incomplete"
    return [mapped[id(o)] for o in source]


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    moved_frames, train_heights = set(), set()
    shared_payload = independent_band = chain_witness = False
    blank_bottom = occupied_bottom = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            background, objects = _decode(inp)
            packed = _unshear(inp, out, background)
            packed_background, decoded = _decode(packed)
            assert packed_background == background, "background changed"
            settled = _check_settled(objects, decoded, inp.shape[1])
            if split == "test":
                assert any(a["c"] != b["c"] for a, b in zip(objects, settled)), "test must exercise packing"
            # Shear and translation must preserve every foreground color count.
            for color in range(10):
                if color != background:
                    assert np.count_nonzero(inp == color) == np.count_nonzero(out == color)
            if split == "train":
                train_heights.add(inp.shape[0])
                blank_bottom |= bool(np.all(inp[-1] == background))
                occupied_bottom |= bool(np.any(inp[-1] != background))
                moved_frames.update(a["frame"] for a, b in zip(objects, settled) if a["c"] != b["c"])
                chain_witness |= _witness(objects, settled, inp.shape[1])
                for i, a in enumerate(objects):
                    for j, b in enumerate(objects[i+1:], i+1):
                        shared_payload |= (a["frame"] != b["frame"] and np.array_equal(
                            a["patch"][1:-1, 1:-1], b["patch"][1:-1, 1:-1]))
                        independent_band |= (a["frame"] == b["frame"] and not _band(a, b)
                                             and settled[i]["c"] == settled[j]["c"])
    assert moved_frames == set(FRAMES), "training must visibly demonstrate both frame instructions"
    assert shared_payload, "training must separate payload identity from direction"
    assert independent_band, "training must teach edge reuse by disjoint row spans"
    assert chain_witness, "training must distinguish stable sliding from first-fit jumping"
    assert len(train_heights) >= 2 and blank_bottom and occupied_bottom, "teach absolute canvas-row anchoring"


def measure_puzzle(puzzle):
    """Small grid-derived measurements; no grids or execution traces are retained."""
    records = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            inp = _grid(pair["input"])
            background, objects = _decode(inp)
            settled = _pack(objects, inp.shape[1])
            overlaps = [SIZE-abs(a["r"]-b["r"]) for i, a in enumerate(objects)
                        for b in objects[i+1:] if _band(a, b)]
            records.append(dict(split=split, index=index, shape=list(inp.shape),
                background=background, count=len(objects),
                classes=[sum(o["frame"] == f for o in objects) for f in FRAMES],
                layers=[max(((o["c"]//SIZE+1 if f == 8 else
                              (inp.shape[1]-SIZE-o["c"])//SIZE+1)
                             for o in settled if o["frame"] == f), default=0) for f in FRAMES],
                travels=[abs(a["c"]-b["c"]) for a, b in zip(objects, settled)],
                overlap_widths=sorted(set(overlaps)),
                crossings=sum(a["frame"] == 8 and b["frame"] == 2 and _band(a, b)
                              and a["c"] > b["c"] for a in objects for b in objects),
                witness=bool(_witness(objects, settled, inp.shape[1])),
                patterned=sum(len(np.unique(o["patch"][1:-1, 1:-1])) > 1 for o in objects),
                frame_as_payload=any(any(c in FRAMES for c in np.unique(o["patch"][1:-1, 1:-1]))
                                     for o in objects),
                bottom_blank=bool(np.all(inp[-1] == background)),
                output_width=int(np.asarray(pair["output"]).shape[1])))
    return records


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and bounded count/mutation checks; write no files."""
    fingerprints, shapes, palettes, counts, structures = set(), set(), set(), set(), set()
    backgrounds, layers, directions, overlap_widths, modes = set(), set(), set(), set(), set()
    stationary = moving = patterned = plain = crossings = noncrossings = frame_payload = 0
    boundary = single_class = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        assert _fingerprint(p) == _fingerprint(generate_puzzle(seed))
        for row in measure_puzzle(p):
            shapes.add(tuple(row["shape"]))
            counts.add(row["count"])
            backgrounds.add(row["background"])
            layers.update(row["layers"])
            overlap_widths.update(row["overlap_widths"])
            stationary += row["travels"].count(0)
            moving += sum(t > 0 for t in row["travels"])
            patterned += row["patterned"] > 0
            plain += row["patterned"] == 0
            crossings += row["crossings"] > 0
            noncrossings += row["crossings"] == 0
            frame_payload += row["frame_as_payload"]
            single_class += 0 in row["classes"]
            boundary += row["output_width"] == 30
            if row["witness"]:
                directions.update(f for f, n in zip(FRAMES, row["classes"]) if n >= 3)
            if row["split"] == "test":
                modes.add("wide" if row["shape"][0] < 10 else "tall")
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                palettes.add(tuple(map(int, np.unique(inp))))
                background, objects = _decode(inp)
                # A color-free geometry signature, rather than recolor hashes.
                structures.add((inp.shape, tuple((o["r"], o["c"], o["frame"]) for o in objects)))
                if split == "test":
                    assert inp.shape[0] < min(q["input"].shape[0] for q in p["train"]) or (
                        inp.shape[0] > max(q["input"].shape[0] for q in p["train"]))
                    assert not np.array_equal(out, _shear(inp, background)), "held-out B ablation must differ"
        # Both inherited operations are necessary in the first demonstration.
        pair = p["train"][0]
        bg, objects = _decode(pair["input"])
        packed = _render(pair["input"].shape, bg, _pack(objects, pair["input"].shape[1]))
        assert not np.array_equal(pair["output"], packed)  # omit A
        assert not np.array_equal(pair["output"], _shear(pair["input"], bg))  # omit B
    assert len(fingerprints) == 42 and len(structures) >= 150
    assert len(shapes) >= 45 and len(palettes) >= 25 and len(counts) >= 5
    assert len(backgrounds) >= 7 and {1, 2, 3, 4} <= layers
    assert directions == set(FRAMES) and modes == {"wide", "tall"}
    assert {1, 2, 3, 4, 5} <= overlap_widths
    assert all((stationary, moving, patterned, plain, crossings, noncrossings,
                frame_payload, boundary, single_class))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            q = generate_puzzle(917, a, b)
            assert (len(q["train"]), len(q["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    # Hand-specified chain. Bottom row 12 stays fixed; top row shifts 12.
    objects = []
    for r, c, payload in ((8, 1, 3), (4, 7, 4), (0, 13, 6)):
        patch = np.full((5, 5), 8, dtype=np.int8)
        patch[1:4, 1:4] = payload
        objects.append(dict(r=r, c=c, frame=8, patch=patch))
    inp = _render((13, 18), 0, objects)
    out = solve(inp)
    assert [o["c"] for o in _pack(objects, 18)] == [0, 5, 10]
    assert [o["c"] for o in _edge_first(objects, 18)] == [0, 5, 0]
    assert out.shape == (13, 30)
    assert "".join(map(str, out[0])) == "0"*22 + "8"*5 + "0"*3
    assert "".join(map(str, out[12])) == "8"*5 + "0"*25
    # Dense valid inputs need not have a majority background. Nine squares
    # occupy 225/240 cells; each independent band packs to columns 0,5,10.
    dense = [dict(objects[0], r=r, c=c) for r in (0, 5, 10) for c in (1, 6, 11)]
    dense_input = _render((15, 16), 0, dense)
    assert np.count_nonzero(dense_input == 0) == 15
    dense_output = solve(dense_input)
    dense_packed = _unshear(dense_input, dense_output, 0)
    assert [(o["r"], o["c"]) for o in _decode(dense_packed)[1]] == [
        (r, c) for r in (0, 5, 10) for c in (0, 5, 10)]
    # Solver domain extends beyond the generated test schedule, including a
    # stationary 26x5 scene with one complete blank row retained at the bottom.
    tall = [dict(objects[0], r=r, c=0) for r in (0, 5, 10, 15, 20)]
    tall_input = _render((26, 5), 0, tall)
    tall_output = solve(tall_input)
    assert tall_output.shape == (26, 30)
    assert np.all(tall_output[-1] == 0) and np.array_equal(tall_output[0, 25:30], tall_input[0])
    wide = [dict(objects[0], r=0, c=c) for c in (1, 6, 11, 16, 21)]
    wide_input = _render((5, 26), 0, wide)
    wide_packed = _unshear(wide_input, solve(wide_input), 0)
    assert [o["c"] for o in _decode(wide_packed)[1]] == [0, 5, 10, 15, 20]
    # Verify constraints reject plausible wrong outputs, not only random noise.
    p = generate_puzzle(0)
    for kind in ("omit_a", "omit_b", "top_anchor", "jump", "payload"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        index = 1 if kind == "jump" else 0
        pair = bad["train"][index]
        bg, objs = _decode(pair["input"])
        packed = _render(pair["input"].shape, bg, _pack(objs, pair["input"].shape[1]))
        if kind == "omit_a":
            pair["output"] = packed
        elif kind == "omit_b":
            pair["output"] = _shear(pair["input"], bg)
        elif kind == "top_anchor":
            h, w = packed.shape
            wrong = np.full((h, h+w-1), bg, dtype=np.int8)
            for r in range(h):
                wrong[r, r:r+w] = packed[r]
            pair["output"] = wrong
        elif kind == "jump":
            pair["output"] = _shear(_render(packed.shape, bg, _edge_first(objs, packed.shape[1])), bg)
        else:
            rr, cc = np.argwhere(pair["output"] != bg)[0]
            pair["output"][rr, cc] = (int(pair["output"][rr, cc])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"validator accepted {kind}")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "geometries": len(structures), "palettes": len(palettes), "object_counts": sorted(counts),
            "backgrounds": sorted(backgrounds), "packing_layers": sorted(layers),
            "boundary_outputs": boundary, "patterned_pairs": patterned,
            "plain_pairs": plain, "single_class_pairs": single_class,
            "stationary_objects": stationary, "moving_objects": moving,
            "frame_as_payload_pairs": frame_payload, "mutations_rejected": 5}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
