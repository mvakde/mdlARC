"""Finite, beacon-terminated pinwheel rails with marker-directed echoes.

The rule is fixed and clockwise. Four spoke colors identify four temporal
phases. Input beacons terminate their own phase, not an infinite full maze.
No parent modules, fixtures, seeds, or examples are used by solve.
"""
from __future__ import annotations

import json
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# Five full four-stage cycles cover every possible in-canvas return on 30x30.
# This is a geometric safety bound, not a restriction on where beacons may sit.
MAX_STAGES = 20


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(length, center, phase, stages=MAX_STAGES):
    """Unit-step temporal path; corners and stage ends are generation aids."""
    dy, dx = DIRECTIONS[phase]
    y, x = center
    path, corners, ends = [], [], []

    def walk(vy, vx, n):
        nonlocal y, x
        for _ in range(n):
            y, x = y + vy, x + vx
            path.append((y, x))

    radius = max(2, length)
    walk(dy, dx, radius)
    corners.append(len(path) - 1)
    dy, dx = dx, -dy  # clockwise in row/column coordinates
    for k in range(stages):
        walk(dy, dx, 2 + 4*k)
        corners.append(len(path) - 1)
        if radius > 2:
            walk(dx, -dy, radius - 2)
            corners.append(len(path) - 1)
        walk(dy, dx, radius)
        corners.append(len(path) - 1)
        ends.append(len(path) - 1)
        dy, dx = dx, -dy
    assert len(set(path)) == len(path), "a rail must not revisit a cell"
    return path, sorted(set(corners)), ends


def _cross_candidates(a, bg):
    candidates = []
    h, w = a.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if a[y, x] != bg:
                continue
            colors = tuple(int(a[y+dy, x+dx]) for dy, dx in DIRECTIONS)
            if bg in colors or len(set(colors)) != 4:
                continue
            lengths = []
            for (dy, dx), color in zip(DIRECTIONS, colors):
                n = 0
                while (_inside((y+(n+1)*dy, x+(n+1)*dx), a.shape)
                       and a[y+(n+1)*dy, x+(n+1)*dx] == color):
                    n += 1
                lengths.append(n)
            if len(set(lengths)) == 1 and 1 <= lengths[0] <= 4:
                candidates.append(((y, x), lengths[0], colors))
    return candidates


def _parse(a):
    a = _grid(a)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert int(counts.max()) > a.size / 2, "background must be the strict majority"
    candidates = _cross_candidates(a, bg)
    assert len(candidates) == 1, "one uniquely identifiable hollow four-color cross required"
    center, length, colors = candidates[0]
    rails, source_cells = [], set()
    for phase, color in enumerate(colors):
        path, corners, ends = _trace(length, center, phase)
        index = {p: i for i, p in enumerate(path)}
        landmarks = {tuple(map(int, p)) for p in np.argwhere(a == color)}
        assert len(landmarks) >= length + 1, "each spoke needs a separate outer beacon"
        assert landmarks <= set(index), "beacon is not on its phase's winding"
        last = max(index[p] for p in landmarks)
        assert last >= ends[0], "at least one complete winding stage per rail"
        finite = path[:last+1]
        cells = {p for p in finite if _inside(p, a.shape)}
        assert not source_cells & cells, "source rails must be disjoint"
        source_cells.update(cells)
        rails.append(dict(color=color, phase=phase, path=finite, cells=cells,
                          landmarks=landmarks, last=last, corners=corners,
                          stage=sum(last > e for e in ends), ends=ends))
    markers, offsets = [], set()
    for color in (int(c) for c in values if c != bg and c not in colors):
        points = np.argwhere(a == color)
        assert len(points) == 1, "every extra color is one echo marker"
        p = tuple(map(int, points[0]))
        associations = [(i, q) for i, rail in enumerate(rails) for q in rail["landmarks"]
                        if max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1]
        assert len(associations) == 1, "marker must have one adjacent supplied landmark"
        i, q = associations[0]
        assert i not in {m["rail"] for m in markers}, "at most one marker per rail"
        assert p not in source_cells, "marker must be clear of completed source rails"
        offset = p[0]-q[0], p[1]-q[1]
        offsets.add(offset)
        copied = {(r+offset[0], c+offset[1]) for r, c in rails[i]["path"]
                  if _inside((r+offset[0], c+offset[1]), a.shape)}
        markers.append(dict(color=color, point=p, rail=i, offset=offset, cells=copied))
    assert len(offsets) <= 1, "all echoes in one loom share the visible displacement"
    assert len(markers) <= 4
    return dict(bg=bg, center=center, length=length, colors=colors,
                rails=rails, source=source_cells, markers=markers)


def _paint(a, info, rail_sets=None, copy_sets=None):
    result = a.copy()
    rail_sets = [r["cells"] for r in info["rails"]] if rail_sets is None else rail_sets
    copy_sets = [m["cells"] for m in info["markers"]] if copy_sets is None else copy_sets
    source = set().union(*rail_sets)
    claimed = set()
    for marker, cells in zip(info["markers"], copy_sets):
        for p in cells:
            if a[p] == info["bg"] and p not in source:
                assert p not in claimed, "companion overlap is not a legal family member"
                claimed.add(p)
                result[p] = marker["color"]
    for rail, cells in zip(info["rails"], rail_sets):
        for p in cells:
            if a[p] == info["bg"]:
                result[p] = rail["color"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from geometry and visible colors alone, with fixed clockwise turns."""
    a = _grid(input_grid)
    return _paint(a, _parse(a))


def _segment(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    n = max(abs(dr), abs(dc))
    if not n or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return set()
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return {(a[0]+i*sr, a[1]+i*sc) for i in range(n+1)}


def _straight_shortcut(a, info):
    """A plausible B-ablation: indiscriminately interpolate aligned beacons."""
    sets = []
    for rail in info["rails"]:
        landmarks = sorted(rail["landmarks"])
        cells = set(landmarks)
        for i, p in enumerate(landmarks):
            for q in landmarks[i+1:]:
                line = _segment(p, q)
                if not ((line - {p, q}) & rail["landmarks"]):
                    cells.update(line)
        sets.append(cells)
    # This diagnostic uses source priority but only one color at newly colliding
    # source cells; the comparison need not declare those ambiguous lines valid.
    result = a.copy()
    source = set().union(*sets)
    for marker in info["markers"]:
        dr, dc = marker["offset"]
        for r, c in sets[marker["rail"]]:
            p = r+dr, c+dc
            if _inside(p, a.shape) and a[p] == info["bg"] and p not in source:
                result[p] = marker["color"]
    for rail, cells in zip(info["rails"], sets):
        for p in cells:
            if a[p] == info["bg"]:
                result[p] = rail["color"]
    return result


def _features(a, b, info=None):
    info = _parse(a) if info is None else info
    no_echo = _paint(a, info, copy_sets=[set() for _ in info["markers"]])
    reentry, outside_echo, overlap, visible = 0, 0, 0, []
    clipped_copies, without_spoke_copies = [], []
    for rail in info["rails"]:
        flags = [_inside(p, a.shape) for p in rail["path"]]
        if False in flags:
            reentry += any(flags[flags.index(False)+1:])
    for marker in info["markers"]:
        rail = info["rails"][marker["rail"]]
        dr, dc = marker["offset"]
        clipped_copies.append({(r+dr, c+dc) for r, c in rail["cells"]
                               if _inside((r+dr, c+dc), a.shape)})
        without_spoke_copies.append({(r+dr, c+dc) for r, c in rail["path"][info["length"]:]
                                    if _inside((r+dr, c+dc), a.shape)})
        outside_echo += sum(not _inside((r, c), a.shape)
                            and _inside((r+dr, c+dc), a.shape) for r, c in rail["path"])
        overlap += sum(a[p] == info["bg"] for p in marker["cells"] & info["source"])
        visible.append(int(np.count_nonzero((a == info["bg"]) & (b == marker["color"]))))
    unlimited_paths = [_trace(info["length"], info["center"], i)[0] for i in range(4)]
    unlimited_sets = [{p for p in path if _inside(p, a.shape)} for path in unlimited_paths]
    unlimited_copies = []
    for marker in info["markers"]:
        dr, dc = marker["offset"]
        unlimited_copies.append({(r+dr, c+dc) for r, c in unlimited_paths[marker["rail"]]
                                 if _inside((r+dr, c+dc), a.shape)})
    unlimited = _paint(a, info, unlimited_sets, unlimited_copies)
    distance_paths = []
    cy, cx = info["center"]
    for rail in info["rails"]:
        index = {p: i for i, p in enumerate(rail["path"])}
        far = max(rail["landmarks"], key=lambda p: ((p[0]-cy)**2+(p[1]-cx)**2, index[p]))
        distance_paths.append(rail["path"][:index[far]+1])
    distance_sets = [{p for p in path if _inside(p, a.shape)} for path in distance_paths]
    distance_copies = []
    for marker in info["markers"]:
        dr, dc = marker["offset"]
        distance_copies.append({(r+dr, c+dc) for r, c in distance_paths[marker["rail"]]
                               if _inside((r+dr, c+dc), a.shape)})
    return dict(shape=list(a.shape), length=info["length"], center=list(info["center"]),
                markers=len(info["markers"]), offset=list(info["markers"][0]["offset"]) if info["markers"] else None,
                stops=[r["last"] for r in info["rails"]], stages=[r["stage"] for r in info["rails"]],
                beacons=[len(r["landmarks"])-info["length"] for r in info["rails"]],
                reentry=int(reentry), outside_echo=int(outside_echo), overlap=int(overlap),
                visible=visible, echo_difference=int(np.count_nonzero(b != no_echo)),
                stop_difference=int(np.count_nonzero(b != unlimited)),
                distance_difference=int(np.count_nonzero(b != _paint(a, info, distance_sets, distance_copies))),
                spoke_copy_difference=int(np.count_nonzero(b != _paint(a, info, copy_sets=without_spoke_copies))),
                clip_difference=int(np.count_nonzero(b != _paint(a, info, copy_sets=clipped_copies))),
                straight_difference=int(np.count_nonzero(b != _straight_shortcut(a, info))))


def _make_pair(rng, length, kind, marked, offset):
    """Sample compatible geometry; all accepted inputs are checked by parsing."""
    for attempt in range(160):
        if kind == "clear":
            shape = tuple(int(x) for x in rng.integers(23, 31, size=2))
            center = tuple((n-1)//2 for n in shape)
            cap = 3
        elif kind == "teach_window":
            shape = tuple(int(x) for x in rng.integers(23, 31, size=2))
            center = (length+3, (shape[1]-1)//2)
            cap = 3
        elif kind == "later":
            shape = tuple(int(x) for x in rng.integers(21, 31, size=2))
            center = tuple(int(rng.integers(length, n-length)) for n in shape)
            cap = 7
        elif kind == "window":
            # A genuinely narrower query window than every training canvas.
            radius = max(2, length)
            shape = (2*radius+6, int(rng.integers(21, 31)))
            # Enough room for an unambiguous marker on every phase, even when
            # four cardinal echoes are requested. Later stages still clip.
            center = (radius+2 if offset[0] >= 0 else radius+3,
                      int(rng.integers(radius+3, shape[1]-radius-3)))
            cap = 5
        else:
            shape = tuple(int(x) for x in rng.integers(15, 31, size=2))
            center = tuple(int(rng.integers(length, n-length)) for n in shape)
            cap = int(rng.integers(2, 4))
        palette = [int(c) for c in rng.permutation(10)]
        bg, colors = palette[0], palette[1:5]
        a = np.full(shape, bg, dtype=np.int8)
        finite_paths, landmark_sets, full_source, last_indices = [], [], set(), []
        for phase, color in enumerate(colors):
            path, corners, ends = _trace(length, center, phase)
            visible = [i for i in range(ends[cap]+1) if _inside(path[i], shape)]
            minimum = ends[1] if kind == "clear" else ends[0]
            options = [i for i in visible if i >= minimum]
            if not options:
                break
            # Unequal finite stops are deliberate, not a common square crop.
            if kind == "teach_window" and phase == 2:
                terminal = max(options)
            elif kind in ("later", "window"):
                late = [i for i in options if i >= ends[min(3, cap)]]
                terminal = int(rng.choice(late or options[-max(1, len(options)//3):]))
            else:
                terminal = int(rng.choice(options))
            beacons = {terminal}
            for i in corners:
                if i <= terminal and _inside(path[i], shape):
                    if kind == "clear" or rng.random() < .55:
                        beacons.add(i)
            # Never extend the visible initial radial spoke with a same-color
            # beacon. L=1's radius-two extension is supplied only by the answer.
            beacons = {i for i in beacons if i >= max(2, length)}
            beacons.add(terminal)
            landmarks = {path[i] for i in range(length)} | {path[i] for i in beacons}
            finite = path[:terminal+1]
            cells = {p for p in finite if _inside(p, shape)}
            finite_paths.append(finite)
            landmark_sets.append(landmarks)
            last_indices.append(terminal)
            full_source.update(cells)
            for p in landmarks:
                a[p] = color
        if len(finite_paths) != 4 or len(set(last_indices)) == 1:
            continue
        all_landmarks = set().union(*landmark_sets)
        mark_options = []
        for phase in range(4):
            options = []
            for q in sorted(landmark_sets[phase]):
                p = q[0]+offset[0], q[1]+offset[1]
                if (not _inside(p, shape) or p in full_source
                        or sum(max(abs(p[0]-v[0]), abs(p[1]-v[1])) == 1
                               for v in all_landmarks) != 1):
                    continue
                copy = {(r+offset[0], c+offset[1]) for r, c in finite_paths[phase]
                        if _inside((r+offset[0], c+offset[1]), shape)}
                if len(copy-full_source-all_landmarks-{p}) >= 2:
                    options.append(p)
            mark_options.append(options)
        eligible = [i for i in range(4) if mark_options[i]]
        if len(eligible) < marked:
            continue
        for j, phase in enumerate(rng.choice(eligible, marked, replace=False)):
            p = mark_options[phase][int(rng.integers(len(mark_options[phase])))]
            a[p] = palette[5+j]
        # Rotate the complete scene and cues; fixed clockwise geometry is
        # rotation-equivariant. This also distributes the teaching crop edges.
        a = np.rot90(a, int(rng.integers(4))).copy()
        try:
            info = _parse(a)
            b = _paint(a, info)
            f = _features(a, b, info)
            assert np.any(a != b) and f["straight_difference"] > 0
            assert f["stop_difference"] > 0
            assert len(set(f["stops"])) > 1
            assert all(n >= 2 for n in f["visible"])
            if kind == "clear":
                assert f["distance_difference"] > 0 and f["spoke_copy_difference"] > 0
            if kind == "teach_window":
                assert f["reentry"] > 0
                assert f["clip_difference"] > 0
            if kind == "later":
                assert max(f["stages"]) >= 4
            if marked:
                assert f["overlap"] > 0
            return {"input": a, "output": b}
        except AssertionError:
            continue
    raise RuntimeError(f"bounded geometry sampling exhausted for {kind}, L={length}")


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _fingerprint(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    regime = str(rng.choice(("later", "window", "echo")))
    cardinal = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    length_order = [2, 3] if rng.random() < .5 else [3, 2]
    plans = [(length_order[0], "clear", 1, cardinal[int(rng.integers(4))]),
             (length_order[1], "teach_window", 2, (1, 0))]
    for _ in range(int(num_train)-2):
        options = cardinal if regime == "echo" else OFFSETS
        plans.append((int(rng.integers(1, 5)), "varied", int(rng.integers(0, 3 if regime == "echo" else 5)),
                      options[int(rng.integers(len(options)))]))
    training_lengths = sorted({p[0] for p in plans})
    for _ in range(int(num_test)):
        plans.append((int(rng.choice(training_lengths)), "later" if regime == "later" else "window",
                      int(rng.integers(3, 5)) if regime == "echo" else int(rng.integers(1, 5)),
                      diagonal[int(rng.integers(4))] if regime == "echo" else OFFSETS[int(rng.integers(8))]))
    for i, plan in enumerate(plans):
        for _ in range(20):
            pair = _make_pair(rng, *plan)
            key = _fingerprint(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if i < num_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("distinct-input sampling exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _interval_expected(a, info):
    """Validation cross-check: closed-form stage starts, no unit-step tracer.

    Rasterize entire directed intervals, cut the phase at its recovered temporal
    index, and only then translate/clip. The input parser's ordering is shared;
    this is an author implementation cross-check, not an independent oracle.
    """
    radius = max(2, info["length"])
    paths = []
    for phase, rail in enumerate(info["rails"]):
        cells = set()
        limit = rail["last"]
        intervals = [((-1, 0), (-radius, 0))]
        for k in range(MAX_STAGES):
            q, s = divmod(k, 4)
            y, x = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                    (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))[s]
            dy, dx = ((0, 1), (1, 0), (0, -1), (-1, 0))[s]
            p0 = (y, x)
            p1 = (y+(2+4*k)*dy, x+(2+4*k)*dx)
            p2 = (p1[0]+(radius-2)*dx, p1[1]-(radius-2)*dy)
            p3 = (p2[0]+radius*dy, p2[1]+radius*dx)
            intervals.extend(((p0, p1), (p1, p2), (p2, p3)))
        remaining = limit+1
        first = True
        for p, q in intervals:
            dr, dc = q[0]-p[0], q[1]-p[1]
            n = max(abs(dr), abs(dc))
            for j in range(0 if first else 1, n+1):
                if remaining == 0:
                    break
                r, c = p[0]+j*int(np.sign(dr)), p[1]+j*int(np.sign(dc))
                for _ in range(phase):
                    r, c = c, -r
                cells.add((r+info["center"][0], c+info["center"][1]))
                remaining -= 1
            first = False
            if remaining == 0:
                break
        assert remaining == 0
        paths.append(cells)
    source = [{p for p in cells if _inside(p, a.shape)} for cells in paths]
    copies = []
    for marker in info["markers"]:
        dr, dc = marker["offset"]
        copies.append({(r+dr, c+dc) for r, c in paths[marker["rail"]]
                       if _inside((r+dr, c+dc), a.shape)})
    return _paint(a, info, source, copies)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = _fingerprint(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(a)
            assert np.array_equal(b, _interval_expected(a, info)), "incorrect finite winding or echo"
            assert np.all(b[a != info["bg"]] == a[a != info["bg"]]), "input cells must survive"
            assert set(np.unique(b)) == set(np.unique(a)), "palette must be preserved"
            assert np.any(a != b)
            f = _features(a, b, info)
            assert len(set(f["stops"])) > 1, "independent beacon termination must be visible"
            assert f["straight_difference"] > 0, "winding must matter"
            assert f["stop_difference"] > 0, "beacon termination must matter"
            assert all(n >= 2 for n in f["visible"]), "echo must be observable"
            if f["markers"]:
                assert f["overlap"] > 0, "source-over-copy priority must be observable"
            features[split].append(f)
    train = features["train"]
    assert {2, 3} <= {f["length"] for f in train}, "two scales must teach the changing jog"
    assert any(f["markers"] == 1 for f in train) and any(f["markers"] == 2 for f in train)
    assert any(f["reentry"] for f in train), "boundary re-entry needs a demonstration"
    assert any(f["clip_difference"] for f in train), "translation before clipping needs a demonstration"
    assert any(f["distance_difference"] for f in train), "temporal order must beat center-distance order"
    assert any(f["spoke_copy_difference"] for f in train), "echoing supplied spokes must be demonstrated"
    assert all(f["length"] in {t["length"] for t in train} for f in features["test"])


def self_test() -> dict:
    """Bounded geometry, mutation, count, determinism and 42-seed checks."""
    # Check closed-form interval endpoints against each procedural phase,
    # including the radius-two equivalence and coordinate rotation.
    p1 = _trace(1, (0, 0), 0)[0]
    p2 = _trace(2, (0, 0), 0)[0]
    assert p1 == p2
    for length in range(1, 5):
        traces = [_trace(length, (0, 0), phase)[0] for phase in range(4)]
        for i, trace in enumerate(traces):
            assert not set(trace) & set().union(*(set(p) for j, p in enumerate(traces) if j != i))
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a, b in zip(trace, trace[1:]))
        # Admissible members outside the sampler's beacon schedule: interior
        # milestones and terminals much later than stage seven, near corners.
        for center in ((length, length), (29-length, 29-length)):
            a = np.zeros((30, 30), dtype=np.int8)
            for phase in range(4):
                path, _, ends = _trace(length, center, phase)
                long_path = _trace(length, center, phase, stages=32)[0]
                assert {p for p in path if _inside(p, a.shape)} == {
                    p for p in long_path if _inside(p, a.shape)}, "continuation bound loses re-entry"
                last = max(i for i, p in enumerate(path) if _inside(p, a.shape))
                supplied = set(path[:length]) | {path[last]}
                interior = [p for i, p in enumerate(path[:last])
                            if i >= ends[0] and _inside(p, a.shape)]
                supplied.update(interior[::max(1, len(interior)//4)])
                for p in supplied:
                    a[p] = phase+1
            info = _parse(a)
            assert np.array_equal(solve(a), _interval_expected(a, info))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows, shapes, layouts, palettes = [], set(), set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                info = _parse(a)
                f = _features(a, b, info)
                rows.append(f)
                shapes.add(a.shape)
                layouts.add((a.shape, tuple(f["center"]), f["length"], tuple(f["stops"]), tuple(f["beacons"])))
                palettes.add((info["bg"], *info["colors"]))
                assert np.array_equal(solve(a), b)
                rotated = np.rot90(a).copy()
                assert np.array_equal(solve(rotated), np.rot90(b))
    assert len(shapes) >= 40 and len(layouts) >= 180 and len(palettes) >= 100
    assert {f["length"] for f in rows} == {1, 2, 3, 4}
    assert {f["markers"] for f in rows} == {0, 1, 2, 3, 4}
    assert {tuple(f["offset"]) for f in rows if f["offset"]} == set(OFFSETS)
    assert sum(f["reentry"] for f in rows) > 20
    assert any(f["outside_echo"] for f in rows), "translate-before-clip must be exercised"
    assert max(max(f["stages"]) for f in rows) >= 5
    signature = lambda p: tuple(_fingerprint(pair[k]) for s in ("train", "test")
                               for pair in p[s] for k in ("input", "output"))
    assert len({signature(p) for p in puzzles}) == 42
    assert signature(generate_puzzle(0)) == signature(puzzles[0])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(123456, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    pair = bad["test"][0]
    p = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][p] = pair["input"][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing completion pixel accepted")
    return dict(seeds=42, pairs=len(rows), shapes=len(shapes), structural_layouts=len(layouts),
                ordered_palettes=len(palettes), lengths=sorted({f["length"] for f in rows}),
                marker_counts=sorted({f["markers"] for f in rows}),
                offsets=sorted({tuple(f["offset"]) for f in rows if f["offset"]}),
                reentering_rails=sum(f["reentry"] for f in rows),
                outside_source_echo_cells=sum(f["outside_echo"] for f in rows),
                clip_order_differences=sum(f["clip_difference"] for f in rows),
                max_stage=max(max(f["stages"]) for f in rows))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
