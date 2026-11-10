"""Recover a run-expanded canvas, then exchange glyph and segment arrangements.

The decoder and solver use input pixels only. Construction samples geometry
directly; it neither imports nor invokes either parent's puzzle generator.
"""

from itertools import combinations
import hashlib

import numpy as np


_MOTIFS = (
    frozenset({(0, 0)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}),
    frozenset({(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}),
    frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)}),
    # Adjacent columns of this wide motif differ; its width survives collapse.
    frozenset({(-1, -2), (-1, 0), (-1, 2), (0, -2), (0, -1),
               (0, 0), (0, 1), (0, 2), (1, -1), (1, 1)}),
    frozenset({(-2, -1), (0, -1), (2, -1), (-2, 0), (-1, 0),
               (0, 0), (1, 0), (2, 0), (-1, 1), (1, 1)}),
)
_FEATURED = (3, 4, 5)
_HOST_SEEDS = (
    0, 39023253, 152913964, 680990111, 405001883, 46988787,
    550549174, 178642700, 589812765, 129958666, 16463233, 884855421,
    89531838, 917571569, 312669528, 644629403, 180632569, 830039298,
    838791098, 781663505, 844250761, 352719045, 1181669, 674075218,
    59778038, 875866279, 569595023, 504876432, 169555138, 4869288,
    963098006, 952155907, 542058575, 599110676, 329285844, 426888828,
    770256870, 103169252, 712811913, 994480874, 143338269, 30003855,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10,
             "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4,
             "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid colors must be 0..9")
    return grid.astype(np.int8, copy=False)


def _collapse(grid):
    """Collapse maximal adjacent equal vectors independently, without cropping.

    Separated recurrences survive. Comparing full vectors, rather than local
    monochrome patches, retains seams exposed elsewhere in the canvas.
    """
    rows = np.r_[0, np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0)) + 1]
    scene = grid[np.ix_(rows, cols)].copy()
    heights = np.diff(np.r_[rows, grid.shape[0]])
    widths = np.diff(np.r_[cols, grid.shape[1]])
    return scene, heights, widths


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr, dc) != (0, 0), "line needs distinct endpoints")
    _require(dr == 0 or dc == 0 or abs(dr) == abs(dc), "unsupported line slope")
    length = max(abs(dr), abs(dc))
    sr = (dr > 0) - (dr < 0)
    sc = (dc > 0) - (dc < 0)
    return frozenset((a[0] + i * sr, a[1] + i * sc) for i in range(length + 1))


def _components(cells):
    unseen = set(cells)
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (r + dr, c + dc)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        groups.append(frozenset(component))
    return tuple(groups)


def _straight(cells):
    _require(len(cells) >= 4, "source segment must contain at least four cells")
    # For a valid allowed line, lexicographic extrema are its endpoints.
    endpoints = min(cells), max(cells)
    _require(_line(*endpoints) == cells, "source segment is not a complete straight line")
    return endpoints


def _glyphs(cells):
    groups = _components(cells)
    _require(len(groups) in (2, 3), "need two or three glyph components")
    motifs, centers = [], []
    for group in groups:
        rs, cs = zip(*group)
        _require((min(rs) + max(rs)) % 2 == (min(cs) + max(cs)) % 2 == 0,
                 "glyph bounding box must have odd dimensions")
        center = (min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2
        motif = frozenset((r - center[0], c - center[1]) for r, c in group)
        _require((0, 0) in motif, "glyph center must be occupied")
        centers.append(center)
        motifs.append(motif)
    _require(all(m == motifs[0] for m in motifs), "glyphs must be unchanged translated copies")
    centers = tuple(sorted(centers))
    for a, b in combinations(centers, 2):
        _line(a, b)
    if len(centers) == 3:
        a, b, c = centers
        _require((b[0] - a[0]) * (c[1] - a[1]) !=
                 (b[1] - a[1]) * (c[0] - a[0]), "three centers must be noncollinear")
        # Pairwise H/V/45-degree edges and noncollinearity imply a right
        # isosceles triangle, in an axial or diagonal basis.
    return centers, motifs[0]


def _stamps(centers, motif):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in motif)


def _figure(centers):
    return frozenset(cell for a, b in combinations(centers, 2) for cell in _line(a, b))


def _separated(first, second):
    return all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
               for a in first for b in second)


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _analyze(scene):
    """Parse a compact scene, with no generator metadata or motif lookup."""
    bg = int(scene[0, 0])
    _require(bool(np.all(scene[0] == bg) and np.all(scene[-1] == bg)
                  and np.all(scene[:, 0] == bg) and np.all(scene[:, -1] == bg)),
             "input scene must have a uniform background border")
    palette = [int(c) for c in np.unique(scene)]
    _require(len(palette) == 3, "scene must have background and two foreground colors")
    foreground = [c for c in palette if c != bg]
    cells = {c: frozenset(map(tuple, np.argwhere(scene == c).tolist())) for c in foreground}
    candidates = []
    for edge, ink in (foreground, foreground[::-1]):
        try:
            endpoints = _straight(cells[edge])
            centers, motif = _glyphs(cells[ink])
            _require(_separated(cells[edge], cells[ink]), "input roles touch")
            figure = _figure(centers)
            stamps = _stamps(endpoints, motif)
            _require(_inside(stamps, scene.shape), "endpoint glyph would be clipped")
            _require(len(stamps) == 2 * len(motif) and len(_components(stamps)) == 2,
                     "endpoint copies would overlap or touch")
            _require(stamps.isdisjoint(figure), "output roles would collide")
            candidates.append({"background": bg, "edge": edge, "ink": ink,
                               "endpoints": endpoints, "centers": centers,
                               "motif": motif, "figure": figure, "stamps": stamps})
        except ValueError:
            continue
    _require(len(candidates) == 1, "foreground roles must have a unique structural interpretation")
    return candidates[0]


def _exchange(scene, analysis):
    output = np.full(scene.shape, analysis["background"], dtype=np.int8)
    for cell in analysis["figure"]:
        output[cell] = analysis["edge"]
    for cell in analysis["stamps"]:
        output[cell] = analysis["ink"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the compact coordinate system before reading centers or endpoints."""
    scene, _, _ = _collapse(_grid(input_grid))
    return _exchange(scene, _analyze(scene))


def _orientation(endpoints):
    a, b = endpoints
    return "horizontal" if a[0] == b[0] else "vertical" if a[1] == b[1] else "diagonal"


def _place(rng, points, motif, shape):
    """Translate center locations, leaving a one-cell input background border."""
    placed = []
    for axis in (0, 1):
        low = 1 - min(p[axis] for p in points) - min(p[axis] for p in motif)
        high = shape[axis] - 2 - max(p[axis] for p in points) - max(p[axis] for p in motif)
        if high < low:
            return None
        placed.append(int(rng.integers(low, high + 1)))
    return tuple((r + placed[0], c + placed[1]) for r, c in points)


def _sample_centers(rng, count, motif, shape):
    distance = int(rng.integers(4, 14))
    if count == 2:
        dr, dc = ((0, 1), (1, 0), (1, 1), (1, -1))[int(rng.integers(4))]
        relative = ((0, 0), (distance * dr, distance * dc))
    elif rng.random() < 0.5:
        relative = ((0, 0), (0, distance), (distance, 0))
    else:
        relative = ((0, 0), (distance, distance), (distance, -distance))
    turns, reflected = int(rng.integers(4)), bool(rng.integers(2))
    transformed = []
    for r, c in relative:
        if reflected:
            c = -c
        for _ in range(turns):
            r, c = c, -r
        transformed.append((r, c))
    return _place(rng, transformed, motif, shape)


def _sample_scene(rng, count, motif, orientation):
    """Generate coordinates, normalize redundant slack, reject lost geometry.

    Normalization is only a sampling convenience: the accepted logical scene
    itself has no adjacent equal full vectors. No invisible grid divisions are
    ever retained. The recovered motif must still equal the sampled motif.
    """
    for _ in range(6000):
        shape = tuple(map(int, rng.integers(13, 26, size=2)))
        centers = _sample_centers(rng, count, motif, shape)
        distance = int(rng.integers(4, 18))
        dr, dc = {"horizontal": (0, 1), "vertical": (1, 0),
                  "diagonal": (1, 1 if rng.random() < 0.5 else -1)}[orientation]
        endpoints = _place(rng, ((0, 0), (distance * dr, distance * dc)), motif, shape)
        if centers is None or endpoints is None:
            continue
        edge, glyphs = _line(*endpoints), _stamps(centers, motif)
        if len(glyphs) != count * len(motif) or not _separated(edge, glyphs):
            continue
        raw = np.zeros(shape, dtype=np.int8)
        for cell in edge:
            raw[cell] = 1
        for cell in glyphs:
            raw[cell] = 2
        scene, _, _ = _collapse(raw)
        try:
            a = _analyze(scene)
        except ValueError:
            continue
        if (a["edge"] == 1 and a["ink"] == 2 and len(a["centers"]) == count
                and a["motif"] == motif and _orientation(a["endpoints"]) == orientation):
            return scene
    raise RuntimeError("bounded geometry sampling exhausted")


def _axis_profile(rng, size, mode):
    """Positive factors with a bounded total; no saved intermediate arrays."""
    bands = np.ones(size, dtype=int)
    if mode == "peak":
        peak = int(rng.integers(size))
        bands[peak] = int(rng.integers(4, min(7, 32 - size)))
        budget = min(30 - int(bands.sum()), int(rng.integers(2, 8)))
        for _ in range(budget):
            bands[int(rng.integers(size))] += 1
    else:
        budget = int(rng.integers(2, min(30 - size, size + 5) + 1))
        cap = 3 if mode == "mild" else 4
        for _ in range(budget):
            eligible = np.flatnonzero(bands < cap)
            if not len(eligible):
                break
            bands[int(rng.choice(eligible))] += 1
    return bands


def _physical_motifs(scene, analysis, rows, cols):
    """Bounding-box patterns after expansion, used only to guarantee a lesson."""
    rr, cc = np.r_[0, np.cumsum(rows)], np.r_[0, np.cumsum(cols)]
    expanded = np.repeat(np.repeat(scene, rows, axis=0), cols, axis=1)
    drs, dcs = zip(*analysis["motif"])
    result = set()
    for r, c in analysis["centers"]:
        box = expanded[rr[r + min(drs)]:rr[r + max(drs) + 1],
                       cc[c + min(dcs)]:cc[c + max(dcs) + 1]]
        mask = box == analysis["ink"]
        result.add((mask.shape, mask.tobytes()))
    return result


def _expand(rng, scene, mode, teach_distortion=False):
    analysis = _analyze(scene)
    for _ in range(100):
        rows = _axis_profile(rng, scene.shape[0], mode)
        cols = _axis_profile(rng, scene.shape[1], mode)
        if teach_distortion and len(_physical_motifs(scene, analysis, rows, cols)) < 2:
            continue
        return np.repeat(np.repeat(scene, rows, axis=0), cols, axis=1)
    raise RuntimeError("bounded expansion sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one deterministic puzzle with local geometric color roles."""
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    featured = _MOTIFS[int(rng.choice(_FEATURED))]
    other = _MOTIFS[int(rng.choice((0, 1, 2)))]
    counts = [2, 3]
    rng.shuffle(counts)
    directions = list(rng.choice(("horizontal", "vertical", "diagonal"), size=2, replace=False))
    base = tuple(map(int, rng.choice(10, size=3, replace=False)))
    shared_bg = bool(rng.integers(2))
    # Half the puzzles shift to stronger bands; the rest to a new combination
    # of a taught motif and the other center count, at comparable band scales.
    stronger_test = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_direction = None
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            if split == "train" and i < 2:
                count = counts[i]
                motif = featured if i == 0 else other
                direction = str(directions[i])
            elif split == "test" and not stronger_test:
                count, motif = counts[1], featured
                direction = str(rng.choice(directions))
            else:
                count = int(rng.integers(2, 4))
                motif = (featured, other)[int(rng.integers(2))]
                direction = str(rng.choice(("horizontal", "vertical", "diagonal")))
                if split == "train" and not stronger_test and motif == featured:
                    count = counts[0]  # Reserve the motif/count recombination.
            if split == "train" and i == 0:
                palette = base
            elif split == "train" and i == 1:
                palette = (base[0], base[2], base[1])
            elif rng.random() < 0.4:
                palette = base if rng.random() < 0.5 else (base[0], base[2], base[1])
            else:
                bg = base[0] if shared_bg else int(rng.integers(10))
                foreground = rng.choice([c for c in range(10) if c != bg], size=2, replace=False)
                palette = (bg, int(foreground[0]), int(foreground[1]))
            mode = "peak" if stronger_test and split == "test" else "mild" if stronger_test else "mixed"
            for _ in range(32):
                # Some large-motif arrangements cannot survive canonical
                # compression with room for the endpoint copies. Try other
                # directions rather than impose invisible spacing or clip.
                choices = [direction] + [str(d) for d in rng.permutation(
                    ("horizontal", "vertical", "diagonal")) if d != direction]
                if split == "train" and i == 1:
                    choices = [d for d in choices if d != first_direction]
                for actual_direction in choices:
                    try:
                        scene = _sample_scene(rng, count, motif, actual_direction)
                        break
                    except RuntimeError:
                        continue
                else:
                    raise RuntimeError("bounded compatible-direction sampling exhausted")
                scene = np.asarray(palette, dtype=np.int8)[scene]
                source = _expand(rng, scene, mode, teach_distortion=split == "train" and i < 2)
                signature = (source.shape, source.tobytes())
                if signature in seen:
                    continue
                output = _exchange(scene, _analyze(scene))
                puzzle[split].append({"input": source, "output": output})
                if split == "train" and i == 0:
                    first_direction = actual_direction
                seen.add(signature)
                break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate semantics and teaching evidence directly from grids, not seeds."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            source, output = _grid(pair["input"]), _grid(pair["output"])
            key = source.shape, source.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            scene, rows, cols = _collapse(source)
            _require(np.array_equal(np.repeat(np.repeat(scene, rows, axis=0), cols, axis=1), source),
                     "input is not exact row/column replication")
            again, _, _ = _collapse(scene)
            _require(np.array_equal(again, scene), "compact scene has an invisible boundary")
            _require(max(rows) > 1 and max(cols) > 1, "both collapse axes must affect the answer")
            _require(len(set(rows)) > 1 and len(set(cols)) > 1,
                     "profiles must expose independent nonuniform scaling")
            a = _analyze(scene)
            _require(output.shape == scene.shape, "must preserve the entire compact canvas")
            _require(np.array_equal(output, _exchange(scene, a)), "wrong arrangement exchange")
            # A-only and B-only ablations both fail for every generated pair.
            _require(not np.array_equal(output, scene), "omitting exchange must change the answer")
            _require(source.shape[0] > output.shape[0] and source.shape[1] > output.shape[1],
                     "omitting either collapse axis must change output dimensions")
            for p in a["centers"]:
                _require(scene[p] == a["ink"] and output[p] == a["edge"], "old center must change role")
            for p in a["endpoints"]:
                _require(scene[p] == a["edge"] and output[p] == a["ink"], "old endpoint must change role")
            # Background cells outside the new structures stay background.
            fg = set(map(tuple, np.argwhere(output != a["background"]).tolist()))
            _require(fg == a["figure"] | a["stamps"], "stray output pixels or unerased structures")
            a["distorted"] = len(_physical_motifs(scene, a, rows, cols)) > 1
            a["max_band"] = int(max(max(rows), max(cols)))
            (training if split == "train" else testing).append(a)
    _require({len(a["centers"]) for a in training} == {2, 3}, "teach both center-count branches")
    _require(len({_orientation(a["endpoints"]) for a in training}) >= 2, "teach multiple segment directions")
    _require(len({(a["background"], a["edge"], a["ink"]) for a in training}) >= 2, "teach local color roles")
    _require(sum(a["distorted"] for a in training) >= 2, "teach global recovery from unequal pixel glyphs")
    training_motifs = {a["motif"] for a in training}
    _require(any(len(m) > 1 and frozenset((-r, -c) for r, c in m) != m for m in training_motifs),
             "teach unchanged asymmetric glyph copying")
    _require(all(a["motif"] in training_motifs for a in testing), "test glyphs must have copy demonstrations")
    scale_shift = (all(a["max_band"] <= 3 for a in training)
                   and all(a["max_band"] >= 4 for a in testing))
    trained_combinations = {(a["motif"], len(a["centers"])) for a in training}
    recombination_shift = all((a["motif"], len(a["centers"])) not in trained_combinations
                              for a in testing)
    _require(scale_shift or recombination_shift, "tests must exercise a visible structural shift")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _fixture_test():
    """A known geometric answer with a motif absent from the sampling catalog."""
    motif = frozenset({(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, -1), (1, 0)})
    assert motif not in _MOTIFS
    scene = np.zeros((11, 15), dtype=np.int8)
    for c in range(1, 14):
        scene[9, c] = 1
    for r, c in ((2, 7), (6, 3), (6, 11)):
        for dr, dc in motif:
            scene[r + dr, c + dc] = 8
    # Build the known triangle answer explicitly, without _line/_figure or
    # the generator's answer-building path.
    expected = np.zeros_like(scene)
    for i in range(5):
        expected[2 + i, 7 - i] = 1
        expected[2 + i, 7 + i] = 1
    expected[6, 3:12] = 1
    for r, c in ((9, 1), (9, 13)):
        for dr, dc in motif:
            expected[r + dr, c + dc] = 8
    for rows, cols in ((np.ones(11, dtype=int), np.ones(15, dtype=int)),
                       (np.full(11, 2), np.full(15, 2)),
                       (np.array([1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 2]),
                        np.array([2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 2, 1, 3, 1, 2]))):
        source = np.repeat(np.repeat(scene, rows, axis=0), cols, axis=1)
        before = source.copy()
        assert np.array_equal(solve(source), expected)
        assert np.array_equal(source, before), "solve must not modify its input"
        assert np.array_equal(solve(source.T), expected.T)
        palette = np.roll(np.arange(10, dtype=np.int8), 5)
        assert np.array_equal(solve(palette[source]), palette[expected])


def self_test():
    """Bounded 42-seed suite, all count combinations, and semantic witnesses.

    Retains only hashes/counters; writes no payloads, logs, caches or traces.
    Returns descriptive coverage, which is not independent semantic approval.
    """
    _fixture_test()
    puzzles, shapes, palettes, geometries, profiles = set(), set(), set(), set(), set()
    split_counts = {s: set() for s in ("train", "test")}
    split_directions = {s: set() for s in ("train", "test")}
    boxes, backgrounds, peak_tests = set(), set(), 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene, rows, cols = _collapse(pair["input"])
                a = _analyze(scene)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add((pair["input"].shape, scene.shape))
                palettes.add((a["background"], a["edge"], a["ink"]))
                backgrounds.add(a["background"])
                profiles.add((tuple(rows), tuple(cols)))
                split_counts[split].add(len(a["centers"]))
                split_directions[split].add(_orientation(a["endpoints"]))
                mr, mc = zip(*a["motif"])
                boxes.add((max(mr) - min(mr) + 1, max(mc) - min(mc) + 1))
                # Color-free geometric diversity after all expansion is removed.
                geometries.add((scene.shape, a["endpoints"], a["centers"], a["motif"]))
                if split == "test":
                    peak_tests += int(max(max(rows), max(cols)) >= 4)
                # Separated all-background border vectors are equal but survive.
                assert np.array_equal(scene[0], scene[-1]) and scene.shape[0] > 2
                assert np.array_equal(scene[:, 0], scene[:, -1]) and scene.shape[1] > 2
    assert len(puzzles) == 42 and len(geometries) >= 120
    assert len(shapes) >= 70 and len(profiles) >= 150 and len(palettes) >= 50
    assert len(backgrounds) == 10
    assert all(v == {2, 3} for v in split_counts.values())
    assert all(len(v) == 3 for v in split_directions.values())
    assert {(1, 1), (3, 3), (3, 5), (5, 3)} <= boxes
    assert peak_tests > 0
    first = generate_puzzle(0)
    assert _signature(first) == _signature(generate_puzzle(0))
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(761 + 10 * n + t, n, t)
            assert len(p["train"]) == n and len(p["test"]) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutation checks: dimensions, source erasure and orientation copying matter.
    for replacement in (first["train"][0]["input"], _collapse(first["train"][0]["input"])[0]):
        bad = {s: [dict(pair) for pair in first[s]] for s in ("train", "test")}
        bad["train"][0]["output"] = replacement
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("parent-omission mutation accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzles),
            "geometries": len(geometries), "shape_combinations": len(shapes),
            "expansion_profiles": len(profiles), "palettes": len(palettes),
            "background_colors": len(backgrounds), "motif_boxes": sorted(boxes),
            "peak_test_pairs": peak_tests, "independent_solver": False}
