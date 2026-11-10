"""Shape-key docking: rotation-matched temporary joints assemble a rooted union.

Only standard-library modules and NumPy are used. Nothing is written by this
module, including self_test(); all diagnostics are small in-memory aggregates.
"""
from collections import deque
from hashlib import sha256
import json

import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
BG, KEY, ROOT = 0, 8, 9


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR8:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _shift(cells, offset):
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _rotate(cells, turn):
    result = set(cells)
    for _ in range(turn % 4):
        result = {(-c, r) for r, c in result}
    return result


def _minimum(cells):
    return min(r for r, c in cells), min(c for r, c in cells)


def _normal(cells):
    r, c = _minimum(cells)
    return tuple(sorted((y - r, x - c) for y, x in cells))


def _canonical(cells, reflected=False):
    variants = [_normal(_rotate(cells, k)) for k in range(4)]
    if reflected:
        variants += [_normal(_rotate({(r, -c) for r, c in cells}, k)) for k in range(4)]
    return min(variants)


def _crop(body, root, color):
    cells = body | root
    top, left = _minimum(cells)
    h = max(r for r, c in cells) - top + 1
    w = max(c for r, c in cells) - left + 1
    if max(h, w) > 30:
        raise ValueError("assembled crop exceeds ARC dimensions")
    result = np.zeros((h, w), dtype=np.int8)
    for r, c in body:
        result[r - top, c - left] = color
    for r, c in root:
        result[r - top, c - left] = ROOT
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a


def _parse(input_grid):
    a = _grid(input_grid)
    colors = set(map(int, np.unique(a)))
    ordinary = colors - {BG, KEY, ROOT}
    if len(ordinary) != 1 or not {BG, KEY, ROOT} <= colors:
        raise ValueError("expected background 0, keys 8, one anchor 9 and one body color")
    root_cells = {tuple(map(int, q)) for q in np.argwhere(a == ROOT)}
    if len(root_cells) != 1:
        raise ValueError("exactly one anchor pixel is required")
    pieces, owner = [], None
    for cells in _components(map(tuple, np.argwhere(a != BG))):
        ports = _components(q for q in cells if a[q] == KEY)
        if not ports or any(len({_normal(_rotate(p, k)) for k in range(4)}) != 4 for p in ports):
            raise ValueError("every fragment needs asymmetric key shapes")
        body = {q for q in cells if a[q] not in (BG, KEY)}
        if not body:
            raise ValueError("a fragment needs ordinary pixels")
        if cells & root_cells:
            owner = len(pieces)
        pieces.append({"ports": ports, "body": body})
    if owner is None or not 2 <= len(pieces) <= 4:
        raise ValueError("expected two through four fragments and an anchored fragment")
    return {"pieces": pieces, "owner": owner, "root": root_cells, "color": ordinary.pop()}


def _assemble(scene, mode="correct"):
    """Recover poses from input pixels. Mutant modes are diagnostic hypotheses."""
    pieces = scene["pieces"]
    groups = {}
    for i, piece in enumerate(pieces):
        for j, port in enumerate(piece["ports"]):
            if mode == "identity":
                key = _normal(port)
            elif mode == "area":
                key = len(port)
            else:
                key = _canonical(port, mode == "reflection")
            groups.setdefault(key, []).append((i, j))
    edges = [[] for _ in pieces]
    matched = 0
    for members in groups.values():
        if len(members) > 2:
            raise ValueError("a key has more than one possible mate")
        if len(members) == 2:
            (i, j), (u, v) = members
            if i == u:
                raise ValueError("a joint cannot join a fragment to itself")
            edges[i].append((j, u, v))
            edges[u].append((v, i, j))
            matched += 1
    poses = {scene["owner"]: (0, False, (0, 0))}
    queue = deque([scene["owner"]])
    order = []

    def transform(cells, pose):
        turn, flip, offset = pose
        if flip:
            cells = {(r, -c) for r, c in cells}
        return _shift(_rotate(cells, turn), offset)

    while queue:
        i = queue.popleft()
        order.append(i)
        for j, u, v in sorted(edges[i]):
            target = transform(pieces[i]["ports"][j], poses[i])
            source = pieces[u]["ports"][v]
            candidates = []
            for flip in ((False, True) if mode == "reflection" else (False,)):
                raw = {(r, -c) for r, c in source} if flip else source
                for turn in range(4):
                    rotated = _rotate(raw, turn)
                    if _normal(rotated) != _normal(target) and mode != "area":
                        continue
                    tr, tc = _minimum(target)
                    sr, sc = _minimum(rotated)
                    candidates.append((turn, flip, (tr - sr, tc - sc)))
            if not candidates:
                raise ValueError("joint masks do not register")
            pose = candidates[0]
            if mode == "no_rotation":
                tr, tc = _minimum(target)
                sr, sc = _minimum(source)
                pose = (0, False, (tr - sr, tc - sc))
            if mode == "centers":
                source_body = _rotate(pieces[u]["body"], pose[0])
                parent_body = transform(pieces[i]["body"], poses[i])
                tr, tc = _minimum(parent_body)
                sr, sc = _minimum(source_body)
                pose = (pose[0], False, (tr - sr, tc - sc))
            if u in poses:
                # The correct family is a tree. A false reflected match can add
                # a contradictory closing edge, which invalidates that hypothesis.
                if mode in ("correct", "reflection") and transform(source, poses[u]) != target:
                    raise ValueError("inconsistent docking constraints")
                continue
            poses[u] = pose
            queue.append(u)
    if mode == "correct" and (len(poses) != len(pieces) or matched != len(pieces) - 1):
        raise ValueError("matched keys must form one tree")
    body, guides = set(), set()
    total = 0
    for i in order:
        new = transform(pieces[i]["body"], poses[i])
        total += len(new)
        guides |= set.union(*(transform(p, poses[i]) for p in pieces[i]["ports"]))
        if mode == "xor":
            body ^= new
        elif mode == "opaque":
            r0, c0 = _minimum(new)
            r1, c1 = max(r for r, c in new), max(c for r, c in new)
            body = {q for q in body if not (r0 <= q[0] <= r1 and c0 <= q[1] <= c1)} | new
        else:
            body |= new
    root = scene["root"]
    info = {"fragments": len(pieces), "matched_classes": matched,
            "singleton_keys": sum(len(x) == 1 for x in groups.values()),
            "overlap": total - len(body), "covered_guides": len(guides & body),
            "uncovered_guides": len(guides - body),
            "root_degree": len(edges[scene["owner"]]),
            "relative_turns": sorted({p[0] for p in poses.values()}),
            "key_areas": sorted({len(p) for x in pieces for p in x["ports"]})}
    if mode == "paint":
        body |= guides
    elif mode == "erase":
        body -= guides
    return _crop(body, root, scene["color"]), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Dock exact rotation-congruent keys, union bodies, erase keys, tight crop."""
    return _assemble(_parse(input_grid))[0]


def _alternative(scene, mode):
    try:
        return _assemble(scene, mode)[0]
    except ValueError:
        return None


def _key_vocabulary():
    # A finite procedural vocabulary, not saved puzzles or examples. Every key
    # has a 2x3 bounding box, is eight-connected and has a unique rotational pose.
    result = {}
    for bits in range(1, 64):
        cells = {(i // 3, i % 3) for i in range(6) if bits & (1 << i)}
        if not 3 <= len(cells) <= 5 or len(_components(cells)) != 1:
            continue
        if max(r for r, c in cells) != 1 or max(c for r, c in cells) != 2 or _minimum(cells) != (0, 0):
            continue
        if len({_normal(_rotate(cells, k)) for k in range(4)}) == 4:
            result.setdefault(_canonical(cells), cells)
    return [result[k] for k in sorted(result)]


def _port_origins(h, w):
    return ((0, (w - 3) // 2), ((h - 2) // 2, w - 3),
            (h - 2, (w - 3) // 2), ((h - 2) // 2, 0))


def _payload(rng, h, w, ports, style):
    hub = h // 2, w // 2
    cells = {hub}
    # All port pixels attach to a common skeleton, allowing hollow bodies,
    # branches, stepped outlines, filled masses and irregular lobes alike.
    for port in ports:
        for r, c in port:
            cells.update((y, c) for y in range(min(r, hub[0]), max(r, hub[0]) + 1))
            cells.update((hub[0], x) for x in range(min(c, hub[1]), max(c, hub[1]) + 1))
    if style == "frame":
        cells |= {(r, c) for r in range(h) for c in range(w)
                  if r in (0, h - 1) or c in (0, w - 1)}
    elif style == "mass":
        cells |= {(r, c) for r in range(h) for c in range(w)
                  if not (r < h // 3 and c < w // 3)}
    elif style == "steps":
        for r in range(h):
            c = min(w - 1, r * w // h)
            cells.add((r, c))
            cells.add((r, max(0, c - 1)))
        cells |= {(hub[0], c) for c in range(w)}
    elif style == "branches":
        for r in range(h):
            cells.add((r, hub[1]))
            if r % 2 == 0:
                cells.update((r, c) for c in range(int(rng.integers(hub[1] + 1)),
                                                  int(rng.integers(hub[1], w)) + 1))
    else:
        for _ in range(2 * h * w):
            r, c = sorted(cells)[int(rng.integers(len(cells)))]
            dr, dc = DIR8[int(rng.integers(len(DIR8)))]
            if 0 <= r + dr < h and 0 <= c + dc < w:
                cells.add((r + dr, c + dc))
    cells |= set.union(*ports)
    # Keep the hub component only, retaining every key by its skeleton paths.
    cells = next(p for p in _components(cells) if hub in p)
    return cells


def _stage(rng, pieces):
    arrays, turns = [], []
    for a in pieces:
        turn = int(rng.integers(4))
        turns.append(turn)
        a = np.rot90(a, turn).copy()
        occupied = np.argwhere(a != BG)
        lo, hi = occupied.min(0), occupied.max(0) + 1
        arrays.append(a[lo[0]:hi[0], lo[1]:hi[1]])
    n = len(arrays)
    columns = int(rng.choice([1, 2, n]))
    rows = (n + columns - 1) // columns
    if rows * 10 + 1 > 30 or columns * 10 + 1 > 30:
        columns, rows = 2, (n + 1) // 2
    order = list(map(int, rng.permutation(n)))
    heights = [max(arrays[order[j]].shape[0] for j in range(i * columns, min(n, (i + 1) * columns)))
               for i in range(rows)]
    widths = [max(arrays[order[j]].shape[1] for j in range(c, n, columns)) for c in range(columns)]
    gaps_r = [int(rng.integers(1, 3)) for _ in range(rows - 1)]
    gaps_c = [int(rng.integers(1, 3)) for _ in range(columns - 1)]
    pads = list(map(int, rng.integers(0, 3, 4)))
    h, w = sum(heights) + sum(gaps_r), sum(widths) + sum(gaps_c)
    pads[1] = min(pads[1], 30 - h - pads[0])
    pads[3] = min(pads[3], 30 - w - pads[2])
    if min(pads) < 0:
        raise ValueError("staging does not fit")
    grid = np.zeros((h + pads[0] + pads[1], w + pads[2] + pads[3]), dtype=np.int8)
    y = pads[0]
    for row in range(rows):
        x = pads[2]
        for col in range(columns):
            j = row * columns + col
            if j < n:
                a = arrays[order[j]]
                dy = int(rng.integers(heights[row] - a.shape[0] + 1))
                dx = int(rng.integers(widths[col] - a.shape[1] + 1))
                grid[y + dy:y + dy + a.shape[0], x + dx:x + dx + a.shape[1]] = a
            if col < columns - 1:
                x += widths[col] + gaps_c[col]
        if row < rows - 1:
            y += heights[row] + gaps_r[row]
    return grid, turns[0]


def _make_pair_once(rng, n, color, caps, branch):
    vocabulary = _key_vocabulary()
    chosen = [vocabulary[int(i)] for i in rng.permutation(len(vocabulary))]
    edge_masks = chosen[:n - 1]
    used = {_canonical(p) for p in edge_masks}
    cap_masks = []
    if caps == 2:
        candidates = [p for p in chosen if _canonical(p) not in used
                      and _canonical({(r, -c) for r, c in p}) not in used
                      and _canonical(p) != _canonical({(r, -c) for r, c in p})]
        cap = candidates[0]
        cap_masks = [cap, set(_normal({(r, -c) for r, c in cap}))]
    elif caps:
        cap_masks = [next(p for p in chosen if _canonical(p) not in used)]
    sizes = [(int(rng.integers(8, 10)), int(rng.integers(8, 10))) for _ in range(n)]
    ports = [[] for _ in range(n)]
    sides = [set() for _ in range(n)]
    origins = [(0, 0)] * n
    for child in range(1, n):
        parent = 0 if branch else child - 1
        side = int(rng.choice([s for s in range(4) if s not in sides[parent]]))
        other = (side + 2) % 4
        sides[parent].add(side)
        sides[child].add(other)
        a = _shift(edge_masks[child - 1], _port_origins(*sizes[parent])[side])
        b = _shift(edge_masks[child - 1], _port_origins(*sizes[child])[other])
        ports[parent].append(a)
        ports[child].append(b)
        ar, ac = _minimum(_shift(a, origins[parent]))
        br, bc = _minimum(b)
        origins[child] = ar - br, ac - bc
    for cap in cap_masks:
        node = int(rng.choice([i for i in range(n) if len(sides[i]) < 4]))
        side = int(rng.choice([s for s in range(4) if s not in sides[node]]))
        sides[node].add(side)
        ports[node].append(_shift(cap, _port_origins(*sizes[node])[side]))
    bodies, grids = [], []
    root = None
    styles = ("frame", "mass", "steps", "branches", "growth")
    for i, (h, w) in enumerate(sizes):
        full = _payload(rng, h, w, ports[i], str(rng.choice(styles)))
        key_cells = set.union(*ports[i])
        body = full - key_cells
        a = np.zeros((h, w), dtype=np.int8)
        for q in body:
            a[q] = color
        for q in key_cells:
            a[q] = KEY
        if i == 0:
            options = sorted(body)
            anchor = options[int(rng.integers(len(options)))]
            a[anchor] = ROOT
            root = _shift({anchor}, origins[0])
        bodies.append(_shift(body, origins[i]))
        grids.append(a)
    # Labels come from the constructed world coordinates, independently of the
    # input parser and docking traversal used by solve(). Root orientation is
    # recovered from its staged key geometry, so rotate the expected world too.
    input_grid, turn = _stage(rng, grids)
    expected = _crop(_rotate(set.union(*bodies), turn), _rotate(root, turn), color)
    return {"input": input_grid, "output": expected}


MUTANTS = ("identity", "area", "reflection", "no_rotation", "centers", "paint", "erase", "xor", "opaque")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < lo or (hi is not None and int(value) > hi)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _count(seed, "seed", 0, None)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    fixed_color = int(rng.integers(1, 8)) if rng.random() < .4 else None
    puzzle, seen, excluded = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            n = (2 if index == 0 else 3) if split == "train" else 4
            for attempt in range(180):
                color = fixed_color or int(rng.integers(1, 8))
                caps = 2 if split == "train" and index == 0 else int(rng.integers(3))
                try:
                    branch = n >= 3 and bool(rng.integers(2))
                    if split == "train" and index == 1:
                        branch = False  # Teach a pose inherited through an intermediate piece.
                    pair = _make_pair_once(rng, n, color, caps, branch)
                    scene = _parse(pair["input"])
                    predicted, info = _assemble(scene)
                except (ValueError, IndexError):
                    continue
                if not np.array_equal(predicted, pair["output"]):
                    raise AssertionError("constructed labels disagree with pixel docking")
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature in seen:
                    continue
                rejects = {m for m in MUTANTS if not np.array_equal(_alternative(scene, m), predicted)}
                if split == "train":
                    if index == 0 and not {"identity", "area", "reflection", "no_rotation", "centers", "paint", "xor", "opaque"} <= rejects:
                        continue
                    if index == 1 and not {"erase"} <= rejects:
                        continue
                    excluded |= rejects
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded docking construction exhausted")
    assert set(MUTANTS) <= excluded
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, exclusions, infos = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            signature = a.shape, a.tobytes()
            assert signature not in seen
            seen.add(signature)
            scene = _parse(a)
            expected, info = _assemble(scene)
            assert np.array_equal(b, expected), "incorrect docking, union, marker handling or crop"
            assert not np.array_equal(a, b)
            assert int(np.sum(b == ROOT)) == 1 and not np.any(b == KEY)
            assert np.any(b[0]) and np.any(b[-1]) and np.any(b[:, 0]) and np.any(b[:, -1])
            infos[split].append(info)
            if split == "train":
                exclusions |= {m for m in MUTANTS if not np.array_equal(_alternative(scene, m), expected)}
    assert set(MUTANTS) <= exclusions, "training fails to distinguish a named simpler alternative"
    assert {2, 3} <= {i["fragments"] for i in infos["train"]}
    assert all(i["fragments"] == 4 for i in infos["test"])
    assert any(i["covered_guides"] for i in infos["train"])
    assert any(i["singleton_keys"] >= 2 for i in infos["train"])
    assert infos["train"][1]["fragments"] == 3 and infos["train"][1]["root_degree"] == 1
    return None


def _fingerprint(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    fingerprints, inputs, outputs, topologies, colors, keys = set(), set(), set(), set(), set(), set()
    covered = overlap = singleton = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = _fingerprint(puzzle)
        assert fingerprint == _fingerprint(generate_puzzle(seed))
        fingerprints.add(fingerprint)
        for split in puzzle:
            for pair in puzzle[split]:
                scene = _parse(pair["input"])
                predicted, info = _assemble(scene)
                assert np.array_equal(solve(pair["input"].tolist()), predicted)
                inputs.add(pair["input"].shape)
                outputs.add(predicted.shape)
                colors.add(scene["color"])
                keys.update(_canonical(p) for x in scene["pieces"] for p in x["ports"])
                topologies.add((info["fragments"], info["root_degree"], info["singleton_keys"]))
                covered += info["covered_guides"] > 0
                overlap += info["overlap"] > 0
                singleton += info["singleton_keys"] > 0
                if seed < 3:
                    for turn in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(predicted, turn))
                    padded = np.pad(pair["input"], ((1, 0), (0, 1)))
                    assert np.array_equal(solve(padded), predicted)
    assert len(fingerprints) == 42 and len(inputs) >= 15 and len(outputs) >= 20
    assert len(topologies) >= 8 and colors == set(range(1, 8)) and len(keys) >= 8
    assert covered >= 42 and overlap >= 100 and singleton >= 42
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            puzzle = generate_puzzle(123, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = generate_puzzle(17)
    bad["test"][0]["output"][0, 0] = KEY
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(inputs),
            "output_shapes": len(outputs), "structural_profiles": len(topologies), "key_shapes": len(keys),
            "covered_guide_pairs": covered, "overlapping_pairs": overlap,
            "note": "Generator-owned finite checks; no independent or blind solver."}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
