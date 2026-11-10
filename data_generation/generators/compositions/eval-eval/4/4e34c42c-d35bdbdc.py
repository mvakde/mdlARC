"""Relay seams: translate patterned ports and fuse their directed color arrows.

Only NumPy and the standard library are used. Generation writes no artifacts.
The input-only parser and solver never read generation coordinates or examples.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_BORDER = tuple((r, c) for r in range(-2, 3) for c in range(-2, 3)
                if max(abs(r), abs(c)) == 2)
_CORES = {
    "square": tuple((r, c) for r in range(-1, 2) for c in range(-1, 2)),
    "cross": ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
}


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "color outside 0..9"
    return a.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _components(points):
    todo, result = set(points), []
    while todo:
        p = min(todo)
        todo.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _tight(cells, background):
    assert cells, "empty result"
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    h = max(r for r, c in cells) - r0 + 1
    w = max(c for r, c in cells) - c0 + 1
    assert max(h, w) <= 30, "assembly exceeds 30"
    out = np.full((h, w), background, dtype=np.int8)
    for (r, c), value in cells.items():
        out[r-r0, c-c0] = value
    return out


def _shift(p, delta):
    return p[0] + delta[0], p[1] + delta[1]


def _ordering(a, b):
    forward = a["value"] == b["body"]
    backward = b["value"] == a["body"]
    if forward == backward:
        return None
    return (a, b) if forward else (b, a)


def _parse(input_grid):
    """Recover all conventions from pixels, including roles under recoloring."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert counts.max() * 2 > grid.size, "background must be a majority"
    bg = int(colors[counts.argmax()])
    tokens = []
    h, w = grid.shape
    for r in range(2, h-2):
        for c in range(2, w-2):
            border = tuple(int(grid[r+dr, c+dc]) for dr, dc in _BORDER)
            corners = [border[i] for i, (dr, dc) in enumerate(_BORDER)
                       if abs(dr) == abs(dc) == 2]
            if not (all(v == bg for v in corners) or all(v != bg for v in corners)):
                continue
            if any(v == bg for v, (dr, dc) in zip(border, _BORDER)
                   if abs(dr) != 2 or abs(dc) != 2):
                continue
            markers = set(border) - {bg}
            if len(markers) != 2 or min(border.count(v) for v in markers) < 4:
                continue
            value = int(grid[r, c])
            if value == bg or value in markers:
                continue
            for kind, support in _CORES.items():
                body_values = {int(grid[r+dr, c+dc]) for dr, dc in support if dr or dc}
                if len(body_values) != 1:
                    continue
                body = body_values.pop()
                if body == bg or body in markers:
                    continue
                if any(grid[r+dr, c+dc] != bg for dr in range(-1, 2)
                       for dc in range(-1, 2) if (dr, dc) not in support):
                    continue
                cells = {(r+dr, c+dc) for dr, dc in _BORDER if grid[r+dr, c+dc] != bg}
                cells.update((r+dr, c+dc) for dr, dc in support)
                tokens.append(dict(center=(r, c), cells=cells, body=body, value=value,
                                   kind=kind, key=(kind, border), markers=markers,
                                   notched=all(v == bg for v in corners)))
    assert len(tokens) >= 4, "missing framed ports"
    marker_sets = {tuple(sorted(t["markers"])) for t in tokens}
    assert len(marker_sets) == 1, "inconsistent collar roles"
    markers = set(next(iter(marker_sets)))
    covered = set()
    for i, t in enumerate(tokens):
        r, c = t["center"]
        assert all(max(abs(r-u["center"][0]), abs(c-u["center"][1])) >= 5
                   for u in tokens[:i]), "overlapping frames"
        assert not covered.intersection(t["cells"]), "ambiguous frame parsing"
        covered.update(t["cells"])
    foreground = set(zip(*np.where(grid != bg)))
    wire = foreground - covered
    line_values = {int(grid[p]) for p in wire}
    assert len(line_values) == 1, "wire role is not unique"
    line = line_values.pop()
    assert line not in markers | {bg}
    assert all(t["body"] != line and t["value"] != line for t in tokens)
    pieces = _components(foreground)
    assert 2 <= len(pieces) <= 4, "fragment count"
    owner = {p: i for i, part in enumerate(pieces) for p in part}
    for i, t in enumerate(tokens):
        r, c = t["center"]
        t["index"], t["piece"] = i, owner[(r, c)]
        contacts = {q for p in t["cells"] for dr, dc in _D4
                    if (q := (p[0]+dr, p[1]+dc)) in wire}
        aligned = {(r+3*dr, c+3*dc) for dr, dc in _D4} & wire
        assert contacts == aligned and len(contacts) in (1, 2), "unaligned wire contact"
        t["eligible"] = len(contacts) == 1
    for i, part in enumerate(pieces):
        ts = [t for t in tokens if t["piece"] == i]
        assert part & wire and sum(t["eligible"] for t in ts) == 2, "two end ports required"
    return grid, bg, line, tokens, pieces, wire


def _candidate_pairs(tokens, *, use_collar=True, use_arrow=True, endpoints=True):
    pairs = []
    for i, a in enumerate(tokens):
        for b in tokens[i+1:]:
            if a["piece"] == b["piece"]:
                continue
            if endpoints and not (a["eligible"] and b["eligible"]):
                continue
            if use_collar and a["key"] != b["key"]:
                continue
            if use_arrow and _ordering(a, b) is None:
                continue
            pairs.append((a, b))
    return pairs


def _registration_offsets(pieces, matches, background):
    """Check the geometric constraints independently of arrow compatibility."""
    degree = {}
    adjacency = [[] for _ in pieces]
    allowed = {}
    for a, b in matches:
        degree[a["index"]] = degree.get(a["index"], 0) + 1
        degree[b["index"]] = degree.get(b["index"], 0) + 1
        i, j = a["piece"], b["piece"]
        delta = (a["center"][0]-b["center"][0], a["center"][1]-b["center"][1])
        adjacency[i].append((j, delta))
        adjacency[j].append((i, (-delta[0], -delta[1])))
    assert max(degree.values()) <= 1, "multiple possible partners"
    assert len(matches) == len(pieces)-1, "not an open registration chain"
    offsets, stack = {0: (0, 0)}, [0]
    while stack:
        i = stack.pop()
        for j, delta in adjacency[i]:
            position = _shift(offsets[i], delta)
            if j in offsets:
                assert offsets[j] == position, "inconsistent registration"
            else:
                offsets[j] = position
                stack.append(j)
    assert len(offsets) == len(pieces), "disconnected registration"
    assert sorted(len(a) for a in adjacency) == [1, 1] + [2]*(len(pieces)-2)
    # Check translated original supports before deleting anything. At a seam
    # the collar agrees exactly; only the two directed core labels may disagree.
    for a, b in matches:
        i, j = sorted((a["piece"], b["piece"]))
        allowed[i, j] = {_shift(p, offsets[a["piece"]]) for p in a["cells"]}
    occupancy = {}
    for i, part in enumerate(pieces):
        for p in part:
            q = _shift(p, offsets[i])
            if q in occupancy:
                j = occupancy[q]
                assert q in allowed.get(tuple(sorted((i, j))), set()), "extraneous overlap"
            else:
                occupancy[q] = i
    _tight({p: 1 for p in occupancy}, background)
    return offsets


def _collar_only_layout_count(info):
    """Count up to two geometrically possible assemblies without reading arrows.

    Eight end ports bound this search to at most C(28,3)=3276 subsets, and
    generated collar groups are much smaller. It stops at the second solution.
    """
    candidates = _candidate_pairs(info["tokens"], use_arrow=False)
    count = 0
    for matches in itertools.combinations(candidates, len(info["pieces"])-1):
        try:
            _registration_offsets(info["pieces"], matches, info["background"])
        except AssertionError:
            continue
        count += 1
        if count == 2:
            break
    return count


def _reconstruct(input_grid):
    grid, bg, line, tokens, pieces, wire = _parse(input_grid)
    matches = _candidate_pairs(tokens)
    assert matches, "no relay seams"
    offsets = _registration_offsets(pieces, matches, bg)
    result = {_shift(p, offsets[next(i for i, part in enumerate(pieces) if p in part)]): line
              for p in wire}
    for a, b in matches:
        source, donor = _ordering(a, b)
        for p in source["cells"]:
            q = _shift(p, offsets[source["piece"]])
            assert q not in result, "wire enters seam"
            result[q] = int(grid[p])
        result[_shift(source["center"], offsets[source["piece"]])] = donor["value"]
    out = _tight(result, bg)
    info = dict(grid=grid, background=bg, line=line, tokens=tokens, pieces=pieces,
                wire=wire, matches=matches, offsets=offsets)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the input's visible geometry and directed color pairs."""
    return _reconstruct(input_grid)[0]


def _pattern(rng, marker_colors, kind, notched, used):
    for _ in range(128):
        support = [p for p in _BORDER if not notched or abs(p[0]) != abs(p[1])]
        n = len(support)
        count = int(rng.integers(4, n-3))
        order = rng.permutation(n)
        values = [marker_colors[0]]*n
        for i in order[:count]:
            values[int(i)] = marker_colors[1]
        pattern = dict(zip(support, values))
        signature = (kind, tuple(pattern.get(p, -1) for p in _BORDER))
        if signature not in used:
            used.add(signature)
            return dict(kind=kind, pattern=pattern)
    raise RuntimeError("bounded collar sampling exhausted")


def _token(center, key, relation):
    r, c = center
    body, value = relation
    cells = {(r+dr, c+dc): v for (dr, dc), v in key["pattern"].items()}
    cells.update({(r+dr, c+dc): body for dr, dc in _CORES[key["kind"]]})
    cells[r, c] = value
    return cells


def _segment(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr = int(np.sign(b[0]-a[0]))
    dc = int(np.sign(b[1]-a[1]))
    return {(a[0]+i*dr, a[1]+i*dc) for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)}


def _geometry(rng, n, internal_piece):
    """Grow off-lattice chains. No shape table, seed restart or latent solver data."""
    for _ in range(100):
        nodes, wires, inner = [(0, 0)], [], {}
        all_wire = set()
        good = True
        for i in range(n):
            start = nodes[-1]
            found = False
            for _ in range(80):
                dr, dc = _D4[int(rng.integers(4))]
                vr, vc = -dc, dr
                long = i == internal_piece
                lateral = 0 if long else int(rng.choice([0, 0, 0, -3, -2, -1, 1, 2, 3]))
                distance = int(rng.integers(13, 17) if long else
                               rng.integers(8, 11) if lateral else rng.integers(6, 10))
                end = (start[0]+dr*distance+vr*lateral,
                       start[1]+dc*distance+vc*lateral)
                new_nodes = [end]
                if long:
                    mid_distance = int(rng.integers(6, distance-5))
                    midpoint = (start[0]+dr*mid_distance, start[1]+dc*mid_distance)
                    new_nodes.insert(0, midpoint)
                else:
                    midpoint = None
                old_centers = nodes + list(inner.values())
                if any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) < 6
                       for j, a in enumerate(new_nodes)
                       for b in old_centers+new_nodes[:j]):
                    continue
                if any(max(abs(p[0]-a[0]), abs(p[1]-a[1])) <= 3
                       for a in new_nodes for p in all_wire):
                    continue
                bend = int(rng.integers(4, distance-3)) if lateral else distance//2
                p1 = (start[0]+dr*bend, start[1]+dc*bend)
                p2 = (p1[0]+vr*lateral, p1[1]+vc*lateral)
                path = _segment(start, p1) | _segment(p1, p2) | _segment(p2, end)
                centers = [start]+new_nodes
                path = {p for p in path if all(max(abs(p[0]-a[0]), abs(p[1]-a[1])) > 2
                                               for a in centers)}
                if any(max(abs(p[0]-a[0]), abs(p[1]-a[1])) <= 3
                       for p in path for a in old_centers if a != start):
                    continue
                if any(p in all_wire or any((p[0]+x, p[1]+y) in all_wire for x, y in _D4)
                       for p in path):
                    continue
                proposed = old_centers + new_nodes
                if any(max(a[axis] for a in proposed)-min(a[axis] for a in proposed)+5 > 30
                       for axis in (0, 1)):
                    continue
                nodes.append(end)
                wires.append(path)
                all_wire.update(path)
                if midpoint is not None:
                    inner[i] = midpoint
                found = True
                break
            if not found:
                good = False
                break
        if good:
            return nodes, wires, inner
    raise RuntimeError("bounded geometry sampling exhausted")


def _pack(parts, background, rng):
    normalized = []
    for part in parts:
        r0 = min(r for r, c in part)
        c0 = min(c for r, c in part)
        normalized.append({(r-r0, c-c0): v for (r, c), v in part.items()})
    shapes = [(max(r for r, c in p)+1, max(c for r, c in p)+1) for p in normalized]
    mh, mw = max(h for h, w in shapes), max(w for h, w in shapes)
    area = sum(map(len, normalized))
    canvases = [(int(rng.integers(max(mh, 12), 31)), int(rng.integers(max(mw, 12), 31)))
                for _ in range(12)] + [(30, 30)]*16
    for h, w in canvases:
        if h*w <= 2*area:
            continue
        halo = np.zeros((h, w), dtype=bool)
        grid = np.full((h, w), background, dtype=np.int8)
        order = sorted(range(len(parts)), key=lambda i: -(shapes[i][0]*shapes[i][1]+rng.random()*50))
        for i in order:
            ph, pw = shapes[i]
            feasible = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in normalized[i]:
                feasible &= ~halo[r:r+h-ph+1, c:c+w-pw+1]
            choices = np.argwhere(feasible)
            if not len(choices):
                break
            r0, c0 = map(int, choices[int(rng.integers(len(choices)))])
            for (r, c), value in normalized[i].items():
                r, c = r+r0, c+c0
                grid[r, c] = value
                halo[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
        else:
            return grid
    raise RuntimeError("bounded separated packing exhausted")


def _make_pair(rng, parameters, mode, test):
    n = 4 if test else 2 if mode == "selection" else 3 if mode == "repeated" else int(rng.integers(2, 4))
    palette = list(map(int, rng.permutation(10))) if parameters["varied"] else parameters["palette"]
    bg, line, m1, m2 = palette[:4]
    identities = palette[4:]
    used = set()

    def make_key(active=False):
        kind = parameters["core"]
        if kind == "mixed":
            kind = ("square" if mode == "selection" else "cross") if active and mode != "random" else str(rng.choice(["square", "cross"]))
        form = parameters["collar"]
        if form == "mixed":
            notched = (mode == "repeated") if active and mode != "random" else bool(rng.integers(2))
        else:
            notched = form == "notched"
        return _pattern(rng, (m1, m2), kind, notched, used)

    # At most two seams share one collar. Disjoint identity triples make the
    # unique visible matching explicit without assigning global color meanings.
    shared = n >= 3 and (mode == "repeated" or test or rng.random() < .6)
    groups = [0 if shared and i < 2 else i+1 for i in range(n-1)]
    seam_keys, seam_relations = [None]*(n-1), [None]*(n-1)
    for group in sorted(set(groups)):
        indices = [i for i, g in enumerate(groups) if g == group]
        key = make_key(active=True)
        ids = list(map(int, rng.permutation(identities)))
        for k, i in enumerate(indices):
            x, y, z = ids[3*k:3*k+3] if len(indices) == 2 else ids[:3]
            # Selection teaches a changed center. Other pairs may show an
            # unchanged center or a monochrome source, never a bidirectional tie.
            if mode != "selection" and not (mode == "repeated" and k == 0):
                coin = rng.random()
                if coin < .22:
                    z = y
                elif coin < .36:
                    x = y
            source, donor = (x, y), (y, z)
            pair = (source, donor) if rng.random() < .5 else (donor, source)
            seam_keys[i], seam_relations[i] = key, pair
    terminal_keys = [make_key(), make_key()]
    terminal_relations = [seam_relations[0][1], seam_relations[-1][0]]
    internal_piece = int(rng.integers(n)) if mode == "selection" or rng.random() < .28 else None
    nodes, wires, inner = _geometry(rng, n, internal_piece)
    parts = []
    for i in range(n):
        part = {p: line for p in wires[i]}
        left_key = terminal_keys[0] if i == 0 else seam_keys[i-1]
        left_rel = terminal_relations[0] if i == 0 else seam_relations[i-1][1]
        right_key = terminal_keys[1] if i == n-1 else seam_keys[i]
        right_rel = terminal_relations[1] if i == n-1 else seam_relations[i][0]
        part.update(_token(nodes[i], left_key, left_rel))
        part.update(_token(nodes[i+1], right_key, right_rel))
        if i in inner:
            # Put a copy on the opposite piece from its compatible partner.
            seam_index = 0 if i == 0 else i-1
            relation = seam_relations[seam_index][0 if i == seam_index else 1]
            part.update(_token(inner[i], seam_keys[seam_index], relation))
        parts.append(part)
    expected = {p: line for wire in wires for p in wire}
    for i in range(n-1):
        a, b = seam_relations[i]
        if a[1] == b[0]:
            source, donor = a, b
        else:
            source, donor = b, a
        expected.update(_token(nodes[i+1], seam_keys[i], (source[0], donor[1])))
    inp, out = _pack(parts, bg, rng), _tight(expected, bg)
    # Rotate the entire example together: no individual fragment rotation.
    rotation = int(rng.integers(4))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": inp.copy(), "output": out.copy()}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return tuple(a.shape), a.tobytes()


def _features(inp):
    out, info = _reconstruct(inp)
    ts, matches = info["tokens"], info["matches"]
    actual = {tuple(sorted((a["index"], b["index"]))) for a, b in matches}

    def extra(**kw):
        return len({tuple(sorted((a["index"], b["index"])))
                    for a, b in _candidate_pairs(ts, **kw)} - actual)

    source_donor = [_ordering(a, b) for a, b in matches]
    seam_centers = [_shift(a["center"], info["offsets"][a["piece"]]) for a, b in matches]
    return dict(fragments=len(info["pieces"]), tokens=len(ts), seams=len(matches),
                shape=tuple(np.asarray(inp).shape), output_shape=tuple(out.shape),
                background=info["background"], line=info["line"],
                kinds=tuple(sorted({t["kind"] for t in ts})),
                notches=tuple(sorted({t["notched"] for t in ts})),
                interior=sum(not t["eligible"] for t in ts),
                changed=sum(s["value"] != d["value"] for s, d in source_donor),
                monochrome=sum(t["body"] == t["value"] for t in ts),
                unchanged=sum(s["value"] == d["value"] for s, d in source_donor),
                source_sides=tuple(s["piece"] < d["piece"] for s, d in source_donor),
                reused=len({a["key"] for a, b in matches}) < len(matches),
                collar_only_layouts=_collar_only_layout_count(info),
                ignore_endpoint_extra=extra(endpoints=False),
                ignore_collar_extra=extra(use_collar=False),
                ignore_arrow_extra=extra(use_arrow=False),
                two_axis=(len({r for r, c in seam_centers}) > 1
                          and len({c for r, c in seam_centers}) > 1),
                wire_cells=len(info["wire"]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, _ = _reconstruct(inp)
            assert np.array_equal(out, expected), "wrong fusion, deletion, wire or crop"
            features[split].append(_features(inp))
    train = features["train"]
    assert any(f["ignore_endpoint_extra"] and f["ignore_collar_extra"] and f["changed"] for f in train), "selection contrasts missing"
    assert any(f["reused"] and f["collar_only_layouts"] == 2 and f["changed"] for f in train), "relational registration not taught"
    assert all(f["fragments"] <= 3 for f in train)
    assert all(f["fragments"] == 4 for f in features["test"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    parameters = dict(palette=list(map(int, rng.permutation(10))), varied=bool(rng.integers(2)),
                      core=str(rng.choice(["square", "cross", "mixed"])),
                      collar=str(rng.choice(["complete", "notched", "mixed"])))
    modes = ["selection", "repeated"] + ["random"]*(nt-2)
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            mode = modes[i] if split == "train" else "random"
            for _ in range(24):
                try:
                    pair = _make_pair(rng, parameters, mode, split == "test")
                    predicted, _ = _reconstruct(pair["input"])
                    assert np.array_equal(predicted, pair["output"])
                    f = _features(pair["input"])
                    if mode == "selection":
                        assert f["ignore_endpoint_extra"] and f["ignore_collar_extra"] and f["changed"]
                    if mode == "repeated":
                        assert f["collar_only_layouts"] == 2 and f["changed"]
                    key = _key(pair["input"])
                    assert key not in seen
                except (AssertionError, RuntimeError):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded pair construction exhausted")
    return puzzle


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _key(p[role])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain features and hashes, never full sample corpora."""
    signatures, shapes, out_shapes, patterns, structures = set(), set(), set(), set(), set()
    values = {name: set() for name in ("background", "line", "kinds", "notches", "fragments", "tokens", "wire_cells")}
    counters = {name: 0 for name in ("interior", "changed", "monochrome", "unchanged", "reused", "two_axis")}
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic"
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                f = _features(pair["input"])
                shapes.add(f["shape"])
                out_shapes.add(f["output_shape"])
                for name in values:
                    values[name].add(f[name])
                for name in counters:
                    counters[name] += bool(f[name])
                inp = pair["input"]
                labels = {}
                normalized = bytes(labels.setdefault(int(v), len(labels)) for v in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape)+normalized).hexdigest())
                _, info = _reconstruct(inp)
                patterns.update(t["key"] for t in info["tokens"])
        # Repack the same fragments: original canvas positions have no meaning.
        pair = puzzle["train"][0]
        _, info = _reconstruct(pair["input"])
        parts = [{p: int(pair["input"][p]) for p in part} for part in info["pieces"]]
        repacked = _pack(parts, info["background"], np.random.default_rng(seed+10000))
        assert np.array_equal(solve(repacked), pair["output"])
        # Arbitrary role permutation, including background, is equivariant.
        mapping = np.random.default_rng(seed+20000).permutation(10).astype(np.int8)
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 60 and len(out_shapes) >= 60 and len(patterns) >= 150
    assert values["background"] == values["line"] == set(range(10))
    assert values["fragments"] == {2, 3, 4}
    assert len(values["tokens"]) >= 5 and len(values["wire_cells"]) >= 20
    assert ("cross", "square") in values["kinds"]
    assert (False, True) in values["notches"]
    assert all(counters[name] >= 10 for name in counters)
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(9000+a*10+b, a, b))
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, False), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A damaged label must fail the semantic validator.
    damaged = generate_puzzle(47, 2, 1)
    out = damaged["train"][0]["output"]
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # A two-way compositional tie and two possible partners are input errors,
    # even if a particular layout could suggest a preference.
    base = generate_puzzle(48, 2, 1)
    selection = next(p for p in base["train"] if _features(p["input"])["fragments"] == 2)
    _, info = _reconstruct(selection["input"])
    source, donor = _ordering(*info["matches"][0])
    tied = selection["input"].copy()
    tied[donor["center"]] = source["body"]
    ambiguous = selection["input"].copy()
    terminal = next(t for t in info["tokens"] if t["piece"] == source["piece"]
                    and t["eligible"] and t["index"] != source["index"])
    r, c = source["center"]
    u, v = terminal["center"]
    ambiguous[u-2:u+3, v-2:v+3] = ambiguous[r-2:r+3, c-2:c+3].copy()
    for invalid in (tied, ambiguous):
        try:
            solve(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError("ambiguous arrow or partner accepted")
    return dict(seeds=42, pairs=total, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), output_shapes=len(out_shapes),
                collar_signatures=len(patterns), color_normalized_inputs=len(structures),
                legal_counts=legal, feature_pair_counts=counters,
                independent_solver=False, blind_prediction=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
