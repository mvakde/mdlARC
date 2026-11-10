"""L-key strip normalization with literal squared-adjacency annotations.

The capped slow arm makes every convention input-readable.  No files, global
examples, or generation metadata are needed by solve().  See task_spec.md.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


BACKGROUNDS = (2, 3, 7, 8)  # Disjoint from all decimal square units digits.


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    pending = set(zip(*np.nonzero(mask)))
    boxes = []
    while pending:
        start = min(pending)
        pending.remove(start)
        todo, cells = [start], [start]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in pending:
                    pending.remove(q)
                    todo.append(q)
                    cells.append(q)
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells) + 1
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells) + 1
        _require(len(cells) == (r1-r0)*(c1-c0), "strips must be full rectangles")
        boxes.append((int(r0), int(r1), int(c0), int(c1)))
    return boxes


def _read(input_grid):
    """Find a unique key, then read strips using only the input."""
    original = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        a = np.rot90(original, rotation)
        h, w = a.shape
        if h < 6 or w < 6:
            continue
        for down in (False, True):
            root = 0 if down else h-1
            opposite = h-1-root
            origin, fast, slow, bg = map(int, (a[root, 0], a[root, 1], a[1, 0], a[opposite, 1]))
            if bg not in BACKGROUNDS or len({origin, fast, slow, bg}) != 4:
                continue
            if not (np.all(a[root, 1:] == fast) and a[opposite, 0] == origin
                    and np.all(a[1:-1, 0] == slow) and np.all(a[opposite, 1:] == bg)
                    and np.all(a[1:-1, -1] == bg) and np.all(a[1:-1, 1:3] == bg)):
                continue
            candidates.append((a.copy(), rotation, down, bg, origin, fast, slow))
    _require(len(candidates) == 1, "input must have exactly one capped L-key")
    a, rotation, down, bg, origin, fast, slow = candidates[0]
    mask = a != bg
    mask[:, 0] = False
    mask[0 if down else -1, :] = False
    boxes = _components(mask)
    n = len(boxes)
    _require(1 <= n <= 13 and a.shape[0] >= 2*n+4, "strip count exceeds packing budget")
    lanes = {}
    for r0, r1, c0, c1 in boxes:
        _require(r1-r0 == 2 and 2 <= c1-c0 <= 26, "strip must be 2 by L, L>=2")
        _require(1 <= r0 and r1 < a.shape[0] and 3 <= c0 and c1 < a.shape[1], "strip touches key or margin")
        block = a[r0:r1, c0:c1]
        _require(np.all(block[0] == block[0, 0]), "normalized top row must be monochrome")
        lanes.setdefault(r0, []).append((r0, r1, c0, c1))
    rows = sorted(lanes)
    _require(len(rows) <= 9 and all(y-x >= 3 for x, y in zip(rows, rows[1:])), "overlapping lane bands")
    groups = []
    for row in sorted(lanes, reverse=not down):
        group = sorted(lanes[row], key=lambda box: box[2])
        _require(all(b[2]-x[3] >= 1 for x, b in zip(group, group[1:])), "strips need a separating cell")
        groups.append(group)
    ordered = [box for group in groups for box in group]
    blocks = [a[r0:r1, c0:c1].copy() for r0, r1, c0, c1 in ordered]
    lengths = [b.shape[1] for b in blocks]
    _require(len({length % 2 for length in lengths}) == 1, "mixed parity would prevent exact centering")
    return {"normal": a, "rotation": rotation, "down": down, "background": bg,
            "origin": origin, "fast": fast, "slow": slow, "groups": groups,
            "boxes": ordered, "blocks": blocks, "lengths": lengths}


def _blank_with_key(d):
    a = d["normal"]
    result = np.full_like(a, d["background"])
    result[:, 0] = a[:, 0]
    root = 0 if d["down"] else a.shape[0]-1
    result[root, :] = a[root, :]
    return result


def _assemble(d, blocks=None, arithmetic=True):
    blocks = d["blocks"] if blocks is None else blocks
    result = _blank_with_key(d)
    maximum = max(b.shape[1] for b in blocks)
    for index, block in enumerate(blocks):
        length = block.shape[1]
        row, column = 2+2*index, 3+(maximum-length)//2
        result[row:row+2, column:column+length] = block
        if arithmetic:
            result[row, 1] = (length-1)**2 % 10
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the normalized, packed, annotated grid using only its input."""
    return _assemble(_read(input_grid))


def _competing_orders(d):
    groups, offset = [], 0
    for group in d["groups"]:
        groups.append(list(range(offset, offset+len(group))))
        offset += len(group)
    return {
        "reverse_fast": [i for g in groups for i in g[::-1]],
        "reverse_slow": [i for g in groups[::-1] for i in g],
        "fast_major": sorted(range(offset), key=lambda i: (d["boxes"][i][2], i)),
        "length_sort": sorted(range(offset), key=lambda i: d["lengths"][i]),
        "color_sort": sorted(range(offset), key=lambda i: int(d["blocks"][i][0, 0])),
    }


def _order_witness(d):
    correct = _assemble(d)
    return all(not np.array_equal(correct, _assemble(d, [d["blocks"][i] for i in order]))
               for order in _competing_orders(d).values())


def _gaps(rng, spare, count):
    # Bounded allocation; no rejection loop or oversized temporary arrays.
    return rng.multinomial(spare, rng.dirichlet(np.full(count, 1.5))).tolist()


def _span(group):
    return sum(block.shape[1] for block in group) + len(group)-1


def _ordinary_groups(rng, blocks):
    groups = []
    # First-fit decreasing bounds the lane budget even with long strips.
    for block in sorted(blocks, key=lambda b: -b.shape[1]):
        legal = [i for i, group in enumerate(groups) if _span(group)+1+block.shape[1] <= 26]
        if legal and (len(groups) >= 9 or rng.random() < .8):
            groups[int(rng.choice(legal))].append(block)
        else:
            groups.append([block])
    return groups


def _payload(rng, length, palette):
    block = np.full((2, length), int(rng.choice(palette)), dtype=np.int8)
    mode = int(rng.integers(4))
    if mode == 1:
        block[1] = rng.choice(palette)
    elif mode == 2:
        motif = rng.choice(palette, size=int(rng.integers(1, min(4, length)+1)))
        block[1] = np.resize(motif, length)
    elif mode == 3:
        block[1] = rng.choice(palette, size=length)
    return block


def _make_pair(rng, pool, configuration, witness=False):
    bg = int(rng.choice(BACKGROUNDS))
    available = [c for c in range(10) if c != bg]
    origin, fast, slow = map(int, rng.choice(available, size=3, replace=False))
    palette = rng.choice(available, size=int(rng.integers(2, 7)), replace=False)
    a, b = map(int, palette[:2])
    if witness:
        lengths = list(pool) + [pool[0], pool[0]]
        blocks = [_payload(rng, length, palette) for length in lengths]
        blocks[0][:] = a
        blocks[1][:] = a
        blocks[4][:] = b
        blocks[5] = blocks[0].copy()
        # One monochrome bar over an asymmetric lower row distinguishes proper
        # rotation from reflection and the two thickness rows.
        blocks[2][0] = a
        blocks[2][1] = a
        blocks[2][1, 0] = b
        groups = [[blocks[0], blocks[1]], [blocks[4], blocks[5]], [blocks[2]], [blocks[3]]]
        # Merge or retain the singleton lanes, while keeping two populated lanes.
        if rng.random() < .65:
            singleton = groups.pop()
            legal = [i for i, g in enumerate(groups) if _span(g)+1+_span(singleton) <= 26]
            if legal:
                groups[int(rng.choice(legal))].extend(singleton)
            else:
                groups.append(singleton)
    else:
        n = int(rng.integers(1, 14))
        lengths = rng.choice(pool, size=n).astype(int).tolist()
        # Force no count template: repair only the finite nine-lane canvas bound.
        for _ in range(n+1):
            blocks = [_payload(rng, length, palette) for length in lengths]
            groups = _ordinary_groups(rng, blocks)
            if len(groups) <= 9:
                break
            largest = max(range(n), key=lambda i: lengths[i])
            lengths[largest] = pool[0]
        else:
            # At most n replacements make every length minimal; from nine
            # existing lanes onward a fitting lane is always used.  Thirteen
            # length-2/3 strips therefore cannot reach this branch.
            raise AssertionError("finite lane repair failed")
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    n, k = sum(map(len, groups)), len(groups)
    minimum_h = max(2*n+4, 3*k+2)
    minimum_w = max(_span(group) for group in groups)+4
    _require(minimum_h <= 30 and minimum_w <= 30, "internal packing failure")
    h, w = int(rng.integers(minimum_h, 31)), int(rng.integers(minimum_w, 31))
    rotation, down = configuration
    normal = np.full((h, w), bg, dtype=np.int8)
    root = 0 if down else h-1
    normal[1:-1, 0] = slow
    normal[0, 0] = normal[-1, 0] = origin
    normal[root, 1:] = fast
    slow_extra = _gaps(rng, h-(3*k+2), k+1)
    distance = 2+slow_extra[0]
    flattened = []
    for lane, group in enumerate(groups):
        row = distance if down else h-2-distance
        fast_extra = _gaps(rng, w-(_span(group)+4), len(group)+1)
        column = 3+fast_extra[0]
        for j, block in enumerate(group):
            length = block.shape[1]
            normal[row:row+2, column:column+length] = block
            column += length+1+fast_extra[j+1]
            flattened.append(block)
        distance += 3+slow_extra[lane+1]
    # Ground truth comes from the sampled block sequence, not from solve().
    output = np.full_like(normal, bg)
    output[:, 0] = normal[:, 0]
    output[root, :] = normal[root, :]
    maximum = max(block.shape[1] for block in flattened)
    for i, block in enumerate(flattened):
        left = 3+(maximum-block.shape[1])//2
        output[2+2*i:4+2*i, left:left+block.shape[1]] = block
        output[2+2*i, 1] = (block.shape[1]-1)**2 % 10
    return {"input": np.rot90(normal, rotation).copy(), "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    pools = ([2, 4, 6, int(rng.choice(np.arange(8, 27, 2)))],
             [3, 5, 7, int(rng.choice(np.arange(9, 26, 2)))])
    rotation = int(rng.integers(4))
    other = (rotation+int(rng.choice((1, 3)))) % 4
    sign = bool(rng.integers(2))
    teaching = [(rotation, sign), (other, not sign)]
    held = [(rotation, not sign), (other, sign)]
    train_configs = [(k, d) for k in range(4) for d in (False, True) if (k, d) not in held]
    parity = int(rng.integers(2))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index < 2
            if witness:
                configuration, pool = teaching[index], pools[(parity+index) % 2]
            else:
                choices = train_configs if split == "train" else held
                configuration = choices[int(rng.integers(len(choices)))]
                pool = pools[int(rng.integers(2))]
            for _ in range(48):
                pair = _make_pair(rng, pool, configuration, witness)
                signature = _key(pair["input"])
                if signature in used:
                    continue
                if witness and not _order_witness(_read(pair["input"])):
                    continue
                used.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded construction failed to obtain a distinct teaching pair")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible geometry, preservation, arithmetic, teaching and shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen, descriptors = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _read(inp)
            descriptors[split].append(d)
            normal, blocks, bg = d["normal"], d["blocks"], d["background"]
            _require(out.shape == normal.shape, "output must preserve the rotated canvas")
            root = 0 if d["down"] else normal.shape[0]-1
            _require(np.array_equal(out[:, 0], normal[:, 0]) and np.array_equal(out[root], normal[root]), "key changed")
            bare = _blank_with_key(d)
            maximum = max(d["lengths"])
            recovered = []
            for i, block in enumerate(blocks):
                length, row = block.shape[1], 2+2*i
                left = 3+(maximum-length)//2
                copied = out[row:row+2, left:left+length]
                _require(np.array_equal(copied, block), "payload rotated, reflected, recolored or reordered incorrectly")
                recovered.extend(map(int, copied.flat))
                _require(int(out[row, 1]) == (length-1)**2 % 10, "incorrect adjacency-square digit")
                _require(normal[row, 1] == bg, "annotation site was not blank")
                bare[row:row+2, left:left+length] = block
                bare[row, 1] = int(out[row, 1])
            _require(np.array_equal(out, bare), "unexpected padding or extra changes")
            _require(Counter(recovered) == Counter(int(c) for b in blocks for c in b.flat), "payload multiplicity changed")
            _require(np.count_nonzero(out != _assemble(d, arithmetic=False)) == len(blocks), "exactly one visible label per strip is required")
            _require(np.array_equal(solve(inp), out), "input-only solve mismatch")
    train, test = descriptors["train"], descriptors["test"]
    taught = {length for d in train for length in d["lengths"]}
    _require(all(set(d["lengths"]) <= taught for d in test), "untaught query length")
    configs = {(d["rotation"], d["down"]) for d in train}
    _require(all((d["rotation"], d["down"]) not in configs for d in test), "query key combination must be held out")
    _require({length % 2 for length in taught} == {0, 1}, "training must teach both centering parities")
    _require(any(_order_witness(d) for d in train), "training leaves traversal shortcuts unresolved")
    _require(any(any(not np.array_equal(b, b[:, ::-1]) and not np.array_equal(b, b[::-1]) for b in d["blocks"]) for d in train), "no rotation/reflection witness")
    length_colors, color_lengths = {}, {}
    for d in train:
        for block in d["blocks"]:
            length, color = block.shape[1], int(block[0, 0])
            length_colors.setdefault(length, set()).add(color)
            color_lengths.setdefault(color, set()).add(length)
    _require(any(len(v) >= 2 for v in length_colors.values()), "color at fixed length is not varied")
    _require(any(len(v) >= 2 for v in color_lengths.values()), "length at fixed color is not varied")
    _require(any(length >= 5 for length in taught), "no modulo-ten teaching case")
    _require(any((length-1) % 10 != (length-1)**2 % 10 for length in taught), "raw adjacency shortcut survives")
    _require(any(len({_key(b) for b in d["blocks"]}) < len(d["blocks"]) for d in train), "no repeated-occurrence witness")


def _digest(puzzle):
    hasher = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                hasher.update(bytes(a.shape))
                hasher.update(a.tobytes())
    return hasher.hexdigest()


def self_test():
    """Streaming 42-seed checks; retain only compact diversity signatures."""
    signatures, shapes, output_shapes, layouts, configs = set(), set(), set(), set(), set()
    counts, colors, lengths, backgrounds = set(), set(), set(), set()
    top_origins, repeat_tests, black_labels = set(), 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        if seed in (0, 17):
            assert _digest(p) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                d = _read(pair["input"])
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                configs.add((d["rotation"], d["down"]))
                counts.add(len(d["blocks"]))
                backgrounds.add(d["background"])
                colors.update((d["origin"], d["fast"], d["slow"]))
                lengths.update(d["lengths"])
                layouts.add((pair["input"].shape, tuple(d["boxes"]), tuple(d["lengths"])))
                top_origins.add(d["down"])
                black_labels += sum((length-1)**2 % 10 == 0 for length in d["lengths"])
                if split == "test":
                    repeat_tests += len(set(d["lengths"])) < len(d["lengths"])
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"])
                # Whole-input rotation leaves the normalized answer unchanged.
                for rotation in range(1, 4):
                    assert np.array_equal(solve(np.rot90(original, rotation)), pair["output"])
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 65 and len(output_shapes) >= 45
    assert len(configs) == 8 and backgrounds == set(BACKGROUNDS)
    assert min(counts) == 1 and max(counts) == 13 and len(counts) >= 10
    assert len(colors) == 10 and max(lengths) >= 24 and min(lengths) == 2
    assert top_origins == {False, True} and repeat_tests >= 10 and black_labels > 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(5000+10*nt+ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Negative evidence: reject wrong arithmetic and a hole in a rectangle.
    p = generate_puzzle(9000)
    d = _read(p["test"][0]["input"])
    p["test"][0]["output"][2, 1] = (int(p["test"][0]["output"][2, 1])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("bad label accepted")
    damaged = d["normal"].copy()
    r0, _, c0, _ = d["boxes"][0]
    damaged[r0, c0] = d["background"]
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged rectangle accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; no puzzle files retained)")
