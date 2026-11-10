"""Generate ordered all-pairs tables of polyomino congruence."""

import hashlib
import numpy as np


_SOURCE_INPUT_HASHES = {
    "263c4081071e0bcdb697dd2fa1ddc488a3b6a1e4dd00a9d821016efe56f6f12b",
    "37797389656e6020fac1b4c28d65e656ac9d8b221bfa3e82f4376b58869374a9",
    "12dcf834f23a9c2fdd10a0e103911313923f5b28c3b8dd0a9cab26b3dc53",
    "2cd9dd23e93ea53dbf95aa0de2e3bcd2bb64ce5d14e7cf5f7905c94dd154f47e",
    "2e410a5da995a01c97cafb90f5659d03be14cee2b8d517f8aba0bb36cdca6198",
}


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    assert cells
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _rotate(shape):
    return _normalize((c, -r) for r, c in shape)


def _reflect(shape):
    return _normalize((r, -c) for r, c in shape)


def _transform(shape, code):
    result = _reflect(shape) if code >= 4 else shape
    for _ in range(code % 4):
        result = _rotate(result)
    return _normalize(result)


def _canonical(shape):
    return min(_transform(shape, code) for code in range(8))


def _rotation_canonical(shape):
    return min(_transform(shape, code) for code in range(4))


def _connected(cells):
    remaining = set(cells)
    stack = [next(iter(remaining))]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for neighbor in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if neighbor in remaining and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(remaining)


def _grid_token(grid):
    return (tuple(grid.shape), bytes(grid.astype(np.uint8).flat))


def _pair_token(pair):
    return (_grid_token(pair["input"]), _grid_token(pair["output"]))


def _source_hash(grid):
    return hashlib.sha256(bytes([*grid.shape]) + bytes(grid.astype(np.uint8).flat)).hexdigest()


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(num_test + 1, 2 + (seed // 4) % 9)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _partitions(n, maximum=None):
    if n == 0:
        yield ()
        return
    for first in range(min(n, maximum or n), 0, -1):
        for rest in _partitions(n-first, first):
            yield (first,) + rest


def _class_groups(rng, n, train):
    choices = list(_partitions(n))
    if train:
        choices = [p for p in choices if len(p) >= 2 and p[0] >= 2]
    # Source partitions receive extra probability, but all partitions remain possible.
    favored = {4: (2, 2), 5: (3, 2), 7: (3, 2, 2)}.get(n)
    if favored in choices and rng.random() < 0.32:
        sizes = favored
    else:
        sizes = choices[int(rng.integers(len(choices)))]
    indices = [int(x) for x in rng.permutation(n)]
    groups, cursor = [], 0
    for size in sizes:
        groups.append(indices[cursor:cursor+size])
        cursor += size
    return groups


def _grow_shape(rng, area, box, elongated):
    h, w = box
    if elongated:
        if w > h:
            length = min(w, max(4, int(rng.integers(w-1, w+1))))
            row = int(rng.integers(h))
            cells = {(row, c) for c in range(length)}
        else:
            length = min(h, max(4, int(rng.integers(h-1, h+1))))
            col = int(rng.integers(w))
            cells = {(r, col) for r in range(length)}
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    area = max(area, len(cells))
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for rr, cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        options = sorted(frontier)
        cells.add(options[int(rng.integers(len(options)))])
    return _normalize(cells)


def _random_shape(rng, forbidden, exact_area=None, target_bbox=None,
                  demonstration=None, large=False):
    for _ in range(600):
        if target_bbox is not None:
            box = target_bbox
            area = exact_area
            elongated = False
        else:
            profile = int(rng.integers(4))
            if profile == 0:
                box = (4, 4)
                area = int(rng.integers(4, 9))
            elif profile == 1:
                box = (5, 5)
                area = int(rng.integers(6, 13))
            elif profile == 2:
                box = (3, 7)
                area = int(rng.integers(8, 15))
            else:
                box = (7, 3)
                area = int(rng.integers(8, 15))
            if large:
                area = max(area, int(rng.integers(10, 15)))
                if box == (4, 4) and area > 12:
                    box = (5, 5)
            elongated = profile >= 2
            if exact_area is not None:
                area = exact_area
                if area > box[0] * box[1]:
                    continue
        shape = _grow_shape(rng, area, box, elongated)
        if shape is None:
            continue
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if target_bbox is not None and (height, width) != target_bbox:
            continue
        if _canonical(shape) in forbidden:
            continue
        if demonstration == "reflection" and (
            _rotation_canonical(shape) == _rotation_canonical(_reflect(shape))
        ):
            continue
        if demonstration == "rotation" and shape == _rotate(shape):
            continue
        return shape
    raise RuntimeError("could not sample required shape")


def _make_shapes(rng, n, train, demonstration, large):
    groups = _class_groups(rng, n, train)
    instances = [None] * n
    used = set()
    first_area = first_bbox = None
    for group_index, group in enumerate(groups):
        if train and group_index == 1:
            # Distinct shapes with the same area, sometimes also the same box.
            try:
                base = _random_shape(rng, used, first_area, first_bbox)
            except RuntimeError:
                base = _random_shape(rng, used, first_area)
        else:
            base = _random_shape(rng, used,
                                 demonstration=demonstration if group_index == 0 else None,
                                 large=large and group_index == 0)
        used.add(_canonical(base))
        if group_index == 0:
            first_area = len(base)
            first_bbox = (max(r for r, _ in base)+1, max(c for _, c in base)+1)
        codes = [int(rng.integers(8)) for _ in group]
        if group_index == 0 and demonstration is not None:
            # An arbitrary common transform prevents a fixed visible orientation.
            common = int(rng.integers(8))
            codes[0] = common
            codes[1] = (common + (4 if demonstration == "reflection" else 1)) % 8
            # For reflection, code composition is subtle; verify below.
            if demonstration == "reflection":
                instances[group[0]] = _transform(base, common)
                instances[group[1]] = _reflect(instances[group[0]])
            else:
                instances[group[0]] = _transform(base, common)
                instances[group[1]] = _rotate(instances[group[0]])
        for i, code in zip(group, codes):
            if instances[i] is None:
                instances[i] = _transform(base, code)
    return instances


def _choose_dimensions(rng, n, regime):
    minimum = 16 if n <= 5 else (20 if n == 6 else 22)
    if regime == "target_test" and rng.random() < 0.45:
        minimum = max(minimum, 23)
    return (int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31)))


def _place_objects(rng, height, width, background, colors, shapes):
    n = len(colors)
    for _ in range(12):
        grid = np.full((height, width), background, dtype=np.int8)
        grid[:n, -1] = colors
        blocked = np.zeros((height, width), dtype=bool)
        for r in range(n):
            blocked[max(0,r-1):min(height,r+2), max(0,width-2):width] = True
        order = sorted(range(n), key=lambda i: -len(shapes[i]) + rng.random() * 0.2)
        placed = []
        style = int(rng.integers(4))
        for index in order:
            shape = shapes[index]
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            options = []
            for top in range(height-sh+1):
                for left in range(width-sw):
                    if not any(blocked[top+r, left+c] for r, c in shape):
                        options.append((top, left))
            if not options:
                break
            if style == 1 and placed:
                # A small neighborhood of earlier objects creates clusters.
                target = placed[int(rng.integers(len(placed)))]
                options.sort(key=lambda p: abs(p[0]-target[0])+abs(p[1]-target[1]))
                options = options[:max(1, len(options)//5)]
            elif style == 2:
                options.sort(key=lambda p: min(p[0], p[1], height-sh-p[0], width-1-sw-p[1]))
                options = options[:max(1, len(options)//4)]
            elif style == 3 and placed and rng.random() < 0.5:
                target = placed[int(rng.integers(len(placed)))]
                options.sort(key=lambda p: abs(p[0]-target[0])+abs(p[1]-target[1]))
                options = options[:max(1, len(options)//3)]
            top, left = options[int(rng.integers(len(options)))]
            placed.append((top, left))
            for r, c in shape:
                rr, cc = top+r, left+c
                grid[rr,cc] = colors[index]
                blocked[max(0,rr-1):min(height,rr+2), max(0,cc-1):min(width,cc+2)] = True
        else:
            return grid
    raise RuntimeError("could not place separated objects")


def _read_input(grid, background):
    height, width = grid.shape
    key = []
    for r in range(height):
        value = int(grid[r,-1])
        if value == background:
            break
        key.append(value)
    assert 4 <= len(key) <= 7
    assert len(set(key)) == len(key)
    assert np.all(grid[len(key):,-1] == background)
    shapes = []
    occupied = {}
    for r, color in enumerate(key):
        assert color != background
        cells = [(int(rr),int(cc)) for rr,cc in np.argwhere(grid == color)
                 if (int(rr),int(cc)) != (r,width-1)]
        assert 4 <= len(cells) <= 14
        assert _connected(cells)
        shape = _normalize(cells)
        assert max(rr for rr,_ in shape) < 7
        assert max(cc for _,cc in shape) < 7
        shapes.append(shape)
        for cell in cells:
            occupied[cell] = color
    assert set(int(x) for x in np.unique(grid)) == {background,*key}
    for (r,c), color in occupied.items():
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                other = occupied.get((r+dr,c+dc))
                assert other is None or other == color
        for kr in range(len(key)):
            assert max(abs(r-kr),abs(c-(width-1))) > 1
    return tuple(key), tuple(shapes)


def _solve_input(grid, background, match, mismatch):
    key, shapes = _read_input(grid, background)
    n = len(key)
    output = np.full((3*n+2,3*n+2), background, dtype=np.int8)
    signatures = [_canonical(shape) for shape in shapes]
    for i,color in enumerate(key):
        start = 3+3*i
        output[0:2,start:start+2] = color
        output[start:start+2,0:2] = color
    for i in range(n):
        for j in range(n):
            if i != j:
                value = match if signatures[i] == signatures[j] else mismatch
                output[3+3*i:5+3*i,3+3*j:5+3*j] = value
    return output


def _generate_pair(rng, regime, conventions, n, demonstration,
                   reserved_color=None, require_reserved=False):
    background, match, mismatch = conventions
    palette = [x for x in range(10) if x not in conventions]
    if regime == "source_train" and reserved_color is not None:
        palette.remove(reserved_color)
    if require_reserved:
        assert reserved_color in palette
        palette.remove(reserved_color)
        sampled = [reserved_color]
        sampled.extend(int(x) for x in rng.choice(palette, size=n-1, replace=False))
        colors = tuple(int(x) for x in rng.permutation(sampled))
    else:
        colors = tuple(int(x) for x in rng.choice(palette, size=n, replace=False))
    large = regime == "target_test" and rng.random() < 0.65
    shapes = _make_shapes(rng, n, regime == "source_train", demonstration, large)
    for _ in range(24):
        height,width = _choose_dimensions(rng,n,regime)
        try:
            inp = _place_objects(rng,height,width,background,colors,shapes)
            out = _solve_input(inp,background,match,mismatch)
            return {"input":inp,"output":out}
        except RuntimeError:
            pass
    raise RuntimeError("pair construction exhausted")


def _infer_conventions(puzzle):
    backgrounds = set()
    match_codes, mismatch_codes = set(), set()
    rotation = reflection = equal_area_mismatch = False
    for pair in puzzle["train"]:
        inp,out = pair["input"],pair["output"]
        colors,counts = np.unique(inp,return_counts=True)
        background = int(colors[int(np.argmax(counts))])
        backgrounds.add(background)
        key,shapes = _read_input(inp,background)
        signatures = [_canonical(s) for s in shapes]
        for i in range(len(key)):
            for j in range(i+1,len(key)):
                value = int(out[3+3*i,3+3*j])
                if signatures[i] == signatures[j]:
                    match_codes.add(value)
                    if shapes[i] != shapes[j] and _rotation_canonical(shapes[i]) == _rotation_canonical(shapes[j]):
                        rotation = True
                    if _rotation_canonical(shapes[i]) != _rotation_canonical(shapes[j]):
                        reflection = True
                else:
                    mismatch_codes.add(value)
                    if len(shapes[i]) == len(shapes[j]):
                        equal_area_mismatch = True
    assert len(backgrounds) == len(match_codes) == len(mismatch_codes) == 1
    assert rotation and reflection and equal_area_mismatch
    background,match,mismatch = next(iter(backgrounds)),next(iter(match_codes)),next(iter(mismatch_codes))
    assert len({background,match,mismatch}) == 3
    return background,match,mismatch


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs,seen_pairs = set(),set()
    for section in ("train","test"):
        for pair in puzzle[section]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype,np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            token = _pair_token(pair)
            assert token[0] not in seen_inputs and token not in seen_pairs
            seen_inputs.add(token[0]); seen_pairs.add(token)
            assert _source_hash(pair["input"]) not in _SOURCE_INPUT_HASHES
    conventions = _infer_conventions(puzzle)
    background,match,mismatch = conventions
    for section in ("train","test"):
        for pair in puzzle[section]:
            inp,out = pair["input"],pair["output"]
            colors,counts = np.unique(inp,return_counts=True)
            assert int(colors[int(np.argmax(counts))]) == background
            key,shapes = _read_input(inp,background)
            assert all(color not in conventions for color in key)
            assert np.array_equal(out,_solve_input(inp,*conventions))
            assert inp.shape != out.shape or not np.array_equal(inp,out)
            assert 2 <= len(set(_canonical(s) for s in shapes)) <= len(key) or section == "test"


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    counts = _resolve_counts(seed,rng,num_train,num_test)
    conventions = tuple(int(x) for x in rng.choice(10,size=3,replace=False))
    object_palette = [x for x in range(10) if x not in conventions]
    reserved_color = (int(rng.choice(object_palette)) if rng.random() < 0.5
                      else None)
    puzzle = {"train":[],"test":[]}
    # Two shuffled demonstrations establish mirror and rotation matching.
    demos = ["reflection","rotation"]
    if rng.random() < 0.5:
        demos.reverse()
    first_n = int(rng.integers(4,7))
    second_n = int(rng.choice([n for n in (4,5,6) if n != first_n]))
    for section,count in (("train",counts[0]),("test",counts[1])):
        for index in range(count):
            if section == "train":
                n = (first_n if index == 0 else second_n if index == 1
                     else int(rng.integers(4,7)))
                demonstration = demos[index] if index < 2 else None
            else:
                n = int(rng.integers(4,8))
                demonstration = None
            for _ in range(20):
                child_rng = np.random.default_rng(int(rng.integers(0,2**63-1,dtype=np.int64)))
                try:
                    pair = _generate_pair(child_rng,section == "train" and "source_train" or "target_test",
                                          conventions,n,demonstration,
                                          reserved_color,section == "test" and index == 0
                                          and reserved_color is not None)
                except RuntimeError:
                    continue
                if _grid_token(pair["input"]) not in {
                    _grid_token(p["input"]) for s in puzzle.values() for p in s
                } and _source_hash(pair["input"]) not in _SOURCE_INPUT_HASHES:
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    backgrounds,codes,dimensions,partitions,shape_classes = set(),set(),set(),set(),set()
    large_count = 0
    novel_test_palettes = 0
    mixed_counts = 0
    mixed_dimensions = mixed_palettes = 0
    test_all_same = test_all_distinct = 0
    long_count = edge_count = near_key_count = clustered_count = 0
    smaller_test_count = 0
    seen = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        conventions = _infer_conventions(puzzle)
        backgrounds.add(conventions[0]); codes.add(conventions[1:])
        train_colors = {color for pair in puzzle["train"]
                        for color in _read_input(pair["input"],conventions[0])[0]}
        local_counts = set()
        local_dimensions,local_palettes = set(),set()
        train_counts = []
        test_counts = []
        for section in ("train","test"):
            for pair in puzzle[section]:
                inp,out = pair["input"],pair["output"]
                key,shapes = _read_input(inp,conventions[0])
                local_counts.add(len(key))
                local_dimensions.add(inp.shape)
                local_palettes.add(tuple(sorted(key)))
                if section == "test":
                    test_counts.append(len(key))
                    novel_test_palettes += bool(set(key)-train_colors)
                else:
                    train_counts.append(len(key))
                signatures = [_canonical(s) for s in shapes]
                sizes = sorted((signatures.count(s) for s in set(signatures)),reverse=True)
                partitions.add(tuple(sizes))
                if section == "test":
                    test_all_same += len(sizes) == 1
                    test_all_distinct += len(sizes) == len(key)
                shape_classes.update(signatures)
                dimensions.add(inp.shape)
                large_count += any(len(s) >= 11 for s in shapes)
                long_count += any(max(max(r for r,c in s)+1,
                                      max(c for r,c in s)+1) >= 6 for s in shapes)
                objects = [{(int(r),int(c)) for r,c in np.argwhere(inp == color)
                            if (int(r),int(c)) != (i,inp.shape[1]-1)}
                           for i,color in enumerate(key)]
                edge_count += any(any(r in (0,inp.shape[0]-1) or c in (0,inp.shape[1]-2)
                                      for r,c in cells) for cells in objects)
                near_key_count += any(any(inp.shape[1]-1-c <= 3 for r,c in cells)
                                      for cells in objects)
                clustered_count += any(
                    max(abs(r-v),abs(c-w)) <= 3
                    for i in range(len(objects)) for j in range(i+1,len(objects))
                    for r,c in objects[i] for v,w in objects[j]
                )
                # Compare via direct transformed-cell equality, apart from
                # the canonical signature used by the output constructor.
                for i in range(len(key)):
                    for j in range(len(key)):
                        if i != j:
                            congruent = any(_transform(shapes[i],code) == shapes[j]
                                            for code in range(8))
                            expected = conventions[1] if congruent else conventions[2]
                            assert np.all(out[3+3*i:5+3*i,3+3*j:5+3*j] == expected)
                seen.add(_pair_token(pair))
        mixed_counts += len(local_counts) > 1
        mixed_dimensions += len(local_dimensions) > 1
        mixed_palettes += len(local_palettes) > 1
        smaller_test_count += any(t <= max(train_counts) for t in test_counts)
    assert len(backgrounds) >= 7 and len(codes) >= 15
    assert len(dimensions) >= 45 and len(partitions) >= 8
    assert len(shape_classes) >= 100 and large_count >= 30
    assert len(seen) >= 250
    assert mixed_counts == 42 and novel_test_palettes >= 8
    assert mixed_dimensions == mixed_palettes == 42
    assert smaller_test_count >= 20
    assert long_count >= 30 and edge_count >= 50
    assert near_key_count >= 30 and clustered_count >= 50
    assert (3,2) in partitions and (2,2) in partitions and (3,2,2) in partitions
    assert test_all_same >= 2 and test_all_distinct >= 2
    a = generate_puzzle(1001,6,2)
    b = generate_puzzle(1001,6,2)
    assert all(np.array_equal(x[k],y[k]) for s in ("train","test")
               for x,y in zip(a[s],b[s]) for k in ("input","output"))
    assert _pair_token(a["train"][0]) != _pair_token(generate_puzzle(1002,6,2)["train"][0])
    for seed,t,v in ((2001,2,1),(2002,10,1),(2003,10,4),
                     (3001,None,1),(3002,None,4),(3003,2,None),(3004,10,None)):
        p = generate_puzzle(seed,t,v)
        assert t is None or len(p["train"]) == t
        assert v is None or len(p["test"]) == v
        validate_generated_puzzle(p)
    for kwargs in ({"num_train":1},{"num_train":11},{"num_test":0},
                   {"num_test":5},{"num_train":2,"num_test":2},
                   {"num_train":3,"num_test":4},{"num_train":4,"num_test":4}):
        try:
            generate_puzzle(4001,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {kwargs}")
