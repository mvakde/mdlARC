"""Collision-controlled circulation of glyphs in fixed colored sockets.

Only NumPy and the standard library are used.  No parent code is imported.
The four literal source colors are a fixed, input-only convention; socket
colors are observable and may be assigned afresh in every example.
"""
from collections import Counter
import json

import numpy as np


SPECS = {
    1: ((-1, 2), ((0, 1), (0, 2))),
    2: ((0, -4), ((0, -1), (0, -2), (0, -3))),
    3: ((3, 0), ((1, 0), (2, 0))),
    6: ((-6, 0), ((-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0))),
}
KINDS = ((1, 3), (1, 6), (2, 3), (2, 6))
COMPLEMENTS = (((1, 3), (2, 6)), ((1, 6), (2, 3)))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _mask_key(mask):
    return sum(int(x) << i for i, x in enumerate(mask.flat))


def _box(center):
    r, c = center
    return {(rr, cc) for rr in range(r - 2, r + 3)
            for cc in range(c - 2, c + 3)}


def _trace_geometry(tokens, shape):
    """Instance-owned stencils, then simultaneous overlaps, then rays."""
    h, w = shape
    if not tokens or any(t not in SPECS for t, _, _ in tokens):
        raise ValueError("unknown or absent source")
    sources = [(r, c) for _, r, c in tokens]
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate source")
    for i, (r, c) in enumerate(sources):
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in sources[i + 1:]):
            raise ValueError("sources must be isolated")
    endpoints, owners = [], {}
    for i, (role, r, c) in enumerate(tokens):
        end, trail = SPECS[role]
        endpoints.append((r + end[0], c + end[1]))
        for dr, dc in trail:
            owners.setdefault((r + dr, c + dc), []).append(i)
    occupied = set(sources) | set(endpoints) | set(owners)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in occupied):
        raise ValueError("complete stencils must fit")
    if (len(set(endpoints)) != len(endpoints)
            or set(sources) & (set(endpoints) | set(owners))
            or set(endpoints) & set(owners)):
        raise ValueError("source/endpoint/connector overwrite")
    crossings = {}
    for p, ids in owners.items():
        if len(ids) > 2:
            raise ValueError("more than two traces meet")
        if len(ids) == 2:
            kind = tuple(sorted(tokens[i][0] for i in ids))
            if kind not in KINDS:
                raise ValueError("unsupported collinear overlap")
            crossings[p] = kind
    if len(crossings) > 2 or len({r + c for r, c in crossings}) != len(crossings):
        raise ValueError("at most two distinct parallel ray lines")
    rays = {}
    for r, c in sorted(crossings):
        ray = []
        rr, cc = r + 1, c - 1
        while rr < h and cc >= 0:
            if (rr, cc) in occupied:
                raise ValueError("a ray would overwrite trace geometry")
            ray.append((rr, cc))
            rr, cc = rr + 1, cc - 1
        rays[(r, c)] = tuple(ray)
    return {"tokens": tuple(tokens), "endpoints": tuple(endpoints),
            "owners": owners, "crossings": crossings, "rays": rays,
            "occupied": occupied}


def _layout(tokens, centers, shape):
    h, w = shape
    if not 2 <= len(centers) <= 7:
        raise ValueError("two to seven sockets required")
    boxes = []
    for i, (r, c) in enumerate(centers):
        if not (2 <= r < h - 2 and 2 <= c < w - 2):
            raise ValueError("socket must fit")
        if any(abs(r - rr) < 6 and abs(c - cc) < 6 for rr, cc in centers[:i]):
            raise ValueError("sockets need a blank gap")
        boxes.append(_box((r, c)))
    info = _trace_geometry(tokens, shape)
    covered = set().union(*boxes)
    if covered & info["occupied"]:
        raise ValueError("trace geometry touches a socket")
    if any(abs(r - rr) <= 3 and abs(c - cc) <= 3
           for _, r, c in tokens for rr, cc in centers):
        raise ValueError("input sources need a blank gap from socket borders")
    groups, addressed = [], set()
    for origin, ray in info["rays"].items():
        ray_set = set(ray)
        group = [i for i, p in enumerate(centers) if p in ray_set]
        group.sort(key=lambda i: centers[i][0])
        for i, box in enumerate(boxes):
            if box & ray_set and i not in group:
                raise ValueError("a ray may traverse only the center of a socket")
        if addressed & set(group):
            raise ValueError("socket belongs to two cycles")
        if len(group) > 4:
            raise ValueError("at most four sockets per ray")
        addressed.update(group)
        groups.append(tuple(group))
    info.update(centers=tuple(centers), boxes=boxes, covered=covered,
                groups=tuple(groups), addressed=addressed, shape=shape)
    return info


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    cards = []
    for r in range(h - 4):
        for c in range(w - 4):
            ink = int(a[r, c])
            if not ink or a[r, c + 4] != ink or a[r + 4, c] != ink:
                continue
            block = a[r:r + 5, c:c + 5]
            if not (np.all(block[0] == ink) and np.all(block[4] == ink)
                    and np.all(block[:, 0] == ink) and np.all(block[:, 4] == ink)):
                continue
            inside = block[1:4, 1:4]
            if not np.all((inside == 0) | (inside == ink)) or not np.any(inside):
                raise ValueError("socket interiors are nonempty monochrome masks")
            cards.append(((r + 2, c + 2), ink, inside != 0))
    if len({ink for _, ink, _ in cards}) != len(cards):
        raise ValueError("socket carrier colors must be distinct")
    if len({_mask_key(mask) for _, _, mask in cards}) != len(cards):
        raise ValueError("socket masks must be distinct within an input")
    rest = a.copy()
    for (r, c), _, _ in cards:
        rest[r - 2:r + 3, c - 2:c + 3] = 0
    tokens = [(int(rest[r, c]), int(r), int(c)) for r, c in np.argwhere(rest)]
    info = _layout(tokens, [p for p, _, _ in cards], a.shape)
    info.update(input=a, colors=tuple(ink for _, ink, _ in cards),
                masks=tuple(mask.copy() for _, _, mask in cards))
    return info


def _render(info, groups=None, reverse=False, carry_ink=False):
    out = info["input"].copy()
    for (role, _, _), p in zip(info["tokens"], info["endpoints"]):
        out[p] = role
    for p, ids in info["owners"].items():
        out[p] = 4 if len(ids) == 2 else 5
    for ray in info["rays"].values():
        for p in ray:
            if p not in info["covered"]:
                out[p] = 4
    for group in info["groups"] if groups is None else groups:
        for j, destination in enumerate(group):
            source = group[(j + (-1 if reverse else 1)) % len(group)]
            r, c = info["centers"][destination]
            inside = out[r - 1:r + 2, c - 1:c + 2]
            inside[:] = 0
            ink = info["colors"][source if carry_ink else destination]
            inside[info["masks"][source]] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read frames and sparse sources; derive both geometry and permutation."""
    return _render(_decode(input_grid))


def _chain(rng, shape, kind, count):
    h, w = shape
    first = int(rng.integers(5, 8 if count < 4 else 6))
    gaps = [int(rng.integers(6, 9 if count < 4 else 7)) for _ in range(count - 1)]
    distances = [first]
    for gap in gaps:
        distances.append(distances[-1] + gap)
    far = distances[-1]
    if h - far - 2 <= 1 or w - far - 2 <= 1:
        return None
    r = int(rng.integers(1, h - far - 2))
    c = int(rng.integers(far + 2, w))
    horizontal, vertical = kind
    hg = int(rng.integers(1, 3 if horizontal == 1 else 4))
    vg = int(rng.integers(1, 3 if vertical == 3 else 6))
    tokens = [(horizontal, r, c - hg if horizontal == 1 else c + hg),
              (vertical, r - vg if vertical == 3 else r + vg, c)]
    centers = [(r + d, c - d) for d in distances]
    return tokens, centers


def _shape(rng, counts):
    if len(counts) == 2:
        return int(rng.integers(22, 31)), int(rng.integers(26, 31))
    low_h, low_w = {2: (18, 19), 3: (23, 24), 4: (28, 29)}[counts[0]]
    return int(rng.integers(low_h, 31)), int(rng.integers(low_w, 31))


def _try_layout(tokens, centers, shape):
    try:
        return _layout(tokens, centers, shape)
    except ValueError:
        return None


def _sample_layout(rng, kinds, counts, inactive_count, extra_count):
    for _ in range(500):
        shape = _shape(rng, counts)
        tokens, centers = [], []
        success = True
        for kind, count in zip(kinds, counts):
            placed = False
            for _ in range(100 if tokens else 15):
                candidate = _chain(rng, shape, kind, count)
                if candidate is None:
                    continue
                ts, cs = candidate
                info = _try_layout(tokens + ts, centers + cs, shape)
                if info is None or len(info["crossings"]) != len(tokens) // 2 + 1:
                    continue
                if sorted(map(len, info["groups"])) != sorted(counts[:len(tokens) // 2 + 1]):
                    continue
                tokens += ts
                centers += cs
                placed = True
                break
            if not placed:
                success = False
                break
        if not success:
            continue
        reference = _layout(tokens, centers, shape)
        origins = reference["crossings"]
        # Unaddressed sockets are genuine distractors. Sometimes put one on
        # the same infinite diagonal, northeast of the collision origin.
        wanted = inactive_count
        for slot in range(wanted):
            candidates = []
            if slot == 0 and rng.random() < 0.45:
                for r, c in origins:
                    candidates.extend((r - d, c + d) for d in range(5, 11))
            h, w = shape
            candidates.extend((int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
                              for _ in range(180))
            for center in candidates:
                info = _try_layout(tokens, centers + [center], shape)
                if info is not None and len(info["addressed"]) == sum(counts):
                    centers.append(center)
                    break
            else:
                break
        if len(centers) == sum(counts):
            continue  # Every generated scene has at least one inactive socket.
        for _ in range(extra_count):
            for _ in range(100):
                h, w = shape
                role = int(rng.choice(tuple(SPECS)))
                candidate = (role, int(rng.integers(h)), int(rng.integers(w)))
                info = _try_layout(tokens + [candidate], centers, shape)
                if info is not None and info["crossings"] == origins:
                    tokens.append(candidate)
                    break
        return tokens, centers, shape
    raise RuntimeError("bounded layout search exhausted")


def _masks(rng, number, familiar):
    result, used = [], set()
    for _ in range(number):
        available = sorted(familiar - used)
        if available and rng.random() < 0.30:
            key = int(rng.choice(available))
        else:
            for _attempt in range(2048):
                size = int(rng.choice((2, 3, 4, 5, 6, 7, 8),
                                      p=(.04, .06, .20, .23, .23, .20, .04)))
                bits = rng.choice(9, size=size, replace=False)
                key = sum(1 << int(i) for i in bits)
                if key not in used:
                    break
            else:
                raise RuntimeError("bounded mask sampling exhausted")
        used.add(key)
        result.append(np.array([(key >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3))
    return result


def _pair(rng, kinds, counts, familiar, used, extras):
    for _ in range(20):
        tokens, centers, shape = _sample_layout(
            rng, kinds, counts, int(rng.integers(1, 3)), extras)
        masks = _masks(rng, len(centers), familiar)
        colors = [int(x) for x in rng.choice(np.arange(1, 10), len(centers), replace=False)]
        inp = np.zeros(shape, dtype=np.int8)
        for role, r, c in tokens:
            inp[r, c] = role
        for (r, c), ink, mask in zip(centers, colors, masks):
            inp[r - 2:r + 3, c - 2:c + 3] = ink
            inside = inp[r - 1:r + 2, c - 1:c + 2]
            inside[:] = 0
            inside[mask] = ink
        key = _key(inp)
        if key in used:
            continue
        used.add(key)
        familiar.update(_mask_key(mask) for mask in masks)
        return {"input": inp, "output": solve(inp)}
    raise RuntimeError("could not produce a distinct input")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    trained = COMPLEMENTS[int(rng.integers(2))]
    novel = tuple(kind for kind in KINDS if kind not in trained)
    used, familiar = set(), set()
    train = [_pair(rng, [trained[0]], [3], familiar, used, int(rng.integers(2))),
             _pair(rng, [trained[1]], [2], familiar, used, int(rng.integers(2)))]
    for i in range(2, num_train):
        multi = i == 2 or rng.random() < .35
        if multi:
            kinds = [trained[int(rng.integers(2))] for _ in range(2)]
            counts = [2, 2]
        else:
            kinds = [trained[int(rng.integers(2))]]
            counts = [int(rng.choice((2, 3)))]
        train.append(_pair(rng, kinds, counts, familiar, used, int(rng.integers(3))))
    train = [train[int(i)] for i in rng.permutation(num_train)]
    test = []
    for _ in range(num_test):
        multi = num_train >= 3 and rng.random() < .45
        counts = [2, 2] if multi else [int(rng.choice((2, 3, 4), p=(.40, .50, .10)))]
        kinds = [novel[int(rng.integers(2))] for _ in counts]
        test.append(_pair(rng, kinds, counts, familiar, used, int(rng.integers(3))))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    info = _decode(pair["input"])
    out = _grid(pair["output"])
    assert out.shape == info["shape"]
    assert len(info["crossings"]) in (1, 2)
    assert all(2 <= len(g) <= 4 for g in info["groups"])
    assert len(info["addressed"]) < len(info["centers"])
    # Pixel-level obligations, including fixed frames and exact mask transport.
    for (role, r, c), endpoint in zip(info["tokens"], info["endpoints"]):
        assert out[r, c] == role and out[endpoint] == role
    for p, owners in info["owners"].items():
        assert out[p] == (4 if len(owners) == 2 else 5)
    for ray in info["rays"].values():
        for p in ray:
            if p not in info["covered"]:
                assert out[p] == 4
    source_for = {i: i for i in range(len(info["centers"]))}
    for group in info["groups"]:
        for j, dest in enumerate(group):
            source_for[dest] = group[(j + 1) % len(group)]
    output_masks = []
    for i, ((r, c), ink) in enumerate(zip(info["centers"], info["colors"])):
        block = out[r - 2:r + 3, c - 2:c + 3]
        assert np.all(block[0] == ink) and np.all(block[4] == ink)
        assert np.all(block[:, 0] == ink) and np.all(block[:, 4] == ink)
        expected = np.where(info["masks"][source_for[i]], ink, 0)
        assert np.array_equal(block[1:4, 1:4], expected)
        output_masks.append(_mask_key(block[1:4, 1:4] != 0))
    assert Counter(output_masks) == Counter(map(_mask_key, info["masks"]))
    assert np.array_equal(out, _render(info))  # Also catches any stray painted cell.
    assert not np.array_equal(out, _render(info, groups=()))  # Omitting B.
    assert not np.array_equal(out, _render(info, groups=(tuple(range(len(info["centers"]))),)))
    assert not np.array_equal(out, _render(info, carry_ink=True))
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            infos[split].append(info)
    roles = {t for info in infos["train"] for t, _, _ in info["tokens"]}
    assert roles == set(SPECS)
    trained = {kind for info in infos["train"] for kind in info["crossings"].values()}
    assert any(trained == set(pair) for pair in COMPLEMENTS)
    assert any(3 in map(len, info["groups"]) for info in infos["train"])
    assert any(2 in map(len, info["groups"]) for info in infos["train"])
    # Three distinct masks expose direction, wraparound, and the full cycle.
    witness = next(info for info in infos["train"] if 3 in map(len, info["groups"]))
    assert not np.array_equal(_render(witness), _render(witness, reverse=True))
    truncated = tuple(g[:2] for g in witness["groups"])
    assert not np.array_equal(_render(witness), _render(witness, groups=truncated))
    for info in infos["test"]:
        assert not (set(info["crossings"].values()) & trained)
        if len(info["groups"]) == 2:
            assert any(len(x["groups"]) == 2 for x in infos["train"])


def self_test():
    """Bounded checks of semantics, ablations, counts, and 42-seed diversity."""
    shapes, structures, masks, palettes, kinds, lengths = set(), set(), set(), set(), set(), set()
    group_sizes, token_counts, card_counts, gaps = set(), set(), set(), set()
    repeated = multi = northeast_inactive = familiar_tests = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        learned_masks = set()
        for split in ("train", "test"):
            for p in puzzle[split]:
                info = _decode(p["input"])
                assert np.array_equal(solve(np.array(p["input"], copy=True)), p["output"])
                shapes.add(info["shape"])
                # This signature discards carrier colors and glyph identities.
                structures.add((info["shape"], info["centers"], info["tokens"]))
                keys = set(map(_mask_key, info["masks"]))
                masks.update(keys)
                if split == "train":
                    learned_masks.update(keys)
                else:
                    familiar_tests += bool(keys & learned_masks)
                palettes.add(info["colors"])
                kinds.update(info["crossings"].values())
                lengths.update(len(ray) for ray in info["rays"].values())
                group_sizes.update(map(len, info["groups"]))
                token_counts.add(len(info["tokens"]))
                card_counts.add(len(info["centers"]))
                repeated += len({t for t, _, _ in info["tokens"]}) < len(info["tokens"])
                multi += len(info["groups"]) == 2
                for group in info["groups"]:
                    gaps.update(info["centers"][b][0] - info["centers"][a][0]
                                for a, b in zip(group, group[1:]))
                for i, (r, c) in enumerate(info["centers"]):
                    northeast_inactive += (i not in info["addressed"] and any(
                        r + c == rr + cc and r < rr for rr, cc in info["crossings"]))
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(structures) >= 180
    assert len(masks) >= 180 and len(palettes) >= 150
    assert kinds == set(KINDS) and group_sizes == {2, 3, 4}
    assert len(lengths) >= 10 and len(token_counts) >= 4 and len(card_counts) >= 3
    assert gaps >= {6, 7, 8} and repeated > 20 and multi >= 42 and familiar_tests >= 10
    assert northeast_inactive > 0
    # The specified empty/singleton-cycle boundary convention is supported
    # independently of the generator's choice to expose actual movement.
    for first_center in ((10, 3), (10, 12)):
        a = np.zeros((25, 25), dtype=np.int8)
        a[4, 20], a[8, 18] = 2, 6
        centers = (first_center, (20, 20))
        for i, (r, c) in enumerate(centers):
            a[r - 2:r + 3, c - 2:c + 3] = 7 + i
            if i == 0:
                a[r - 1:r + 2, c - 1:c + 2] = 0
                a[r, c] = 7
        expected = a.copy()
        expected[4, 16], expected[2, 18] = 2, 6
        expected[4, 17:20] = 5
        expected[3:8, 18] = 5
        expected[4, 18] = 4
        for d in range(1, 19):
            r, c = 4 + d, 18 - d
            if not any(abs(r - rr) <= 2 and abs(c - cc) <= 2 for rr, cc in centers):
                expected[r, c] = 4
        assert np.array_equal(solve(a), expected)
    # All legal explicit count combinations and invalid types/relationships.
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            validate_generated_puzzle(generate_puzzle(1234 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(789, 7, 3), generate_puzzle(789, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for split in a for x, y in zip(a[split], b[split])
               for k in ("input", "output"))
    return {"seeds": 42, "shapes": len(shapes), "layouts": len(structures),
            "masks": len(masks), "palettes": len(palettes), "multi_scenes": multi,
            "repeated_source_scenes": repeated, "northeast_inactive": northeast_inactive,
            "familiar_test_scenes": familiar_tests, "group_sizes": sorted(group_sizes)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
