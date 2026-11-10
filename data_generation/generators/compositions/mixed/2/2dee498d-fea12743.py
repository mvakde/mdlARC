"""Global motif-column pruning followed by a strict binary-union search.

Only NumPy and the standard library are used. Generation and self_test perform
no file I/O; solve knows neither the seed nor any previously generated examples.
"""
from __future__ import annotations

import hashlib
import numpy as np

OPERAND_COLOR = 8
RESULT_COLOR = 3


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _slot(i, h, w):
    row, col = divmod(i, 2)
    return (slice(1 + row * (h + 1), (row + 1) * (h + 1)),
            slice(1 + col * (w + 1), (col + 1) * (w + 1)))


def _extract(value):
    grid = _grid(value)
    H, W = grid.shape
    if H < 7 or W < 5 or (H - 4) % 3 or (W - 3) % 2:
        raise ValueError("expected a 3-row, 2-column array with one-cell gutters")
    h, w = (H - 4) // 3, (W - 3) // 2
    bg = int(grid[0, 0])
    colors = set(map(int, np.unique(grid)))
    if len(colors) != 2 or colors & {OPERAND_COLOR, RESULT_COLOR}:
        raise ValueError("input needs two colors, excluding highlight colors 8 and 3")
    fg = next(iter(colors - {bg}))
    gutter = np.ones(grid.shape, dtype=bool)
    raw = []
    for i in range(6):
        sl = _slot(i, h, w)
        gutter[sl] = False
        raw.append((grid[sl] == fg).copy())
    if not np.all(grid[gutter] == bg):
        raise ValueError("broken border or gutter")
    if any(not np.all(np.any(mask, axis=0)) for mask in raw):
        raise ValueError("source motif columns must be nonempty")
    return grid, h, w, bg, fg, raw


def _keys(mask):
    return [tuple(map(bool, col)) for col in mask.T]


def _compact(raw, global_seen=True):
    """Padding is never scanned: only the actual source motif columns count."""
    seen = set()
    compact = []
    for mask in raw:
        if not global_seen:
            seen = set()
        kept = []
        for col in mask.T:
            key = tuple(map(bool, col))
            if key not in seen:
                seen.add(key)
                kept.append(col.copy())
        compact.append(np.stack(kept, axis=1) if kept else
                       np.zeros((mask.shape[0], 0), dtype=bool))
    return compact


def _relations(masks):
    relations = []
    for a in range(6):
        if masks[a].shape[1] == 0:
            continue
        for b in range(a + 1, 6):
            A, B = masks[a], masks[b]
            if A.shape != B.shape:
                continue
            if not (np.any(A & ~B) and np.any(B & ~A)):
                continue
            union = A | B
            for c in range(6):
                if c not in (a, b) and np.array_equal(union, masks[c]):
                    relations.append((a, b, c))
    return relations


def _render(h, w, bg, fg, compact, relation=None):
    result = np.full((3 * h + 4, 2 * w + 3), bg, dtype=np.int8)
    for i, mask in enumerate(compact):
        color = fg
        if relation is not None:
            a, b, c = relation
            color = RESULT_COLOR if i == c else OPERAND_COLOR if i in (a, b) else fg
        block = result[_slot(i, h, w)]
        block[:, :mask.shape[1]] = np.where(mask, color, bg)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Deduplicate globally; compare unpadded, equally wide compacted masks."""
    _, h, w, bg, fg, raw = _extract(input_grid)
    compact = _compact(raw)
    relations = _relations(compact)
    if len(relations) != 1:
        raise ValueError("family members require exactly one compacted strict union")
    return _render(h, w, bg, fg, compact, relations[0])


def _orientation(pair):
    a, b = pair
    return "H" if a // 2 == b // 2 else "V" if a % 2 == b % 2 else "D"


def _mask(values, h):
    return np.array([[(v >> r) & 1 for v in values] for r in range(h)], dtype=bool)


def _choose_roles(rng, orientation, echo):
    candidates = []
    for a in range(6):
        for b in range(a + 1, 6):
            if _orientation((a, b)) != orientation:
                continue
            for c in range(6):
                if c in (a, b):
                    continue
                echoes = [i for i in range(6) if i not in (a, b, c) and i > a]
                if not echo or echoes:
                    candidates.append(((a, b, c), echoes))
    relation, echoes = candidates[int(rng.integers(len(candidates)))]
    return relation, int(rng.choice(echoes)) if echo else None


def _latent_masks(rng, h, k, relation, echo_slot, widths):
    """Plant one equation using globally distinct nonzero column vectors."""
    total = sum(widths)
    # A common horizontal spine gives related silhouettes without forcing it
    # into every pair. The unrestricted mode permits disconnected masks.
    spine = int(rng.integers(h)) if total <= 2 ** (h - 1) and rng.random() < .45 else None
    pool = [v for v in range(1, 2 ** h) if spine is None or (v >> spine) & 1]
    for _ in range(160):
        used = set()
        A, B, C = [], [], []
        for j in range(k):
            for _ in range(500):
                x, y = map(int, rng.choice(pool, size=2, replace=False))
                z = x | y
                if len({x, y, z}) == 3 and not used & {x, y, z}:
                    used.update((x, y, z))
                    A.append(x); B.append(y); C.append(z)
                    break
            else:
                break
        if len(A) != k:
            continue
        if not any(x & y for x, y in zip(A, B)):
            # Overlap makes Boolean union differ from XOR and pixel addition.
            continue
        a, b, c = relation
        values = [None] * 6
        values[a], values[b], values[c] = A, B, C
        if echo_slot is not None:
            values[echo_slot] = []
        remaining = [v for v in pool if v not in used]
        rng.shuffle(remaining)
        for i in range(6):
            if values[i] is None:
                values[i] = remaining[:widths[i]]
                del remaining[:widths[i]]
        masks = [_mask(v, h) for v in values]
        if _relations(masks) == [relation]:
            return values, masks
    raise RuntimeError("bounded latent-mask construction exhausted")


def _repeat_indices(rng, k, w):
    ids = list(range(k))
    # Repeat before the final first occurrence: taking a prefix is wrong.
    ids.insert(int(rng.integers(1, k)), 0)
    while len(ids) < w:
        pos = int(rng.integers(1, len(ids) + 1))
        ids.insert(pos, int(rng.choice(ids[:pos])))
    return ids


def _raw_values(rng, values, relation, echo_slot, w):
    raw = []
    if echo_slot is not None:
        ids = _repeat_indices(rng, len(values[relation[0]]), w)
        for i in range(6):
            if i == echo_slot:
                raw.append(raw[relation[0]].copy())
            else:
                raw.append([values[i][j] for j in ids])
        return raw
    earlier = []
    for own in values:
        sequence = own.copy()
        if len(own) >= 2:
            sequence.insert(int(rng.integers(1, len(own))), own[0])
        if earlier:
            sequence.insert(int(rng.integers(len(sequence) + 1)), int(rng.choice(earlier)))
        while len(sequence) < w:
            pos = int(rng.integers(1, len(sequence) + 1))
            allowed = sequence[:pos] + earlier
            sequence.insert(pos, int(rng.choice(allowed)))
        raw.append(sequence)
        earlier.extend(own)
    return raw


def _make_pair(rng, palette, orientation, mode, test):
    h = 8 if test else int(rng.integers(5, 8))
    k = int(rng.integers(2, 5 if test else 4))
    echo = mode == "echo"
    relation, echo_slot = _choose_roles(rng, orientation, echo)
    widths = [k] * 6
    if echo:
        widths[echo_slot] = 0
    elif rng.random() < .65:
        free = [i for i in range(6) if i not in relation]
        rng.shuffle(free)
        widths[free[0]] = max(1, k - 1)
        widths[free[1]] = k + 1
        # The third distractor remains width k, so width alone cannot select
        # exactly the three role motifs.
    values, masks = _latent_masks(rng, h, k, relation, echo_slot, widths)
    w = int(rng.integers(max(widths) + 2, min(13, max(widths) + 5) + 1))
    bg, fg = palette
    for _ in range(80):
        raw_values = _raw_values(rng, values, relation, echo_slot, w)
        raw_masks = [_mask(v, h) for v in raw_values]
        compact = _compact(raw_masks)
        _require(all(np.array_equal(a, b) for a, b in zip(masks, compact)),
                 "expansion changed first occurrences")
        if not echo and (relation in _relations(raw_masks)
                         or relation in _relations(_compact(raw_masks, False))):
            continue
        if echo and len(_relations(raw_masks)) < 2:
            continue
        source = _render(h, w, bg, fg, raw_masks)
        target = _render(h, w, bg, fg, compact, relation)
        return {"input": source, "output": target}
    raise RuntimeError("bounded column-expansion construction exhausted")


def _key(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = np.array([c for c in range(10) if c not in (8, 3)])
    palettes = [tuple(map(int, rng.choice(colors, size=2, replace=False))) for _ in range(2)]
    orientations = list(rng.permutation(["H", "V", "D"]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mode = ("echo" if index == 0 else "cross") if split == "train" else str(rng.choice(["echo", "cross"]))
            orientation = str(orientations[index % 3]) if split == "train" else str(rng.choice(orientations))
            palette = palettes[index % 2] if split == "train" else palettes[int(rng.integers(2))]
            for _ in range(60):
                pair = _make_pair(rng, palette, orientation, mode, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    source, h, w, bg, fg, raw = _extract(pair["input"])
    target = _grid(pair["output"])
    _require(source.shape == target.shape, "canvas changed")
    compact = _compact(raw)
    relations = _relations(compact)
    _require(len(relations) == 1, "union must be unique after pruning")
    relation = relations[0]
    _require(np.any(compact[relation[0]] & compact[relation[1]]),
             "missing overlap witness against XOR")
    expected = _render(h, w, bg, fg, compact, relation)
    _require(np.array_equal(expected, target), "wrong compacted/recolored output")
    _require(np.array_equal(solve(source), target), "solve disagrees")
    _require(all(mask.shape[1] < w for mask in compact), "no deletion in a motif")
    all_keys = [key for mask in compact for key in _keys(mask)]
    _require(len(all_keys) == len(set(all_keys)), "retained vectors not globally unique")
    _require(sum(m.shape[1] == compact[relation[0]].shape[1] for m in compact) >= 4,
             "width alone exposes the role triple")
    # Check every output cell against visible first-occurrence provenance,
    # including unchanged surviving distractors and vacated suffixes.
    seen = set()
    for i, mask in enumerate(raw):
        kept = []
        for j, key in enumerate(_keys(mask)):
            if key not in seen:
                seen.add(key)
                kept.append(j)
        block = target[_slot(i, h, w)]
        color = RESULT_COLOR if i == relation[2] else OPERAND_COLOR if i in relation[:2] else fg
        for dest, src in enumerate(kept):
            _require(np.array_equal(block[:, dest], np.where(mask[:, src], color, bg)),
                     "first-occurrence provenance violated")
        _require(np.all(block[:, len(kept):] == bg), "vacated suffix not cleared")
    no_union = _render(h, w, bg, fg, compact)
    _require(not np.array_equal(no_union, target), "B ablation failed")
    local = _compact(raw, False)
    return {"h": h, "w": w, "palette": (bg, fg), "relation": relation,
            "widths": tuple(m.shape[1] for m in compact),
            "raw_relations": len(_relations(raw)),
            "cross_witness": relation not in _relations(local) and relation not in _relations(raw),
            "echo_witness": len(_relations(raw)) >= 2 and any(m.shape[1] == 0 for m in compact),
            "local_differs": any(not np.array_equal(a, b) for a, b in zip(local, compact)),
            "crop_fails": any(not np.array_equal(m, raw[i][:, :m.shape[1]])
                              for i, m in enumerate(compact) if m.shape[1])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(_analyze_pair(pair))
    train = records["train"]
    _require(any(a["echo_witness"] for a in train), "missing raw-tie pruning demonstration")
    _require(any(a["cross_witness"] for a in train), "missing global-before-union demonstration")
    _require(any(a["crop_fails"] for a in train), "missing first-occurrence order witness")
    _require(all(a["local_differs"] for a in train), "global equality not exercised")
    _require(len({_orientation(a["relation"][:2]) for a in train}) >= 2,
             "training must vary operand placement")
    _require(all(a["h"] < 8 for a in train), "training height regime")
    for a in records["test"]:
        _require(a["h"] == 8, "missing test height extrapolation")
        _require(a["palette"] in {t["palette"] for t in train}, "unseen test palette")
    return None


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape)); digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming 42-seed study; no saved grids, logs, or traces."""
    signatures, shapes, structures, role_positions, widths, palettes = (set() for _ in range(6))
    echo_count = cross_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                shapes.add(tuple(np.asarray(pair["input"]).shape))
                role_positions.add(info["relation"])
                widths.add(info["widths"])
                palettes.add(info["palette"])
                echo_count += info["echo_witness"]
                cross_count += info["cross_witness"]
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                structures.add(hashlib.sha256(bytes(pair["input"].shape) + normalized).hexdigest())
        _require(_fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle), "nondeterminism")
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 15 and len(structures) >= 200, "geometry/structure diversity")
    _require(len(role_positions) >= 30 and len(widths) >= 25 and len(palettes) >= 20,
             "placement/retained-width/palette diversity")
    _require(echo_count >= 42 and cross_count >= 42, "causal witnesses absent")
    for a, b in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(1000 + a * 10 + b, a, b)
        _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.5, 1), (4, True), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "normalized_inputs": len(structures), "role_triples": len(role_positions),
            "width_profiles": len(widths), "palettes": len(palettes),
            "echo_witness_pairs": echo_count, "cross_witness_pairs": cross_count}
