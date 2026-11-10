"""Competing self-stamps: input-only geometric expansion with count-priority occlusion."""
from __future__ import annotations

import itertools
import json
import numpy as np


# Shapes are (height, width). Full carrier rectangles make the extent explicit.
_SHAPES = ((3, 3), (3, 4), (4, 3), (3, 5), (5, 3),
           (4, 4), (4, 5), (5, 4), (5, 5))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _components(g, bg):
    seen = np.zeros(g.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(g != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                if (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]
                        and g[nr, nc] != bg and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        rows, cols = zip(*cells)
        box = min(rows), min(cols), max(rows)+1, max(cols)+1
        r0, c0, r1, c1 = box
        _require(len(cells) == (r1-r0)*(c1-c0), "cards must be full rectangles")
        result.append((box, g[r0:r1, c0:c1].copy()))
    return result


def _read_input(value):
    """Decode roles and objects using only the input, without puzzle conventions."""
    g = _grid(value)
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    _require(len(np.unique(border)) == 1, "uniform empty perimeter required")
    bg = int(border[0])
    objects = _components(g, bg)
    _require(2 <= len(objects) <= 4, "two to four cards required")
    for (a, _), (b, _) in itertools.combinations(objects, 2):
        _require(a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1],
                 "cards need a background-cell gap")
    palettes = [set(map(int, np.unique(card))) for _, card in objects]
    _require(all(len(p) == 2 for p in palettes), "each card has two colors")
    common = set.intersection(*palettes)
    _require(len(common) == 1, "one common accent color required")
    accent = common.pop()
    carriers, cards = set(), []
    for (box, card), colors in zip(objects, palettes):
        carrier = (colors - {accent}).pop()
        _require(carrier not in carriers, "carrier colors identify distinct cards")
        carriers.add(carrier)
        h, w = card.shape
        _require(3 <= h <= 5 and 3 <= w <= 5, "card sides must be 3..5")
        count = int(np.count_nonzero(card == accent))
        _require(2 <= count < card.size / 2, "accents must be sparse but plural")
        cards.append({"card": card, "mask": card == accent, "count": count,
                      "carrier": carrier, "box": box})
    counts = [c["count"] for c in cards]
    _require(len(set(counts)) == len(counts), "accent counts must be distinct")
    return bg, accent, cards


def _stamp_layer(card, accent, shape, bg):
    """A card's self-controlled candidate image and opaque footprint."""
    h, w = card.shape
    mask = card == accent
    footprint = np.repeat(np.repeat(mask, h, axis=0), w, axis=1)
    image = np.full(shape, bg, dtype=np.int8)
    opaque = np.zeros(shape, dtype=bool)
    image[:h*h, :w*w] = np.tile(card, (h, w))
    opaque[:h*h, :w*w] = footprint
    return image, opaque


def solve(input_grid: np.ndarray) -> np.ndarray:
    bg, accent, cards = _read_input(input_grid)
    shape = (max(c["card"].shape[0] for c in cards)**2,
             max(c["card"].shape[1] for c in cards)**2)
    output = np.full(shape, bg, dtype=np.int8)
    priority = np.full(shape, 100, dtype=np.int16)
    for c in cards:
        image, opaque = _stamp_layer(c["card"], accent, shape, bg)
        wins = opaque & (c["count"] < priority)
        output[wins] = image[wins]
        priority[wins] = c["count"]
    return output


def _render_cards(cards, accent, bg, order=None):
    """Generation path: paint entire stamps back to front."""
    shape = (max(c.shape[0] for c in cards)**2,
             max(c.shape[1] for c in cards)**2)
    out = np.full(shape, bg, dtype=np.int8)
    if order is None:
        order = sorted(range(len(cards)),
                       key=lambda i: int(np.count_nonzero(cards[i] == accent)),
                       reverse=True)
    for i in order:
        card = cards[i]
        h, w = card.shape
        for r, c in np.argwhere(card == accent):
            out[r*h:(r+1)*h, c*w:(c+1)*w] = card
    return out


def _analyze(value):
    """Direct stamp enumeration for validation; also supplies coverage witnesses."""
    bg, accent, cards = _read_input(value)
    shape = (max(c["card"].shape[0] for c in cards)**2,
             max(c["card"].shape[1] for c in cards)**2)
    owner = np.full(shape, -1, dtype=int)
    score = np.full(shape, 100, dtype=int)
    expected = np.full(shape, bg, dtype=np.int8)
    covers = []
    for i, fact in enumerate(cards):
        card, count = fact["card"], fact["count"]
        h, w = card.shape
        cover = np.zeros(shape, dtype=bool)
        for r in range(h):
            for c in range(w):
                if card[r, c] != accent:
                    continue
                for u in range(h):
                    for v in range(w):
                        rr, cc = r*h+u, c*w+v
                        cover[rr, cc] = True
                        if count < score[rr, cc]:
                            expected[rr, cc] = card[u, v]
                            score[rr, cc] = count
                            owner[rr, cc] = i
        covers.append(cover)
    stack = np.stack(covers)
    whole = []
    for i, fact in enumerate(cards):
        h, w = fact["card"].shape
        whole.append(any(np.all(owner[r*h:(r+1)*h, c*w:(c+1)*w] == i)
                         for r, c in np.argwhere(fact["mask"])))
    low = min(range(len(cards)), key=lambda i: cards[i]["count"])
    partial = any(np.any(owner[r*h:(r+1)*h, c*w:(c+1)*w] == i)
                  and np.any(owner[r*h:(r+1)*h, c*w:(c+1)*w] != i)
                  for i, f in enumerate(cards)
                  for h, w in [f["card"].shape]
                  for r, c in np.argwhere(f["mask"]))
    # A lower-density pattern contests the minimum, but absolute count wins.
    density_trap = any(
        i != low and f["count"] * cards[low]["card"].size
        < cards[low]["count"] * f["card"].size
        and np.any(covers[i] & covers[low])
        for i, f in enumerate(cards))
    return {"bg": bg, "accent": accent, "cards": cards, "owner": owner,
            "expected": expected, "covers": covers, "whole": whole,
            "contested": int(np.count_nonzero(stack.sum(axis=0) > 1)),
            "max_depth": int(stack.sum(axis=0).max()), "partial": partial,
            "density_trap": density_trap, "low": low}


def _counts_for_shapes(rng, shapes):
    caps = [(h*w-1)//2 for h, w in shapes]
    # Enumerating this small count space avoids rejection or capacity failures.
    choices = [x for x in itertools.product(*(range(2, min(cap, 8)+1) for cap in caps))
               if len(set(x)) == len(x)]
    return choices[int(rng.integers(len(choices)))]


def _scatter(rng, cards, bg):
    total = sum(card.size for card in cards)
    for attempt in range(60):
        # Empty perimeter fixes the background even if it is not color zero.
        h = int(rng.integers(11, 23)) if attempt < 45 else 26
        w = int(rng.integers(11, 23)) if attempt < 45 else 26
        if h*w < total*2:
            continue
        g = np.full((h, w), bg, dtype=np.int8)
        order = rng.permutation(len(cards))
        for index in order:
            card = cards[int(index)]
            ch, cw = card.shape
            candidates = []
            for r in range(1, h-ch):
                for c in range(1, w-cw):
                    if np.all(g[r-1:r+ch+1, c-1:c+cw+1] == bg):
                        candidates.append((r, c))
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            g[r:r+ch, c:c+cw] = card
        else:
            return g
    raise RuntimeError("free card placement exhausted")


def _make_pair(rng, shapes, counts, carriers, bg, accent, anchor, density_witness=False):
    for _ in range(400):
        cards = []
        for (h, w), count, carrier in zip(shapes, counts, carriers):
            card = np.full((h, w), carrier, dtype=np.int8)
            anchor_index = anchor[0]*w + anchor[1]
            pool = [j for j in range(h*w) if j != anchor_index]
            indices = rng.choice(pool, count-1, replace=False)
            card.flat[indices] = accent
            card[anchor] = accent
            cards.append(card)
        # A temporary placement is used only to ask the input-derived analyzer.
        temp = np.full((8, 6*len(cards)+1), bg, dtype=np.int8)
        for i, card in enumerate(cards):
            h, w = card.shape
            temp[1:1+h, 1+6*i:1+6*i+w] = card
        facts = _analyze(temp)
        if (not all(facts["whole"]) or facts["contested"] == 0
                or facts["max_depth"] != len(cards)
                or not np.any(facts["owner"] == -1)):
            continue
        # For mixed shapes, demonstrate partial geometric occlusion explicitly.
        if len(set(shapes)) > 1 and not facts["partial"]:
            continue
        target = _render_cards(cards, accent, bg)
        wrong_order = sorted(range(len(cards)), key=lambda i: counts[i])
        if np.array_equal(target, _render_cards(cards, accent, bg, wrong_order)):
            continue
        if density_witness:
            density_order = sorted(range(len(cards)),
                                   key=lambda i: counts[i]/cards[i].size, reverse=True)
            if np.array_equal(target, _render_cards(cards, accent, bg, density_order)):
                continue
            area_order = sorted(range(len(cards)), key=lambda i: cards[i].size, reverse=True)
            if np.array_equal(target, _render_cards(cards, accent, bg, area_order)):
                continue
        source = _scatter(rng, cards, bg)
        return {"input": source, "output": target}
    raise RuntimeError("unable to construct readable competing self-stamps")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = [int(x) for x in rng.permutation(10)]
    bg, accent = colors[:2]
    carriers = colors[2:6]
    depth_shift = bool(rng.integers(2))
    first_shape = _SHAPES[int(rng.integers(1, len(_SHAPES)))]
    small = (3, 4) if rng.integers(2) else (4, 3)
    third = ((3, 5), (5, 3), (4, 4))[int(rng.integers(3))]
    # Training vocabulary leaves (3,3), (4,5) or (5,4) available for a new shape.
    train_pool = sorted(set((first_shape, small, (5, 5), third)))
    novel_pool = [s for s in _SHAPES if s not in train_pool]
    puzzle, seen = {"train": [], "test": []}, set()
    anchor0 = (int(rng.integers(2)), int(rng.integers(2)))
    for section, count in (("train", nt), ("test", ns)):
        for index in range(count):
            if section == "train" and index == 0:
                shapes = [first_shape]*2
                counts = (2, int(rng.integers(3, min(8, (first_shape[0]*first_shape[1]-1)//2)+1)))
                ids = carriers[:2]
                anchor = anchor0
            elif section == "train" and index == 1:
                # The minimum is larger than one rival but denser than another.
                shapes = [third, (5, 5), small]
                counts = (3, 4, 5)
                ids = [carriers[1], carriers[0], carriers[2]]
                anchor = (1-anchor0[0], 1-anchor0[1])
            else:
                k = (4 if depth_shift else 3) if section == "test" else int(rng.integers(2, 4))
                shapes = [train_pool[int(rng.integers(len(train_pool)))] for _ in range(k)]
                if section == "test" and not depth_shift:
                    shapes[0] = novel_pool[int(rng.integers(len(novel_pool)))]
                counts = _counts_for_shapes(rng, shapes)
                ids = [carriers[int(i)] for i in rng.permutation(4)[:k]]
                anchor = (int(rng.integers(2)), int(rng.integers(2)))
            for _ in range(20):
                pair = _make_pair(rng, shapes, counts, ids, bg, accent, anchor,
                                  density_witness=section == "train" and index == 1)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("duplicate input capacity exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    facts = {"train": [], "test": []}
    seen = set()
    roles = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            key = _key(source)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(source)
            roles.add((a["bg"], a["accent"]))
            _require(np.array_equal(target, a["expected"]), "wrong stamp or occlusion")
            _require(np.array_equal(target, solve(source)), "input-only solve mismatch")
            _require(all(a["whole"]), "each card needs a wholly visible copy")
            _require(a["contested"] > 0 and a["max_depth"] == len(a["cards"]),
                     "missing collision evidence")
            _require(np.any(target == a["bg"]), "blank output territory required")
            card_arrays = [c["card"] for c in a["cards"]]
            # Count-blind max-priority overlay must differ on every example.
            reversed_priority = sorted(range(len(card_arrays)),
                                       key=lambda i: a["cards"][i]["count"])
            _require(not np.array_equal(target, _render_cards(
                card_arrays, a["accent"], a["bg"], reversed_priority)), "A ablation vacuous")
            # Merely selecting the global minimum and self-stamping it also fails.
            only_min = np.full(target.shape, a["bg"], dtype=np.int8)
            card = a["cards"][a["low"]]["card"]
            h, w = card.shape
            for r, c in np.argwhere(card == a["accent"]):
                only_min[r*h:(r+1)*h, c*w:(c+1)*w] = card
            _require(not np.array_equal(target, only_min), "global selection shortcut survives")
            # Ignoring the self-addresses and making every macrocell active fails.
            all_active = np.full(target.shape, a["bg"], dtype=np.int8)
            for f in sorted(a["cards"], key=lambda f: f["count"], reverse=True):
                h, w = f["card"].shape
                all_active[:h*h, :w*w] = np.tile(f["card"], (h, w))
            _require(not np.array_equal(target, all_active), "B ablation vacuous")
            facts[section].append(a)
    _require(len(roles) == 1, "background/accent conventions must be stable")
    t0, t1 = facts["train"][:2]
    _require(len(t0["cards"]) == 2 and len(t1["cards"]) == 3, "2/3-card demonstrations")
    _require(len({c["card"].shape for c in t0["cards"]}) == 1, "aligned first demonstration")
    _require(t1["density_trap"] and t1["partial"], "density and partial-overlap evidence")
    density_order = sorted(range(len(t1["cards"])),
                           key=lambda i: t1["cards"][i]["count"]/t1["cards"][i]["card"].size,
                           reverse=True)
    _require(not np.array_equal(t1["expected"], _render_cards(
        [f["card"] for f in t1["cards"]], t1["accent"], t1["bg"], density_order)),
        "density alternative must actually change a visible cell")
    area_order = sorted(range(len(t1["cards"])),
                        key=lambda i: t1["cards"][i]["card"].size, reverse=True)
    _require(not np.array_equal(t1["expected"], _render_cards(
        [f["card"] for f in t1["cards"]], t1["accent"], t1["bg"], area_order)),
        "area alternative must actually change a visible cell")
    lo0, lo1 = t0["cards"][t0["low"]], t1["cards"][t1["low"]]
    _require(lo0["count"] == 2 and lo1["count"] == 3, "minimum count evidence")
    _require(lo0["carrier"] != lo1["carrier"]
             and lo0["carrier"] in {c["carrier"] for c in t1["cards"]}
             and lo1["carrier"] in {c["carrier"] for c in t0["cards"]},
             "priority must reverse between shared carrier identities")
    train_shapes = {c["card"].shape for f in facts["train"] for c in f["cards"]}
    for a in facts["test"]:
        _require(len(a["cards"]) == 4 or any(c["card"].shape not in train_shapes for c in a["cards"]),
                 "missing structural train/test shift")


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for s in ("train", "test") for pair in puzzle[s])


def self_test():
    """Bounded coverage of 42 seeds, legal counts, ablations and metamorphisms."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed repetition")
    metrics = {"puzzles": 42, "pairs": 0, "partial_pairs": 0, "density_traps": 0}
    input_shapes, output_shapes, card_shapes, masks, role_pairs, counts = (set() for _ in range(6))
    shifts = set()
    for puzzle in puzzles:
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a = _analyze(pair["input"])
                metrics["pairs"] += 1
                metrics["partial_pairs"] += int(a["partial"])
                metrics["density_traps"] += int(a["density_trap"])
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                role_pairs.add((a["bg"], a["accent"]))
                counts.add(len(a["cards"]))
                for c in a["cards"]:
                    card_shapes.add(c["card"].shape)
                    masks.add((c["card"].shape, c["mask"].tobytes()))
                if section == "test":
                    shifts.add("depth" if len(a["cards"]) == 4 else "shape")
    _require(len(input_shapes) >= 35 and len(output_shapes) >= 6, "canvas diversity")
    _require(len(card_shapes) == 9 and len(masks) >= 200, "structural diversity")
    _require(len(role_pairs) >= 25 and counts == {2, 3, 4}, "roles or counts lack variation")
    _require(shifts == {"depth", "shape"} and metrics["density_traps"] >= 42, "shift/evidence coverage")
    _require(_puzzle_key(generate_puzzle(123, 10, 4))
             == _puzzle_key(generate_puzzle(123, 10, 4)), "determinism")
    # Every legal count combination; no automatic/partial-count convention is used.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ns, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "counts ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Same cards, new positions and arbitrary role recoloring: output is equivariant.
    p = puzzles[0]["train"][1]
    g = p["input"]
    bg, accent, cards = _read_input(g)
    relocated = _scatter(np.random.default_rng(991), [c["card"] for c in cards], bg)
    _require(np.array_equal(solve(relocated), p["output"]), "source placement leaked")
    perm = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    _require(np.array_equal(solve(perm[g]), perm[p["output"]]), "literal color dependency")
    # The rule extends beyond the sampling anchors and the sampled count cap (8).
    # These masks have no common active address; counts 11/12 remain valid on 5x5.
    extended = np.zeros((8, 14), dtype=np.int8)
    for col, carrier, indices in ((1, 2, range(11)), (8, 3, range(13, 25))):
        card = np.full((5, 5), carrier, dtype=np.int8)
        card.flat[list(indices)] = 1
        extended[1:6, col:col+5] = card
    _require(np.array_equal(solve(extended), _analyze(extended)["expected"]),
             "solver relied on shared anchors or count cap")
    metrics.update(input_shapes=len(input_shapes), output_shapes=len(output_shapes),
                   card_shapes=len(card_shapes), distinct_masks=len(masks), role_pairs=len(role_pairs))
    return metrics


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
