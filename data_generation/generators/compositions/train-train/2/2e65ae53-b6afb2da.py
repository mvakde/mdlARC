"""Distributed nine-slice key: composition of ARC 2e65ae53 (A) and b6afb2da (B).

Each input holds a background and several separated solid rectangles painted in a
"hidden" colour.  A few cells of some rectangles already show their final colour.
Every cell of a rectangle belongs to one of nine slice roles -- its four corner
cells, its four non-corner edge runs (top, bottom, left, right) and its strict
interior (B's corner/edge/interior topology, refined by A's fixed positional
registration).  All rectangles in one input share one hidden 3x3 colour key, one
colour per role.  The revealed cells are scattered witnesses of that key (A's
distributed clue integration).  The output reconstructs the key from all
witnesses and repaints every rectangle by nine-slice scaling: corners stay single
cells, edges stretch along one axis, the interior stretches along both.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Roles and key symmetry classes
# ---------------------------------------------------------------------------

ROLE_NAMES = {
    (0, 0): "TL", (0, 1): "T", (0, 2): "TR",
    (1, 0): "L", (1, 1): "I", (1, 2): "R",
    (2, 0): "BL", (2, 1): "B", (2, 2): "BR",
}
ALL_ROLES = tuple(sorted(ROLE_NAMES))

# Orbit maps: cells with equal ids share one colour (before optional merges).
KEY_CLASSES = {
    "ring": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],          # B-like three-role key
    "axes": [[0, 1, 0], [2, 3, 2], [0, 1, 0]],
    "mirror_lr": [[0, 1, 0], [2, 3, 2], [4, 5, 4]],
    "mirror_tb": [[0, 1, 2], [3, 4, 5], [0, 1, 2]],
    "rot180": [[0, 1, 2], [3, 4, 3], [2, 1, 0]],
    "transpose": [[0, 1, 2], [1, 3, 4], [2, 4, 5]],
    "free": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
}
NON_RING_CLASSES = tuple(k for k in KEY_CLASSES if k != "ring")


def _role_axis(index, length):
    if index == 0:
        return 0
    if index == length - 1:
        return 2
    return 1


def _role_map(height, width):
    """3x3 role index for every cell of a height x width rectangle."""
    rows = np.array([_role_axis(i, height) for i in range(height)])
    cols = np.array([_role_axis(j, width) for j in range(width)])
    return rows[:, None], cols[None, :]


def _render(key, height, width):
    rows, cols = _role_map(height, width)
    return np.asarray(key)[rows, cols]


def _is_ring(key):
    key = np.asarray(key)
    corners = {key[0, 0], key[0, 2], key[2, 0], key[2, 2]}
    edges = {key[0, 1], key[1, 0], key[1, 2], key[2, 1]}
    return len(corners) == 1 and len(edges) == 1


# ---------------------------------------------------------------------------
# Solver (input only)
# ---------------------------------------------------------------------------

def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(height):
        for c in range(width):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _analyse(grid):
    """Return (background, hidden, boxes, key) inferred from the input alone."""
    grid = np.asarray(grid, dtype=int)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    comps = _components(grid != background)
    if not comps:
        raise ValueError("no rectangles")
    boxes = []
    for cells in comps:
        ys = [p[0] for p in cells]
        xs = [p[1] for p in cells]
        top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
        if len(cells) != (bottom - top + 1) * (right - left + 1):
            raise ValueError("component is not a solid rectangle")
        boxes.append((top, left, bottom - top + 1, right - left + 1))
    # Hidden colour: the most frequent colour inside the rectangles.
    totals = {}
    for top, left, h, w in boxes:
        patch = grid[top:top + h, left:left + w]
        for v, n in zip(*np.unique(patch, return_counts=True)):
            totals[int(v)] = totals.get(int(v), 0) + int(n)
    hidden = max(sorted(totals), key=lambda v: totals[v])
    key = {}
    for top, left, h, w in boxes:
        patch = grid[top:top + h, left:left + w]
        rows, cols = _role_map(h, w)
        rows = np.broadcast_to(rows, (h, w))
        cols = np.broadcast_to(cols, (h, w))
        for y in range(h):
            for x in range(w):
                v = int(patch[y, x])
                if v == hidden:
                    continue
                role = (int(rows[y, x]), int(cols[y, x]))
                if key.get(role, v) != v:
                    raise ValueError("conflicting witnesses")
                key[role] = v
    return background, hidden, boxes, key


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    _, _, boxes, key = _analyse(grid)
    if set(key) != set(ALL_ROLES):
        raise ValueError("key not fully witnessed")
    table = np.zeros((3, 3), dtype=int)
    for (r, c), v in key.items():
        table[r, c] = v
    out = grid.copy()
    for top, left, h, w in boxes:
        out[top:top + h, left:left + w] = _render(table, h, w)
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for value, name in ((num_train, "num_train"), (num_test, "num_test")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _make_key(rng, cls, avoid):
    """Sample a 3x3 key of class `cls` using colours not in `avoid`."""
    orbit = np.array(KEY_CLASSES[cls])
    k = int(orbit.max()) + 1
    pool = [c for c in range(10) if c not in avoid]
    for _ in range(200):
        if k <= len(pool):
            colours = list(rng.choice(pool, size=k, replace=False))
            if cls != "ring" and k >= 4 and rng.random() < 0.3:
                a, b = rng.choice(k, size=2, replace=False)
                colours[a] = colours[b]
        else:
            m = int(rng.integers(5, min(len(pool), 7) + 1))
            palette = list(rng.choice(pool, size=m, replace=False))
            colours = [palette[i] for i in range(m)] + list(rng.choice(palette, size=k - m))
            colours = [colours[i] for i in rng.permutation(k)]
        key = np.array([[int(colours[orbit[r, c]]) for c in range(3)] for r in range(3)])
        if len(set(key.ravel().tolist())) < 3:
            continue
        if cls != "ring" and _is_ring(key):
            continue
        return key
    raise RuntimeError("could not sample key")


def _place(rng, height, width, sizes):
    """Place rectangles with at least one background cell between any two (8-sep)."""
    occupied = np.zeros((height, width), dtype=bool)
    boxes = []
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
    placed = {}
    for i in order:
        h, w = sizes[i]
        if h > height or w > width:
            return None
        ok = False
        for _ in range(80):
            top = int(rng.integers(0, height - h + 1))
            left = int(rng.integers(0, width - w + 1))
            y0, y1 = max(0, top - 1), min(height, top + h + 1)
            x0, x1 = max(0, left - 1), min(width, left + w + 1)
            if not occupied[y0:y1, x0:x1].any():
                occupied[top:top + h, left:left + w] = True
                placed[i] = (top, left, h, w)
                ok = True
                break
        if not ok:
            return None
    for i in range(len(sizes)):
        boxes.append(placed[i])
    return boxes


def _sample_sizes(rng, count, regime):
    sizes = []
    for i in range(count):
        if regime == "test" and i == 0:
            # Elongated band: one side 3, the other long.
            long_side = int(rng.integers(8, 14))
            sizes.append((3, long_side) if rng.random() < 0.5 else (long_side, 3))
            continue
        if regime == "test" and i == 1:
            sizes.append((int(rng.integers(6, 10)), int(rng.integers(6, 10))))
            continue
        hi = 8 if regime == "train" else 9
        h = int(rng.integers(3, hi + 1))
        w = int(rng.integers(3, hi + 1))
        sizes.append((h, w))
    # Every example needs at least one rectangle that is longer than 3 somewhere
    if all(h == 3 and w == 3 for h, w in sizes):
        sizes[0] = (3, int(rng.integers(4, 7)))
    return sizes


def _role_cells(box):
    top, left, h, w = box
    rows, cols = _role_map(h, w)
    rows = np.broadcast_to(rows, (h, w))
    cols = np.broadcast_to(cols, (h, w))
    cells = {role: [] for role in ALL_ROLES}
    for y in range(h):
        for x in range(w):
            cells[(int(rows[y, x]), int(cols[y, x]))].append((top + y, left + x))
    return cells


def _make_pair(rng, regime, cls, used_keys, allow_full_key):
    for _attempt in range(300):
        background, hidden = (int(v) for v in rng.choice(10, size=2, replace=False))
        key = _make_key(rng, cls, {background, hidden})
        if any(np.array_equal(key, k) for k in used_keys):
            continue
        if regime == "train":
            count = int(rng.integers(2, 6))
            height, width = int(rng.integers(10, 21)), int(rng.integers(10, 21))
        else:
            count = int(rng.integers(4, 8))
            height, width = int(rng.integers(16, 31)), int(rng.integers(16, 31))
        sizes = _sample_sizes(rng, count, regime)
        boxes = _place(rng, height, width, sizes)
        if boxes is None:
            continue
        # --- distribute witnesses -----------------------------------------
        n = len(boxes)
        order = [int(i) for i in rng.permutation(n)]
        if regime == "test":
            blanks = int(rng.integers(1, max(1, n - 2) + 1))
        else:
            blanks = int(rng.integers(0, n - 1))  # 0..n-2, leaves >= 2 donors
        donors = order[blanks:]
        reveal = {i: set() for i in range(n)}
        full_key_box = None
        if allow_full_key and regime == "train" and rng.random() < 0.5:
            full_key_box = donors[0]
            reveal[full_key_box] = set(ALL_ROLES)
        else:
            for role in ALL_ROLES:
                reveal[donors[int(rng.integers(len(donors)))]].add(role)
            extra = 0.2 if regime == "train" else 0.08
            for i in donors:
                for role in ALL_ROLES:
                    if rng.random() < extra:
                        reveal[i].add(role)
            if any(len(reveal[i]) == 9 for i in donors):
                continue
        grid = np.full((height, width), background, dtype=int)
        for top, left, h, w in boxes:
            grid[top:top + h, left:left + w] = hidden
        full_p = 0.5 if regime == "train" else 0.15
        for i, box in enumerate(boxes):
            cells = _role_cells(box)
            for role in reveal[i]:
                pts = cells[role]
                if i == full_key_box or rng.random() < full_p or len(pts) == 1:
                    chosen = pts
                else:
                    m = int(rng.integers(1, max(1, len(pts) // 2) + 1))
                    idx = rng.choice(len(pts), size=m, replace=False)
                    chosen = [pts[j] for j in idx]
                for y, x in chosen:
                    grid[y, x] = key[role]
        out = grid.copy()
        for top, left, h, w in boxes:
            out[top:top + h, left:left + w] = _render(key, h, w)
        pair = {"input": grid, "output": out}
        try:
            _check_pair(pair)
        except AssertionError:
            continue
        return pair, key, boxes, reveal
    raise RuntimeError("could not build pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    classes = []
    ring_used = False
    for i in range(num_train):
        if i >= 2 and not ring_used and rng.random() < 0.3:
            classes.append("ring")
            ring_used = True
        else:
            classes.append(str(rng.choice(NON_RING_CLASSES)))
    classes = [classes[j] for j in rng.permutation(num_train)]
    # At most one ring (three-role) key, never among the first two draws, so >= 2 non-ring trains.
    full_key_index = int(rng.integers(num_train)) if (num_train >= 3 and rng.random() < 0.3) else -1
    used_keys, seen = [], set()
    train, test = [], []
    for i in range(num_train):
        while True:
            pair, key, _, _ = _make_pair(rng, "train", classes[i], used_keys, i == full_key_index)
            sig = pair["input"].tobytes() + bytes(pair["input"].shape)
            if sig not in seen:
                break
        seen.add(sig)
        used_keys.append(key)
        train.append({"input": pair["input"].tolist(), "output": pair["output"].tolist()})
    for _ in range(num_test):
        cls = str(rng.choice(NON_RING_CLASSES))
        while True:
            pair, key, _, _ = _make_pair(rng, "test", cls, used_keys, False)
            sig = pair["input"].tobytes() + bytes(pair["input"].shape)
            if sig not in seen:
                break
        seen.add(sig)
        used_keys.append(key)
        test.append({"input": pair["input"].tolist(), "output": pair["output"].tolist()})
    return {"train": train, "test": test}


# ---------------------------------------------------------------------------
# Validation and shortcut models
# ---------------------------------------------------------------------------

def _check_pair(pair):
    grid = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    assert grid.ndim == 2 and grid.shape == out.shape
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert grid.min() >= 0 and grid.max() <= 9 and out.min() >= 0 and out.max() <= 9
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    assert len(counts) < 2 or counts[order[0]] > counts[order[1]], "background not a strict majority"
    try:
        background, hidden, boxes, key = _analyse(grid)
    except ValueError as err:
        raise AssertionError(str(err))
    assert len(boxes) >= 2, "need at least two rectangles"
    assert set(key) == set(ALL_ROLES), "every slice role must be witnessed"
    assert hidden not in key.values() and background not in key.values()
    # Hidden colour: strictly more frequent than any key colour in the input.
    hid_n = int((grid == hidden).sum())
    for v in set(key.values()):
        assert hid_n > int((grid == v).sum()), "hidden colour not dominant"
    assert sum(int((grid[t:t + h, l:l + w] == hidden).any()) for t, l, h, w in boxes) >= len(boxes) - 1
    # 8-separation of rectangles and minimum size 3x3.
    occ = np.full(grid.shape, -1)
    for i, (top, left, h, w) in enumerate(boxes):
        assert h >= 3 and w >= 3, "rectangles must be at least 3x3"
        y0, y1 = max(0, top - 1), min(grid.shape[0], top + h + 1)
        x0, x1 = max(0, left - 1), min(grid.shape[1], left + w + 1)
        assert not ((occ[y0:y1, x0:x1] >= 0) & (occ[y0:y1, x0:x1] != i)).any(), "rectangles touch"
        occ[top:top + h, left:left + w] = i
    # Integration needed: some rectangle still hidden-containing beyond its own clues.
    per_box_roles = []
    for top, left, h, w in boxes:
        patch = grid[top:top + h, left:left + w]
        rows, cols = _role_map(h, w)
        rows = np.broadcast_to(rows, (h, w))
        cols = np.broadcast_to(cols, (h, w))
        roles = {(int(rows[y, x]), int(cols[y, x])) for y in range(h) for x in range(w) if patch[y, x] != hidden}
        per_box_roles.append(roles)
    assert sum(len(r) < 9 for r in per_box_roles) >= 1
    assert not np.array_equal(grid, out)
    assert np.array_equal(solve(grid), out), "output does not match rule"
    assert any(max(h, w) > 3 for _, _, h, w in boxes), "nine-slice needs a long side"
    return background, hidden, boxes, key, per_box_roles


def _shortcut_ring(grid):
    """B-only: one corner colour, one edge colour, one interior colour (majority)."""
    grid = np.asarray(grid)
    _, hidden, boxes, key = _analyse(grid)
    def pick(roles):
        vals = [key[r] for r in roles if r in key]
        return max(sorted(set(vals)), key=vals.count) if vals else hidden
    corner = pick([(0, 0), (0, 2), (2, 0), (2, 2)])
    edge = pick([(0, 1), (1, 0), (1, 2), (2, 1)])
    inner = pick([(1, 1)])
    table = np.array([[corner, edge, corner], [edge, inner, edge], [corner, edge, corner]])
    out = grid.copy()
    for top, left, h, w in boxes:
        out[top:top + h, left:left + w] = _render(table, h, w)
    return out


def _shortcut_local(grid):
    """No A: each rectangle completed only from its own witnesses."""
    grid = np.asarray(grid)
    _, hidden, boxes, _ = _analyse(grid)
    out = grid.copy()
    for top, left, h, w in boxes:
        patch = grid[top:top + h, left:left + w]
        rows, cols = _role_map(h, w)
        rows = np.broadcast_to(rows, (h, w))
        cols = np.broadcast_to(cols, (h, w))
        local = {}
        for y in range(h):
            for x in range(w):
                if patch[y, x] != hidden:
                    local[(int(rows[y, x]), int(cols[y, x]))] = int(patch[y, x])
        for y in range(h):
            for x in range(w):
                role = (int(rows[y, x]), int(cols[y, x]))
                if role in local:
                    out[top + y, left + x] = local[role]
    return out


def _shortcut_uniform(grid):
    """Stretch the 3x3 key uniformly (thirds) instead of nine-slice scaling."""
    grid = np.asarray(grid)
    _, _, boxes, key = _analyse(grid)
    table = np.zeros((3, 3), dtype=int)
    for (r, c), v in key.items():
        table[r, c] = v
    out = grid.copy()
    for top, left, h, w in boxes:
        rows = (np.arange(h) * 3) // h
        cols = (np.arange(w) * 3) // w
        out[top:top + h, left:left + w] = table[rows[:, None], cols[None, :]]
    return out


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    keys = []
    nonring_train = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            grid = np.asarray(pair["input"], dtype=int)
            sig = (grid.shape, grid.tobytes())
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            _, _, boxes, key, per_box = _check_pair(pair)
            table = tuple(key[r] for r in ALL_ROLES)
            keys.append(table)
            out = np.asarray(pair["output"], dtype=int)
            assert not np.array_equal(_shortcut_local(grid), out), "per-rectangle shortcut suffices"
            assert not np.array_equal(_shortcut_uniform(grid), out), "uniform scaling suffices"
            is_ring = _is_ring(np.array(table).reshape(3, 3))
            if split == "train" and not is_ring:
                nonring_train += 1
            if split == "test":
                assert not is_ring, "test key must need positional registration"
                assert all(len(r) < 9 for r in per_box), "test must not contain a full key"
                assert not np.array_equal(_shortcut_ring(grid), out)
    assert nonring_train >= 2
    assert len(set(keys)) == len(keys), "each example must have its own key"


def self_test() -> None:
    stats = {"classes": set(), "counts": set(), "shapes": set(), "blank": 0, "full": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        assert all(np.array_equal(np.asarray(a["input"]), np.asarray(b["input"]))
                   for a, b in zip(puzzle["train"] + puzzle["test"], again["train"] + again["test"]))
        for pair in puzzle["train"] + puzzle["test"]:
            grid = np.asarray(pair["input"])
            _, hidden, boxes, key = _analyse(grid)
            stats["counts"].add(len(boxes))
            stats["shapes"].add(grid.shape)
            for top, left, h, w in boxes:
                patch = grid[top:top + h, left:left + w]
                if (patch == hidden).all():
                    stats["blank"] += 1
                if not (patch == hidden).any():
                    stats["full"] += 1
            stats["classes"].add(tuple(key[r] for r in ALL_ROLES))
    assert len(stats["counts"]) >= 5, stats["counts"]
    assert len(stats["shapes"]) >= 40
    assert stats["blank"] >= 20
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (3, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test ok")
