from hashlib import sha256

import numpy as np


_EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS = _EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    contiguous = np.ascontiguousarray(grid)
    return contiguous.shape, contiguous.astype(np.int8, copy=False).tobytes()


def _puzzle_equal(a: dict, b: dict) -> bool:
    if set(a.keys()) != {"train", "test"}:
        return False
    if set(b.keys()) != {"train", "test"}:
        return False

    for section in ("train", "test"):
        if len(a[section]) != len(b[section]):
            return False
        for pa, pb in zip(a[section], b[section]):
            if set(pa.keys()) != {"input", "output"}:
                return False
            if set(pb.keys()) != {"input", "output"}:
                return False
            if not np.array_equal(pa["input"], pb["input"]):
                return False
            if not np.array_equal(pa["output"], pb["output"]):
                return False
    return True


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    _require(
        int(np.count_nonzero(counts == max_count)) == 1,
        "The input background must be the unique most frequent color.",
    )
    return int(values[int(np.argmax(counts))])


def _is_connected(mask: np.ndarray, neighbors: tuple = _NEIGHBORS) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False

    height, width = mask.shape
    start = (int(cells[0, 0]), int(cells[0, 1]))
    stack = [start]
    visited = {start}

    while stack:
        row, col = stack.pop()
        for dr, dc in neighbors:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and bool(mask[nr, nc])
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                stack.append((nr, nc))

    return len(visited) == int(np.count_nonzero(mask))


def _component_arrays(grid: np.ndarray, background: int) -> list[np.ndarray]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for start_row in range(height):
        for start_col in range(width):
            if seen[start_row, start_col] or int(grid[start_row, start_col]) == background:
                continue

            stack = [(start_row, start_col)]
            seen[start_row, start_col] = True
            cells: list[tuple[int, int]] = []

            while stack:
                row, col = stack.pop()
                cells.append((row, col))

                for dr, dc in _NEIGHBORS:
                    nr, nc = row + dr, col + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) != background
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            min_row = min(row for row, _ in cells)
            max_row = max(row for row, _ in cells)
            min_col = min(col for _, col in cells)
            max_col = max(col for _, col in cells)

            piece = np.full(
                (max_row - min_row + 1, max_col - min_col + 1),
                -1,
                dtype=np.int16,
            )
            for row, col in cells:
                piece[row - min_row, col - min_col] = int(grid[row, col])

            components.append(piece)

    return components


def _tight_colored_array(mask: np.ndarray, colored_grid: np.ndarray) -> np.ndarray:
    cells = np.argwhere(mask)
    _require(len(cells) > 0, "Cannot create an empty piece.")

    min_row = int(cells[:, 0].min())
    max_row = int(cells[:, 0].max())
    min_col = int(cells[:, 1].min())
    max_col = int(cells[:, 1].max())

    piece = np.full(
        (max_row - min_row + 1, max_col - min_col + 1),
        -1,
        dtype=np.int16,
    )

    for row, col in cells:
        row_i, col_i = int(row), int(col)
        piece[row_i - min_row, col_i - min_col] = int(
            colored_grid[row_i, col_i]
        )

    return piece


def _piece_orientations(piece: np.ndarray) -> list[np.ndarray]:
    orientations: list[np.ndarray] = []
    seen: set[tuple] = set()
    current = np.array(piece, dtype=np.int16, copy=True)

    for _ in range(4):
        key = current.shape, current.tobytes()
        if key not in seen:
            seen.add(key)
            orientations.append(current.copy())
        current = np.rot90(current).copy()

    return orientations


def _solve_pieces(pieces: list[np.ndarray], side: int, marker: int,
                  limit: int = 2) -> list[np.ndarray]:
    """Exhaustive colored exact cover, with duplicate geometric rotations removed.

    Nothing here knows the partition, scattering positions, seed, or answer.
    Even a marked bar may have two legal anchor orientations: search both.
    """
    if not 3 <= len(pieces) <= 6:
        return []
    if sum(int(np.sum(p >= 0)) for p in pieces) != side * side:
        return []
    marked = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    if len(marked) != 1:
        return []
    candidates = []
    for i, piece in enumerate(pieces):
        placements = {}
        for oriented in _piece_orientations(piece):
            h, w = oriented.shape
            if h > side or w > side:
                continue
            cells = [(int(r) * side + int(c), int(oriented[r, c]))
                     for r, c in np.argwhere(oriented >= 0)]
            base = sum(1 << pos for pos, _ in cells)
            if i == marked[0]:
                if oriented[0, 0] != marker or np.sum(oriented == marker) != 1:
                    continue
                offsets = [0]
            else:
                offsets = [r * side + c for r in range(side - h + 1)
                           for c in range(side - w + 1)]
            for offset in offsets:
                mask = base << offset
                placements.setdefault(mask, (mask, offset, cells))
        if not placements:
            return []
        candidates.append(list(placements.values()))
    full = (1 << (side * side)) - 1
    placed = [None] * len(pieces)
    solutions = []
    signatures = set()

    def search(occupied, remaining):
        if len(solutions) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            signature = tuple(p[0] for p in placed)
            if signature in signatures:
                return
            signatures.add(signature)
            answer = np.empty((side, side), dtype=np.int8)
            for _, offset, cells in placed:
                for pos, color in cells:
                    answer.flat[pos + offset] = color
            solutions.append(answer)
            return
        # Each cover must occupy the first uncovered cell with exactly one
        # remaining piece. This enumerates every packing without permutations.
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in candidates[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    placed[i] = candidate
                    search(occupied | candidate[0], rest)
                    if len(solutions) >= limit:
                        return
            placed[i] = None

    anchor = marked[0]
    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for candidate in candidates[anchor]:
        placed[anchor] = candidate
        search(candidate[0], rest)
        if len(solutions) >= limit:
            break
    return solutions


def _read_input(grid: np.ndarray) -> dict:
    background = _background_color(grid)
    colors, counts = np.unique(grid, return_counts=True)
    singletons = [int(c) for c, n in zip(colors, counts)
                  if n == 1 and c != background]
    _require(len(singletons) == 1, "Exactly one visible singleton marker is required.")
    marker = singletons[0]
    area = int(np.sum(grid != background))
    side = int(round(area ** 0.5))
    _require(side * side == area, "Foreground area must be a perfect square.")
    pieces = _component_arrays(grid, background)
    _require(3 <= len(pieces) <= 6, "There must be three to six separated rigid objects.")
    carriers = []
    anchor_carrier = None
    for p in pieces:
        vals = set(map(int, p[p >= 0])) - {marker}
        _require(len(vals) == 1, "Separate objects must not touch, even diagonally.")
        _require(np.sum(p >= 0) >= 3, "Objects must have at least three cells.")
        carriers.append(next(iter(vals)))
        if np.any(p == marker):
            anchor_carrier = carriers[-1]
    _require(anchor_carrier is not None, "Marker must attach to a carrier.")
    _require(len(set(carriers)) == len(pieces), "Carrier colors must be distinct.")
    return dict(background=background, marker=marker, side=side, pieces=pieces,
                anchor_carrier=anchor_carrier, carriers=carriers)


def solve(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    """Predict using visible demonstrations and input only (not a reference oracle)."""
    markers = {int(p['output'][0, 0]) for p in train_pairs}
    _require(len(markers) == 1, "Demonstrations must establish one marker color.")
    facts = _read_input(np.array(test_input, copy=True))
    _require(facts['marker'] in markers, "Test marker must match training.")
    answers = _solve_pieces(facts['pieces'], facts['side'], facts['marker'])
    _require(len(answers) == 1, "Input must have exactly one colored anchored packing.")
    return answers[0]


def _grow_partition(rng, side, count, diagonal, balanced):
    neighbors = _NEIGHBORS if diagonal else _EDGE
    labels = np.full((side, side), -1, dtype=np.int8)
    if balanced and count == 4:
        points = [0, side - 1, side * (side - 1), side * side - 1]
    else:
        points = [0] + list(map(int, rng.choice(np.arange(1, side * side),
                                               count - 1, replace=False)))
    frontier = [set() for _ in range(count)]
    sizes = [1] * count
    weights = rng.uniform(.7, 1.4, count) if balanced else np.exp(rng.uniform(-1.4, 1.4, count))
    adjacent = []
    for r in range(side):
        for c in range(side):
            adjacent.append([nr * side + nc for dr, dc in neighbors
                             if 0 <= (nr := r + dr) < side and 0 <= (nc := c + dc) < side])
    for i, pos in enumerate(points):
        labels.flat[pos] = i
    for i, pos in enumerate(points):
        frontier[i].update(p for p in adjacent[pos] if labels.flat[p] < 0)
    for _ in range(side * side - count):
        eligible = [i for i in range(count) if frontier[i]]
        i = min(eligible, key=lambda j: sizes[j] / weights[j] + float(rng.random()) * .4)
        choices = sorted(frontier[i])
        if diagonal and rng.random() < .9:
            edge_choices = [p for p in choices if any(
                0 <= (rr := p // side + dr) < side and
                0 <= (cc := p % side + dc) < side and labels[rr, cc] == i
                for dr, dc in _EDGE)]
            if edge_choices:
                choices = edge_choices
        pos = choices[int(rng.integers(len(choices)))]
        labels.flat[pos] = i
        sizes[i] += 1
        for f in frontier:
            f.discard(pos)
        frontier[i].update(p for p in adjacent[pos] if labels.flat[p] < 0)
    return labels


def _cut_partition(rng, side, count, diagonal):
    """Random rectangular cuts, then local steps and notches; keep some solids."""
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        eligible = [i for i, (r, c, h, w) in enumerate(rectangles)
                    if h * w >= 6 and (h >= 2 or w >= 2)]
        if not eligible:
            return None
        i = int(rng.choice(eligible))
        r, c, h, w = rectangles.pop(i)
        choices = [(0, k) for k in range(1, h) if min(k, h-k) * w >= 3]
        choices += [(1, k) for k in range(1, w) if min(k, w-k) * h >= 3]
        if not choices:
            return None
        axis, k = choices[int(rng.integers(len(choices)))]
        rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                          [(r, c, h, k), (r, c+k, h, w-k)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    protected = int(rng.integers(count))
    neighbors = _NEIGHBORS if diagonal else _EDGE
    for _ in range(int(rng.integers(1, side * 3 + 1))):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = _EDGE[int(rng.integers(4))]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < side and 0 <= nc < side) or (r, c) == (0, 0):
            continue
        a, b = int(labels[r, c]), int(labels[nr, nc])
        if a == b or protected in (a, b) or np.sum(labels == a) <= 3:
            continue
        labels[r, c] = b
        if not _is_connected(labels == a, neighbors):
            labels[r, c] = a
    return labels


def _inset_partition(rng, side, count, diagonal):
    """Carve independent inserts out of a wrapping object, at edges or inside."""
    labels = np.zeros((side, side), dtype=np.int8)
    for label in range(1, count):
        placed = False
        for _ in range(35):
            h = int(rng.integers(1, max(2, side // 2 + 1)))
            w = int(rng.integers(1, max(2, side // 2 + 1)))
            if h*w < 3:
                continue
            r = int(rng.integers(side-h+1))
            c = int(rng.integers(side-w+1))
            if (r == 0 and c == 0) or np.any(labels[r:r+h, c:c+w]):
                continue
            mask = np.ones((h, w), dtype=bool)
            if h >= 2 and w >= 2 and h*w > 4 and rng.random() < .65:
                # Both convex blocks and concave stepped inserts remain common.
                mask[:int(rng.integers(1, h)), :int(rng.integers(1, w))] = False
            if mask.sum() < 3:
                continue
            proposal = labels.copy()
            proposal[r:r+h, c:c+w][mask] = label
            if _is_connected(proposal == 0, _NEIGHBORS if diagonal else _EDGE):
                labels = proposal
                placed = True
                break
        if not placed:
            return None
    return labels


def _partition_pieces(labels, colors, marker):
    # Relabel by the actual top-left object, never by a prescribed seed region.
    anchor = int(labels[0, 0])
    order = [anchor] + [i for i in range(len(colors)) if i != anchor]
    output = np.empty(labels.shape, dtype=np.int8)
    for i, label in enumerate(order):
        output[labels == label] = colors[i]
    output[0, 0] = marker
    pieces = [_tight_colored_array(labels == label, output) for label in order]
    return output, pieces


def _fallback_partition(side, count):
    # A procedural guaranteed-unique safety net: distinct-length bars in
    # isolated cavities plus an asymmetric triomino. The marked enclosing
    # frame fixes every cavity; distinct areas prevent insert permutations.
    side = max(side, 2 * count)
    labels = np.zeros((side, side), dtype=np.int8)
    labels[1, 1:3] = 1
    labels[2, 1] = 1
    for i in range(2, count):
        labels[2*i, 1:i+3] = i
    return labels


def _scatter_fallback(pieces, background, rotation, easy):
    """Guaranteed fit for the enclosing frame and its small, unique inserts."""
    frame = np.rot90(pieces[0], rotation)
    side = frame.shape[0]
    height, width = side+2, side+len(pieces)+5
    foreground = sum(int(np.sum(p >= 0)) for p in pieces)
    largest = int(np.sum(frame >= 0)) - 1
    while height*width - foreground <= largest:
        if height <= width:
            height += 1
        else:
            width += 1
    canvas = np.full((height, width), background, dtype=np.int8)
    view = canvas[1:side+1, 1:side+1]
    view[frame >= 0] = frame[frame >= 0]
    row = 1
    for i, piece in enumerate(pieces[1:]):
        p = np.rot90(piece, 0 if easy and rotation == 0 else 1) if i == 0 else piece
        h, w = p.shape
        view = canvas[row:row+h, side+2:side+2+w]
        view[p >= 0] = p[p >= 0]
        row += h+1
    return canvas


def _pack_in_box(rng, oriented, height, width):
    """Bounded backtracking on real masks, including interleaving tight boxes."""
    full = (1 << (height*width))-1
    first_col = sum(1 << (r*width) for r in range(height))
    last_col = first_col << (width-1)
    options = []
    for p in oriented:
        h, w = p.shape
        base = sum(1 << (int(r)*width+int(c)) for r, c in np.argwhere(p >= 0))
        placements = []
        for r in range(height-h+1):
            for c in range(width-w+1):
                mask = base << (r*width+c)
                horizontal = mask | ((mask & ~first_col) >> 1) | ((mask & ~last_col) << 1)
                dilation = (horizontal | (horizontal << width) | (horizontal >> width)) & full
                placements.append((mask, dilation, r, c, float(rng.random())))
        options.append(placements)
    order = sorted(range(len(oriented)), key=lambda i: -oriented[i].size)
    placed, budget = {}, [320]

    def search(depth, blocked, maxr, maxc):
        if depth == len(order):
            return True
        budget[0] -= 1
        if budget[0] < 0:
            return False
        i = order[depth]
        h, w = oriented[i].shape
        choices = [p for p in options[i] if not p[0] & blocked]
        choices.sort(key=lambda p: max(maxr, p[2]+h)*max(maxc, p[3]+w) + p[4]*5)
        for p in choices[:20]:
            occupied = blocked | p[1]
            if any(not any(not q[0] & occupied for q in options[j]) for j in order[depth+1:]):
                continue
            placed[i] = p
            if search(depth+1, occupied, max(maxr, p[2]+h), max(maxc, p[3]+w)):
                return True
        return False

    if not search(0, 0, 0, 0):
        return None
    canvas = np.full((height, width), -1, dtype=np.int8)
    for i, p in enumerate(oriented):
        _, _, r, c, _ = placed[i]
        view = canvas[r:r+p.shape[0], c:c+p.shape[1]]
        view[p >= 0] = p[p >= 0]
    return canvas


def _frame_scatter(rng, canvas, background, layout):
    cells = np.argwhere(canvas >= 0)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)+1
    cropped = canvas[r0:r1, c0:c1]
    h, w = cropped.shape
    if h <= 20 and w <= 20 and rng.random() < .13:
        H = W = 20
    else:
        extra = 2 if layout == 'dense' else 7
        H = h+int(rng.integers(min(extra, 30-h)+1))
        W = w+int(rng.integers(min(extra, 30-w)+1))
    foreground = int(np.sum(cropped >= 0))
    largest = max(int(np.sum(cropped == color)) for color in np.unique(cropped) if color >= 0)
    while H*W-foreground <= largest:
        if H < 30 and (H <= W or W == 30):
            H += 1
        elif W < 30:
            W += 1
        else:
            return None
    result = np.full((H, W), background, dtype=np.int8)
    top, left = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
    result[top:top+h, left:left+w] = np.where(cropped >= 0, cropped, background)
    return result


def _scatter_pieces(rng, pieces, background, rotation, layout, easy=False):
    rotations = [rotation] + ([rotation] * (len(pieces)-1) if easy else
                             list(map(int, rng.integers(4, size=len(pieces)-1))))
    # Establish rotations with an asymmetric object whenever rotations are used.
    if not easy:
        for i, p in enumerate(pieces[1:], 1):
            if len(_piece_orientations((p >= 0).astype(np.int8))) == 4:
                rotations[i] = int(rng.integers(1, 4))
                break
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    if layout == 'dense':
        area = sum(int(np.sum(p >= 0)) for p in pieces)
        for _ in range(3):
            aspect, expansion = float(rng.uniform(.8, 1.25)), float(rng.uniform(1.7, 2.3))
            h = min(30, max(max(p.shape[0] for p in oriented), int(round((area*expansion*aspect)**.5))))
            w = min(30, max(max(p.shape[1] for p in oriented), int(round((area*expansion/aspect)**.5))))
            packed = _pack_in_box(rng, oriented, h, w)
            if packed is not None:
                framed = _frame_scatter(rng, packed, background, layout)
                if framed is not None:
                    return framed
    order = sorted(range(len(pieces)), key=lambda i: (-oriented[i].size, i))
    for attempt in range(24):
        canvas = np.full((30, 30), -1, dtype=np.int8)
        blocked = 0
        maxr = maxc = 0
        aspect = float(rng.uniform(.65, 1.55))
        okay = True
        if attempt:
            order = list(map(int, rng.permutation(len(pieces))))
        for step, i in enumerate(order):
            p = oriented[i]
            h, w = p.shape
            coords = [(int(r), int(c)) for r, c in np.argwhere(p >= 0)]
            base = sum(1 << (r * 30 + c) for r, c in coords)
            options = []
            for r in range(31-h):
                for c in range(31-w):
                    if (base << (r*30+c)) & blocked:
                        continue
                    hr, wc = max(maxr, r+h), max(maxc, c+w)
                    if layout == 'loose' and attempt < 8:
                        score = float(rng.random())
                    else:
                        score = hr*wc + abs(hr-aspect*wc) * 2 + float(rng.random()) * (8 if layout == 'dense' else 35)
                    options.append((score, r, c))
            if not options:
                okay = False
                break
            _, r, c = min(options)
            for pr, pc in coords:
                rr, cc = r+pr, c+pc
                canvas[rr, cc] = p[pr, pc]
                for nr in range(max(0, rr-1), min(30, rr+2)):
                    for nc in range(max(0, cc-1), min(30, cc+2)):
                        blocked |= 1 << (nr*30+nc)
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not okay:
            continue
        framed = _frame_scatter(rng, canvas, background, layout)
        if framed is not None:
            return framed
    return None


def _generate_pair(rng, side, count, colors, background, marker, rotation,
                   diagonal, require_diagonal, style, layout, easy, anchor_2d=False):
    for attempt in range(250):
        # Keep the requested geometry mode for most proposals, with a broad
        # connected-growth escape route for difficult small partitions.
        method = style if attempt < 100 else 'growth'
        if method == 'cuts':
            labels = _cut_partition(rng, side, count, diagonal)
        elif method == 'insets':
            labels = _inset_partition(rng, side, count, diagonal)
        else:
            labels = _grow_partition(rng, side, count, diagonal, method == 'balanced')
        if labels is None:
            continue
        areas = [int(np.sum(labels == i)) for i in range(count)]
        if min(areas) < 3:
            continue
        output, pieces = _partition_pieces(labels, colors, marker)
        if anchor_2d and min(pieces[0].shape) == 1:
            continue
        if not all(_is_connected(p >= 0) for p in pieces):
            continue
        has_diagonal = any(not _is_connected(p >= 0, _EDGE) for p in pieces)
        if require_diagonal and not has_diagonal:
            continue
        # Preserve a few clear diagonal attachments instead of producing a
        # cloud of mostly isolated pixels held together by corner contacts.
        if any(_component_count(p >= 0) > 3 for p in pieces):
            continue
        if not any(len(_piece_orientations((p >= 0).astype(np.int8))) == 4 for p in pieces):
            continue
        answers = _solve_pieces(pieces, side, marker)
        if len(answers) != 1 or not np.array_equal(answers[0], output):
            continue
        inp = _scatter_pieces(rng, pieces, background, rotation, layout, easy)
        if inp is not None:
            return dict(input=inp, output=output)
    # This path is deliberately independent of random success. A diagonal
    # demonstration is retried by the puzzle scheduler if this fallback is used.
    labels = _fallback_partition(side, count)
    output, pieces = _partition_pieces(labels, colors, marker)
    answers = _solve_pieces(pieces, output.shape[0], marker)
    _require(len(answers) == 1 and np.array_equal(answers[0], output),
             "Fallback construction must also have exactly one answer.")
    inp = _scatter_fallback(pieces, background, rotation, easy)
    return dict(input=inp, output=output)


def _validate_count_argument(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None.')
    if not minimum <= int(value) <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}.')
    return int(value)


def _choose_counts(seed, rng, num_train, num_test):
    a = _validate_count_argument(num_train, 'num_train', 2, 10)
    b = _validate_count_argument(num_test, 'num_test', 1, 4)
    if a is not None and b is not None:
        if a <= b:
            raise ValueError('num_train must exceed num_test.')
    elif a is None and b is None:
        b = 1 + seed % 4
        a = max(b + 1, 2 + (seed // 4 + seed) % 9)
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    else:
        b = int(rng.integers(1, min(4, a-1) + 1))
    return a, b


def _schedule(rng, values, count):
    return [int(v) for _ in range((count+len(values)-1)//len(values))
            for v in rng.permutation(values)][:count]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer.')
    rng = np.random.default_rng(seed)
    ntrain, ntest = _choose_counts(int(seed), rng, num_train, num_test)
    # Only the marker's literal color and upper-left meaning are semantic
    # conventions. Other choices describe observable sampling distributions.
    palette = list(map(int, rng.permutation(10)))
    marker, base_background = palette[:2]
    carrier_pool = palette[2:]
    palette_mode = str(rng.choice(['stable', 'carriers', 'vary'], p=[.25, .25, .5]))
    extensions = bool(rng.random() < .32)
    diagonal_mode = bool(rng.random() < .65)
    easy = bool(rng.random() < .12)
    # Preserve the actual combined source challenge as a deliberate mixture
    # component. Elsewhere, size and marker orientation are sampled separately.
    source_challenge = bool(not easy and rng.random() < .25)
    if source_challenge:
        extensions = False
    size_mode = ('larger' if source_challenge else 'easy' if easy else
                 str(rng.choice(['larger', 'overlap'], p=[.4, .6])))
    orientation_mode = ('source' if source_challenge else 'same' if easy else
                        str(rng.choice(['source', 'heldout', 'mixed', 'same'])))
    if size_mode == 'easy':
        side = int(rng.integers(5, 11))
        train_sides, test_sides = [side]*ntrain, [side]*ntest
    elif size_mode == 'larger':
        train_sides = _schedule(rng, list(range(4, 11)), ntrain)
        test_sides = _schedule(rng, list(range(11, 17)), ntest)
    else:
        train_sides = _schedule(rng, list(range(4, 15)), ntrain)
        test_sides = [train_sides[int(rng.integers(ntrain))]]
        test_sides += _schedule(rng, list(range(4, 17)), ntest-1)
    if orientation_mode in ('source', 'heldout'):
        held = 2 if orientation_mode == 'source' else int(rng.choice([0, 1, 3]))
        train_rotations = _schedule(rng, [k for k in range(4) if k != held], ntrain)
        test_rotations = [held]*ntest
    elif orientation_mode == 'same':
        k = int(rng.integers(4))
        train_rotations, test_rotations = [k]*ntrain, [k]*ntest
    else:
        train_rotations = _schedule(rng, [0, 1, 2, 3], ntrain)
        test_rotations = _schedule(rng, [0, 1, 2, 3], ntest)
    counts = _schedule(rng, [3, 4, 5, 6], ntrain+ntest) if extensions else [4]*(ntrain+ntest)
    backgrounds = _schedule(rng, [c for c in range(10) if c != marker], ntrain+ntest)
    styles = [str(v) for _ in range((ntrain+ntest+3)//4)
              for v in rng.permutation(['balanced', 'growth', 'cuts', 'insets'])]
    puzzle = dict(train=[], test=[])
    seen = set()
    diagonal_seen = False
    previous_anchor = None
    for i, (side, rotation) in enumerate(zip(train_sides+test_sides, train_rotations+test_rotations)):
        split = 'train' if i < ntrain else 'test'
        regime = 'source_train' if split == 'train' else 'target_test'
        count = counts[i]
        side = max(side, count if count >= 5 else 4)
        background = backgrounds[i] if palette_mode == 'vary' else base_background
        if palette_mode == 'stable':
            colors = carrier_pool[:count]
        else:
            available = [c for c in range(10) if c not in (background, marker)]
            anchors = [c for c in available if c != previous_anchor]
            anchor = int(rng.choice(anchors))
            colors = [anchor] + list(map(int, rng.choice([c for c in available if c != anchor], count-1, replace=False)))
        previous_anchor = colors[0]
        diagonal = diagonal_mode and (regime == 'source_train' or diagonal_seen)
        must_diagonal = diagonal_mode and i == 0
        style = 'growth' if must_diagonal else styles[i]
        layout = str(rng.choice(['dense', 'staggered', 'loose'], p=[.6, .25, .15]))
        for retry in range(12):
            child = np.random.default_rng(int(rng.integers(2**63)))
            pair = _generate_pair(child, side, count, colors, background, marker,
                                  rotation, diagonal, must_diagonal, style, layout, easy, source_challenge)
            key = _grid_key(pair['input'])
            if key not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS:
                break
        else:
            raise AssertionError('Could not scatter a distinct pair.')
        seen.add(key)
        if split == 'train':
            diagonal_seen |= any(not _is_connected(p >= 0, _EDGE)
                                 for p in _read_input(pair['input'])['pieces'])
        puzzle[split].append(pair)
    return puzzle


def _pair_digest(pair):
    payload = b''.join(bytes(g.shape) + g.astype(np.int8).tobytes()
                       for g in (pair['input'], pair['output']))
    return sha256(payload).hexdigest()


_SOURCE_DIGESTS = {
    'a85d58aed4ce70ebe7043dfeaf1709abb000baab93811452b67c930eba2848aa',
    '2e1059b1857012f717b20b2e7c1638cdf6581bce8519b491c07e7190c8d313de',
    '4727946bd21c110f24c7e56481c407ce919e1399cb9ab0adeeda0fa594b225ce',
    'b7fda2c2b18fbd19d8026bb1ff64f8f98b27f22a4a01eae7f83d6001e5e4b019',
}


def _corner_names(piece, marker):
    h, w = piece.shape
    r, c = map(int, np.argwhere(piece == marker)[0])
    return tuple(name for rr, cc, name in [(0, 0, 'TL'), (0, w-1, 'TR'),
                 (h-1, 0, 'BL'), (h-1, w-1, 'BR')] if (r, c) == (rr, cc))


def _component_count(mask):
    remaining = set((int(r), int(c)) for r, c in np.argwhere(mask))
    count = 0
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            r, c = stack.pop()
            for dr, dc in _EDGE:
                p = (r+dr, c+dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
    return count


def _interface_run(owners):
    longest = 0
    for a in (owners, owners.T):
        for c in range(a.shape[1]-1):
            previous, run = None, 0
            for r in range(a.shape[0]):
                x, y = int(a[r, c]), int(a[r, c+1])
                key = tuple(sorted((x, y))) if x != y else None
                run = run+1 if key is not None and key == previous else int(key is not None)
                previous = key
                longest = max(longest, run)
    return longest


def pair_features(pair):
    """Measurements reconstructed from returned grids, not generation metadata."""
    inp, out = pair['input'], pair['output']
    facts = _read_input(inp)
    pieces, marker = facts['pieces'], facts['marker']
    masks, rotation_required, asymmetric_rotation = [], False, False
    symmetry = diagonal = bars = blocks = triominoes = interior = 0
    holes = concavities = thin_tips = 0
    perimeters, edge_components = [], []
    for p in pieces:
        color = next(int(c) for c in np.unique(p) if c >= 0 and c != marker)
        omask = (out == color) | ((out == marker) if color == facts['anchor_carrier'] else False)
        masks.append(omask)
        orientations = len(_piece_orientations((p >= 0).astype(np.int8)))
        symmetry += orientations < 4
        diagonal += not _is_connected(p >= 0, _EDGE)
        bars += min(p.shape) == 1
        blocks += bool(np.all(p >= 0))
        triominoes += int(np.sum(p >= 0)) == 3
        tight = _tight_colored_array(omask, out)
        changed = not np.array_equal(tight, p)
        rotation_required |= changed
        asymmetric_rotation |= changed and orientations == 4
        interior += not (np.any(omask[0]) or np.any(omask[-1]) or
                         np.any(omask[:, 0]) or np.any(omask[:, -1]))
        edge_components.append(_component_count(p >= 0))
        padded = np.pad(omask.astype(np.int8), 1)
        adj = padded[:-2, 1:-1]+padded[2:, 1:-1]+padded[1:-1, :-2]+padded[1:-1, 2:]
        thin_tips += int(np.sum(omask & (adj == 1)))
        perimeters.append(int(np.sum((4-adj)[omask])))
        holes += _component_count(~padded.astype(bool))-1
        corners = padded[:-1, :-1]+padded[:-1, 1:]+padded[1:, :-1]+padded[1:, 1:]
        concavities += int(np.sum(corners == 3))
    corner_owners = [facts['anchor_carrier'] if v == marker else int(v)
                     for v in [out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]]]
    areas = [int(m.sum()) for m in masks]
    fg = inp != facts['background']
    rr, cc = np.nonzero(fg)
    cropped_fg = fg[rr.min():rr.max()+1, cc.min():cc.max()+1]
    boxes = []
    for p in pieces:
        color = next(int(c) for c in np.unique(p) if c >= 0 and c != marker)
        points = np.argwhere(inp == color)
        if color == facts['anchor_carrier']:
            points = np.concatenate([points, np.argwhere(inp == marker)])
        boxes.append((*map(int, points.min(axis=0)), *map(int, points.max(axis=0))))
    overlaps = sum(max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])
                   for i, a in enumerate(boxes) for b in boxes[i+1:])
    # Normalize carrier identities by first occurrence, preserving marker.
    mapping, signature = {}, []
    for v in out.flat:
        v = int(v)
        mapping.setdefault(v, len(mapping))
        signature.append(mapping[v])
    marked_piece = next(p for p in pieces if np.any(p == marker))
    owners = out.copy()
    owners[owners == marker] = facts['anchor_carrier']
    facts.update(count=len(pieces), corners=_corner_names(marked_piece, marker),
                 area_ratio=max(areas)/min(areas), min_area=min(areas), max_area=max(areas),
                 symmetric=symmetry, diagonal=diagonal, bars=bars, blocks=blocks,
                 triominoes=triominoes, interior=interior,
                 corner_owners=len(set(corner_owners)), density=float(fg.mean()),
                 blank_row=bool(np.any(~cropped_fg.any(axis=1))),
                 blank_col=bool(np.any(~cropped_fg.any(axis=0))),
                 overlapping_boxes=int(overlaps), rotation_required=rotation_required,
                 asymmetric_rotation=asymmetric_rotation,
                 marked_bar=min(marked_piece.shape) == 1, holes=holes,
                 concavities=concavities, thin_tips=thin_tips,
                 perimeters=perimeters, edge_components=edge_components,
                 longest_interface=_interface_run(owners),
                 structure=(out.shape, tuple(signature)))
    return facts


def _validate_pair(pair, regime):
    _require(regime in ('source_train', 'target_test'), 'Unknown regime.')
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema.')
    for grid in pair.values():
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'Grids must be 2D NumPy arrays.')
        _require(all(1 <= d <= 30 for d in grid.shape), 'Grid dimensions must be 1..30.')
        _require(np.issubdtype(grid.dtype, np.integer), 'Integer grids required.')
        _require(bool(np.all((grid >= 0) & (grid <= 9))), 'Colors must be 0..9.')
    inp, out = pair['input'], pair['output']
    f = _read_input(inp)
    _require(out.shape == (f['side'], f['side']), 'Output size must follow input area.')
    _require(out[0, 0] == f['marker'], 'Marker belongs at the upper-left.')
    _require(not np.any(out == f['background']), 'Discard only the input background.')
    _require(not np.array_equal(inp, out), 'Scattered input cannot already be the answer.')
    for c in range(10):
        if c != f['background']:
            _require(np.sum(inp == c) == np.sum(out == c), 'Preserve every colored cell.')
    answers = _solve_pieces(f['pieces'], f['side'], f['marker'])
    _require(len(answers) == 1, 'Require exactly one rotation-only colored packing.')
    _require(np.array_equal(answers[0], out), 'Incorrect rigid-object assembly.')
    return pair_features(pair)


def _validate_generated_puzzle_impl(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'Illegal pair counts.')
    seen_inputs, seen_pairs, facts = set(), set(), {}
    for split in ('train', 'test'):
        facts[split] = []
        for pair in puzzle[split]:
            f = _validate_pair(pair, 'source_train' if split == 'train' else 'target_test')
            ik = _grid_key(pair['input'])
            pk = (ik, _grid_key(pair['output']))
            _require(ik not in seen_inputs and pk not in seen_pairs, 'Duplicate pair or input.')
            _require(_pair_digest(pair) not in _SOURCE_DIGESTS, 'Exact source copy.')
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            facts[split].append(f)
    train, test = facts['train'], facts['test']
    _require(len({f['marker'] for f in train+test}) == 1, 'Marker color must be consistent.')
    for role in ('background', 'anchor_carrier', 'count'):
        observed = {f[role] for f in train}
        _require(len(observed) > 1 or all(f[role] in observed for f in test),
                 f'Training must demonstrate variation in {role} before a test changes it.')
    if any(f['diagonal'] for f in test):
        _require(any(f['diagonal'] for f in train), 'Demonstrate diagonal attachments in training.')
    if any(f['rotation_required'] for f in test):
        _require(any(f['asymmetric_rotation'] for f in train), 'Demonstrate rigid rotation on an asymmetric object.')
    for pair in puzzle['test']:
        prediction = solve([{k: v.copy() for k, v in p.items()} for p in puzzle['train']], pair['input'].copy())
        _require(np.array_equal(prediction, pair['output']), 'Visible-data prediction disagrees.')


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f'Malformed generated puzzle: {exc}') from exc


def self_test() -> None:
    records, puzzles, train_counts, test_counts = [], [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        train_counts.add(len(p['train']))
        test_counts.add(len(p['test']))
        for split in ('train', 'test'):
            for pair in p[split]:
                records.append((seed, split, pair_features(pair), pair))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, 'Counts must vary.')
    train = [f for _, s, f, _ in records if s == 'train']
    test = [f for _, s, f, _ in records if s == 'test']
    all_f = train+test
    _require({4, 5, 6, 10} <= {f['side'] for f in train}, 'Restore source training scales.')
    _require({14, 15, 16} <= {f['side'] for f in test}, 'Exercise large test scales.')
    _require({3, 4, 5, 6} == {f['count'] for f in all_f}, 'Exercise all object counts.')
    _require(sum(f['count'] == 4 for f in all_f) > len(all_f)//2, 'Retain ordinary four-piece examples.')
    for split_f in (train, test):
        _require(sum(f['diagonal'] > 0 for f in split_f) >= 5, 'Exercise diagonal attachments in each split.')
        _require(sum(f['symmetric'] > 0 for f in split_f) >= 5, 'Exercise harmless symmetry in each split.')
        _require(set(c for f in split_f for c in f['corners']) == {'TL', 'TR', 'BL', 'BR'}, 'Use every marker corner in each split.')
    _require(sum(f['triominoes'] for f in all_f) >= 10, 'Exercise triominoes.')
    _require(sum(f['bars'] for f in all_f) >= 8, 'Exercise thin bars.')
    _require(sum(f['blocks'] for f in all_f) >= 20, 'Exercise simple solid objects.')
    _require(sum(f['marked_bar'] for f in all_f) >= 3, 'Exercise marked bars with multiple possible anchor rotations.')
    _require(all(max(f['edge_components']) <= 3 for f in all_f), 'Keep diagonal attachments visually coherent.')
    _require(sum(f['holes'] > 0 for f in all_f) >= 10, 'Exercise enclosed pockets.')
    _require(sum(f['longest_interface'] >= 6 for f in all_f) >= 20, 'Exercise long straight interfaces.')
    _require(sum(f['concavities'] >= 4 and f['thin_tips'] >= 2 for f in all_f) >= 20,
             'Combine stepped notches and thin extensions.')
    _require(sum(f['interior'] > 0 for f in all_f) >= 15, 'Interior objects must be intentional.')
    _require(sum(f['corner_owners'] < 4 for f in all_f) >= len(all_f)//5, 'Broaden corner ownership.')
    _require(sum(f['area_ratio'] >= 4 for f in all_f) >= len(all_f)//5, 'Broaden piece-area budgets.')
    _require(sum(f['density'] >= .35 for f in test) >= len(test)//3, 'Restore compact test scattering.')
    _require(sum(f['overlapping_boxes'] > 0 for f in all_f) >= 12, 'Allow interleaving bounding boxes.')
    _require(sum(f['interior'] and f['overlapping_boxes'] and f['area_ratio'] >= 4 for f in all_f) >= 8,
             'Combine interior objects, uneven areas and interleaved scattering.')
    _require(any(not f['blank_row'] and not f['blank_col'] for f in test), 'Remove obligatory blank separators.')
    _require(any(pair['input'].shape == (20, 20) for _, _, _, pair in records), 'Allow 20 by 20 inputs.')
    _require(len({f['structure'] for f in all_f}) >= len(all_f)*.9, 'Require structural novelty.')
    _require(len({pair['input'].shape for _, _, _, pair in records}) >= 35, 'Vary canvas dimensions.')
    varying_background = varying_anchor = overlapping_size = larger = same_corners = other_holdout = 0
    combined_source = 0
    for seed in range(42):
        tr = [f for n, s, f, _ in records if n == seed and s == 'train']
        te = [f for n, s, f, _ in records if n == seed and s == 'test']
        varying_background += len({f['background'] for f in tr}) > 1
        varying_anchor += len({f['anchor_carrier'] for f in tr}) > 1
        overlapping_size += bool({f['side'] for f in tr} & {f['side'] for f in te})
        larger += min(f['side'] for f in te) > max(f['side'] for f in tr)
        tc = {c for f in tr for c in f['corners']}
        ec = {c for f in te for c in f['corners']}
        same_corners += bool(tc & ec)
        other_holdout += bool((ec-tc)-{'BR'})
        combined_source += (min(f['side'] for f in te) > max(f['side'] for f in tr)
                            and ec == {'BR'} and 'BR' not in tc)
    _require(varying_background >= 10 and varying_anchor >= 15, 'Vary visible roles within puzzles.')
    _require(overlapping_size >= 10 and larger >= 5 and same_corners >= 10 and other_holdout >= 3,
             'Size and orientation regimes must not collapse.')
    _require(combined_source >= 2, 'Preserve the combined source scale/orientation challenge.')
    for side in (4, 10, 16):
        for count in (3, 4, 5, 6):
            labels = _fallback_partition(side, count)
            output, pieces = _partition_pieces(labels, list(range(2, count+2)), 1)
            for rotation in range(4):
                pair = dict(input=_scatter_fallback(pieces, 0, rotation, False), output=output)
                _validate_pair(pair, 'target_test')
    a = generate_puzzle(707, 5, 2)
    _require(_puzzle_equal(a, generate_puzzle(707, 5, 2)), 'Determinism failed.')
    _require(not _puzzle_equal(a, generate_puzzle(708, 5, 2)), 'Seeds must differ.')
    for i, (nt, ne) in enumerate([(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]):
        p = generate_puzzle(800+i, nt, ne)
        validate_generated_puzzle(p)
        _require(nt is None or len(p['train']) == nt, 'Explicit train count ignored.')
        _require(ne is None or len(p['test']) == ne, 'Explicit test count ignored.')
    for nt, ne in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2, 1.5)]:
        try:
            generate_puzzle(1000, nt, ne)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted.')
    # A negative semantic test checks that validation compares the actual answer.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted.')
