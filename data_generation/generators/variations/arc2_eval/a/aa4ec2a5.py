import hashlib

import numpy as np

_NEIGHBORS4 = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

_NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if dr != 0 or dc != 0
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if (
            isinstance(num_train, (bool, np.bool_))
            or not isinstance(num_train, (int, np.integer))
        ):
            raise ValueError("num_train must be an integer or None")
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        if (
            isinstance(num_test, (bool, np.bool_))
            or not isinstance(num_test, (int, np.integer))
        ):
            raise ValueError("num_test must be an integer or None")
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = 1 + (seed % 4)
        candidate = 2 + ((3 * seed + seed // 4) % 9)
        num_train = candidate if candidate > num_test else num_test + 1
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + (seed % (11 - low))
    else:
        high = min(4, num_train - 1)
        num_test = 1 + (seed % high)

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("impossible train/test count combination")

    return int(num_train), int(num_test)


def _analyze_input(
    grid: np.ndarray,
    background: int,
    foreground: int,
) -> dict:
    height, width = grid.shape
    foreground_mask = grid == foreground

    # Label the original objects with orthogonal connectivity.
    labels = np.full(
        (height, width),
        -1,
        dtype=np.int16,
    )
    component_count = 0

    for row in range(height):
        for col in range(width):
            if not foreground_mask[row, col]:
                continue
            if labels[row, col] >= 0:
                continue

            labels[row, col] = component_count
            queue = [(row, col)]
            head = 0

            while head < len(queue):
                rr, cc = queue[head]
                head += 1

                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc

                    if not (
                        0 <= nr < height
                        and 0 <= nc < width
                    ):
                        continue

                    if (
                        foreground_mask[nr, nc]
                        and labels[nr, nc] < 0
                    ):
                        labels[nr, nc] = component_count
                        queue.append((nr, nc))

            component_count += 1

    # Flood-fill boundary-connected background.
    background_mask = grid == background
    exterior = np.zeros(
        (height, width),
        dtype=bool,
    )
    queue = []

    for row in range(height):
        for col in (0, width - 1):
            if (
                background_mask[row, col]
                and not exterior[row, col]
            ):
                exterior[row, col] = True
                queue.append((row, col))

    for col in range(width):
        for row in (0, height - 1):
            if (
                background_mask[row, col]
                and not exterior[row, col]
            ):
                exterior[row, col] = True
                queue.append((row, col))

    head = 0
    while head < len(queue):
        rr, cc = queue[head]
        head += 1

        for dr, dc in _NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc

            if not (
                0 <= nr < height
                and 0 <= nc < width
            ):
                continue

            if (
                background_mask[nr, nc]
                and not exterior[nr, nc]
            ):
                exterior[nr, nc] = True
                queue.append((nr, nc))

    holes = background_mask & ~exterior

    # Label cavity regions and identify their unique surrounding objects.
    seen_holes = np.zeros(
        (height, width),
        dtype=bool,
    )
    hollow_ids = set()
    hole_count = 0

    for row in range(height):
        for col in range(width):
            if not holes[row, col] or seen_holes[row, col]:
                continue

            seen_holes[row, col] = True
            queue = [(row, col)]
            adjacent_ids = set()
            head = 0

            while head < len(queue):
                rr, cc = queue[head]
                head += 1

                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc

                    if not (
                        0 <= nr < height
                        and 0 <= nc < width
                    ):
                        continue

                    if holes[nr, nc]:
                        if not seen_holes[nr, nc]:
                            seen_holes[nr, nc] = True
                            queue.append((nr, nc))
                    elif labels[nr, nc] >= 0:
                        adjacent_ids.add(
                            int(labels[nr, nc])
                        )

            _require(
                len(adjacent_ids) == 1,
                "each cavity must belong to exactly one object",
            )
            hollow_ids.update(adjacent_ids)
            hole_count += 1

    all_ids = set(range(component_count))
    solid_ids = all_ids - hollow_ids

    if hollow_ids:
        hollow_object_mask = np.isin(
            labels,
            list(hollow_ids),
        )
    else:
        hollow_object_mask = np.zeros_like(holes)

    if solid_ids:
        solid_object_mask = np.isin(
            labels,
            list(solid_ids),
        )
    else:
        solid_object_mask = np.zeros_like(holes)

    # Dilate all original objects by one cell using eight-neighbor
    # adjacency, but retain only exterior background cells.
    outline = np.zeros(
        (height, width),
        dtype=bool,
    )

    for rr, cc in np.argwhere(foreground_mask):
        for dr, dc in _NEIGHBORS8:
            nr = int(rr) + dr
            nc = int(cc) + dc

            if (
                0 <= nr < height
                and 0 <= nc < width
                and exterior[nr, nc]
            ):
                outline[nr, nc] = True

    far_background = exterior & ~outline

    return {
        "labels": labels,
        "component_count": component_count,
        "hollow_ids": hollow_ids,
        "solid_ids": solid_ids,
        "hole_count": hole_count,
        "holes": holes,
        "exterior": exterior,
        "outline": outline,
        "hollow_objects": hollow_object_mask,
        "solid_objects": solid_object_mask,
        "far_background": far_background,
    }


def _construct_output(
    grid: np.ndarray,
    params: dict,
) -> np.ndarray:
    analysis = _analyze_input(
        grid,
        params["background"],
        params["foreground"],
    )

    output = np.full(
        grid.shape,
        params["background"],
        dtype=np.int8,
    )
    output[analysis["outline"]] = params["outline"]
    output[analysis["solid_objects"]] = params["foreground"]
    output[analysis["hollow_objects"]] = params["hollow"]
    output[analysis["holes"]] = params["cavity"]
    return output


# Exact source contents are excluded; their dimensions remain available.
_SOURCE_INPUT_DIGESTS = {
    ((26, 27), 'fc67d992a75ecc1350e692dce359d62286e3655e13a8b8a9c6954df72c55aa6d'),
    ((23, 25), 'e7747dc0494341bdd3c2b0e49b66ca42aecec78780a6398ff67affa2b36a9026'),
    ((28, 23), '9a3bea1fa26212a8e04be8cc86feffbaa79706178b39be869978e9ee268b56bf'),
    ((20, 15), '991fc25af34a2fc3a7bbca6ffc385a90ae85d6a29f79cf542e1abd3306aca2c0'),
}


def _regions(mask, neighbors=_NEIGHBORS4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    regions = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        region, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    region.add(q)
                    stack.append(q)
        regions.append(region)
    return regions


def _holes(mask, neighbors=_NEIGHBORS4):
    """Object-local enclosure, with an explicit outside padding cell."""
    padded = np.pad(mask, 1)
    outside = np.zeros_like(padded)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _crop(mask):
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()


def _unambiguous(mask):
    # An alternating 2x2 is the only local configuration permitting a
    # diagonal contact/leak. Excluding it also keeps cavity partitions clear.
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _orient(rng, mask):
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return _crop(mask)


def _grow(rng, height, width, steps):
    """Connected unions of rectangles, with varied lobes and bent branches."""
    mask = np.zeros((height, width), dtype=bool)
    mask[height // 2, width // 2] = True
    for _ in range(steps):
        r, c = np.argwhere(mask)[int(rng.integers(mask.sum()))]
        dr, dc = _NEIGHBORS4[int(rng.integers(4))]
        length, thick = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        rr, cc = np.clip(r + dr * length, 0, height - 1), np.clip(c + dc * length, 0, width - 1)
        candidate = mask.copy()
        candidate[min(r, rr):min(height, max(r, rr) + thick),
                  min(c, cc):min(width, max(c, cc) + thick)] = True
        if _unambiguous(candidate):
            mask = candidate
    # Accidental enclosed pockets are filled for this cavity-free body.
    mask |= _holes(mask)
    return _crop(mask)


def _cavity(rng, h, w, irregular):
    cavity = np.ones((h, w), dtype=bool)
    if irregular and h >= 2 and w >= 2:
        # A stepped L, optionally extended into a thicker bent polyomino.
        cavity[:] = False
        cut_r, cut_c = int(rng.integers(1, h)), int(rng.integers(1, w))
        cavity[:cut_r, :] = True
        cavity[:, :cut_c] = True
        # Additional overlapping rectangles produce stepped/bent pockets,
        # without introducing islands or merging diagonal-only contacts.
        for _ in range(int(rng.integers(1, 6))):
            r, c = np.argwhere(cavity)[int(rng.integers(cavity.sum()))]
            rr, cc = int(rng.integers(h)), int(rng.integers(w))
            candidate = cavity.copy()
            candidate[min(r, rr):max(r, rr) + 1, min(c, cc):max(c, cc) + 1] = True
            if not candidate.all() and _unambiguous(candidate) and not _holes(candidate).any():
                cavity = candidate
        cavity = np.rot90(cavity, 2 * int(rng.integers(2)))
    return cavity


def _sculpt_body(rng, mask):
    """Cut open bays and steps independently of the already sealed holes."""
    holes = _holes(mask)
    protected = holes | _dilate(holes)
    h, w = mask.shape
    for _ in range(int(rng.integers(2, 9))):
        candidate = mask.copy()
        side = int(rng.integers(4))
        if side < 2:
            a = int(rng.integers(w))
            b = int(rng.integers(a + 1, w + 1))
            depth = int(rng.integers(1, max(2, h // 2 + 1)))
            cut = (slice(0, depth) if side == 0 else slice(h - depth, h), slice(a, b))
        else:
            a = int(rng.integers(h))
            b = int(rng.integers(a + 1, h + 1))
            depth = int(rng.integers(1, max(2, w // 2 + 1)))
            cut = (slice(a, b), slice(0, depth) if side == 2 else slice(w - depth, w))
        candidate[cut] &= protected[cut]
        if (_unambiguous(candidate) and len(_regions(candidate)) == 1
                and _bulk(candidate) and np.array_equal(_holes(candidate), holes)):
            mask = candidate
    return mask


def _attach_tail(rng, mask):
    mask = _orient(rng, mask)
    length = int(rng.integers(3, 7))
    h, w = mask.shape
    extended = np.pad(mask, ((0, length), (2, 2)))
    col = int(rng.choice(np.flatnonzero(mask[-1]))) + 2
    thickness = int(rng.integers(1, 3))
    extended[h - 1:, col:col + thickness] = True
    reach = int(rng.integers(1, 5))
    if rng.random() < .5:
        extended[-thickness:, max(0, col - reach):col + 1] = True
    else:
        extended[-thickness:, col:min(w + 4, col + reach + 1)] = True
    # A tail must not make a diagonal contact with a pre-existing lobe.
    before = np.pad(_holes(mask), ((0, length), (2, 2)))
    if not _unambiguous(extended) or not np.array_equal(_holes(extended), before):
        return mask
    return _crop(extended)


def _chamber(rng, h, w, irregular=False, multiple=False, lobes=True):
    """Carve independently sampled pockets into an asymmetric, thick body."""
    h, w = max(h, 3), max(w, 3)
    if irregular:
        h, w = max(h, 5), max(w, 5)
    if multiple:
        h, w = max(h, 5), max(w, 8 if irregular else 5)
    mask = np.ones((h, w), dtype=bool)
    if multiple:
        split = int(rng.integers(3 if irregular else 2, w - (3 if irregular else 2)))
        intervals = [(1, split), (split + 1, w - 1)]
        if w >= 11 and not irregular and rng.random() < .35:
            intervals = [(1, 3), (4, 6), (7, w - 1)]
    else:
        left = int(rng.integers(1, max(2, w - (2 if irregular else 1))))
        right = int(rng.integers(left + (2 if irregular else 1), w))
        intervals = [(left, right)]
    for left, right in intervals:
        min_height = 2 if irregular else 1
        top = int(rng.integers(1, h - min_height))
        bottom = int(rng.integers(top + min_height, h))
        hole = _cavity(rng, bottom - top, right - left, irregular)
        mask[top:bottom, left:right] &= ~hole
    if lobes:
        if rng.random() < .7:
            mask = _sculpt_body(rng, mask)
        # Grow outward only: neither body growth nor tails alter the holes.
        pad = int(rng.integers(1, 3))
        expanded = np.pad(mask, pad)
        for _ in range(int(rng.integers(1, 5))):
            side = int(rng.integers(4))
            if side < 2:
                start = int(rng.integers(pad, pad + w))
                end = min(pad + w, start + int(rng.integers(1, 5)))
                rr = slice(0, pad + 1) if side == 0 else slice(pad + h - 1, None)
                candidate = expanded.copy()
                candidate[rr, start:end] = True
            else:
                start = int(rng.integers(pad, pad + h))
                end = min(pad + h, start + int(rng.integers(1, 5)))
                cc = slice(0, pad + 1) if side == 2 else slice(pad + w - 1, None)
                candidate = expanded.copy()
                candidate[start:end, cc] = True
            # A sculpted edge may no longer reach this sampled attachment.
            if (len(_regions(candidate)) == 1 and _unambiguous(candidate)
                    and np.array_equal(_holes(candidate), _holes(expanded))):
                expanded = candidate
        mask = _crop(expanded)
    return mask


def _contrast(rng, irregular):
    # Comparable silhouettes differ in whether a one-cell passage is open.
    h, w = int(rng.integers(5, 9)), int(rng.integers(5, 10))
    closed = _chamber(rng, h, w, irregular, False, True)
    if not _bulk(closed):
        closed = np.ones((h, w), dtype=bool)
        closed[1:-2, 1:-1] = False
    # Choose a clear orthogonal opening; unusual lobe alignments can split
    # a thin top wall, so try the other axes before the simple fallback.
    for direction in range(4):
        candidate = np.rot90(closed, direction).copy()
        cavity = _holes(candidate)
        rr, cc = np.argwhere(cavity)[0]
        candidate[:rr + 1, cc] = False
        if len(_regions(candidate)) == 1 and _unambiguous(candidate) and not _holes(candidate).any():
            opened = np.rot90(candidate, -direction)
            break
    else:
        closed = np.ones((h, w), dtype=bool)
        closed[1:-2, 1:-1] = False
        opened = closed.copy()
        opened[0, 2] = False
    rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
    masks = [closed, opened]
    return [_crop(np.fliplr(np.rot90(m, rotation)) if flip else np.rot90(m, rotation)) for m in masks]


def _make_mask(rng, hollow, scale, irregular, multiple):
    # Object scale is chosen independently of its topological class.
    if scale == 'small':
        h, w = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    elif scale == 'large':
        h, w = int(rng.integers(6, 13)), int(rng.integers(6, 13))
    else:
        h, w = int(rng.integers(2, 10)), int(rng.integers(2, 10))
    if hollow:
        mask = _chamber(rng, h, w, irregular and rng.random() < .65,
                        multiple and rng.random() < .35, rng.random() < .7)
        if rng.random() < .22:
            mask = _attach_tail(rng, mask)
    else:
        style = int(rng.integers(5))
        if style == 0:
            mask = np.ones((h, w), dtype=bool)
            if rng.random() < .55:
                mask = np.ones((1, int(rng.integers(1, 7))), dtype=bool)
        elif style in (1, 2) and min(h, w) >= 3:
            # U/C/open bay, including thick masses around very deep bays.
            mask = np.ones((h, w), dtype=bool)
            left = int(rng.integers(1, w - 1))
            right = int(rng.integers(left + 1, w))
            depth = int(rng.integers(1, h))
            mask[:depth, left:right] = False
        else:
            mask = _grow(rng, h, w, int(rng.integers(3, 14)))
    return _orient(rng, mask)


def _pack(rng, masks, h, w, margin, gap, align, attempts=5):
    """Random free placement of bounding boxes; no fixed spatial slots."""
    order = sorted(range(len(masks)), key=lambda i: -masks[i].size)
    for attempt in range(attempts):
        occupied = np.zeros((h, w), dtype=bool)
        grid = np.zeros_like(occupied)
        anchors = []
        for i in order:
            mask = masks[i]
            mh, mw = mask.shape
            if mh + 2 * margin > h or mw + 2 * margin > w:
                break
            # Integral-image rectangle sums enumerate all feasible offsets.
            prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = prefix[mh:, mw:] - prefix[:-mh, mw:] - prefix[mh:, :-mw] + prefix[:-mh, :-mw]
            valid = sums == 0
            if margin:
                valid[:margin] = False
                valid[-margin:] = False
                valid[:, :margin] = False
                valid[:, -margin:] = False
            candidates = np.argwhere(valid)
            if not len(candidates):
                break
            if align and anchors and rng.random() < .7:
                ar, ac = anchors[int(rng.integers(len(anchors)))]
                matching = candidates[(candidates[:, 0] == ar) | (candidates[:, 1] == ac)]
                if len(matching):
                    candidates = matching
            r, c = candidates[int(rng.integers(len(candidates)))]
            anchors.append((r, c))
            grid[r:r + mh, c:c + mw] |= mask
            occupied[max(0, r - gap):min(h, r + mh + gap),
                     max(0, c - gap):min(w, c + mw + gap)] = True
        else:
            if grid.mean() < .46 and (~grid & ~_dilate(grid)).any():
                return grid
    return None


def _dilate(mask):
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + dr:1 + dr + mask.shape[0],
                                    1 + dc:1 + dc + mask.shape[1]] for dr, dc in _NEIGHBORS8])


def _scene(rng, required, kinds, scale_bias, irregular, multiple):
    masks = list(required)
    for hollow in kinds:
        scale = str(rng.choice(['small', 'medium', 'large'], p=scale_bias))
        masks.append(_make_mask(rng, hollow, scale, irregular, multiple))
    # Spacing, aspect ratio, margins and scale have independent draws.
    gap = int(rng.choice([1, 1, 2, 3, 4]))
    margin = int(rng.choice([0, 0, 1, 2, 3, 4]))
    align = rng.random() < .3
    area = sum(m.size for m in masks)
    ratio = float(np.exp(rng.uniform(-.75, .75)))
    target = area / float(rng.uniform(.18, .42))
    h = int(np.clip(round(np.sqrt(target * ratio)), 6, 30))
    w = int(np.clip(round(np.sqrt(target / ratio)), 6, 30))
    h = max(h, max(m.shape[0] for m in masks) + 2 * margin)
    w = max(w, max(m.shape[1] for m in masks) + 2 * margin)
    h, w = min(h, 30), min(w, 30)
    for grow in (0, 3, 7, 30):
        grid = _pack(rng, masks, min(30, h + grow), min(30, w + grow), margin, gap, align)
        if grid is not None:
            return grid
    # Crowded scenes retain their requested objects but resample oversized
    # optional bodies. Required training evidence is never discarded.
    for shrink in range(3):
        masks = list(required) + [_make_mask(rng, k, 'small', False, False) for k in kinds]
        grid = _pack(rng, masks, 30, 30, min(margin, 1), 1, False, 12)
        if grid is not None:
            return grid
    # Guaranteed finite fallback: up to two <=13-cell required masks fit
    # across the top; at most eight small objects fit along lower shelves.
    grid = np.zeros((30, 30), dtype=bool)
    col = 0
    bottom = 0
    for mask in required:
        mh, mw = mask.shape
        grid[:mh, col:col + mw] = mask
        col += mw + 1
        bottom = max(bottom, mh)
    r, c = bottom + 2, 0
    for hollow in kinds:
        mask = np.ones((3, 3), bool) if hollow else np.ones((1, 1), bool)
        if hollow:
            mask[1, 1] = False
        if c + 3 > 30:
            r, c = r + 5, 0
        grid[r:r + len(mask), c:c + mask.shape[1]] = mask
        c += 5
    return grid


def _input_signature(grid):
    return tuple(grid.shape), grid.astype(np.uint8).tobytes()


def _is_source(grid):
    return (tuple(grid.shape), hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest()) in _SOURCE_INPUT_DIGESTS


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    colors = list(map(int, rng.permutation(10)))
    introduced = dict(zip(('outline', 'hollow', 'cavity'), colors[:3]))
    available = colors[3:]
    variable_palette = rng.random() < .7
    irregular, multiple = bool(rng.random() < .7), bool(rng.random() < .45)
    shift = str(rng.choice(['matched', 'count', 'scale']))
    contrast = _contrast(rng, irregular)
    special = _orient(rng, _chamber(rng, int(rng.integers(5, 9)), int(rng.integers(8, 11)),
                                   irregular, multiple, True))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_count = int(rng.integers(2, 6))
    previous_bg_fg = None
    for split, count in (('source_train', nt), ('target_test', nq)):
        for index in range(count):
            required = []
            if split == 'source_train' and index == 0:
                required = contrast
                n = first_count
            elif split == 'source_train' and index == 1:
                required = [special] if irregular or multiple else []
                if variable_palette:
                    required.append(_make_mask(rng, False, 'medium', False, False))
                n = int(rng.choice([v for v in range(max(1, len(required)), 8) if v != first_count]))
            else:
                upper = 9 if split == 'target_test' and shift == 'count' else 8
                n = int(rng.integers(1, upper))
            if variable_palette:
                bg, fg = map(int, rng.choice(available, size=2, replace=False))
                if split == 'source_train' and index == 1:
                    choices = [c for c in available if c not in previous_bg_fg]
                    bg, fg = map(int, rng.choice(choices, size=2, replace=False))
            else:
                bg, fg = available[:2]
            previous_bg_fg = bg, fg
            params = dict(introduced, background=bg, foreground=fg)
            hollow_probability = float(rng.uniform(.12, .88))
            kinds = [bool(rng.random() < hollow_probability) for _ in range(n - len(required))]
            scale_bias = [.33, .5, .17]
            if split == 'target_test' and shift == 'scale':
                scale_bias = [.2, .4, .4]
            # Pass all shape policy explicitly; no hidden state is returned.
            for attempt in range(32):
                mask = _scene(rng, required, kinds, scale_bias, irregular, multiple)
                inp = np.where(mask, fg, bg).astype(np.int8)
                if _input_signature(inp) not in seen and not _is_source(inp):
                    break
            else:
                # Preserve the teaching bodies even in the uniqueness
                # fallback. Nine isolated pixels offer 511 visible patterns
                # against at most thirteen prior inputs.
                base = np.full((30, 30), bg, np.int8)
                col = 0
                for body in required:
                    h, w = body.shape
                    base[:h, col:col + w][body] = fg
                    col += w + 1
                for i, hollow in enumerate(kinds):
                    r, c = 16 + 5 * (i // 6), 5 * (i % 6)
                    base[r:r + 3, c:c + 3] = fg
                    if hollow:
                        base[r + 1, c + 1] = bg
                for code in range(1, 512):
                    inp = base.copy()
                    for bit in range(9):
                        if code & (1 << bit):
                            inp[27, 1 + 3 * bit] = fg
                    if _input_signature(inp) not in seen and not _is_source(inp):
                        break
            output = _construct_output(inp, params)
            seen.add(_input_signature(inp))
            puzzle['train' if split == 'source_train' else 'test'].append({'input': inp, 'output': output})
    validate_generated_puzzle(puzzle)
    return puzzle

def _pixel_zones(grid):
    """Inspect returned pixels independently of the global constructor.

    Cavity ownership is reconstructed one object at a time in its bounding
    box, then cross-checked against whole-scene background connectivity.
    """
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 2 and counts[0] != counts[1], 'input must have a clear binary background')
    bg, fg = int(values[np.argmax(counts)]), int(values[np.argmin(counts)])
    body = grid == fg
    components = _regions(body)
    _require(len(components) == len(_regions(body, _NEIGHBORS8)), 'diagonal object contact')
    _require(_unambiguous(body), 'diagonal-only leak or cavity contact')
    holes, hollow = np.zeros_like(body), np.zeros_like(body)
    labels = np.full(grid.shape, -1, int)
    objects = []
    for i, cells in enumerate(components):
        rr, cc = np.array(sorted(cells)).T
        r, c = int(rr.min()), int(cc.min())
        local = body[r:rr.max() + 1, c:cc.max() + 1].copy()
        # Only this component participates in local enclosure detection.
        local[:] = False
        local[rr - r, cc - c] = True
        local_holes = _holes(local)
        _require(np.array_equal(local_holes, _holes(local, _NEIGHBORS8)), 'ambiguous hole topology')
        nh = len(_regions(local_holes))
        _require(nh == len(_regions(local_holes, _NEIGHBORS8)), 'diagonally joined cavities')
        labels[rr, cc] = i
        if nh:
            hollow[rr, cc] = True
            hr, hc = np.nonzero(local_holes)
            _require(not body[hr + r, hc + c].any(), 'nested objects')
            _require(not holes[hr + r, hc + c].any(), 'shared cavity ownership')
            holes[hr + r, hc + c] = True
        objects.append({'mask': local, 'holes': local_holes, 'hole_count': nh,
                        'origin': (r, c), 'cells': cells})
    global_holes = _holes(body)
    _require(np.array_equal(holes, global_holes), 'collective enclosure')
    _require(np.array_equal(global_holes, _holes(body, _NEIGHBORS8)), 'diagonal exterior leak')
    # Construct the eight-neighbor union by array shifts, not constructor loops.
    outline = _dilate(body) & ~body & ~holes
    return {'background': bg, 'foreground': fg, 'objects': objects, 'labels': labels,
            'solid': body & ~hollow, 'hollow': hollow, 'cavity': holes,
            'outline': outline, 'far': ~body & ~holes & ~outline}


def _learn_colors(train_pairs):
    learned = {}
    for pair in train_pairs:
        zones = _pixel_zones(pair['input'])
        out = pair['output']
        for role in ('outline', 'hollow', 'cavity'):
            if zones[role].any():
                colors = np.unique(out[zones[role]])
                _require(len(colors) == 1, 'nonuniform ' + role)
                value = int(colors[0])
                _require(role not in learned or value == learned[role], 'inconsistent introduced colors')
                learned[role] = value
    _require(len(learned) == 3 and len(set(learned.values())) == 3, 'training must reveal all output colors')
    return learned


def _predict(train_pairs, test_input):
    # The signature deliberately provides no test label, seed or metadata.
    colors = _learn_colors(train_pairs)
    zones = _pixel_zones(test_input)
    out = test_input.copy()
    for role, color in colors.items():
        out[zones[role]] = color
    return out


def _separated_runs(mask):
    for rows in (mask, mask.T):
        for row in rows:
            indices = np.flatnonzero(row)
            if len(indices) > 1 and np.any(np.diff(indices) > 1):
                return True
    return False


def _bulk(mask):
    return bool(np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]))


def _features(grid, zones=None):
    z = _pixel_zones(grid) if zones is None else zones
    objects = z['objects']
    solid = [o for o in objects if not o['hole_count']]
    hollow = [o for o in objects if o['hole_count']]
    body = grid == z['foreground']
    irregular = sum(len(hole) < (max(r for r, c in hole) - min(r for r, c in hole) + 1)
                    * (max(c for r, c in hole) - min(c for r, c in hole) + 1)
                    for o in hollow for hole in _regions(o['holes']))
    outlines = [_dilate(z['labels'] == i) & z['outline'] for i in range(len(objects))]
    shared = any(np.any(a & b) for i, a in enumerate(outlines) for b in outlines[i + 1:])
    touching = any(np.any(_dilate(a) & b) for i, a in enumerate(outlines) for b in outlines[i + 1:])
    edge_sides = [bool(body[0].any()), bool(body[-1].any()), bool(body[:, 0].any()), bool(body[:, -1].any())]
    box_overlap = {o['mask'].shape for o in solid} & {o['mask'].shape for o in hollow}
    small_hollow_large_solid = any(h['mask'].size * 2 <= s['mask'].size for h in hollow for s in solid)
    offcenter = 0
    distant_body = 0
    for o in hollow:
        rr, cc = np.nonzero(o['holes'])
        h, w = o['mask'].shape
        offcenter += int(rr.min() != h - 1 - rr.max() or cc.min() != w - 1 - cc.max())
        near = o['holes'].copy()
        for _ in range(3):
            near |= _dilate(near)
        distant_body += int(np.any(o['mask'] & ~near))
    open_notch = sum(_separated_runs(o['mask']) for o in solid)
    narrow_bay = 0
    for o in solid:
        filled = o['mask'] | _dilate(o['mask'])
        narrow_bay += int(_holes(filled).any())
    return {
        'height': grid.shape[0], 'width': grid.shape[1],
        'objects': len(objects), 'hollow_objects': len(hollow), 'solid_objects': len(solid),
        'holes': sum(o['hole_count'] for o in hollow), 'irregular_holes': int(irregular),
        'multiple_hole_objects': sum(o['hole_count'] > 1 for o in hollow),
        'offcenter_holes': offcenter, 'bulky_hollows': sum(_bulk(o['mask']) for o in hollow),
        'hollow_distant_body': distant_body,
        'hollow_open_bays': sum(_separated_runs(o['mask'] | o['holes']) for o in hollow),
        'horizontal_solids': sum(o['mask'].shape[1] > o['mask'].shape[0] for o in solid),
        'vertical_solids': sum(o['mask'].shape[0] > o['mask'].shape[1] for o in solid),
        'horizontal_hollows': sum(o['mask'].shape[1] > o['mask'].shape[0] for o in hollow),
        'vertical_hollows': sum(o['mask'].shape[0] > o['mask'].shape[1] for o in hollow),
        'open_notches': int(open_notch), 'bay_closed_by_outline': narrow_bay,
        'single_pixels': sum(o['mask'].sum() == 1 for o in solid),
        'short_lines': sum(min(o['mask'].shape) == 1 and 1 < o['mask'].sum() <= 6 for o in solid),
        'tiny_polyominoes': sum(2 <= o['mask'].sum() <= 4 and min(o['mask'].shape) > 1 for o in solid),
        'tiny_hollows': sum(max(o['mask'].shape) <= 4 for o in hollow),
        'small_hollow_large_solid': small_hollow_large_solid,
        'matching_class_boxes': bool(box_overlap),
        'single_pixel_holes': sum(len(h) == 1 for o in hollow for h in _regions(o['holes'])),
        'narrow_holes': sum(min(_crop(o['holes']).shape) == 1 for o in hollow),
        'foreground_edge': any(edge_sides), 'edge_sides': edge_sides,
        'corner_clip': bool(body[0, 0] or body[0, -1] or body[-1, 0] or body[-1, -1]),
        'shared_outline': shared, 'touching_only_outline': touching and not shared,
        'separated_outlines': len(objects) > 1 and not touching,
        'generous_margin': not body[:3].any() and not body[-3:].any() and not body[:, :3].any() and not body[:, -3:].any(),
        'midrow_occupied': bool(body[grid.shape[0] // 2].any()),
        'midcol_occupied': bool(body[:, grid.shape[1] // 2].any()),
        'crosses_midline': any((min(r for r, c in o['cells']) < grid.shape[0] // 2 < max(r for r, c in o['cells']))
                              or (min(c for r, c in o['cells']) < grid.shape[1] // 2 < max(c for r, c in o['cells'])) for o in objects),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[k], list) for k in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid pair counts')
    seen = set()
    inspections = {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D array')
                _require(np.issubdtype(grid.dtype, np.integer), 'noninteger grid')
                _require(all(1 <= n <= 30 for n in grid.shape), 'grid size outside ARC bounds')
                _require(bool(((0 <= grid) & (grid <= 9)).all()), 'illegal color')
            inp, out = pair['input'], pair['output']
            _require(inp.shape == out.shape and not np.array_equal(inp, out), 'invalid output dimensions or no change')
            signature = _input_signature(inp)
            _require(signature not in seen and not _is_source(inp), 'duplicate or exact source input')
            seen.add(signature)
            zones = _pixel_zones(inp)
            _require(zones['far'].any() and zones['outline'].any(), 'background and outline must be visible')
            _require(np.all(out[zones['far']] == zones['background']), 'far background changed')
            _require(np.all(out[zones['solid']] == zones['foreground']), 'solid pixels changed')
            inspections[split].append(zones)
    learned = _learn_colors(puzzle['train'])
    train_zones = inspections['train']
    _require(any(z['solid'].any() for z in train_zones), 'no cavity-free training evidence')
    if len({z['foreground'] for z in train_zones}) > 1:
        _require(len({z['foreground'] for z in train_zones if z['solid'].any()}) > 1,
                 'varying input foreground must demonstrate preservation in more than one color')
    _require(any(any(not o['hole_count'] and _separated_runs(o['mask']) for o in z['objects']) for z in train_zones),
             'training permits the separated-runs shortcut')
    _require(any(any(o['hole_count'] and _bulk(o['mask']) for o in z['objects']) for z in train_zones),
             'training must show recoloring beyond a thin ring')
    _require(len({len(z['objects']) for z in inspections['train'] + inspections['test']}) > 1,
             'object counts must vary within a puzzle')
    # Both splits share the same geometric rule. Generalized cavity shapes
    # are exercised in training before any test requires those extensions.
    for feature in ('irregular_holes', 'multiple_hole_objects'):
        train_has = any(_features(p['input'], z)[feature] for p, z in zip(puzzle['train'], train_zones))
        test_has = any(_features(p['input'], z)[feature] for p, z in zip(puzzle['test'], inspections['test']))
        _require(not test_has or train_has, 'test cavity extension not demonstrated: ' + feature)
    for split in puzzle:
        for pair, z in zip(puzzle[split], inspections[split]):
            _require(len(set(learned.values()) | {z['background'], z['foreground']}) == 5, 'color-role collision')
            expected = pair['input'].copy()
            for role, color in learned.items():
                expected[z[role]] = color
            _require(np.array_equal(expected, pair['output']), 'incorrect whole-body/fill/exterior transformation')


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
                                             for k in ('input', 'output')) for s in ('train', 'test'))


def _regressions():
    # Source train 2's notched mass, source tiny objects, and source loops.
    notch = np.array([[int(c) for c in row] for row in (
        '11111100011111', '11111111111111', '11111111111111',
        '11111111111111', '11111111111111', '11111111111111',
        '11111111111111', '00000000111111', '00000000111111',
        '00000000111111', '00000000111111', '00000000111111')], bool)
    fixtures = [np.ones((1, 1), bool), np.array([[0, 1], [1, 1]], bool), notch,
                np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], bool),
                np.array([[1, 1, 1, 1, 1, 0, 0], [1, 0, 0, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], bool)]
    _require(_separated_runs(notch) and not _holes(notch).any(), 'source open notch regression')
    params = dict(background=4, foreground=1, outline=2, hollow=8, cavity=6)
    for mask, cavity_area in zip(fixtures, (0, 0, 0, 2, 2)):
        inp = np.where(np.pad(mask, 2), 1, 4).astype(np.int8)
        out = _construct_output(inp, params)
        z = _pixel_zones(inp)
        _require(int(z['cavity'].sum()) == cavity_area, 'source fixture topology')
        _require(np.all(out[z['solid']] == 1) and np.all(out[z['hollow']] == 8)
                 and np.all(out[z['cavity']] == 6) and np.all(out[z['outline']] == 2), 'source fixture colors')
    # A one-cell door closes after outlining; its bay was exterior in input.
    opened = np.ones((7, 7), bool); opened[1:6, 1:6] = False; opened[0, 3] = False
    closed = opened.copy(); closed[0, 3] = True
    multi = np.ones((6, 10), bool); multi[1:4, 1:3] = False; multi[2:5, 5:9] = False; multi[2, 8] = True
    for mask, expected_holes in ((opened, 0), (closed, 1), (multi, 2)):
        inp = np.where(np.pad(mask, 2), 1, 4).astype(np.int8)
        out = _construct_output(inp, params)
        z = _pixel_zones(inp)
        _require(sum(o['hole_count'] for o in z['objects']) == expected_holes, 'open/sealed/multiple regression')
        _require(np.all(out[z['solid']] == 1) and np.all(out[z['cavity']] == 6), 'class regression')
        if not expected_holes:
            _require(out[5, 5] == 4 and out[2, 5] == 2, 'outline must not turn an open bay into a cavity')


def self_test() -> None:
    _regressions()
    for request in ((None, None), (7, 3), (2, 1), (10, 1), (10, 4),
                    (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(12345, *request)
        _require(_same_puzzle(p, generate_puzzle(12345, *request)), 'nondeterminism')
        for split, requested in zip(('train', 'test'), request):
            _require(requested is None or len(p[split]) == requested, 'count request not honored')
        validate_generated_puzzle(p)
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(77, *request)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _require(not _same_puzzle(generate_puzzle(100), generate_puzzle(101)), 'cross-seed repetition')
    train_counts, test_counts, dimensions, silhouettes, layouts, palettes = set(), set(), set(), set(), set(), set()
    records, variable, fixed, combinations = [], 0, 0, set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p['train'])); test_counts.add(len(p['test']))
        colors = _learn_colors(p['train'])
        palettes.add(tuple(colors[k] for k in ('outline', 'hollow', 'cavity')))
        input_palettes = set()
        for split in p:
            for pair in p[split]:
                z = _pixel_zones(pair['input'])
                f = _features(pair['input'], z)
                f['test'] = split == 'test'
                records.append(f)
                combinations.add((f['objects'], f['hollow_objects']))
                dimensions.add(pair['input'].shape)
                input_palettes.add((z['background'], z['foreground']))
                layouts.add(_input_signature(pair['input'] == z['foreground']))
                for o in z['objects']:
                    silhouettes.add(_input_signature(o['mask']))
                # Prediction gets only training copies and the current input.
                prediction = _predict([{'input': a['input'].copy(), 'output': a['output'].copy()}
                                       for a in p['train']], pair['input'].copy())
                _require(np.array_equal(prediction, pair['output']), 'label-free prediction mismatch')
        variable += len(input_palettes) > 1
        fixed += len(input_palettes) == 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, 'count diversity')
    _require(len(dimensions) >= 50 and len(silhouettes) >= 150 and len(layouts) >= .9 * len(records), 'structural diversity')
    _require(len(palettes) >= 30 and variable >= 12 and fixed >= 4, 'palette diversity')
    _require((3, 1) in combinations and len(combinations) >= 20, 'object/class count diversity')
    for name in ('irregular_holes', 'multiple_hole_objects', 'offcenter_holes', 'bulky_hollows',
                 'hollow_distant_body', 'hollow_open_bays', 'horizontal_solids', 'vertical_solids',
                 'horizontal_hollows', 'vertical_hollows',
                 'open_notches', 'bay_closed_by_outline', 'single_pixels', 'short_lines', 'tiny_polyominoes',
                 'tiny_hollows', 'small_hollow_large_solid', 'matching_class_boxes', 'single_pixel_holes',
                 'narrow_holes', 'foreground_edge', 'corner_clip', 'shared_outline', 'touching_only_outline',
                 'separated_outlines', 'generous_margin', 'crosses_midline'):
        _require(sum(bool(f[name]) for f in records) >= 3, 'missing measured diversity: ' + name)
    _require(all(any(f['edge_sides'][side] for f in records) for side in range(4)), 'missing clipping direction')
    _require(any(f['objects'] == 1 for f in records) and any(f['objects'] >= 7 for f in records), 'scene-count range')
    _require(any(f['height'] <= 10 and f['width'] <= 10 for f in records), 'small canvases missing')
    for split in (False, True):
        subset = [f for f in records if f['test'] == split]
        _require(any(f['height'] == 30 and f['width'] == 30 for f in subset), 'large canvases missing from split')
        _require(any(not f['foreground_edge'] for f in subset), 'clipping compulsory')
        _require(any(f['midrow_occupied'] and f['midcol_occupied'] for f in subset), 'fixed empty midlines')
    _require(sum(bool(f['irregular_holes'] and f['open_notches'] and f['small_hollow_large_solid'])
                 for f in records) >= 3, 'compatible geometry axes do not combine')
    # Corrupted cells must be rejected, even when all color roles remain present.
    p = generate_puzzle(900, 3, 1)
    for role in ('solid', 'hollow', 'cavity', 'outline', 'far'):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        for pair in q['train']:
            z = _pixel_zones(pair['input'])
            if z[role].any():
                r, c = np.argwhere(z[role])[0]
                pair['output'][r, c] = (int(pair['output'][r, c]) + 1) % 10
                break
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted corruption of ' + role)
