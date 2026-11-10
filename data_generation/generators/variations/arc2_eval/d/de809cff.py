from itertools import permutations

import numpy as np


# Collision-free base-11 encodings of the three original source inputs.
# Rejecting an identical input is stronger than merely rejecting an identical
# input/output pair.
_SOURCE_INPUT_CODES = {
    (20, 20): {
        9761743568815890666665348126865118978467516541096530435023699332659501287102520380464522496541047315868986509496693716012434949203411940678020540220083118999856327774913523048446678862638707441443524060618681767309137063756795380677192168589423337251593978475889655883328321958835602686110339984355297866418520287576304527796335033755425772805867975522202979316507187203140138125298328875205305106134116597332907666,
        36064014027398033671959939795382638587015973576969469624041879478651682169253721415071622393258578787548667985603109332525887132406096076943892255028778875633994766373689248996666396221795564579371438579272414192444363276754213598489744690935595824692071290602343827261721594854137570271200342953923476233645497327878593011762539383120039071429370110918555375381099094370699247756549673774410009227512795936743463091,
    },
    (30, 30): {
        667735501513418353522287038671471979080256281921597822588232314264820060256378909905631751108862432264601596191921647279972306412778742196206602331460845146777697236503181721072036365782068393120419972954899693756005035686784795604760080252138048124425901147412243748209035174494171848397772814362414330607319512734744453107086620449550466222189124211591011170245081306442987954677185117365853414758577596317376106689542448146387397188667775452426014169416332705180175962426813505506703706255749388239612611400293732572830249148593111177178179682931332173382138284144317923998020412188288078017148804397347491191912636581102798856166668921830192842669991168218111644374200417721347592367871791655949987942507308180638870611195245469358610871726566167494652327560755054627597248811318088682158231001268802056262022242959173090162282861942061818593209433551109313270041528860501683712922050495283750874757003911366697719713013361954573,
    },
}


def _check_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_optional_count("num_train", num_train, 2, 10)
    _check_optional_count("num_test", num_test, 1, 4)

    key = int(seed)

    if num_train is None and num_test is None:
        num_test = 1 + key % 4
        low = max(2, num_test + 1)
        num_train = low + ((3 * key + key // 4) % (11 - low))
    elif num_train is None:
        num_test = int(num_test)
        low = max(2, num_test + 1)
        num_train = low + key % (11 - low)
    elif num_test is None:
        num_train = int(num_train)
        upper = min(4, num_train - 1)
        num_test = 1 + key % upper
    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _components(
    grid: np.ndarray,
    color: int,
) -> list[tuple[int, int, int, int, int]]:
    mask = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape
    result = []

    for start_row, start_col in np.argwhere(mask):
        start_row, start_col = int(start_row), int(start_col)
        if seen[start_row, start_col]:
            continue

        stack = [(start_row, start_col)]
        seen[start_row, start_col] = True
        cells = []

        while stack:
            row, col = stack.pop()
            cells.append((row, col))

            for other_row, other_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if (
                    0 <= other_row < height
                    and 0 <= other_col < width
                    and mask[other_row, other_col]
                    and not seen[other_row, other_col]
                ):
                    seen[other_row, other_col] = True
                    stack.append((other_row, other_col))

        rows = [point[0] for point in cells]
        cols = [point[1] for point in cells]
        result.append(
            (
                min(rows),
                min(cols),
                max(rows) + 1,
                max(cols) + 1,
                int(color),
            )
        )

    return result


def _overlap_area(
    first: tuple[int, int, int, int, int],
    second: tuple[int, int, int, int, int],
) -> int:
    row_overlap = max(
        0,
        min(first[2], second[2]) - max(first[0], second[0]),
    )
    col_overlap = max(
        0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    return row_overlap * col_overlap




def _grid_code(grid: np.ndarray) -> int:
    value = 0
    for cell in grid.ravel():
        value = value * 11 + int(cell)
    return value


def _array_signature(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )




def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False

    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False

        for first_pair, second_pair in zip(
            first[section],
            second[section],
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False
            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        (
            _array_signature(pair["input"]),
            _array_signature(pair["output"]),
        )
        for section in ("train", "test")
        for pair in puzzle[section]
    )




# Input-only reconstruction deliberately has no access to output markers.
def _smooth(grid, axis, radius):
    colors = np.unique(grid)
    votes = (grid[None, :, :] == colors[:, None, None]).astype(np.int16)
    padding = [(0, 0)] * 3
    padding[axis + 1] = (1, 0)
    cumulative = np.pad(np.cumsum(votes, axis=axis + 1), padding)
    indices = np.arange(grid.shape[axis])
    counts = (np.take(cumulative, np.minimum(indices + radius + 1, grid.shape[axis]), axis=axis + 1)
              - np.take(cumulative, np.maximum(indices - radius, 0), axis=axis + 1))
    result = colors[counts.argmax(axis=0)]
    old_votes = (counts * votes).sum(axis=0)
    result[old_votes == counts.max(axis=0)] = grid[old_votes == counts.max(axis=0)]
    return result.astype(np.int8)


def _paint(shape, background, rectangles):
    grid = np.full(shape, background, dtype=np.int8)
    for r0, c0, r1, c1, color in rectangles:
        grid[r0:r1, c0:c1] = color
    return grid


def _geometry_ok(rectangles, shape):
    if not 2 <= len(rectangles) <= 4:
        return False
    h, w = shape
    for r0, c0, r1, c1, _ in rectangles:
        if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
            return False
        if min(r1 - r0, c1 - c0) < 5 or (r1-r0)*(c1-c0) < 36:
            return False
    edges = []
    degrees = [0] * len(rectangles)
    for i, a in enumerate(rectangles):
        for j in range(i):
            b = rectangles[j]
            if _overlap_area(a, b):
                if a[4] == b[4]:
                    return False
                degrees[i] += 1
                degrees[j] += 1
                edges.append((i,j))
    # An alternating spatial chain; its covering order is independent.
    return len(edges) == len(rectangles)-1 and sorted(degrees) == [1,1]+[2]*(len(rectangles)-2)


def _scene_supported(clean, background, rectangles):
    if not _geometry_ok(rectangles, clean.shape):
        return False
    if np.count_nonzero(clean == background) < max(12, clean.size * .10):
        return False
    for r0,c0,r1,c1,color in rectangles:
        visible = clean[r0:r1,c0:c1] == color
        if visible.sum() < max(18, .25 * visible.size):
            return False
        # Straight surviving segments identify the full visible bounding box.
        if min(visible[0].sum(),visible[-1].sum(),visible[:,0].sum(),visible[:,-1].sum()) < 2:
            return False
    return True


def _fit_scene(input_grid):
    """Robust rectangle fits, followed by competing-boundary checks.

    Smoothing proposes bounding boxes only. All fitting scores use the original
    input, and every answer is recomputed from the fitted rectangles. Tied or
    nearly tied visible scenes are rejected, even if a supplied label fits one.
    """
    colors = tuple(int(c) for c in np.unique(input_grid))
    assert len(colors) == 3
    proposals = {}
    for radius in (1, 2):
        for axis in (0, 1):
            smooth = _smooth(_smooth(input_grid, axis, radius), 1-axis, radius)
            by_color = {c: [box for box in _components(smooth, c)
                            if np.count_nonzero(smooth[box[0]:box[2],box[1]:box[3]] == c) >= 18]
                        for c in colors}
            for background in colors:
                boxes = tuple(box for c in colors if c != background for box in by_color[c])
                if not _geometry_ok(boxes, input_grid.shape):
                    continue
                for ordered in permutations(boxes):
                    key = (background, ordered)
                    if key in proposals:
                        continue
                    clean = _paint(input_grid.shape, background, ordered)
                    if _scene_supported(clean, background, ordered):
                        proposals[key] = int(np.count_nonzero(clean != input_grid))
    assert proposals, 'No supported rectangle composition'
    # Different permutations can depict exactly the same visible scene. They
    # must not crowd out genuinely different reconstruction hypotheses.
    initial = []
    seen_scenes = set()
    for key in sorted(proposals, key=lambda key: (proposals[key], key)):
        scene_key = (key[0], _paint(input_grid.shape, *key).tobytes())
        if scene_key not in seen_scenes:
            initial.append(key)
            seen_scenes.add(scene_key)
        if len(initial) == 8:
            break
    solutions = {}
    for background, initial_boxes in initial:
        boxes = list(initial_boxes)
        clean = _paint(input_grid.shape, background, boxes)
        loss = int(np.count_nonzero(clean != input_grid))
        for _ in range(5):
            improved = False
            for i in range(len(boxes)):
                for side in range(4):
                    best = None
                    for delta in (-1, 1):
                        trial = list(boxes)
                        b = list(trial[i]); b[side] += delta; trial[i] = tuple(b)
                        if not _geometry_ok(trial, input_grid.shape):
                            continue
                        scene = _paint(input_grid.shape, background, trial)
                        cost = int(np.count_nonzero(scene != input_grid))
                        if cost < loss and _scene_supported(scene, background, trial):
                            if best is None or cost < best[0]:
                                best = (cost, trial, scene)
                    if best is not None:
                        loss, boxes, clean = best
                        improved = True
            if not improved:
                break
        if _scene_supported(clean, background, boxes):
            solutions[(background, clean.tobytes())] = (loss, clean, tuple(boxes))
    assert solutions
    ranked = sorted(solutions.items(), key=lambda item: (item[1][0],item[0]))
    (background, _), (loss, clean, boxes) = ranked[0]
    assert loss <= max(8, .15 * input_grid.size), 'Too little intact evidence'
    assert all(other[1][0] > loss+1 for other in ranked[1:]), 'Competing supported scenes'
    for order in permutations(boxes):
        scene = _paint(input_grid.shape, background, order)
        if not np.array_equal(scene, clean) and _scene_supported(scene, background, order):
            assert np.count_nonzero(scene != input_grid) > loss+1, 'Uncertain covering order'
    # Moving any straight boundary is another plausible reconstruction. Hidden
    # geometry may tie, but a different visible scene must cost at least 2 more.
    for i in range(len(boxes)):
        for side in range(4):
            for delta in (-1, 1):
                trial = list(boxes)
                b=list(trial[i]); b[side]+=delta; trial[i]=tuple(b)
                if not _geometry_ok(trial, input_grid.shape):
                    continue
                scene = _paint(input_grid.shape, background, trial)
                if np.array_equal(scene, clean) or not _scene_supported(scene, background, trial):
                    continue
                assert np.count_nonzero(scene != input_grid) > loss+1, 'Uncertain damaged boundary'
    for r0,c0,r1,c1,color in boxes:
        visible = clean[r0:r1,c0:c1] == color
        observed = input_grid[r0:r1,c0:c1] == color
        assert np.count_nonzero(visible & observed) >= .75 * visible.sum()
    return clean, background, tuple(c for c in colors if c != background), boxes


def _ring_positions(shape, center):
    row, col = center
    return [(r,c) for r in range(max(0,row-1),min(shape[0],row+2))
            for c in range(max(0,col-1),min(shape[1],col+2)) if (r,c) != center]


def _make_output(clean, centers, fills, highlight):
    # Compose constraints before painting, so an accidental conflicting overlap
    # cannot introduce a hidden center-order convention.
    demands = {}
    center_set = set(centers)
    for point in centers:
        ring_color = fills[1] if clean[point] == fills[0] else fills[0]
        for q in _ring_positions(clean.shape, point):
            if q in center_set:
                continue
            assert q not in demands or demands[q] == ring_color, 'Conflicting marker rings'
            demands[q] = ring_color
    result = clean.copy()
    for point,color in demands.items():
        result[point] = color
    for point in centers:
        result[point] = highlight
    return result


def _analyze_input(input_grid, highlight):
    clean, background, fills, rectangles = _fit_scene(input_grid)
    assert highlight not in set(np.unique(input_grid))
    centers = [tuple(map(int,p)) for p in np.argwhere((clean != background) & (input_grid == background))]
    assert centers, 'At least one erasure must demonstrate the marker'
    output = _make_output(clean, centers, fills, highlight)
    return {'clean': clean, 'background': background, 'fills': fills,
            'rectangles': rectangles, 'centers': centers, 'expected': output}


def solve(train_pairs, test_input):
    """Prediction interface: accepts demonstrations and an unlabeled input only."""
    extras = [set(map(int,np.unique(p['output']))) - set(map(int,np.unique(p['input']))) for p in train_pairs]
    assert extras and all(len(e) == 1 and e == extras[0] for e in extras)
    highlight = next(iter(extras[0]))
    return _analyze_input(test_input, highlight)['expected']


def _sample_scene(rng, shape, count):
    # Incremental rectangles form a spatial path. Each new rectangle overlaps
    # just its predecessor. Coordinates, sizes, offset signs and depth are free.
    h,w = shape
    rectangles = []
    for _ in range(180):
        rh = int(rng.integers(5, max(6, min(h, int(h*.80))+1)))
        rw = int(rng.integers(5, max(6, min(w, int(w*.80))+1)))
        if rh*rw < 40:
            continue
        r0=int(rng.integers(0,h-rh+1));c0=int(rng.integers(0,w-rw+1))
        # Edge contact has probability mass, but margins and offsets stay free.
        if rng.random()<.28: r0=int(rng.choice([0,h-rh]))
        if rng.random()<.28: c0=int(rng.choice([0,w-rw]))
        box=(r0,c0,r0+rh,c0+rw,1+len(rectangles)%2)
        if rectangles:
            overlap = _overlap_area(box,rectangles[-1])
            if not 4 <= overlap <= .48*min(rh*rw,(rectangles[-1][2]-rectangles[-1][0])*(rectangles[-1][3]-rectangles[-1][1])):
                continue
            if any(_overlap_area(box,other) for other in rectangles[:-1]):
                continue
        rectangles.append(box)
        if len(rectangles) == count:
            order = rng.permutation(count)
            ordered = [rectangles[int(i)] for i in order]
            clean = _paint(shape,0,ordered)
            if _scene_supported(clean,0,ordered):
                components = [b for c in (1,2) for b in _components(clean,c)]
                if sorted(components) == sorted(rectangles):
                    return clean
            return None
    return None


def _fallback_scene(rng, shape):
    # A parameterized two-rectangle construction with long exposed sides.
    h,w=shape
    a=(0,1,max(7,h*2//3),max(7,w*2//3),1)
    b=(max(3,h//3),max(3,w//3),h,w,2)
    clean=_paint(shape,0,[a,b])
    if rng.integers(2): clean=np.flipud(clean)
    if rng.integers(2): clean=np.fliplr(clean)
    return clean.copy()


def _corrupt(rng, clean, style, demonstrate):
    # Geometry is expressed using local roles 0=background, 1/2=fill.
    foreground = [tuple(map(int,p)) for p in np.argwhere(clean != 0)]
    area=len(foreground)
    if style == 0:
        target=int(rng.integers(3,7))
    elif style == 1:
        target=int(rng.integers(7,max(9,min(19,int(area*.07))+1)))
    else:
        target=int(rng.integers(max(10,int(area*.045)),max(13,min(37,int(area*.11))+1)))
    concentrated = not demonstrate and rng.random() < .30
    focus = int(rng.integers(1,3)) if concentrated else None
    centers=[]
    def add(p):
        if p in centers or clean[p] == 0 or (focus is not None and clean[p] != focus):
            return False
        for q in centers:
            d=max(abs(p[0]-q[0]),abs(p[1]-q[1]))
            if (style == 0 and not demonstrate and d < 3) or (clean[p] != clean[q] and d <= 2):
                return False
        # Leave dense evidence even for clusters (at most four holes per 3x3).
        if sum(max(abs(p[0]-q[0]),abs(p[1]-q[1])) <= 1 for q in centers) >= 4:
            return False
        centers.append(p)
        return True
    if demonstrate:
        for color in (1,2):
            candidates=[p for p in foreground if clean[p] == color and
                        len(_ring_positions(clean.shape,p)) == 8 and
                        all(clean[q] == color for q in _ring_positions(clean.shape,p))]
            rng.shuffle(candidates)
            if not candidates: return None
            add(candidates[0])
        anchor=centers[0]
        offsets=[(0,1),(1,0),(1,1),(1,-1)]
        rng.shuffle(offsets)
        for dr,dc in offsets:
            q=(anchor[0]+dr,anchor[1]+dc)
            if 0<=q[0]<clean.shape[0] and 0<=q[1]<clean.shape[1] and clean[q]==clean[anchor] and add(q):
                break
        else: return None
    # Random clustered growth includes distance-two sharing, adjacent pairs and
    # small nonlinear groups; only same-fill neighborhoods may intersect.
    candidates=foreground.copy();rng.shuffle(candidates)
    if style>0 and rng.random()<.30:
        candidates.sort(key=lambda p: 0 if p[0] in (0,clean.shape[0]-1) and p[1] in (0,clean.shape[1]-1) else 1)
    for p in candidates:
        if len(centers)>=target: break
        if add(p) and style>0 and rng.random()<.40:
            offsets=[(0,1),(1,0),(1,1),(1,-1),(0,2),(2,0),(2,2)]
            rng.shuffle(offsets)
            for dr,dc in offsets[:int(rng.integers(1,4))]:
                q=(p[0]+dr,p[1]+dc)
                if len(centers)<target and 0<=q[0]<clean.shape[0] and 0<=q[1]<clean.shape[1] and clean[q]==clean[p]:
                    add(q)
    if demonstrate and not all(any(clean[p]==color for p in centers) for color in (1,2)):
        return None
    corrupted=clean.copy()
    for p in centers: corrupted[p]=0
    # Independent count choices allow absence, interior errors, seam bites and
    # occasional ordinary noise in an eventual marker neighborhood.
    wrong_target=0 if rng.random()<.25 else int(rng.integers(1, max(3, min(10,int(area*.025))+1)))
    seam=[p for p in foreground if any(clean[q] not in (0,clean[p]) for q in _ring_positions(clean.shape,p))]
    seam_set = set(seam)
    interior=[p for p in foreground if p not in seam_set]
    rng.shuffle(seam);rng.shuffle(interior)
    wrong_pool=seam+interior if rng.random()<.75 else interior+seam
    used=set(centers);wrong=[]
    allow_near=rng.random()<.30
    for p in wrong_pool:
        if len(wrong)>=wrong_target: break
        if p in used: continue
        if any(max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=1 for q in wrong): continue
        if not allow_near and any(max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=1 for q in centers):continue
        corrupted[p]=3-int(clean[p]);used.add(p);wrong.append(p)
    exterior=[tuple(map(int,p)) for p in np.argwhere(clean==0)]
    max_speckles=max(2,min(22,int(len(exterior)*.13)))
    speckle_target=int(rng.integers(1,max_speckles+1)) if style else int(rng.integers(1,min(5,max_speckles)+1))
    rng.shuffle(exterior)
    for p in exterior[:speckle_target]:corrupted[p]=int(rng.integers(1,3))
    return corrupted


def _generate_pair(rng, shape, count, style, demonstrate, palette, highlight):
    for attempt in range(72):
        clean=_sample_scene(rng,shape,count) if attempt < 60 else _fallback_scene(rng,shape)
        if clean is None: continue
        corrupted=_corrupt(rng,clean,style,demonstrate)
        if corrupted is None:continue
        try:
            recovered,bg,_,_ = _fit_scene(corrupted)
            if bg!=0 or not np.array_equal(recovered,clean):continue
            centers=[tuple(map(int,p)) for p in np.argwhere((clean!=0)&(corrupted==0))]
            if demonstrate:
                if not any(clean[p]==clean[q] and max(abs(p[0]-q[0]),abs(p[1]-q[1]))==1
                           for i,p in enumerate(centers) for q in centers[:i]):continue
            mapping=np.array(palette,dtype=np.int8)
            actual_clean=mapping[clean]
            actual_input=mapping[corrupted]
            output=_make_output(actual_clean,centers,tuple(palette[1:]),highlight)
            # Tie behavior in proposal smoothing must not make palette a cue.
            observed=_analyze_input(actual_input,highlight)
            if not np.array_equal(observed['expected'],output):continue
            return {'input':actual_input,'output':output}
        except AssertionError:
            continue
    # Deterministic evidence-preserving fallback: incrementally accept defects
    # only if input reconstruction still agrees. This handles any legal seed.
    clean=_fallback_scene(rng,shape)
    corrupted=clean.copy()
    candidates=[tuple(map(int,p)) for p in np.argwhere(clean!=0)]
    rng.shuffle(candidates)
    accepted=[]
    desired=[1,2,1] if demonstrate else [int(rng.integers(1,3))]*3
    for color in desired:
        for p in candidates:
            if clean[p]!=color or p in accepted:continue
            if accepted and len(accepted)==2 and demonstrate:
                if max(abs(p[0]-accepted[0][0]),abs(p[1]-accepted[0][1]))!=1:continue
            if any(clean[p]!=clean[q] and max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=2 for q in accepted):continue
            trial=corrupted.copy();trial[p]=0
            try:
                fit,bg,_,_=_fit_scene(trial)
                if bg==0 and np.array_equal(fit,clean):
                    corrupted=trial;accepted.append(p);break
            except AssertionError:continue
    assert len(accepted)==3
    mapping=np.array(palette,dtype=np.int8)
    output=_make_output(mapping[clean],accepted,tuple(palette[1:]),highlight)
    return {'input':mapping[corrupted], 'output':output}


def pair_features(pair):
    inp,out=pair['input'],pair['output']
    extra=set(map(int,np.unique(out)))-set(map(int,np.unique(inp)))
    assert len(extra)==1
    highlight=next(iter(extra))
    a=_analyze_input(inp,highlight)
    assert np.array_equal(out,a['expected'])
    clean,bg,fills,rects,centers=(a[k] for k in ('clean','background','fills','rectangles','centers'))
    errors=inp!=clean
    wrong=[tuple(map(int,p)) for p in np.argwhere(errors & (clean!=bg) & (inp!=bg))]
    seam=lambda p:any(clean[q] in fills and clean[q]!=clean[p] for q in _ring_positions(inp.shape,p))
    distances=[max(abs(p[0]-q[0]),abs(p[1]-q[1])) for i,p in enumerate(centers) for q in centers[:i]]
    sizes=[(r1-r0,c1-c0) for r0,c0,r1,c1,_ in rects]
    clipped=sum(r in (0,inp.shape[0]-1) or c in (0,inp.shape[1]-1) for r,c in centers)
    links = {i: [j for j in range(len(rects)) if j != i and _overlap_area(rects[i], rects[j])]
             for i in range(len(rects))}
    start = min((i for i in links if len(links[i]) == 1), key=lambda i: rects[i][:4])
    path = [start]
    while len(path) < len(rects):
        path.append(next(j for j in links[path[-1]] if j not in path))
    covering = tuple(int(i < j) for i,j in zip(path,path[1:]))
    overlaps = tuple(sorted(_overlap_area(rects[i], rects[j]) for i,j in zip(path,path[1:])))
    offsets = tuple((rects[j][0]-rects[i][0], rects[j][1]-rects[i][1])
                    for i,j in zip(path,path[1:]))
    exposed = tuple(sorted(round(float(np.mean(clean[r0:r1,c0:c1] == color)), 3)
                           for r0,c0,r1,c1,color in rects))
    return {'background':bg,'highlight':highlight,'palette':tuple(sorted(fills)),
            'shape':inp.shape,'rectangles':len(rects),'rectangle_sizes':tuple(sorted(sizes)),
            'overlap_areas':overlaps, 'offsets':offsets, 'visible_fractions':exposed,
            'covering_pattern':covering,
            'unequal_sizes':len(set(sizes))>1,'holes':len(centers),'wrong_fill':len(wrong),
            'seam_errors':sum(seam(p) for p in wrong),
            'interior_errors':sum(not seam(p) for p in wrong),
            'speckles':int(np.count_nonzero(errors & (clean==bg))),
            'adjacent':1 in distances,'distance_two':2 in distances,
            'cluster3':any(sum(max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=2 for q in centers)>=3 for p in centers),
            'horizontal_adjacency':any(p[0]==q[0] and abs(p[1]-q[1])==1 for i,p in enumerate(centers) for q in centers[:i]),
            'vertical_adjacency':any(p[1]==q[1] and abs(p[0]-q[0])==1 for i,p in enumerate(centers) for q in centers[:i]),
            'diagonal_adjacency':any(abs(p[0]-q[0])==abs(p[1]-q[1])==1 for i,p in enumerate(centers) for q in centers[:i]),
            'clipped':clipped,'corner_markers':sum(r in (0,inp.shape[0]-1) and c in (0,inp.shape[1]-1) for r,c in centers),
            'corner_occupied':any(clean[p]!=bg for p in ((0,0),(0,-1),(-1,0),(-1,-1))),
            'all_margins':all(np.all(edge==bg) for edge in (clean[0],clean[-1],clean[:,0],clean[:,-1])),
            'concentrated':len(set(int(clean[p]) for p in centers))==1,
            'noise_near_hole':any(max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=1 for p in wrong for q in centers),
            'ring_into_background':any(clean[q]==bg for p in centers for q in _ring_positions(inp.shape,p)),
            'ring_crosses_seam':any(seam(p) for p in centers),
            'axis': 'horizontal' if np.ptp([(b[1]+b[3])/2 for b in rects])>np.ptp([(b[0]+b[2])/2 for b in rects]) else 'vertical',
            'rich_combination':len(rects)>=3 and len(set(sizes))>1 and any(seam(p) for p in wrong) and 1 in distances}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2<=len(puzzle['train'])<=10 and 1<=len(puzzle['test'])<=4
    assert len(puzzle['train'])>len(puzzle['test'])
    signatures=set();highlight=None;analyses=[]
    for pair in puzzle['train']+puzzle['test']:
        assert isinstance(pair,dict) and set(pair)=={'input','output'}
        for grid in pair.values():
            assert isinstance(grid,np.ndarray) and grid.ndim==2
            assert np.issubdtype(grid.dtype,np.integer)
            assert all(1<=d<=30 for d in grid.shape) and np.all((grid>=0)&(grid<=9))
        inp,out=pair['input'],pair['output']
        assert inp.shape==out.shape and not np.array_equal(inp,out)
        signature=_array_signature(inp)
        assert signature not in signatures;signatures.add(signature)
        assert _grid_code(inp) not in _SOURCE_INPUT_CODES.get(inp.shape,set())
        extra=set(map(int,np.unique(out)))-set(map(int,np.unique(inp)))
        assert len(extra)==1
        h=next(iter(extra))
        if highlight is None:highlight=h
        assert highlight==h
        a=_analyze_input(inp,highlight)
        assert np.array_equal(out,a['expected']), 'Input-inferred transformation disagrees'
        analyses.append(a)
    train_a=analyses[:len(puzzle['train'])]
    # At least one clear demonstration teaches both directions and center
    # precedence. The rest may concentrate on either role or have sparse holes.
    assert any(set(int(a['clean'][p]) for p in a['centers'])==set(a['fills']) for a in train_a)
    assert any(any(a['clean'][p]==a['clean'][q] and max(abs(p[0]-q[0]),abs(p[1]-q[1]))==1
                   for i,p in enumerate(a['centers']) for q in a['centers'][:i]) for a in train_a)
    assert len(set(pair['input'].shape for pair in puzzle['train']+puzzle['test']))>=2
    for pair in puzzle['test']:
        assert np.array_equal(solve(puzzle['train'],pair['input'].copy()),pair['output'])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or int(seed)<0:
        raise ValueError('seed must be a nonnegative integer')
    num_train,num_test=_resolve_counts(int(seed),num_train,num_test)
    rng=np.random.default_rng(seed)
    highlight=int(rng.integers(10))
    available=[c for c in range(10) if c!=highlight]
    base_palette=tuple(map(int,rng.choice(available,3,replace=False)))
    palette_mode=int(rng.integers(4)) # fixed, recurring, changing backgrounds, or free
    size_mode=int(rng.integers(3)) # source-scale transfer, shared, mixed ranges
    demonstration=int(rng.integers(num_train))
    puzzle={'train':[],'test':[]};seen=set();previous_shape=None
    for split,total in (('train',num_train),('test',num_test)):
        for index in range(total):
            if size_mode==0:
                low,high=(15,23) if split=='train' else (24,31)
            elif size_mode==1:low,high=17,31
            else:low,high=14,31
            shape=(int(rng.integers(low,high)),int(rng.integers(low,high)))
            if rng.random()<.18:
                side=int(rng.integers(low,high));shape=(side,side)
            if shape==previous_shape:shape=(low if shape[0]!=low else high-1,shape[1])
            previous_shape=shape
            probabilities=[.42,.43,.15] if min(shape)>=23 else [.57,.43,0]
            count=int(rng.choice([2,3,4],p=probabilities))
            style=int(rng.choice([0,1,2],p=[.24,.39,.37] if split=='train' else [.18,.35,.47]))
            teach=split=='train' and index==demonstration
            palette=base_palette
            if palette_mode==1 and rng.random()<.45:
                palette=(base_palette[0],*map(int,rng.choice([c for c in available if c!=base_palette[0]],2,replace=False)))
            elif palette_mode==2:
                palette=(int(rng.choice([c for c in available if c not in base_palette[1:]])),*base_palette[1:])
            elif palette_mode==3:
                palette=tuple(map(int,rng.choice(available,3,replace=False)))
            for _ in range(5):
                pair=_generate_pair(rng,shape,count,style,teach,palette,highlight)
                sig=_array_signature(pair['input'])
                if sig not in seen and _grid_code(pair['input']) not in _SOURCE_INPUT_CODES.get(shape,set()):
                    break
            else:raise AssertionError('Could not make a distinct pair')
            seen.add(sig);puzzle[split].append(pair)
    return puzzle


def self_test() -> None:
    puzzles=[generate_puzzle(seed) for seed in range(42)]
    features=[]
    grouped=[]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        group=[]
        for split in ('train','test'):
            for pair in puzzle[split]:
                f=pair_features(pair);f['split']=split;features.append(f);group.append(f)
                # A separate simultaneous implementation, vectorized by color,
                # checks the ring composer, including preservation of centers.
                a=_analyze_input(pair['input'],f['highlight'])
                clean=a['clean'];holes=(pair['input']==a['background'])&(clean!=a['background'])
                expected=clean.copy()
                for color in a['fills']:
                    mask=holes&(clean==color);paint=np.zeros_like(mask)
                    for dr in (-1,0,1):
                        for dc in (-1,0,1):
                            if dr==dc==0:continue
                            rs=slice(max(0,dr),min(mask.shape[0],mask.shape[0]+dr))
                            cs=slice(max(0,dc),min(mask.shape[1],mask.shape[1]+dc))
                            src_r=slice(max(0,-dr),min(mask.shape[0],mask.shape[0]-dr))
                            src_c=slice(max(0,-dc),min(mask.shape[1],mask.shape[1]-dc))
                            paint[rs,cs]|=mask[src_r,src_c]
                    expected[paint]=next(c for c in a['fills'] if c!=color)
                expected[holes]=f['highlight']
                assert np.array_equal(expected,pair['output'])
        grouped.append(group)
    assert _puzzles_equal(puzzles[7],generate_puzzle(7))
    assert len({_puzzle_signature(p) for p in puzzles})==42
    assert len({len(p['train']) for p in puzzles})>=4
    assert len({len(p['test']) for p in puzzles})>=3
    for nt,ne in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        puzzle=generate_puzzle(1701+(nt or 0)*10+(ne or 0),nt,ne)
        validate_generated_puzzle(puzzle)
        if nt is not None:assert len(puzzle['train'])==nt
        if ne is not None:assert len(puzzle['test'])==ne
    for nt,ne in ((1,1),(11,1),(2,0),(10,5),(2,2),(3,4),(True,1),(2,1.5)):
        try:generate_puzzle(3,nt,ne)
        except ValueError:pass
        else:raise AssertionError('Illegal counts accepted')
    assert len({f['shape'] for f in features})>=65
    assert {f['rectangles'] for f in features}=={2,3,4}
    assert len({f['highlight'] for f in features})>=8
    assert len({f['background'] for f in features})==10
    assert max(f['holes'] for f in features)>=24 and min(f['holes'] for f in features)<=4
    assert max(f['speckles'] for f in features)>=15
    assert any(f['wrong_fill']==0 for f in features)
    assert len({f['rectangle_sizes'] for f in features}) >= 100
    assert len({f['offsets'] for f in features}) >= 100
    assert len({f['overlap_areas'] for f in features}) >= 50
    assert len({f['covering_pattern'] for f in features}) >= 7
    assert len({f['visible_fractions'] for f in features}) >= 100
    assert sum(len({f['axis'] for f in group}) == 2 for group in grouped) >= 25
    assert sum(len({f['rectangles'] for f in group}) >= 2 for group in grouped) >= 25
    assert sum(len({f['background'] for f in group}) >= 2 for group in grouped) >= 10
    assert sum(len({f['palette'] for f in group}) == 1 for group in grouped) >= 5
    assert sum(len({f['palette'] for f in group}) < len(group) for group in grouped) >= 20
    assert sum(max(max(f['shape']) for f in group if f['split']=='train') <
               min(min(f['shape']) for f in group if f['split']=='test') for group in grouped) >= 5
    assert sum(max(f['holes'] for f in group if f['split']=='train') >=
               min(f['holes'] for f in group if f['split']=='test') for group in grouped) >= 15
    for key in ('adjacent','distance_two','cluster3','horizontal_adjacency','vertical_adjacency',
                'diagonal_adjacency','clipped','corner_markers','corner_occupied','all_margins',
                'concentrated','noise_near_hole','ring_into_background','ring_crosses_seam',
                'rich_combination','interior_errors','seam_errors','unequal_sizes'):
        assert sum(bool(f[key]) for f in features)>=3,key
    for split in ('train','test'):
        fs=[f for f in features if f['split']==split]
        assert any(f['shape']==(30,30) for f in fs)
        assert any(f['clipped'] for f in fs) and any(not f['clipped'] for f in fs)
        assert any(f['holes']>=17 for f in fs) and any(f['holes']<=5 for f in fs)
        assert {f['rectangles'] for f in fs}=={2,3,4}
    # Semantic rejection probes: a marker center cannot be overwritten by a
    # ring, and a wrong label must not be allowed to supply reconstruction data.
    bad={s:[{k:v.copy() for k,v in p.items()} for p in puzzles[0][s]] for s in ('train','test')}
    bad['test'][0]['output'][0,0]=(int(bad['test'][0]['output'][0,0])+1)%10
    try:validate_generated_puzzle(bad)
    except AssertionError:pass
    else:raise AssertionError('Incorrect output accepted')
    clean=np.ones((8,8),dtype=np.int8);clean[:,4:]=2
    try:_make_output(clean,[(3,3),(3,4)],(1,2),8)
    except AssertionError:pass
    else:raise AssertionError('Conflicting ring colors accepted')
    compatible = [(0,0),(0,1),(1,1),(3,1)]
    assert np.array_equal(_make_output(clean,compatible,(1,2),8),
                          _make_output(clean,list(reversed(compatible)),(1,2),8))
    # Source regression fixtures encode the visible rectangles inferred from
    # the staged source, independently of generation-time parameters.
    source_scenes = {
        (20,20): [((0,1,10,13,3),(7,6,18,20,4),(14,0,20,10,3)),
                  ((0,0,11,11,1),(8,6,20,20,2))],
        (30,30): [((1,8,11,30,6),(5,1,20,21,1),(19,4,30,26,6))],
    }
    for shape,codes in _SOURCE_INPUT_CODES.items():
        for code in codes:
            cells=[]
            for _ in range(shape[0]*shape[1]):
                cells.append(code % 11);code //= 11
            inp=np.array(cells[::-1],dtype=np.int8).reshape(shape)
            clean,bg,_,_=_fit_scene(inp)
            assert bg==0
            assert any(np.array_equal(clean,_paint(shape,0,boxes)) for boxes in source_scenes[shape])
