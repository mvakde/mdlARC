import numpy as np

_DIRECTIONS = ("U", "D", "L", "R")
_DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _need(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validated_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _select_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validated_count(num_train, "num_train")
    num_test = _validated_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        seed_code = abs(int(seed))
        num_test = 1 + seed_code % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed_code // 4) % span
    elif num_train is None:
        minimum_train = num_test + 1
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for the requested train count")
        num_test = int(rng.integers(1, maximum_test + 1))

    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _counter_coordinates(
    top: int,
    left: int,
    interior_height: int,
    interior_width: int,
    direction: str,
) -> list[tuple[int, int]]:
    bottom = top + interior_height + 1
    right = left + interior_width + 1
    coordinates: list[tuple[int, int]] = []

    if direction == "U":
        for row in range(top + 1, bottom):
            for col in range(left + 1, right):
                coordinates.append((row, col))
    elif direction == "D":
        for row in range(bottom - 1, top, -1):
            for col in range(right - 1, left, -1):
                coordinates.append((row, col))
    elif direction == "L":
        for col in range(left + 1, right):
            for row in range(bottom - 1, top, -1):
                coordinates.append((row, col))
    elif direction == "R":
        for col in range(right - 1, left, -1):
            for row in range(top + 1, bottom):
                coordinates.append((row, col))
    else:
        raise AssertionError("unknown direction")

    return coordinates


def _segment_for_rectangle(
    top: int,
    bottom: int,
    left: int,
    right: int,
    direction: str,
    outward_distance: int,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    height, width = grid_shape

    if direction == "U":
        row = top - 1 - outward_distance
        if row < 0:
            return None
        return [(row, col) for col in range(left, right + 1)]

    if direction == "D":
        row = bottom + 1 + outward_distance
        if row >= height:
            return None
        return [(row, col) for col in range(left, right + 1)]

    if direction == "L":
        col = left - 1 - outward_distance
        if col < 0:
            return None
        return [(row, col) for row in range(top, bottom + 1)]

    if direction == "R":
        col = right + 1 + outward_distance
        if col >= width:
            return None
        return [(row, col) for row in range(top, bottom + 1)]

    raise AssertionError("unknown direction")


def _set_segment(
    grid: np.ndarray,
    segment: list[tuple[int, int]],
    color: int,
) -> None:
    for row, col in segment:
        grid[row, col] = color


def _segment_values(
    grid: np.ndarray,
    segment: list[tuple[int, int]],
) -> list[int]:
    return [int(grid[row, col]) for row, col in segment]


def _is_exact_marker_bar(
    grid: np.ndarray,
    segment: list[tuple[int, int]],
    marker: int,
) -> bool:
    if not segment:
        return False
    if any(int(grid[row, col]) != marker for row, col in segment):
        return False

    rows = {row for row, _ in segment}
    cols = {col for _, col in segment}
    height, width = grid.shape

    if len(rows) == 1:
        row = next(iter(rows))
        first = min(cols)
        last = max(cols)
        if first > 0 and int(grid[row, first - 1]) == marker:
            return False
        if last + 1 < width and int(grid[row, last + 1]) == marker:
            return False
    elif len(cols) == 1:
        col = next(iter(cols))
        first = min(rows)
        last = max(rows)
        if first > 0 and int(grid[first - 1, col]) == marker:
            return False
        if last + 1 < height and int(grid[last + 1, col]) == marker:
            return False
    else:
        return False

    return True


def _frame_components(grid: np.ndarray, frame_color: int) -> list[list[tuple[int, int]]]:
    mask = grid == frame_color
    visited = np.zeros(grid.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = grid.shape

    for start_row, start_col in np.argwhere(mask):
        start_row = int(start_row)
        start_col = int(start_col)
        if visited[start_row, start_col]:
            continue

        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        component: list[tuple[int, int]] = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + delta_row
                next_col = col + delta_col
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and mask[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

        components.append(component)

    return components


def _remaining_count(vehicle: dict) -> int:
    return vehicle["capacity"] - vehicle["advanced_count"]


def _move_count(vehicle: dict) -> int:
    return min(vehicle["distance"], _remaining_count(vehicle))


def _stopping_regime(vehicle: dict) -> str:
    remaining = _remaining_count(vehicle)
    distance = vehicle["distance"]

    if remaining == 0:
        return "zero"
    if distance < remaining:
        return "destination"
    if distance == remaining:
        return "exact"
    return "fuel"


def _footprint_shape(spec: dict) -> tuple[int, int]:
    outer_height = spec["interior_height"] + 2
    outer_width = spec["interior_width"] + 2

    if spec["direction"] in ("U", "D"):
        return outer_height + spec["distance"] + 1, outer_width
    return outer_height, outer_width + spec["distance"] + 1


def _draw_initial_vehicle(
    grid: np.ndarray,
    footprint_top: int,
    footprint_left: int,
    spec: dict,
    roles: dict,
) -> None:
    interior_height = spec["interior_height"]
    interior_width = spec["interior_width"]
    outer_height = interior_height + 2
    outer_width = interior_width + 2
    distance = spec["distance"]
    direction = spec["direction"]

    if direction == "U":
        top = footprint_top + distance + 1
        left = footprint_left
    elif direction == "D":
        top = footprint_top
        left = footprint_left
    elif direction == "L":
        top = footprint_top
        left = footprint_left + distance + 1
    elif direction == "R":
        top = footprint_top
        left = footprint_left
    else:
        raise AssertionError("unknown direction")

    bottom = top + outer_height - 1
    right = left + outer_width - 1

    attached = _segment_for_rectangle(
        top, bottom, left, right, direction, 0, grid.shape
    )
    destination = _segment_for_rectangle(
        top, bottom, left, right, direction, distance, grid.shape
    )
    _need(attached is not None and destination is not None, "marker is out of bounds")

    _set_segment(grid, attached, roles["marker"])
    _set_segment(grid, destination, roles["marker"])

    grid[top:bottom + 1, left:right + 1] = roles["frame"]
    counter_coordinates = _counter_coordinates(
        top,
        left,
        interior_height,
        interior_width,
        direction,
    )
    for index, (row, col) in enumerate(counter_coordinates):
        grid[row, col] = (
            roles["advanced"]
            if index < spec["advanced_count"]
            else roles["remaining"]
        )


def _assert_basic_grid(grid: np.ndarray) -> None:
    _need(isinstance(grid, np.ndarray), "grid must be a NumPy array")
    _need(grid.ndim == 2, "grid must be two-dimensional")
    height, width = grid.shape
    _need(1 <= height <= 30 and 1 <= width <= 30, "grid size is invalid")
    _need(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
    _need(grid.size > 0, "grid may not be empty")
    _need(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for first_pair, second_pair in zip(first[split], second[split]):
            if not np.array_equal(first_pair["input"], second_pair["input"]):
                return False
            if not np.array_equal(first_pair["output"], second_pair["output"]):
                return False
    return True


def _expect_value_error(**kwargs) -> None:
    raised = False
    try:
        generate_puzzle(123, **kwargs)
    except ValueError:
        raised = True
    _need(raised, f"expected ValueError for {kwargs}")


def _grid_signature(grid):
    return grid.shape, tuple(map(int,grid.flat))


# Exact source contents serve regression and duplicate checks; sizes remain legal.
_SOURCE_TEXT = (('66666666 99996666 99996999 00006999 05506000 05506070 05506050 00006000', '66666666 99996666 00006999 07506000 05506070 05506070 00006000 66666666'), ('66666666666666666 66669666690000066 66669666690555066 66669666690000066 66666666666666666 66666666666666666 66600000096666669 66605555096666669 66605555096666669 66600000096666669 66666666666666666 66666666666666666 96669000000066666 96669075555066666 96669077555066666 96669000000066666 66666666666666666', '66666666666666666 66669690000066666 66669690777066666 66669690000066666 66666666666666666 66666666666666666 66666666660000009 66666666660777709 66666666660577709 66666666660000009 66666666666666666 66666666666666666 90000000666666666 90777550666666666 90777750666666666 90000000666666666 66666666666666666'), ('666666666666 666666666666 000009666696 055709666696 055709666696 055509666696 000009666696 666666666666 969000000066 969077777066 969077777066 969000000066', '666666666666 666666666666 666660000096 666660777096 666660577096 666660577096 666660000096 666666666666 969000000066 969077777066 969077777066 969000000066'), ('66666666666000006666660000 66666666666055506666660770 66666666666055506666660770 69999999666055506666660770 66666666666055506666660770 66666666666055506666660770 66666666666055706666660770 66666666666000006666660770 66666666666999996666660770 66666666666666666666660770 66666666666666666666660770 66666666666666666666660770 69999999666666666666660770 60000000666666666666660000 60777770666666666999669999 60777770666666666666666666 60777750666666666666666666 60555550666666666666666666 60555550666666666999666666 60555550666666666000666666 60555550666666666070666666 60555550666666666070666666 60555550666666666050666666 60000000666999996050666666 66666666666666666050666666 66666666666666666000669999', '66666666666666666666660000 66666666666666666666660770 66666666666666666666660770 69999999666666666666660770 60000000666666666666660770 60777770666666666666660770 60777770666666666666660770 60777770666666666666660770 60777770666666666666660770 60777550666666666666660770 60555550666666666666660770 60555550666666666666660770 60555550666666666666660770 60555550666666666666660000 60000000666666666999669999 66666666666000006999666666 66666666666055706000666666 66666666666077706070666666 66666666666077706070666666 66666666666077706070666666 66666666666077706070666666 66666666666077706070666666 66666666666000006000666666 66666666666999996666666666 66666666666666666666666666 66666666666666666666669999'))


# The only puzzle-wide color convention is the ordered counter alphabet.
# Frames, backgrounds and matched bars are identified from complete geometry.
def _rectangles(grid):
    rectangles = []
    for color in np.unique(grid):
        for component in _frame_components(grid, int(color)):
            rows, cols = zip(*component)
            t, b, l, r = min(rows), max(rows), min(cols), max(cols)
            if b-t < 2 or r-l < 2 or len(component) != 2*(b-t+r-l):
                continue
            if any(y not in (t,b) and x not in (l,r) for y,x in component):
                continue
            interior_colors = set(map(int, np.unique(grid[t+1:b,l+1:r])))
            if int(color) in interior_colors or len(interior_colors) > 2:
                continue
            rectangles.append(dict(top=t, bottom=b, left=l, right=r,
                                   frame=int(color), colors=interior_colors))
    return rectangles


def _explain_grid(grid, roles, rectangles, background):
    advanced, remaining = roles['advanced'], roles['remaining']
    frame_colors = {v['frame'] for v in rectangles}
    _need(background not in frame_colors | {advanced, remaining}, 'conflicting roles')
    cover = np.zeros(grid.shape, dtype=np.int8)
    swept = np.zeros(grid.shape, dtype=np.int8)
    vehicles = []
    for rectangle in rectangles:
        v = dict(rectangle)
        t,b,l,r = (v[k] for k in ('top','bottom','left','right'))
        attached = []
        for direction in _DIRECTIONS:
            segment = _segment_for_rectangle(t,b,l,r,direction,0,grid.shape)
            if segment is None:
                continue
            marker = int(grid[segment[0]])
            if marker in frame_colors | {background, advanced, remaining}:
                continue
            if _is_exact_marker_bar(grid, segment, marker):
                attached.append((direction,segment,marker))
        _need(len(attached) == 1, 'ambiguous or missing leading bar')
        direction, segment, marker = attached[0]
        destination = None
        for distance in range(1, max(grid.shape)+1):
            candidate = _segment_for_rectangle(t,b,l,r,direction,distance,grid.shape)
            if candidate is None:
                break
            values = _segment_values(grid,candidate)
            if all(c == marker for c in values):
                _need(_is_exact_marker_bar(grid,candidate,marker),'wrong destination length')
                destination = candidate
                break
            _need(all(c == background for c in values),'obstructed corridor')
        _need(destination is not None,'missing destination')
        coordinates = _counter_coordinates(t,l,b-t-1,r-l-1,direction)
        values = [int(grid[p]) for p in coordinates]
        advanced_count = values.count(advanced)
        _need(values == [advanced]*advanced_count + [remaining]*(len(values)-advanced_count),
              'counter is not a direction-relative prefix')
        cover[t:b+1,l:r+1] += 1
        for p in segment + destination:
            cover[p] += 1
        # The whole future corridor is reserved, even for an exhausted vehicle.
        ys = [t,b] + [p[0] for p in destination]
        xs = [l,r] + [p[1] for p in destination]
        bounds = (min(ys),max(ys),min(xs),max(xs))
        y0,y1,x0,x1 = bounds
        swept[y0:y1+1,x0:x1+1] += 1
        v.update(direction=direction, distance=distance, marker=marker,
                 background=background, attached_segment=segment,
                 destination_segment=destination, interior_height=b-t-1,
                 interior_width=r-l-1, advanced_count=advanced_count,
                 capacity=len(values), bounds=bounds)
        vehicles.append(v)
    _need(np.max(cover) <= 1,'shared structures or destinations')
    _need(np.max(swept) <= 1,'overlapping swept corridors')
    _need(np.array_equal(cover == 1, grid != background),'unexplained cells')
    return sorted(vehicles,key=lambda v:(v['top'],v['left']))


def _analyze_grid(grid, roles):
    alphabet = {roles['advanced'],roles['remaining']}
    rectangles = [v for v in _rectangles(grid) if v['colors'] <= alphabet]
    _need(bool(rectangles),'no vehicles')
    explanations = []
    for background in np.unique(grid):
        try:
            explanations.append(_explain_grid(grid,roles,rectangles,int(background)))
        except AssertionError:
            pass
    _need(len(explanations) == 1,'scene lacks a unique complete geometric explanation')
    return explanations[0]


def _transform_vehicles(grid, roles, vehicles):
    output = grid.copy()
    background = vehicles[0]['background']
    for v in vehicles:
        output[v['top']:v['bottom']+1,v['left']:v['right']+1] = background
        _set_segment(output,v['attached_segment'],background)
    for v in vehicles:
        movement = _move_count(v)
        dy,dx = _DELTAS[v['direction']]
        t,l = v['top']+dy*movement,v['left']+dx*movement
        b,r = t+v['interior_height']+1,l+v['interior_width']+1
        output[t:b+1,l:r+1] = v['frame']
        for i,p in enumerate(_counter_coordinates(t,l,v['interior_height'],v['interior_width'],v['direction'])):
            output[p] = roles['advanced'] if i < v['advanced_count']+movement else roles['remaining']
        attached = _segment_for_rectangle(t,b,l,r,v['direction'],0,grid.shape)
        _need(attached is not None,'movement leaves canvas')
        _set_segment(output,attached,v['marker'])
    return output


def _transform_grid(grid, roles):
    return _transform_vehicles(grid,roles,_analyze_grid(grid,roles))


def _role_candidates(pairs):
    _need(bool(pairs),'training required')
    # Interior alphabets come from rectangular frame geometry, not frequency.
    possible = None
    for pair in pairs:
        colors = set()
        for v in _rectangles(pair['input']):
            colors.update(v['colors'])
        current = {(a,b) for a in colors for b in range(10) if a != b}
        current |= {(b,a) for a,b in current}
        possible = current if possible is None else possible & current
    candidates = []
    for advanced,remaining in sorted(possible or ()):
        roles = dict(advanced=advanced,remaining=remaining)
        try:
            if all(np.array_equal(_transform_grid(p['input'],roles),p['output']) for p in pairs):
                candidates.append(roles)
        except AssertionError:
            pass
    return candidates


def solve(train_pairs, test_input):
    roles = _role_candidates(train_pairs)
    _need(len(roles) == 1,'training does not identify the counter states')
    return _transform_grid(test_input,roles[0])


def _unit_step_prediction(grid, roles, vehicles=None):
    """Alternative counter implementation: rotate the actual block, tick, shift."""
    if vehicles is None:
        vehicles = _analyze_grid(grid,roles)
    output = grid.copy()
    background = vehicles[0]['background']
    for v in vehicles:
        output[v['top']:v['bottom']+1,v['left']:v['right']+1] = background
        _set_segment(output,v['attached_segment'],background)
    turns = {'U':0,'D':2,'L':3,'R':1}
    for v in vehicles:
        block = grid[v['top']:v['bottom']+1,v['left']:v['right']+1].copy()
        canonical = np.rot90(block,turns[v['direction']]).copy()
        interior = canonical[1:-1,1:-1].copy()
        t,l = v['top'],v['left']
        dy,dx = _DELTAS[v['direction']]
        steps = 0
        while steps < v['distance']:
            pending = np.flatnonzero(interior.ravel() == roles['remaining'])
            if len(pending) == 0:
                break
            interior.flat[int(pending[0])] = roles['advanced']
            t,l,steps = t+dy,l+dx,steps+1
        canonical[1:-1,1:-1] = interior
        moved = np.rot90(canonical,-turns[v['direction']])
        h,w = moved.shape
        output[t:t+h,l:l+w] = moved
        segment = _segment_for_rectangle(t,t+h-1,l,l+w-1,v['direction'],0,grid.shape)
        _set_segment(output,segment,v['marker'])
    return output


def _sample_puzzle_parameters(rng):
    colors = list(map(int,rng.permutation(10)))
    mode = str(rng.choice(('shared','pair','vehicle'),p=(.30,.30,.40)))
    held_out = str(rng.choice(_DIRECTIONS)) if rng.random() < .20 else None
    if held_out:
        directions = tuple(d for d in _DIRECTIONS if d != held_out)
    else:
        directions = tuple(map(str,rng.choice(_DIRECTIONS,size=int(rng.integers(2,5)),replace=False)))
    return dict(advanced=colors[0],remaining=colors[1],palette=colors[2:],
                mode=mode, directions=directions, held_out=held_out)


def _scene_palette(rng, parameters):
    colors = parameters['palette'][:]
    if parameters['mode'] != 'shared':
        rng.shuffle(colors)
    nf = int(rng.integers(1,4)) if parameters['mode'] == 'vehicle' else 1
    nm = int(rng.integers(1,4)) if parameters['mode'] == 'vehicle' else 1
    return colors[0],colors[1:1+nf],colors[1+nf:1+nf+nm]


def _make_vehicle_spec(rng, direction, stopping=None, limit=(30,30), teaching=False):
    # Work in the travel-up view: a rows of progress, b cells per row.
    axis,cross = limit if direction in ('U','D') else limit[::-1]
    max_a = axis-4
    max_b = cross-2
    if max_a < 1 or max_b < 1:
        return None
    journey = not teaching and axis >= 17 and max_b >= 2 and rng.random() < .25
    if teaching:
        if max_a < 2 or max_b < 2 or axis < 9:
            return None
        a = int(rng.integers(2,min(6,max_a,axis-6)+1))
        b = int(rng.integers(2,min(5,max_b)+1))
    else:
        a = int(rng.integers(1,max_a+1))
        b = int(rng.integers(1,max_b+1))
        if rng.random() < .60:
            b = int(rng.integers(1,min(5,max_b)+1))
        if rng.random() < .28:
            b = 1
        if rng.random() < .15:
            a = min(b,max_a)
        if journey:
            a = int(rng.integers(2,min(10,axis-12)+1))
            b = int(rng.integers(2,min(5,max_b)+1))
    capacity = a*b
    max_distance = axis-a-3
    if max_distance < 1:
        return None
    if stopping is None:
        stopping = str(rng.choice(('destination','fuel','exact','zero'),p=(.32,.30,.23,.15)))
    if teaching and stopping == 'destination':
        upper = min(max_distance,capacity-2)
        if upper < 1:
            return None
        distance = int(rng.integers(1,upper+1))
        remaining = int(rng.integers(distance+1,capacity))
    elif teaching and stopping == 'fuel':
        if max_distance < 2:
            return None
        distance = int(rng.integers(2,max_distance+1))
        remaining = int(rng.integers(1,min(capacity-1,distance-1)+1))
    elif stopping == 'zero':
        distance,remaining = int(rng.integers(8 if journey else 1,max_distance+1)),0
    elif stopping == 'destination':
        if capacity < 2:
            return None
        upper = min(max_distance,capacity-1)
        distance = int(rng.integers(min(8,upper) if journey else 1,upper+1))
        remaining = capacity if rng.random() < .30 else int(rng.integers(distance+1,capacity+1))
    elif stopping == 'exact':
        upper = min(max_distance,capacity)
        distance = int(rng.integers(min(8,upper) if journey else 1,upper+1))
        remaining = distance
    else:
        if max_distance < 2:
            return None
        # Fresh fuel-limited vehicles are possible when capacity < distance.
        upper = min(capacity,max_distance-1)
        remaining = capacity if capacity < max_distance and rng.random() < .45 else int(rng.integers(max(1,upper//2) if journey else 1,upper+1))
        distance = int(rng.integers(remaining+1,max_distance+1))
    ih,iw = (a,b) if direction in ('U','D') else (b,a)
    return dict(direction=direction,interior_height=ih,interior_width=iw,
                distance=distance,advanced_count=capacity-remaining)


def _available_positions(occupied, shape):
    h,w = shape
    H,W = occupied.shape
    if h > H or w > W:
        return np.empty((0,2),dtype=int)
    # Reserve one cell between swept regions. Edges need no outer margin.
    padded = np.pad(occupied,1)
    blocked = np.zeros_like(occupied)
    for dy in range(3):
        for dx in range(3):
            blocked |= padded[dy:dy+H,dx:dx+W]
    integral = np.pad(blocked.astype(int),((1,0),(1,0))).cumsum(0).cumsum(1)
    sums = integral[h:,w:]-integral[:-h,w:]-integral[h:,:-w]+integral[:-h,:-w]
    return np.argwhere(sums == 0)


def _generate_pair(rng, regime, parameters, lessons, forced_direction=None, fallback=0):
    is_test = regime == 'target_test'
    margin = 0 if fallback else int(rng.choice((0,1,2),p=(.65,.20,.15)))
    # Both splits share all dimensions; tests put more weight on crowded scenes.
    if fallback:
        H,W = 30,30
    else:
        H,W = int(rng.integers(9,31)),int(rng.integers(9,31))
        if rng.random() < .16:
            H = W = int(rng.choice((8,12,17,26,30)))
        if lessons:
            H,W = max(H,14),max(W,14)
        H,W = min(H,30-2*margin),min(W,30-2*margin)
    bg,frames,markers = _scene_palette(rng,parameters)
    grid = np.full((H,W),bg,dtype=np.int8)
    occupied = np.zeros((H,W),dtype=bool)
    count = int(rng.choice(np.arange(1,9),p=(.12,.22,.21,.17,.12,.08,.05,.03) if not is_test else (.10,.15,.18,.20,.15,.10,.07,.05)))
    count = max(count,len(lessons),1)
    descriptions = [(d,s,True) for d,s in lessons]
    if forced_direction and not descriptions:
        descriptions.append((forced_direction,None,False))
    stationary = not lessons and rng.random() < .08
    pair_pool = parameters['directions']
    if is_test and parameters['held_out']:
        pair_pool = pair_pool + (parameters['held_out'],)
    if len(pair_pool) == 4 and rng.random() < .30:
        # Occasional complete orientation coverage, without direction/stop ties.
        count = max(count,4)
        missing = [d for d in pair_pool if d not in {x[0] for x in descriptions}]
        rng.shuffle(missing)
        descriptions.extend((d,'zero' if stationary else None,False) for d in missing)
        count = max(count,len(descriptions))
    # Sometimes all cars share a direction; otherwise choose a random subset.
    selected = tuple(map(str,rng.choice(pair_pool,size=int(rng.integers(1,len(pair_pool)+1)),replace=False)))
    while len(descriptions) < count:
        descriptions.append((str(rng.choice(selected)),'zero' if stationary else None,False))
    placements = []
    for index,(direction,stopping,teaching) in enumerate(descriptions):
        placed = False
        for attempt in range(100):
            if fallback and teaching:
                # A bounded geometric fallback uses small informative counters.
                limit = (11,11)
            elif attempt < 25 and count <= 3:
                limit = (H,W)
            elif attempt < 65:
                limit = (min(H,18),min(W,18))
            else:
                limit = (min(H,11),min(W,11)) if teaching else (min(H,9),min(W,9))
            spec = _make_vehicle_spec(rng,direction,stopping,limit,teaching)
            if spec is None:
                continue
            h,w = _footprint_shape(spec)
            # Spread the area budget without equal-sized spatial slots.
            budget = H*W/max(1,count)*1.8
            if h*w > budget and not teaching and attempt < 65:
                continue
            positions = _available_positions(occupied,(h,w))
            if len(positions) == 0:
                continue
            # Edge contacts, free scatter, and aligned lanes all remain possible.
            preference = rng.random()
            if preference < .28:
                edge = positions[(positions[:,0] == 0)|(positions[:,1] == 0)|
                                 (positions[:,0]+h == H)|(positions[:,1]+w == W)]
                if len(edge):
                    positions = edge
            elif preference < .45 and placements:
                aligned = positions[np.isin(positions[:,0],[p[0] for p in placements]) |
                                    np.isin(positions[:,1],[p[1] for p in placements])]
                if len(aligned):
                    positions = aligned
            y,x = map(int,positions[int(rng.integers(len(positions)))])
            local = dict(advanced=parameters['advanced'],remaining=parameters['remaining'],
                         background=bg,frame=int(rng.choice(frames)),marker=int(rng.choice(markers)))
            _draw_initial_vehicle(grid,y,x,spec,local)
            occupied[y:y+h,x:x+w] = True
            placements.append((y,x,h,w))
            placed = True
            break
        if not placed and (teaching or not placements or index == 0 and forced_direction):
            return None
    # Crop to asymmetric random margins, retaining occasional wide separation.
    if not fallback and rng.random() < .65:
        ys,xs = np.nonzero(occupied)
        t = int(rng.integers(0,int(ys.min())+1))
        l = int(rng.integers(0,int(xs.min())+1))
        b = int(rng.integers(int(ys.max())+1,H+1))
        r = int(rng.integers(int(xs.max())+1,W+1))
        grid = grid[t:b,l:r].copy()
    if margin:
        grid = np.pad(grid,margin,constant_values=bg)
    try:
        out = _transform_grid(grid,parameters)
    except AssertionError:
        return None
    return dict(input=grid,output=out)


def _source_pairs():
    return [dict(input=np.array([[int(c) for c in row] for row in inp.split()],dtype=np.int8),
                 output=np.array([[int(c) for c in row] for row in out.split()],dtype=np.int8))
            for inp,out in _SOURCE_TEXT]


def _is_source_pair(pair):
    for inp,out in _SOURCE_TEXT:
        rows = inp.split()
        if pair['input'].shape != (len(rows),len(rows[0])):
            continue
        encoded = ' '.join(''.join(map(str,row)) for row in pair['input'])
        if encoded == inp and ' '.join(''.join(map(str,row)) for row in pair['output']) == out:
            return True
    return False


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    ntrain,ntest = _select_counts(int(seed),rng,num_train,num_test)
    parameters = _sample_puzzle_parameters(rng)
    # Assign lessons to random pair positions and directions. Both stopping causes
    # and every training orientation are demonstrated, including with two pairs.
    directions = list(parameters['directions'])
    rng.shuffle(directions)
    stops = ['destination','fuel'] + ['exact']*(len(directions)-2)
    rng.shuffle(stops)
    lessons = [[] for _ in range(ntrain)]
    pair_order = list(map(int,rng.permutation(ntrain)))
    for i,(direction,stop) in enumerate(zip(directions,stops)):
        lessons[pair_order[i % ntrain]].append((direction,stop))
    puzzle = dict(train=[],test=[])
    seen = set()
    seen_counts = set()
    for split,count in (('train',ntrain),('test',ntest)):
        for i in range(count):
            lesson = lessons[i] if split == 'train' else []
            forced = parameters['held_out'] if split == 'test' and i == 0 else None
            pair = None
            # Bounded random construction followed by small-counter fallback.
            for attempt in range(48):
                candidate = _generate_pair(rng,'source_train' if split == 'train' else 'target_test',
                                           parameters,lesson,forced,fallback=int(attempt >= 24))
                if candidate is None or _is_source_pair(candidate):
                    continue
                object_count = len(_analyze_grid(candidate['input'],parameters))
                if split == 'test' and i == count-1 and seen_counts == {object_count}:
                    continue
                key = _grid_signature(candidate['input'])
                if key not in seen:
                    pair = candidate
                    seen.add(key)
                    seen_counts.add(object_count)
                    break
            _need(pair is not None,'bounded construction failed')
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _need(isinstance(puzzle,dict) and set(puzzle) == {'train','test'},'invalid puzzle schema')
    _need(all(isinstance(puzzle[s],list) for s in ('train','test')),'splits must be lists')
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
          len(puzzle['train']) > len(puzzle['test']),'invalid counts')
    seen = set()
    for split in ('train','test'):
        for p in puzzle[split]:
            _need(isinstance(p,dict) and set(p) == {'input','output'},'invalid pair schema')
            for g in p.values():
                _assert_basic_grid(g)
            _need(p['input'].shape == p['output'].shape,'dimensions changed')
            key = _grid_signature(p['input'])
            _need(key not in seen,'duplicate input')
            seen.add(key)
            _need(not _is_source_pair(p),'exact source copy')
    candidates = _role_candidates(puzzle['train'])
    _need(len(candidates) == 1,'training does not uniquely identify state meanings')
    roles = candidates[0]
    train_vehicles = []
    test_directions = set()
    for split in ('train','test'):
        for p in puzzle[split]:
            vehicles = _analyze_grid(p['input'],roles)
            expected = _transform_vehicles(p['input'],roles,vehicles)
            _need(np.array_equal(expected,p['output']),'incorrect transformation')
            _need(np.array_equal(_unit_step_prediction(p['input'],roles,vehicles),expected),
                  'bulk update differs from rotated unit steps')
            if split == 'train':
                train_vehicles.extend(vehicles)
            else:
                test_directions.update(v['direction'] for v in vehicles)
    regimes = {_stopping_regime(v) for v in train_vehicles}
    _need({'destination','fuel'} <= regimes,'training must show both stopping causes')
    active_directions = {v['direction'] for v in train_vehicles if _move_count(v) > 0}
    _need(len(active_directions) >= 2,'training must demonstrate rotation')
    if test_directions - active_directions:
        _need(len(active_directions) >= 3,'held-out direction needs three demonstrated orientations')
    _need(any(v['interior_height'] > 1 and v['interior_width'] > 1 and
              0 < v['advanced_count'] < v['capacity'] and _move_count(v) > 0
              for v in train_vehicles),'training lacks informative two-dimensional counter')


def puzzle_features(puzzle):
    """Review measurements reconstructed from returned grids, without RNG metadata."""
    roles = _role_candidates(puzzle['train'])
    _need(len(roles) == 1,'ambiguous features')
    roles = roles[0]
    records = []
    for split in ('train','test'):
        for index,pair in enumerate(puzzle[split]):
            g = pair['input']
            vs = _analyze_grid(g,roles)
            objects = []
            for v in vs:
                a,b = (v['interior_height'],v['interior_width']) if v['direction'] in ('U','D') else (v['interior_width'],v['interior_height'])
                move = _move_count(v)
                objects.append(dict(direction=v['direction'],stop=_stopping_regime(v),
                                    rows=a,columns=b,capacity=v['capacity'],distance=v['distance'],
                                    advanced=v['advanced_count'],remaining=_remaining_count(v),movement=move,
                                    fresh=v['advanced_count'] == 0,exhausted=_remaining_count(v) == 0,
                                    row_crossings=(v['advanced_count']+move)//b-v['advanced_count']//b,
                                    partial_end=(v['advanced_count']+move)%b != 0,
                                    frame=v['frame'],marker=v['marker'],bounds=v['bounds']))
            records.append(dict(split=split,index=index,shape=g.shape,count=len(vs),
                                background=vs[0]['background'],advanced=roles['advanced'],remaining=roles['remaining'],
                                palette=tuple(map(int,np.unique(g))),stationary=np.array_equal(g,pair['output']),
                                density=float(np.mean(g != vs[0]['background'])),
                                edge=any(y0 == 0 or x0 == 0 or y1 == g.shape[0]-1 or x1 == g.shape[1]-1
                                         for y0,y1,x0,x1 in (v['bounds'] for v in vs)),vehicles=objects))
    return records


def self_test() -> None:
    original = _source_pairs()
    inferred = _role_candidates(original[:3])
    _need(inferred == [dict(advanced=7,remaining=5)],'source training role regression')
    _need(_role_candidates(original[:1]) == inferred,'non-majority background regression')
    for pair in original:
        _need(np.array_equal(_transform_grid(pair['input'],inferred[0]),pair['output']),'source transform regression')
        _need(np.array_equal(_unit_step_prediction(pair['input'],inferred[0]),pair['output']),'source unit-step regression')
    _need(np.array_equal(solve(original[:3],original[3]['input'].copy()),original[3]['output']),
          'source prediction from training only')
    for tr,te in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(2001,tr,te)
        _need(tr is None or len(p['train']) == tr,'train count ignored')
        _need(te is None or len(p['test']) == te,'test count ignored')
        _need(_puzzles_equal(p,generate_puzzle(2001,tr,te)),'nondeterministic explicit/partial counts')
    for args in (dict(num_train=1),dict(num_train=11),dict(num_test=0),dict(num_test=5),
                 dict(num_train=2,num_test=2),dict(num_train=3,num_test=4),dict(num_train=True)):
        _expect_value_error(**args)
    records,puzzles = [],[]
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        records.extend(puzzle_features(p))
    _need(_puzzles_equal(puzzles[17],generate_puzzle(17)),'automatic nondeterminism')
    _need(not _puzzles_equal(puzzles[17],puzzles[18]),'seed has no effect')
    _need(len({len(p['train']) for p in puzzles}) >= 4,'train count diversity')
    _need(len({len(p['test']) for p in puzzles}) >= 3,'test count diversity')
    _need(len({r['shape'] for r in records}) >= 70,'shape diversity')
    _need(len({_grid_signature(pair['input']) for p in puzzles for split in p.values() for pair in split}) > 250,'grid diversity')
    objects = [v for r in records for v in r['vehicles']]
    _need({(v['direction'],v['stop']) for v in objects} ==
          {(d,s) for d in _DIRECTIONS for s in ('destination','fuel','exact','zero')},'direction/stopping correlation')
    for split in ('train','test'):
        rs = [r for r in records if r['split'] == split]
        vs = [v for r in rs for v in r['vehicles']]
        _need(any(r['count'] == 1 for r in rs) and max(r['count'] for r in rs) >= 5,'object count diversity')
        _need(any(r['stationary'] for r in rs),'missing entirely stationary pairs')
        _need(any(v['fresh'] and v['stop'] == 'fuel' for v in vs),'missing fresh fuel-limited counter')
        _need(max(v['distance'] for v in vs) >= 15 and max(v['movement'] for v in vs) >= 12,'missing long journeys')
        _need(max(v['rows'] for v in vs) >= 12 and max(v['capacity'] for v in vs) >= 45,'missing large counters')
        _need(any(v['columns'] == 1 for v in vs) and any(v['rows'] == 1 for v in vs),'missing thin counters')
        _need(any(v['rows'] == v['columns'] for v in vs),'missing squares')
        _need(any(v['row_crossings'] >= 3 and v['partial_end'] for v in vs),'missing multiple-row partial progression')
        _need(any(r['edge'] for r in rs),'no canvas edge contact')
        _need(any(max(r['shape']) >= 2*min(r['shape']) for r in rs),'missing rectangular canvases')
        _need(any(len({v['direction'] for v in r['vehicles']}) < r['count'] for r in rs),'no repeated directions')
        _need(any(len({v['direction'] for v in r['vehicles']}) == 4 for r in rs),'no all-orientation scenes')
        _need(any(len({v['capacity'] for v in r['vehicles']}) >= 3 for r in rs),'no mixed capacities')
        _need(any(len({v['frame'] for v in r['vehicles']}) >= 2 and len({v['marker'] for v in r['vehicles']}) >= 2 for r in rs),'no per-vehicle palettes')
    per_puzzle = [puzzle_features(p) for p in puzzles]
    _need(sum(len({r['palette'] for r in rs}) > 1 for rs in per_puzzle) >= 20,'no within-puzzle palette diversity')
    _need(any(len({r['background'] for r in rs}) == 1 and len({tuple(sorted({v['frame'] for v in r['vehicles']})) for r in rs}) == 1 for rs in per_puzzle),'missing shared palettes')
    _need(all(len({r['shape'] for r in rs}) >= 2 and len({v['capacity'] for r in rs for v in r['vehicles']}) >= 3 for rs in per_puzzle),'within-puzzle geometry diversity')
    _need(all(len({r['count'] for r in rs}) >= 2 for rs in per_puzzle),'within-puzzle count diversity')
    _need(len({(r['advanced'],r['remaining']) for r in records}) >= 25,'counter palette diversity')
    _need(min(r['density'] for r in records) < .15 and max(r['density'] for r in records) > .65,
          'density range collapsed')
    _need(any(not r['edge'] for r in records),'missing inset scenes')
    _need(all({min(3,int(4*v['advanced']/v['capacity'])) for v in objects
               if v['stop'] == stop and 0 < v['advanced'] < v['capacity']} == {0,1,2,3}
              for stop in ('destination','fuel','exact')), 'fill fraction/stopping coverage')
    _need(any(any(v['fresh'] and min(v['rows'],v['columns']) == 1 for v in r['vehicles'])
              and any(v['distance'] >= 8 and 0 < v['advanced'] < v['capacity'] for v in r['vehicles'])
              and len({v['direction'] for v in r['vehicles']}) < r['count']
              and len({v['stop'] for v in r['vehicles']}) > 1 for r in records),
          'missing combined fresh/thin/long/repeated-direction/stopping scene')
    heldout = 0
    for rs in per_puzzle:
        train_dirs = {v['direction'] for r in rs if r['split'] == 'train' for v in r['vehicles']}
        test_dirs = {v['direction'] for r in rs if r['split'] == 'test' for v in r['vehicles']}
        heldout += bool(test_dirs-train_dirs)
    _need(1 <= heldout < 25,'orientation regimes collapsed')
    # Reject plausible corruptions, not just invalid metadata.
    p = puzzles[0]
    broken = {s:[{k:g.copy() for k,g in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0,0] = (int(broken['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a wrong output')
