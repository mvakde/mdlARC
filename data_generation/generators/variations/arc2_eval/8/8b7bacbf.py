"""Cupped paint sources supply contours along identifiable string routes."""

import numpy as np
from collections import deque


def _is_count(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)

    if num_test is not None:
        if not _is_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)

    key = abs(int(seed))

    if num_train is None and num_test is None:
        num_test = 1 + key % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + ((key // 4 + 3 * key) % span)

    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = minimum_train + key % (11 - minimum_train)

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test for the requested num_train")
        num_test = 1 + key % maximum_test

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "counts must satisfy 2 <= train <= 10, "
            "1 <= test <= 4, and train > test"
        )

    return num_train, num_test


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

def components(cells, steps=D8):
    left = set(cells)
    out = []
    while left:
        p = min(left)
        left.remove(p)
        group = {p}
        queue = [p]
        for r, c in queue:
            for dr, dc in steps:
                q = (r+dr, c+dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        out.append(group)
    return out

def neigh(cells, shape, steps=D8):
    h,w = shape
    return {(r+dr,c+dc) for r,c in cells for dr,dc in steps
            if 0 <= r+dr < h and 0 <= c+dc < w} - set(cells)

def analyze(a):
    bg = int(np.bincount(a.flat, minlength=10).argmax())
    blobs = components(zip(*np.where(a == bg)), D4)
    ext = max(blobs, key=len)
    loops = []
    for cells in blobs:
        if cells is ext:
            continue
        edge = neigh(cells, a.shape, D4)
        colors = {int(a[p]) for p in edge}
        if len(colors) != 1:
            continue
        color = colors.pop()
        boundary = {p for p in neigh(cells, a.shape) if a[p] == color}
        loops.append((cells, boundary, color))
    all_boundary = set().union(*(b for _,b,_ in loops))
    sources=[]
    for color in range(10):
        if color == bg:
            continue
        for cells in components(zip(*np.where(a == color))):
            if len(cells)!=1 or cells & all_boundary:
                continue
            p=next(iter(cells))
            edge = {q for q in neigh(cells,a.shape) if a[q] != bg}
            if len(edge)>=3 and len({int(a[q]) for q in edge})==1:
                sources.append((p,color,int(a[next(iter(edge))])))
    sourceset={p for p,_,_ in sources}
    wires=[]
    for color in range(10):
        if color == bg:
            continue
        for cells in components(set(zip(*np.where(a == color))) - all_boundary - sourceset):
            wires.append((cells,color))
    nodes=[b for _,b,_ in loops]+[c for c,_ in wires]
    graph=[set() for _ in nodes]
    for i,(_,b,_) in enumerate(loops):
        nb=neigh(b,a.shape)
        for j,(cells,col) in enumerate(wires,len(loops)):
            if cells&nb:
                graph[i].add(j);graph[j].add(i)
    # A one-pixel occluder can interrupt a straight or gently turning route.
    for i,(cells,col) in enumerate(wires,len(loops)):
        for j,(other,col2) in enumerate(wires,len(loops)):
            if j<=i or col!=col2:
                continue
            candidates=[]
            for r,c in cells:
                for y,x in other:
                    if max(abs(y-r),abs(x-c))==2:
                        mids=neigh({(r,c)},a.shape)&neigh({(y,x)},a.shape)
                        if any(a[p] not in (bg,col) and p not in all_boundary and p not in sourceset for p in mids):
                            candidates.append(((r,c),(y,x)))
            if candidates:
                graph[i].add(j);graph[j].add(i)
    paints={}
    for p,color,wire in sources:
        start=[i for i,(cells,col) in enumerate(wires,len(loops)) if col==wire and cells&neigh({p},a.shape)]
        seen=set(start);queue=list(start)
        for i in queue:
            for j in graph[i]-seen:
                seen.add(j);queue.append(j)
        for i in seen:
            if i < len(loops):
                assert i not in paints or paints[i]==color
                paints[i]=color
    out=a.copy()
    for i,col in paints.items():
        for p in loops[i][0]:out[p]=col
    return out,dict(bg=bg,loops=loops,sources=sources,wires=wires,
                    graph=graph,paints=paints)


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, items):
    return items[_ri(rng, 0, len(items)-1)]


def _move(cells, dr, dc):
    return {(r+dr, c+dc) for r,c in cells}


def _inside(cells, shape):
    h,w=shape
    return {(r,c) for r,c in cells if 0<=r<h and 0<=c<w}


def _contour(rng, small=False):
    """Sample connected row intervals, then their minimal four-neighbor shell."""
    kind=_pick(rng, ('box','box','diamond','slant','steps','waist','long'))
    h=_ri(rng,1,3 if small else 7)
    w=_ri(rng,1,3 if small else 6)
    cells=set()
    if kind=='long':
        h,w=1,_ri(rng,3,4 if small else 9)
    if kind=='diamond':
        radius=_ri(rng,1,1 if small else 3)
        cells={(r,c) for r in range(-radius,radius+1)
               for c in range(-radius,radius+1) if abs(r)+abs(c)<=radius}
    else:
        left=0
        drift=_pick(rng,(-1,1))
        for r in range(h):
            if kind=='slant':
                left=drift*(r//_ri(rng,1,2))
            elif kind=='steps' and r:
                left+=_pick(rng,(-1,0,1))
            width=w
            if kind=='waist' and h>=3 and 0<r<h-1:
                width=max(1,w-1)
            if kind=='steps':
                width=max(2,w+_ri(rng,-1,1))
            cells.update((r,c) for c in range(left,left+width))
    # Reject disconnected sloping strips by joining adjacent row intervals.
    for r in range(min(p[0] for p in cells)+1,max(p[0] for p in cells)+1):
        prev=[c for rr,c in cells if rr==r-1]
        now=[c for rr,c in cells if rr==r]
        if prev and now and not set(prev)&set(now):
            a,b=sorted((min(now),max(prev))) if min(now)>max(prev) else sorted((max(now),min(prev)))
            cells.update((r,c) for c in range(a,b+1))
    if rng.random()<.5:
        cells={(c,r) for r,c in cells}
    boundary={(r+dr,c+dc) for r,c in cells for dr,dc in D4}-cells
    if kind=='box' and rng.random()<.35:
        boundary|={(r+dr,c+dc) for r,c in cells for dr,dc in D8}-cells
    mr=min(r for r,c in boundary);mc=min(c for r,c in boundary)
    return _move(cells,-mr,-mc),_move(boundary,-mr,-mc)


def _cup(style, rotation):
    if style==3:
        source={(0,0)}
        wire={(0,1),(1,0),(1,1)}
    else:
        source={(0,1)}
        wire={(0,0),(0,2),(1,0),(1,1),(1,2)}
    for _ in range(rotation):
        source={(-c,r) for r,c in source}
        wire={(-c,r) for r,c in wire}
    mr=min(r for r,c in source|wire);mc=min(c for r,c in source|wire)
    return _move(source,-mr,-mc),_move(wire,-mr,-mc)


def _path(rng, starts, goals, allowed, shape):
    """Randomized breadth-first routing; empty cells have equal visible cost."""
    starts=list(sorted(starts))
    rng.shuffle(starts)
    queue=deque(starts)
    previous={p:None for p in starts}
    directions=[D8[i] for i in rng.permutation(8)]
    h,w=shape
    end=None
    while queue:
        p=queue.popleft()
        if p in goals:
            end=p
            break
        r,c=p
        for dr,dc in directions:
            q=(r+dr,c+dc)
            if 0<=q[0]<h and 0<=q[1]<w and q in allowed and q not in previous:
                previous[q]=p
                queue.append(q)
    if end is None:
        return None
    result=[]
    while end is not None:
        result.append(end)
        end=previous[end]
    return result


def _scene(rng, spec, convention, attempt=0):
    """Place contours and endpoints, then connect their visible routes."""
    crossing=spec['crossing']
    touching=spec['touching']
    clipped=spec['clipped']
    active=spec['active']
    inactive=spec['inactive']
    nroute=active+inactive
    small=spec['size']=='small'
    if crossing:
        h=_ri(rng,24,30);w=_ri(rng,24,30)
    elif small:
        h=_ri(rng,8,13);w=_ri(rng,8,15)
    elif spec['size']=='large' or attempt>14:
        h=_ri(rng,25,30);w=_ri(rng,25,30)
    else:
        h=_ri(rng,17,26);w=_ri(rng,17,29)
    if spec.get('full'):
        h=w=30
    if spec.get('compact'):
        h=w=8
    if spec.get('narrow'):
        h,w=_ri(rng,7,10),_ri(rng,18,30)
    shape=(h,w)
    canvas={(r,c) for r in range(h) for c in range(w)}
    bg=spec.get('background',_pick(rng,convention['backgrounds']))
    palette=[int(c) for c in rng.permutation(np.arange(1,10)) if c!=bg]
    # A preferred paint recurs independently of orientation; all colors are
    # still copied from local cups, and new test paint values are permitted.
    preferred=convention['paint']
    if preferred!=bg and rng.random()<.65:
        palette.remove(preferred);palette.insert(0,preferred)
    paint_pool=palette[:active]
    paints=[paint_pool[i%len(paint_pool)] for i in range(active)]
    if active>1 and rng.random()<.3:
        paints=[paints[0]]*active
    available=[c for c in palette if c not in paints]
    wires=available[:nroute]
    if len(wires)!=nroute:
        return None
    loop_palette=[c for c in range(10) if c not in [bg,0]+paints+wires]
    if not loop_palette:
        return None
    rng.shuffle(loop_palette)
    loop_palette=loop_palette[:_ri(rng,1,min(3,len(loop_palette)))]
    # No-paint black remains an uncupped tip. A black cup is deliberately
    # outside this generator's evidence-supported convention.
    counts=list(spec['counts'])
    loop_owner=[]
    for owner,count in enumerate(counts):
        loop_owner.extend([owner]*count)
    if touching:
        loop_owner.append(None)
    for _ in range(spec['detached']):
        loop_owner.append(None)
    a=np.full(shape,bg,dtype=np.int64)
    wire_sets=[set() for _ in range(nroute)]
    source_sets=[set() for _ in range(nroute)]
    center=(h//2+_ri(rng,-1,1),w//2+_ri(rng,-1,1))
    if crossing:
        cr,cc=center
        wire_sets[0]={(cr,c) for c in range(cc-4,cc+5)}
        wire_sets[1]={(r,cc) for r in range(cr-4,cr+5)}
    reserved=set().union(*wire_sets)
    reserved|=neigh(reserved,shape)
    regions=[]
    occupied=set()
    # Dense scenes get softly clustered route endpoints after early attempts.
    # These are movable two-dimensional neighborhoods, not parallel strips;
    # the unrestricted attempts and explicit crossings retain interleaving.
    hubs=[(h*.25,w*.25),(h*.25,w*.75),(h*.75,w*.25),(h*.75,w*.75)]
    rng.shuffle(hubs)
    # Prioritize the contour that will get an incidental touching neighbor.
    first_owner=0
    for index,owner in enumerate(loop_owner):
        placed=False
        inner,boundary=_contour(rng,small or crossing or len(loop_owner)>5 or attempt>14)
        lh=max(r for r,c in boundary)+1;lw=max(c for r,c in boundary)+1
        if lh>h or lw>w:
            return None
        is_touch=touching and owner is None and index==sum(counts)
        for _ in range(100):
            rr=_ri(rng,0,h-lh);cc=_ri(rng,0,w-lw)
            if spec.get('axis') and owner==0:
                cc=w-lw if index==0 else _ri(rng,max(3,w//3),w-lw)
            if attempt>14 and nroute>1 and not crossing and owner is not None:
                hr,hc=hubs[owner]
                rr=max(0,min(h-lh,int(hr+_ri(rng,-6,6)-lh//2)))
                cc=max(0,min(w-lw,int(hc+_ri(rng,-6,6)-lw//2)))
            if clipped and index==0:
                side=_pick(rng,range(4))
                depth=_ri(rng,1,2)
                if side==0:rr=-min(depth,lh-2)
                elif side==1:rr=h-lh+min(depth,lh-2)
                elif side==2:cc=-min(depth,lw-2)
                else:cc=w-lw+min(depth,lw-2)
            if is_touch:
                base=regions[0]
                p=_pick(rng,sorted(base['boundary']))
                q=_pick(rng,sorted(boundary))
                dr,dc=_pick(rng,D8)
                rr,cc=p[0]+dr-q[0],p[1]+dc-q[1]
                if rr<0 or cc<0 or rr+lh>h or cc+lw>w:
                    continue
            ins=_inside(_move(inner,rr,cc),shape)
            edge=_inside(_move(boundary,rr,cc),shape)
            cells=ins|edge
            if not ins or cells & (occupied|reserved):
                continue
            if not is_touch and neigh(cells,shape)&occupied:
                continue
            if is_touch:
                near=neigh(edge,shape)&occupied
                if not near or not near<=regions[0]['boundary']:
                    continue
                if neigh(ins,shape,D4)&regions[0]['inner']:
                    continue
                # Contact must involve boundaries, without broad overlapping
                # shells that would hide which loop a string supplies.
                if len(near)>2:
                    continue
            if crossing and owner in (0,1):
                cr,cc0=center
                if owner==0 and min(c for r,c in cells)<cc0+6:
                    continue
                if owner==1 and min(r for r,c in cells)<cr+6:
                    continue
            color=regions[0]['color'] if is_touch else _pick(rng,loop_palette)
            regions.append(dict(inner=ins,boundary=edge,owner=owner,color=color))
            occupied|=cells
            placed=True
            break
        if not placed:
            return None
    # Place cups in free space. Random placement, reflection and rotation break
    # any active-above-inactive convention. Endpoints may occupy the border.
    for owner in range(nroute):
        placed=False
        for _ in range(100):
            rotation=_ri(rng,0,3)
            style=spec['style'] if owner%2==0 else convention['styles'][-1]
            src,wire=_cup(style,rotation)
            sh=max(r for r,c in src|wire)+1;sw=max(c for r,c in src|wire)+1
            rr=_ri(rng,0,h-sh);cc=_ri(rng,0,w-sw)
            if attempt>14 and nroute>1 and not crossing:
                hr,hc=hubs[owner]
                rr=max(0,min(h-sh,int(hr+_ri(rng,-7,7))))
                cc=max(0,min(w-sw,int(hc+_ri(rng,-7,7))))
            if rng.random()<.55:
                if rng.random()<.5:rr=_pick(rng,(0,h-sh))
                else:cc=_pick(rng,(0,w-sw))
            if style==3 and rng.random()<.14:
                sr,sc=next(iter(src))
                rr=0 if sr==0 else h-sh
                cc=0 if sc==0 else w-sw
            if spec.get('axis'):
                cc=_ri(rng,0,2)
            src=_move(src,rr,cc);wire=_move(wire,rr,cc)
            cells=src|wire
            other=set().union(*wire_sets,*source_sets)
            if cells&(occupied|other) or neigh(cells,shape)&(occupied|other):
                continue
            if crossing:
                cr,cc0=center
                if owner==0 and max(c for r,c in cells)>cc0-6:continue
                if owner==1 and max(r for r,c in cells)>cr-6:continue
            source_sets[owner]=src
            if owner>=active:
                # Inactive endpoints are open strings, optionally with a
                # black tip, never an empty three/five-sided source cup.
                tip=next(iter(src))
                wire={_pick(rng,sorted(wire))}
                source_sets[owner]={tip} if bg!=0 and convention['black_tips'] else set()
            wire_sets[owner]|=wire
            placed=True
            break
        if not placed:
            return None
    loop_cells=set().union(*(q['inner']|q['boundary'] for q in regions))
    sources=set().union(*source_sets)
    # Cups and preseeded crossings are separate route fragments initially.
    for owner in [int(i) for i in rng.permutation(nroute)]:
        own=wire_sets[owner]
        others=set().union(*(s for i,s in enumerate(wire_sets) if i!=owner))
        forbidden=loop_cells|neigh(loop_cells,shape)|sources|neigh(sources,shape)
        forbidden|=others|neigh(others,shape)
        allowed=canvas-forbidden
        allowed|=own
        fragments=components(own)
        if len(fragments)>1:
            tree=set(fragments[0])
            for frag in fragments[1:]:
                path=_path(rng,tree,frag,allowed|frag,shape)
                if path is None:return None
                own.update(path);tree.update(path);tree.update(frag)
            wire_sets[owner]=own
        # Allow the route to leave its cap from any wire cell, but never
        # surround the paint marker with additional string pixels.
        previous_boundary=None
        for index in rng.permutation(len(regions)):
            target=regions[int(index)]
            if target['owner']!=owner:
                continue
            otherloops=loop_cells-(target['inner']|target['boundary'])
            if previous_boundary is not None and spec.get('chain'):
                # Leave the preceding contour through a new port. Its cavity
                # stays blocked, while the exterior of its shell is available.
                otherloops=otherloops-previous_boundary
            blocked=loop_cells|neigh(otherloops,shape)|sources|neigh(sources,shape)
            blocked|=others|neigh(others,shape)
            allowed=canvas-blocked
            goals=neigh(target['boundary'],shape)-blocked
            if not goals:return None
            # Prefer a randomly selected attachment side, but allow all
            # visible ports if the selected side has no free approach.
            side=_ri(rng,0,3)
            axis=side//2
            extreme=(min if side%2==0 else max)(p[axis] for p in goals)
            sidegoals={p for p in goals if p[axis]==extreme}
            starts=own if previous_boundary is None or not spec.get('chain') else previous_boundary
            path=_path(rng,starts,sidegoals,allowed|starts,shape)
            if path is None:
                path=_path(rng,starts,goals,allowed|starts,shape)
            if path is None:return None
            own.update(set(path)-loop_cells)
            previous_boundary=target['boundary']
            # Thin strings must not run along multiple unrelated contours.
            wire_sets[owner]=own
    # A black free end on a supplied route does not replace its cupped paint.
    black_ends=set()
    if bg!=0 and convention['black_tips'] and (spec.get('teach_tip') or rng.random()<.5):
        owner=_ri(rng,0,active-1)
        own=wire_sets[owner]
        allwire=set().union(*wire_sets)
        protected=loop_cells|sources|neigh(loop_cells|sources,shape)
        candidates=sorted(neigh(own,shape)-allwire-protected)
        rng.shuffle(candidates)
        for first in candidates:
            if len(neigh({first},shape)&allwire)!=1:continue
            anchor=next(iter(neigh({first},shape)&own))
            dr,dc=first[0]-anchor[0],first[1]-anchor[1]
            length=_ri(rng,2,5)
            extension={(first[0]+i*dr,first[1]+i*dc) for i in range(length)}
            if not extension<=canvas or extension&protected:continue
            if (neigh(extension,shape)&allwire)!={anchor}:continue
            tip=(first[0]+(length-1)*dr,first[1]+(length-1)*dc)
            own.update(extension-{tip});black_ends.add(tip)
            break
        if spec.get('teach_tip') and not black_ends:return None
    for q in regions:
        for p in q['boundary']:a[p]=q['color']
    for owner in range(nroute):
        for p in wire_sets[owner]:a[p]=wires[owner]
    for owner in range(nroute):
        for p in source_sets[owner]:a[p]=paints[owner] if owner<active else 0
    for p in black_ends:a[p]=0
    b=a.copy()
    for q in regions:
        if q['owner'] is not None and q['owner']<active:
            for p in q['inner']:b[p]=paints[q['owner']]
    # Rotation schedules are independent of paint schedules.
    rotation=spec['rotation']
    a=np.rot90(a,rotation);b=np.rot90(b,rotation)
    if rng.random()<.5:a=np.fliplr(a);b=np.fliplr(b)
    try:
        predicted,info=analyze(a)
    except AssertionError:
        return None
    if not np.array_equal(predicted,b) or np.array_equal(a,b):
        return None
    if len(info['sources'])!=active:
        return None
    if len(info['loops'])!=len(regions):
        return None
    if clipped and not any(any(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for r,c in cell)
                           for cell,edge,col in info['loops']):
        return None
    return {'input':a.copy(),'output':b.copy()}


def _fallback(spec, convention, variant):
    """Guaranteed sparse construction after bounded free-placement attempts."""
    bg=spec.get('background',convention['backgrounds'][variant%len(convention['backgrounds'])])
    paint=spec.get('paint',convention['paint'])
    colors=[c for c in range(1,10) if c not in (bg,paint)]
    wire,other,edge,paint2=colors[:4]
    a=np.full((30,30),bg,dtype=np.int64)
    owned=[]
    def loop(r0,r1,c0,c1,owner):
        cells={(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)}
        border={(r+dr,c+dc) for r,c in cells for dr,dc in D4}-cells
        for p in _inside(border,a.shape):a[p]=edge
        owned.append((_inside(cells,a.shape),owner))
    loop(14,16,25,27,paint)
    a[15,2:24]=wire
    a[15,1]=paint
    a[16,1:3]=wire
    if spec['clipped']:
        loop(-1,1,6,7,paint)
        a[3:16,7]=wire
    if spec['touching'] or spec['inactive'] or spec['detached']:
        if spec['touching']:
            loop(10,11,27,28,None)
            tipr,tipc=5,28
        else:
            loop(7,8,26,27,None)
            tipr,tipc=2,26
        if spec['inactive']:
            a[tipr+1:tipr+4,tipc]=other
            if bg!=0 and convention['black_tips']:a[tipr,tipc]=0
    if spec['crossing']:
        loop(25,27,14,16,paint2)
        a[2:24,15]=other
        a[1,15]=paint2
        a[1:3,16]=other
    if spec.get('teach_tip'):
        a[16:19,10]=wire
        a[19,10]=0
    # Three/five-neighbor cups carry the same source convention.
    if spec['style']==5:
        a[14,1:3]=wire
        if spec['crossing']:a[1:3,14]=other
    b=a.copy()
    for cells,color in owned:
        if color is not None:
            for p in cells:b[p]=color
    a=np.rot90(a,spec['rotation']);b=np.rot90(b,spec['rotation'])
    if variant%2:a=np.fliplr(a);b=np.fliplr(b)
    predicted,_=analyze(a)
    assert np.array_equal(predicted,b)
    return dict(input=a.copy(),output=b.copy())


def _generate_pair(rng, regime, convention, index):
    assert regime in ('source_train','target_test')
    lesson=regime=='source_train' and index<2
    crossing=convention['crossing'] and (index==1 if lesson else rng.random()<.28)
    touching=convention['touching'] and (index==0 if lesson else rng.random()<.35)
    clipped=convention['clipped'] and (index==0 if lesson else rng.random()<.35)
    active=2 if crossing else _pick(rng,(1,1,1,2,2,3))
    inactive=_pick(rng,(0,0,1,1,2))
    inactive=min(inactive,4-active)
    if lesson and index==0:
        active=1
        inactive=1 if not touching else _pick(rng,(0,1))
    if lesson and index==1 and not crossing:
        active=_pick(rng,(1,2))
    size=_pick(rng,('small','medium','medium','large'))
    if convention['shift']=='harder' and regime=='target_test':
        size='large';active=max(active,2)
    if convention['shift']=='smaller' and regime=='target_test':
        size='small';active=1;inactive=0
    if size=='small' and not crossing:
        active=1;inactive=0
    if lesson and (touching or clipped or inactive):size='medium'
    if crossing:active=2;inactive=0;size='large'
    if convention['source_shift'] and regime=='source_train':
        active=1;size='medium'
    if convention['homogeneous']:
        active=1;inactive=0;size='small'
    nroute=active+inactive
    counts=[_ri(rng,1,2 if size=='small' or crossing else 5) for _ in range(nroute)]
    while sum(counts)>9:
        i=int(np.argmax(counts));counts[i]-=1
    if active>1 and not crossing and not lesson and rng.random()<.18:
        counts[active-1]=0
    detached=0 if size=='small' else _pick(rng,(0,0,0,1,2))
    if lesson and index==0 and not touching and inactive==0:detached=1
    orientation=convention['rotation'] if convention['homogeneous'] else _ri(rng,0,3)
    spec=dict(crossing=crossing,touching=touching,clipped=clipped,
              active=active,inactive=inactive,size=size,counts=counts,
              detached=detached,style=convention['styles'][index%len(convention['styles'])],
              rotation=orientation,chain=rng.random()<.4)
    if convention['homogeneous']:
        spec['axis']=True;spec['narrow']=True
        spec['counts']=[_ri(rng,1,3)]
    if lesson:
        spec['paint']=convention['teaching_paints'][index]
    if regime=='source_train' and convention['shift']=='harder' and not convention['black_tips']:
        spec['background']=0
    if lesson and index==0 and convention['black_tips']:
        spec['background']=convention['backgrounds'][1]
        spec['teach_tip']=True
    if size=='small' and not (touching or clipped or crossing or convention['homogeneous']):
        if rng.random()<.28:
            spec['compact']=True;spec['counts']=[1]
        elif rng.random()<.4:spec['narrow']=True
    if size=='large' and rng.random()<.25:spec['full']=True
    local=dict(convention)
    if lesson:local['paint']=spec['paint']
    for attempt in range(90):
        if attempt==55:
            # Preserve the mechanism and all routes, simplify only packing.
            spec=dict(spec,counts=[min(2,c) for c in spec['counts']],detached=0)
            spec['size']='large'
            spec.pop('compact',None);spec.pop('narrow',None)
        candidate=_scene(rng,spec,local,attempt)
        if candidate is None:continue
        info=analyze(candidate['input'])[1]
        if lesson and spec['paint'] not in info['paints'].values():continue
        if crossing and not _features(candidate)['crossings']:continue
        if touching and not _features(candidate)['touching_unsupplied']:continue
        return candidate
    return _fallback(spec,local,_ri(rng,0,1000000))


def _key(grid):
    return (grid.shape,tuple(int(v) for v in grid.flat))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    nt,ne=_resolve_counts(seed,num_train,num_test)
    rng=np.random.default_rng(seed)
    colors=[int(v) for v in rng.permutation(np.arange(1,10))]
    styles=_pick(rng,([3],[5],[3,5]))
    convention=dict(paint=colors[0],teaching_paints=colors[:2],
                    backgrounds=[0,colors[2]] if rng.random()<.75 else [0,colors[2],colors[3]],
                    styles=styles,crossing=rng.random()<.7,touching=rng.random()<.8,
                    clipped=rng.random()<.75,shift=_pick(rng,('harder','comparable','smaller')),
                    homogeneous=rng.random()<.15,rotation=_ri(rng,0,3),black_tips=rng.random()<.65)
    if convention['homogeneous']:convention['crossing']=False
    convention['source_shift']=convention['shift']=='harder' and not convention['black_tips']
    if convention['source_shift']:convention['crossing']=False
    puzzle={'train':[],'test':[]}
    seen=set()
    for split,regime,count in (('train','source_train',nt),('test','target_test',ne)):
        for index in range(count):
            for attempt in range(12):
                pair=_generate_pair(rng,regime,convention,index)
                key=_key(pair['input'])
                if key not in seen and key not in _SOURCE_KEYS:
                    break
            else:
                raise AssertionError('Could not construct a distinct input')
            seen.add(key);puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a=pair['input']
    _,info=analyze(a)
    loops=info['loops'];n=len(loops);wires=info['wires'];graph=info['graph']
    groups=[];left=set(range(len(graph)))
    while left:
        root=min(left);left.remove(root);group={root};todo=[root]
        for i in todo:
            for j in graph[i]&left:
                left.remove(j);group.add(j);todo.append(j)
        groups.append(group)
    active_counts=[];inactive_counts=[];boxes=[];orientations=[];supplied_nodes=set()
    source_without_contours=0;inactive_routed=0;color_orientations=[];route_cells=[]
    for group in groups:
        indices=group&set(range(n))
        # Black singleton end markers are not separate string routes.
        wire_indices={i-n for i in group if i>=n and wires[i-n][1]!=0}
        if not indices and not wire_indices:continue
        count=len(indices)
        if indices&set(info['paints']):
            active_counts.append(count);supplied_nodes.update(group)
        else:
            inactive_counts.append(count)
            inactive_routed+=bool(count and wire_indices)
        if not indices:
            for p,col,wire in info['sources']:
                if any(wires[i][1]==wire and wires[i][0]&neigh({p},a.shape) for i in wire_indices):
                    source_without_contours+=1
        cells=set().union(*(wires[i][0] for i in wire_indices)) if wire_indices else set()
        if cells:
            r0=min(r for r,c in cells);r1=max(r for r,c in cells)
            c0=min(c for r,c in cells);c1=max(c for r,c in cells)
            boxes.append((r0,r1,c0,c1))
            orientations.append('horizontal' if c1-c0>1.5*(r1-r0) else
                                'vertical' if r1-r0>1.5*(c1-c0) else 'mixed')
            route_cells.append((cells,bool(indices&set(info['paints']))))
            for p,color,wire in info['sources']:
                if any(wires[i][1]==wire and wires[i][0]&neigh({p},a.shape) for i in wire_indices):
                    color_orientations.append((int(color),orientations[-1]))
    touches=0;diag=0;orth=0;shape_keys=[];asymmetric=0;concave=0;indented=0
    for i,(cells,edge,color) in enumerate(loops):
        r0=min(r for r,c in cells);c0=min(c for r,c in cells)
        normalized={(r-r0,c-c0) for r,c in cells}
        hh=max(r for r,c in normalized)+1;ww=max(c for r,c in normalized)+1
        shape_keys.append((hh,ww,tuple(sorted(normalized))))
        asymmetric+=normalized!={(hh-1-r,c) for r,c in normalized} or normalized!={(r,ww-1-c) for r,c in normalized}
        concave+=any((r,c) not in normalized for r in range(hh) for c in range(ww))
        indented+=any(len(cs)<max(cs)-min(cs)+1 for cs in
                      ([c for r,c in normalized if r==row] for row in range(hh))) or any(
                      len(rs)<max(rs)-min(rs)+1 for rs in
                      ([r for r,c in normalized if c==col] for col in range(ww)))
        for j in graph[i]:
            if j<n:continue
            points=wires[j-n][0]&neigh(edge,a.shape)
            if points&neigh(edge,a.shape,D4):orth+=1
            if points-neigh(edge,a.shape,D4):diag+=1
        for j in range(i):
            if edge&neigh(loops[j][1],a.shape) and ((i in info['paints']) != (j in info['paints'])):
                touches+=1
    overlaps=sum(max(b[0],c[0])<=min(b[1],c[1]) and max(b[2],c[2])<=min(b[3],c[3])
                 for i,b in enumerate(boxes) for c in boxes[:i])
    crossings=sum(j>=n for i in range(n,len(graph)) for j in graph[i] if j>i)
    source_positions=[(int(p[0]),int(p[1])) for p,color,wire in info['sources']]
    source_degrees=[sum(a[q]==wire for q in neigh({p},a.shape)) for p,col,wire in info['sources']]
    black=[]
    if info['bg']!=0:
        black=[p for p in zip(*np.where(a==0)) if 1<=len({q for q in neigh({p},a.shape) if a[q]!=info['bg']})<=2]
    active_black=sum(any(i in supplied_nodes and cells&neigh({p},a.shape)
                         for i,(cells,col) in enumerate(wires,n) if col!=0) for p in black)
    cap_neighbors=set().union(*(neigh({p},a.shape)|{p} for p,col,wire in info['sources']))
    steps=set();diagonal_steps=0
    for cells,col in wires:
        clean=cells-cap_neighbors
        for r,c in clean:
            if (r,c+1) in clean:steps.add('horizontal')
            if (r+1,c) in clean:steps.add('vertical')
            for dc in (-1,1):
                if (r+1,c+dc) in clean and (r+1,c) not in clean and (r,c+dc) not in clean:
                    steps.add('diagonal');diagonal_steps+=1
    placement=[]
    for active_cells,is_active in route_cells:
        if not is_active:continue
        ar=sum(r for r,c in active_cells)/len(active_cells)
        ac=sum(c for r,c in active_cells)/len(active_cells)
        for inactive_cells,other_active in route_cells:
            if other_active:continue
            ir=sum(r for r,c in inactive_cells)/len(inactive_cells)
            ic=sum(c for r,c in inactive_cells)/len(inactive_cells)
            placement.append(('above' if ar<ir else 'below','left' if ac<ic else 'right'))
    return dict(shape=list(a.shape),background=info['bg'],colors=len(np.unique(a)),
                loops=n,productive_routes=len(active_counts),inactive_routes=sum(c>0 for c in inactive_counts),
                inactive_routed=inactive_routed,placement=placement,
                active_counts=active_counts,inactive_counts=inactive_counts,
                source_count=len(info['sources']),paints=sorted({int(c) for c in info['paints'].values()}),
                source_without_contours=source_without_contours,
                source_colors=[int(c) for p,c,w in info['sources']],
                wire_colors=sorted({int(w) for p,c,w in info['sources']}),
                all_wire_colors=sorted({int(col) for cells,col in wires if len(cells)>1}),
                contour_colors=sorted({int(col) for cells,edge,col in loops}),
                source_positions=source_positions,source_degrees=source_degrees,
                source_on_border=sum(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for r,c in source_positions),
                source_at_corner=sum(r in (0,a.shape[0]-1) and c in (0,a.shape[1]-1) for r,c in source_positions),
                touching_unsupplied=touches,crossings=crossings,bbox_overlaps=overlaps,
                clipped=sum(any(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for r,c in cell) for cell,edge,col in loops),
                detached=sum(not graph[i] for i in range(n)),
                diagonal_attachments=diag,orthogonal_attachments=orth,
                branch_nodes=sum(len(edges)>=3 for edges in graph),
                serial_contours=sum(len(graph[i])>=2 for i in range(n)),
                orientations=orientations,color_orientations=color_orientations,contour_shapes=shape_keys,
                asymmetric=asymmetric,nonrectangular=concave,
                concave=indented,step_axes=sorted(steps),diagonal_steps=diagonal_steps,
                cavity_areas=[len(cell) for cell,edge,col in loops],
                path_cells=sum(len(cells) for cells,col in wires if col!=0),
                black_tips=len(black),active_black_tips=active_black)


def solve(train_pairs, test_input):
    """Predict using observable geometry, with consistency checked on training."""
    observed=set()
    for pair in train_pairs:
        predicted,info=analyze(np.array(pair['input'],copy=True))
        assert np.array_equal(predicted,pair['output'])
        observed.update(info['paints'].values())
    assert len(observed)>=2, 'Demonstrate copying with two paint values'
    return analyze(np.array(test_input,copy=True))[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2<=len(puzzle['train'])<=10 and 1<=len(puzzle['test'])<=4
    assert len(puzzle['train'])>len(puzzle['test'])
    seen=set();features={'train':[],'test':[]}
    training_colors=set();training_degrees=set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            for a in pair.values():
                assert isinstance(a,np.ndarray) and a.ndim==2
                assert np.issubdtype(a.dtype,np.integer)
                assert all(1<=v<=30 for v in a.shape) and np.all((a>=0)&(a<=9))
            a,b=pair['input'],pair['output']
            assert a.shape==b.shape and not np.array_equal(a,b)
            key=_key(a)
            assert key not in seen and key not in _SOURCE_KEYS
            seen.add(key)
            prediction,info=analyze(a)
            assert np.array_equal(prediction,b), 'Route/source/interior prediction mismatch'
            assert np.all(a[a!=b]==info['bg'])
            f=_features(pair);features[split].append(f)
            assert all(c!=0 and c!=info['bg'] for c in f['source_colors'])
            assert set(f['source_degrees'])<={3,5}
            if split=='train':
                training_colors.update(f['paints']);training_degrees.update(f['source_degrees'])
    assert len(training_colors)>=2
    assert any(f['loops']>f['productive_routes'] or f['inactive_routes'] or f['detached']
               for f in features['train'])
    for f in features['test']:
        assert set(f['source_degrees'])<=training_degrees
        for mechanism in ('crossings','touching_unsupplied','clipped','black_tips'):
            if f[mechanism]:assert any(t[mechanism] for t in features['train']), mechanism
    for pair in puzzle['test']:
        assert np.array_equal(solve(puzzle['train'],pair['input'].copy()),pair['output'])


# Immutable source regressions; predictions never receive the expected labels.
_SOURCE_ROWS = (('000000000000000000005500\n022000000000000000050050\n200200000220000000050050\n200200002002000000015510\n022100002002000002100010\n000011111221000020220010\n000000000001000200002010\n000000111110000020022010\n000001000000000012200010\n000110000000000010000010\n000000033000000010000010\n000000300300022100000010\n000000300300200200000010\n000100133111200200000010\n000111000002000200000010\n000000000002002000000010\n000000000000220000000011\n000000000000000000000017', '000000000000000000005500\n022000000000000000057750\n200200000220000000057750\n200200002002000000015510\n022100002002000002100010\n000011111221000027220010\n000000000001000277772010\n000000111110000027722010\n000001000000000012200010\n000110000000000010000010\n000000033000000010000010\n000000377300022100000010\n000000377300277200000010\n000100133111277200000010\n000111000002777200000010\n000000000002772000000010\n000000000000220000000011\n000000000000000000000017'), ('00000000000000000000\n00000002200000002200\n00022020020000020020\n00200220020022020020\n00200212200200202210\n00022111000200200100\n00010000110022001000\n00010000011000010000\n00110000001100100000\n00100000000111000220\n11000000000000002002\n41000000000000002002', '00000000000000000000\n00000002200000002200\n00022024420000024420\n00244224420022024420\n00244212200200202210\n00022111000200200100\n00010000110022001000\n00010000011000010000\n00110000001100100000\n00100000000111000220\n11000000000000002002\n41000000000000002002'), ('00000000000000000022\n00220000000220000200\n02002000002002000200\n02002000002002000322\n01221111111221000300\n01000000000001000300\n00100000000010000030\n11000002211100002230\n01000020020000020020\n11000020020000020020\n00000012200000032200\n00000100000000030000\n00001000000332230000\n00221000003020020000\n02002000333020020000\n02002000343002200000', '00000000000000000022\n00220000000220000244\n02002000002002000244\n02002000002002000322\n01221111111221000300\n01000000000001000300\n00100000000010000030\n11000002211100002230\n01000020020000024420\n11000020020000024420\n00000012200000032200\n00000100000000030000\n00001000000332230000\n00221000003024420000\n02002000333024420000\n02002000343002200000'), ('00220000\n02002000\n02002000\n00220100\n00000010\n00000100\n00000011\n00000014', '00220000\n02442000\n02442000\n00220100\n00000010\n00000100\n00000011\n00000014'), ('615555555555555555555555555538\n111555555555555555555555555533\n555199111111199999995555553355\n555955955555955555559555553555\n555955955555955555559555533555\n555599555555599999991111535555\n555555555555555555555555135555\n555555555555555555555555531555\n555555555555555555555555355155\n555555555555555555555599355155\n555599999995555593355955955155\n555955555559555999333955955155\n555955555559555959553399555115\n553399999995555959555555555515\n503555555555555959555555555515\n555555555555555959555555555115\n555555555555555959555555555155\n555555555555555959555555555155\n550555555555555959555555551155\n553555555555555959555555551555\n553555555555533395555555511555\n553355555533335555559991115555\n555399333335555555595591155555\n555955955555555555955595555555\n555955955555555559555955555555\n555599555555555595559555555555\n555555555555555955595555555555\n555555555555555999955555555555\n555555555555555555555555555555\n555555555555555555555555555555', '615555555555555555555555555538\n111555555555555555555555555533\n555199111111199999995555553355\n555966955555966666669555553555\n555966955555966666669555533555\n555599555555599999991111535555\n555555555555555555555555135555\n555555555555555555555555531555\n555555555555555555555555355155\n555555555555555555555599355155\n555599999995555593355988955155\n555955555559555999333988955155\n555955555559555989553399555115\n553399999995555989555555555515\n503555555555555989555555555515\n555555555555555989555555555115\n555555555555555989555555555155\n555555555555555989555555555155\n550555555555555989555555551155\n553555555555555989555555551555\n553555555555533395555555511555\n553355555533335555559991115555\n555399333335555555596691155555\n555988955555555555966695555555\n555988955555555559666955555555\n555599555555555596669555555555\n555555555555555966695555555555\n555555555555555999955555555555\n555555555555555555555555555555\n555555555555555555555555555555'), ('00000000111111100000000880\n00000001100000111000008008\n00002211000000001188008008\n00020020000000000800800880\n00020020000000000800800000\n00012210000000000088100000\n00110011666000000000011100\n00100006000600088000000010\n00100006000600800800055510\n00100006000600800800500050\n00100000666009988900500050\n00100000000090000990055500\n00100000889900000090000000\n00100008008000000090000000\n00100008008000000090000000\n00100000880000000090008800\n00100000000000008890080080\n01100000000000080080080080\n01000000000000080080098800\n01000009999000098899900000\n11009990009999990000000000\n41009390000000000000000000', '00000000111111100000000880\n00000001100000111000008008\n00002211000000001188008008\n00024420000000000844800880\n00024420000000000844800000\n00012210000000000088100000\n00110011666000000000011100\n00100006444600088000000010\n00100006444600833800055510\n00100006444600833800544450\n00100000666009988900544450\n00100000000090000990055500\n00100000889900000090000000\n00100008338000000090000000\n00100008338000000090000000\n00100000880000000090008800\n00100000000000008890083380\n01100000000000083380083380\n01000000000000083380098800\n01000009999000098899900000\n11009990009999990000000000\n41009390000000000000000000'))


def _read_rows(rows):
    return np.array([[int(c) for c in row] for row in rows.splitlines()],dtype=np.int64)


_SOURCE_KEYS = {_key(_read_rows(rows)) for rows, _ in _SOURCE_ROWS}


def self_test() -> None:
    # The source regression includes incidental red/red contact, the interrupted
    # blue/green crossing, both source cup sizes, black tips, and clipped holes.
    source_pairs=[]
    for inp,out in _SOURCE_ROWS:
        a,b=_read_rows(inp),_read_rows(out)
        assert np.array_equal(analyze(a)[0],b)
        source_pairs.append(dict(input=a,output=b))
    for pair in source_pairs[4:]:
        assert np.array_equal(solve(source_pairs[:4],pair['input'].copy()),pair['output'])
    measurements=[];train_counts=set();test_counts=set();structures=set();puzzles=[]
    homogeneous={'train':0,'test':0};mixed={'train':0,'test':0}
    within_color_orientation=0;shared_backgrounds=0;smaller=0;larger=0
    for seed in range(42):
        puzzle=generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']));test_counts.add(len(puzzle['test']))
        fs={s:[_features(p) for p in puzzle[s]] for s in puzzle}
        allfs=fs['train']+fs['test'];measurements.extend(allfs)
        for s in fs:
            axes={v for f in fs[s] for v in f['orientations']}
            homogeneous[s]+=len(axes)==1
            mixed[s]+=len(axes)>1
        shared_backgrounds+=bool({f['background'] for f in fs['train']}&{f['background'] for f in fs['test']})
        train_area=sum(np.prod(f['shape']) for f in fs['train'])/len(fs['train'])
        test_area=sum(np.prod(f['shape']) for f in fs['test'])/len(fs['test'])
        smaller+=test_area<train_area;larger+=test_area>train_area
        color_axes={}
        for f in allfs:
            for color,axis in f['color_orientations']:
                color_axes.setdefault(color,set()).add(axis)
        within_color_orientation+=any(len(v)>1 for v in color_axes.values())
        for pair in puzzle['train']+puzzle['test']:
            a=pair['input'];bg=int(np.bincount(a.flat,minlength=10).argmax())
            rr,cc=np.where(a!=bg);a=a[rr.min():rr.max()+1,cc.min():cc.max()+1]
            colors={bg:0};normalized=[]
            for value in a.flat:
                value=int(value)
                if value not in colors:colors[value]=len(colors)
                normalized.append(colors[value])
            structures.add((a.shape,tuple(normalized)))
    assert len(train_counts)>=4 and len(test_counts)>=3
    assert len(structures)>=250 and len({tuple(f['shape']) for f in measurements})>=60
    assert {tuple(f['shape']) for f in measurements}>={(8,8),(30,30)}
    assert min(min(f['shape']) for f in measurements)<=8
    assert min(f['loops'] for f in measurements)==1 and max(f['loops'] for f in measurements)>=9
    assert {1,2,3}<={f['productive_routes'] for f in measurements}
    assert max(len(f['paints']) for f in measurements)>=3
    assert max(c for f in measurements for c in f['active_counts'])>=5
    assert max(c for f in measurements for c in f['inactive_counts'])>=4
    assert max(f['source_without_contours'] for f in measurements)>=1
    assert {3,5}<={d for f in measurements for d in f['source_degrees']}
    for key in ('crossings','touching_unsupplied','clipped','bbox_overlaps','black_tips',
                'active_black_tips','source_on_border','source_at_corner','detached',
                'serial_contours','branch_nodes','asymmetric','concave','diagonal_steps',
                'diagonal_attachments','orthogonal_attachments'):
        assert sum(f[key]>0 for f in measurements)>=3,key
    assert any(f['crossings'] and f['clipped'] for f in measurements)
    assert any(f['crossings'] and f['touching_unsupplied'] for f in measurements)
    assert any(f['source_count']>len(set(f['source_colors'])) for f in measurements)
    assert len({shape for f in measurements for shape in f['contour_shapes']})>=70
    assert any(min(s[:2])==1 and max(s[:2])>=8 for f in measurements for s in f['contour_shapes'])
    assert len({f['background'] for f in measurements})>=8
    assert len({f['colors'] for f in measurements})>=5
    assert all(homogeneous[s]>0 and mixed[s]>0 for s in homogeneous)
    assert within_color_orientation>=20 and shared_backgrounds>=20
    assert smaller>=8 and larger>=8
    # Explicit and partial boundary requests, including the smallest lessons.
    for seed in range(8):
        for nt,ne in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
            puzzle=generate_puzzle(1000+seed,nt,ne)
            assert nt is None or len(puzzle['train'])==nt
            assert ne is None or len(puzzle['test'])==ne
    for nt,ne in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1),(2,1.5)):
        try:generate_puzzle(0,nt,ne)
        except ValueError:continue
        raise AssertionError('Illegal count request accepted')
    def puzzle_key(p):
        return tuple(tuple((_key(q['input']),_key(q['output'])) for q in p[s]) for s in ('train','test'))
    assert puzzle_key(puzzles[0])==puzzle_key(generate_puzzle(0))
    assert puzzle_key(puzzles[0])!=puzzle_key(puzzles[1])
    # Deliberately incorrect answers must fail input-derived validation.
    bad={s:[{role:a.copy() for role,a in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    pair=bad['test'][0];r,c=np.argwhere(pair['input']!=pair['output'])[0]
    pair['output'][r,c]=pair['input'][r,c]
    try:validate_generated_puzzle(bad)
    except AssertionError:pass
    else:raise AssertionError('Partial cavity fill accepted')
