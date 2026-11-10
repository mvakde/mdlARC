"""Matched siphons: rail equality selects communicating liquid vessels.

The rail centerlines are abstract, already primed connections with zero
storage capacity. Only enclosed vessel interiors hold liquid. Nothing in
the solver uses sampled geometry, seeds, training labels, or saved puzzles.
"""
from collections import deque
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

def inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]

def add(p, d):
    return p[0]+d[0], p[1]+d[1]

def components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    for rr, cc in np.argwhere(mask):
        p = (int(rr), int(cc))
        if seen[p]:
            continue
        stack, cells = [p], []
        seen[p] = True
        while stack:
            q = stack.pop()
            cells.append(q)
            for d in DIRS:
                z = add(q, d)
                if inside(mask, z) and mask[z] and not seen[z]:
                    seen[z] = True
                    stack.append(z)
        result.append(cells)
    return result

def holes(g, wall):
    return [cells for cells in components(g != wall)
            if not any(r in (0,g.shape[0]-1) or c in (0,g.shape[1]-1) for r,c in cells)]

def roles(g):
    bg = int(g[0,0])
    assert all(np.all(edge == bg) for edge in (g[0],g[-1],g[:,0],g[:,-1]))
    candidates = []
    for wall in map(int, np.unique(g)):
        if wall == bg:
            continue
        rooms = holes(g, wall)
        if len(rooms) < 3:
            continue
        colors = {int(g[p]) for room in rooms for p in room} - {bg}
        if len(colors) == 1:
            water = next(iter(colors))
            interiors = {q for room in rooms for q in room}
            if all(p in interiors for p in map(tuple,np.argwhere(g == water))):
                candidates.append((bg,water,wall,rooms))
    assert len(candidates) == 1, ('roles',len(candidates))
    return candidates[0]

def paired(g, p, d, guides):
    n = (-d[1],d[0])
    a,b = add(p,n),add(p,(-n[0],-n[1]))
    if inside(g,a) and inside(g,b) and int(g[a]) in guides and int(g[b]) in guides:
        return int(g[a]) == int(g[b])
    return None

def analyse(g):
    bg,water,wall,rooms = roles(g)
    labels = np.full(g.shape,-1,int)
    for i,room in enumerate(rooms):
        for p in room:
            labels[p] = i
    guides = set(map(int,np.unique(g))) - {bg,water,wall}
    ports = {}
    for i,room in enumerate(rooms):
        for q in room:
            for d in DIRS:
                z,p = add(q,d),add(add(q,d),d)
                if (inside(g,p) and g[z] == wall and g[p] == bg and labels[p] == -1):
                    match = paired(g,p,d,guides)
                    if match is not None:
                        ports[p,d] = (i,match)
    traces = {}
    for (start,d0),(i,match) in ports.items():
        p,d = start,d0
        path,seen = [p],set()
        for _ in range(g.size*4):
            assert (p,d) not in seen, 'cycle'
            seen.add((p,d))
            z = add(p,d)
            assert inside(g,z), 'boundary pipe'
            if g[z] == bg and labels[z] == -1:
                p=z
                path.append(p)
                continue
            beyond=add(z,d)
            if g[z] == wall and inside(g,beyond) and labels[beyond] >= 0:
                end = (p,(-d[0],-d[1]))
                assert end in ports, 'unmarked endpoint'
                assert ports[end][0] != i, 'self pipe'
                traces[start,d0]=(end,path)
                break
            choices=[]
            for nd in ((-d[1],d[0]),(d[1],-d[0])):
                q=add(p,nd)
                if inside(g,q) and g[q]==bg and labels[q]==-1 and paired(g,q,nd,guides) is True:
                    choices.append(nd)
            assert len(choices)==1, ('bad turn',len(choices))
            d=choices[0]
        else:
            raise AssertionError('trace limit')
    links=[]
    for start,(end,path) in traces.items():
        assert end in traces and traces[end][0] == start and traces[end][1] == path[::-1], 'reciprocity'
        if start < end:
            i,a=ports[start]
            j,b=ports[end]
            links.append((i,j,a and b,path))
    occupied=[p for i,j,opened,path in links for p in path]
    assert len(occupied)==len(set(occupied)), 'conduits intersect'
    assert len(ports)==2*len(links), 'unpaired ports'
    return bg,water,wall,rooms,links

def groups(n, links, ignore=False):
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            i=parent[i]
        return i
    for i,j,opened,path in links:
        if opened or ignore:
            parent[root(i)]=root(j)
    return [tuple(i for i in range(n) if root(i)==r) for r in sorted({root(i) for i in range(n)})]

def allocate(g,analysis,ignore=False,strict=True,local=False):
    bg,water,wall,rooms,links=analysis
    out=g.copy()
    out[out==water]=bg
    for group in ([(i,) for i in range(len(rooms))] if local else groups(len(rooms),links,ignore)):
        cells=[p for i in group for p in rooms[i]]
        volume=sum(g[p]==water for p in cells)
        ordered=sorted(cells,key=lambda p:(-p[0],p[1]))
        if strict and 0 < volume < len(cells):
            assert ordered[volume-1][0] != ordered[volume][0], 'partial level'
        for p in ordered[:volume]:
            out[p]=water
    return out

def answer(g,ignore=False,strict=True,local=False):
    return allocate(g,analyse(g),ignore,strict,local)

def no_lateral_equalization(g,analysis=None):
    """Ablation: keep B's graph and volume, but forbid A's lateral spreading."""
    bg,water,wall,rooms,links=analyse(g) if analysis is None else analysis
    out=g.copy()
    out[out==water]=bg
    for group in groups(len(rooms),links):
        cells=[p for i in group for p in rooms[i]]
        for col in {p[1] for p in cells}:
            column=[p for p in cells if p[1]==col]
            volume=sum(g[p]==water for p in column)
            for p in sorted(column,reverse=True)[:volume]:
                out[p]=water
    return out

def matching_matters(g,out,analysis=None):
    """Ignore all port inequalities: require an energy change, not a tie."""
    analysis=analyse(g) if analysis is None else analysis
    water=analysis[1]
    wrong=allocate(g,analysis,ignore=True,strict=False)
    return np.argwhere(wrong==water)[:,0].sum()>np.argwhere(out==water)[:,0].sum()

def transfers_between_vessels(g,out,rooms,water):
    return any(sum(g[p]==water for p in room)!=sum(out[p]==water for p in room) for room in rooms)

def dilate(mask):
    z=mask.copy()
    z[1:]|=mask[:-1]
    z[:-1]|=mask[1:]
    z[:,1:]|=mask[:,:-1]
    z[:,:-1]|=mask[:,1:]
    return z

def carve_tank(rng,g,r,c,th,tw,bg,wall):
    local=np.zeros((th,tw),bool)
    style=int(rng.integers(4))
    for x in range(1,tw-1):
        roof=1+(int(rng.integers(2)) if style==3 and th>=6 else 0)
        drop=(int(rng.integers(min(3,th-4))) if style in (1,3)
              else (1 if style==2 and x==tw//2 else 0))
        floor=th-1-drop
        local[roof:floor,x]=True
    g[r:r+th,c:c+tw]=wall
    g[r:r+th,c:c+tw][local]=bg
    return [(int(y+r),int(x+c)) for y,x in np.argwhere(local)]

def tank_scene(rng,h,w,n,bg,wall):
    g=np.full((h,w),bg,np.int8)
    busy=np.zeros((h,w),bool)
    rooms=[]
    for i in range(n):
        for _ in range(120):
            th=int(rng.integers(5,8))
            tw=int(rng.integers(4,8))
            r=int(rng.integers(2,h-th-1))
            c=int(rng.integers(2,w-tw-1))
            if busy[max(0,r-2):r+th+2,max(0,c-2):c+tw+2].any():
                continue
            room=carve_tank(rng,g,r,c,th,tw,bg,wall)
            busy[r:r+th,c:c+tw]=True
            rooms.append(room)
            break
        else:
            return None
    return g,rooms,busy

def available_ports(g,room,busy,bg,wall):
    result=[]
    for q in room:
        for d in DIRS:
            z,p=add(q,d),add(add(q,d),d)
            n=(-d[1],d[0])
            if (inside(g,p) and 1<p[0]<g.shape[0]-2 and 1<p[1]<g.shape[1]-2 and
                g[z]==wall and not busy[p] and g[p]==bg and
                all(inside(g,a) and g[a]==bg and not busy[a] for a in (add(p,n),add(p,(-n[0],-n[1]))))):
                result.append((p,d))
    return result

def route_bfs(free,start,target,minlen=6):
    p,d=start
    q,qd=target
    state=(p,d,0,0)
    queue=deque([state])
    prev={state:None}
    final=None
    while queue:
        s=queue.popleft()
        p,d,run,turns=s
        if p==q and d==(-qd[0],-qd[1]) and run>=2:
            final=s
            break
        dirs=[d]
        if run>=2 and turns<4:
            dirs.extend(((-d[1],d[0]),(d[1],-d[0])))
        for nd in dirs:
            z=add(p,nd)
            if inside(free,z) and free[z] and z!=start[0]:
                ns=(z,nd,min(2,run+1) if nd==d else 1,turns+(nd!=d))
                if ns not in prev:
                    prev[ns]=s
                    queue.append(ns)
    if final is None:
        return None
    path=[]
    while final is not None:
        path.append(final[0])
        final=prev[final]
    path=path[::-1]
    cells=set(path)
    if len(path)<minlen or len(cells)!=len(path):
        return None
    for i,p in enumerate(path):
        degree=sum(add(p,d) in cells for d in DIRS)
        if degree != (1 if i in (0,len(path)-1) else 2):
            return None
    return path

def lay_link(rng,g,rooms,busy,reserved,i,j,bg,wall,palette,opened):
    ports_a=available_ports(g,rooms[i],busy,bg,wall)
    ports_b=available_ports(g,rooms[j],busy,bg,wall)
    if not ports_a or not ports_b:
        return None
    pairs=[(a,b) for a in ports_a for b in ports_b]
    rng.shuffle(pairs)
    pairs.sort(key=lambda ab:abs(ab[0][0][0]-ab[1][0][0])+abs(ab[0][0][1]-ab[1][0][1]))
    free=(~dilate(busy)) & (g==bg) & ~dilate(dilate(reserved))
    free[0]=free[-1]=False
    free[:,0]=free[:,-1]=False
    near_reserved=dilate(reserved)
    for a,b in pairs[:24]:
        if near_reserved[a[0]] or near_reserved[b[0]]:
            continue
        f=free.copy()
        f[a[0]]=f[b[0]]=True
        path=route_bfs(f,a,b)
        if path is None:
            continue
        cells=set(path)
        walls={add(p,d) for p in path for d in DIRS if inside(g,add(p,d)) and add(p,d) not in cells and g[add(p,d)]!=wall}
        if any(g[z]!=bg or busy[z] for z in walls):
            continue
        out=g.copy()
        color=int(rng.choice(palette))
        for z in walls:
            out[z]=color
        if not opened:
            p,d=(a if rng.integers(2) else b)
            nd=(-d[1],d[0])
            out[add(p,nd)]=int(rng.choice([v for v in palette if v!=color]))
        # Gaps are directional evidence, not pipe breaks.
        dirs=[(y[0]-x[0],y[1]-x[1]) for x,y in zip(path,path[1:])]
        eligible=[k for k in range(3,len(path)-3) if len(set(dirs[k-2:k+2]))==1]
        if eligible:
            k=int(rng.choice(eligible))
            d=dirs[k]
            nd=(-d[1],d[0])
            for z in (add(path[k],nd),add(path[k],(-nd[0],-nd[1]))):
                out[z]=bg
        new_reserved=reserved.copy()
        for z in path:
            new_reserved[z]=True
        return out,new_reserved,path
    return None

def make_pair(rng,n,partitions,colors,mode='general'):
    bg,water,wall=colors[:3]
    palette=colors[3:]
    for attempt in range(200):
        low={3:19,4:23,5:26,6:29}[n]
        h=int(rng.integers(low,31))
        w=int(rng.integers(low,31))
        scene=tank_scene(rng,h,w,n,bg,wall)
        if scene is None:
            continue
        g,rooms,busy=scene
        reserved=np.zeros(g.shape,bool)
        order=list(map(int,rng.permutation(n)))
        blocks=[]
        off=0
        for size in partitions:
            blocks.append(order[off:off+size])
            off+=size
        successful=True
        paths=[]
        for block in blocks:
            connected={block[0]}
            while len(connected)<len(block):
                candidates=[(i,j) for i in connected for j in block if j not in connected]
                rng.shuffle(candidates)
                for i,j in candidates:
                    laid=lay_link(rng,g,rooms,busy,reserved,i,j,bg,wall,palette,True)
                    if laid is not None:
                        g,reserved,path=laid
                        paths.append(path)
                        connected.add(j)
                        break
                else:
                    successful=False
                    break
            if not successful:
                break
        if not successful:
            continue
        candidates=[(i,j) for i in blocks[0] for j in blocks[1]]
        rng.shuffle(candidates)
        for i,j in candidates:
            laid=lay_link(rng,g,rooms,busy,reserved,i,j,bg,wall,palette,False)
            if laid is not None:
                g,reserved,path=laid
                paths.append(path)
                break
        else:
            continue
        expected=g.copy()
        valid=True
        for block in blocks:
            cells=[p for i in block for p in rooms[i]]
            levels=sorted({r for r,c in cells})
            choices=[level for level in levels if 0<sum(r>=level for r,c in cells)<len(cells)]
            if mode=='elevation' and len(block)>1:
                choices=[level for level in choices if
                         any(all(r<level for r,c in rooms[i]) for i in block) and
                         any(all(r>=level for r,c in rooms[i]) for i in block) and
                         any(any(r>=level for r,c in rooms[i]) and any(r<level for r,c in rooms[i]) for i in block)]
            if not choices:
                valid=False
                break
            level=int(rng.choice(choices))
            target=[p for p in cells if p[0]>=level]
            for p in target:
                expected[p]=water
            # Higher source cells, with jitter to make disconnected supply.
            ranked=sorted(cells,key=lambda p:p[0]+rng.uniform(0, max(2,h//3)))
            for p in ranked[:len(target)]:
                g[p]=water
        if not valid:
            continue
        try:
            a=analyse(g)
            actual=allocate(g,a)
            assert np.array_equal(actual,expected)
            if np.array_equal(actual,g):
                continue
            # Color-free pipes must induce a genuinely different allocation.
            if not matching_matters(g,actual,a):
                continue
            if not transfers_between_vessels(g,actual,rooms,water):
                continue
            if np.array_equal(actual,no_lateral_equalization(g,a)):
                continue
            if mode=='elevation' and not any(opened and bends(path)>0 for i,j,opened,path in a[-1]):
                continue
            if rng.integers(2):
                g=g[:,::-1].copy()
                actual=actual[:,::-1].copy()
            return {'input':g,'output':actual}
        except AssertionError:
            continue
    return fallback_pair(n,partitions,colors,mode)

def first_demo(rng,colors):
    bg,water,wall=colors[:3]
    for attempt in range(200):
        h=int(rng.integers(13,20))
        widths=list(map(int,rng.integers(4,6,size=3)))
        gaps=[int(rng.integers(7,10)),int(rng.integers(3,6))]
        open_index=0
        if rng.integers(2):
            gaps.reverse()
            open_index=1
        w=sum(widths)+sum(gaps)+4
        if w>30:
            continue
        g=np.full((h,w),bg,np.int8)
        rooms=[]
        boxes=[]
        col=2
        for tw in widths:
            r=int(rng.integers(3,6))
            th=int(rng.integers(5,8))
            rooms.append(carve_tank(rng,g,r,col,th,tw,bg,wall))
            boxes.append((r,col,th,tw))
            if len(rooms)<3:
                col+=tw+gaps[len(rooms)-1]
        links=[]
        for i in range(2):
            r,c,th,tw=boxes[i]
            r2,c2,th2,tw2=boxes[i+1]
            candidates=[rr for rr in range(max(r,r2)+1,min(r+th,r2+th2)-1)
                        if (rr,c+tw-2) in rooms[i] and (rr,c2+1) in rooms[i+1]]
            if not candidates:
                break
            rr=int(rng.choice(candidates))
            color=int(rng.choice(colors[3:]))
            g[rr-1,c+tw:c2]=color
            g[rr+1,c+tw:c2]=color
            if i!=open_index:
                g[rr-1,c+tw]=int(rng.choice([v for v in colors[3:] if v!=color]))
            else:
                at=int(rng.integers(c+tw+2,c2-2))
                g[rr-1,at]=g[rr+1,at]=bg
            links.append((i,i+1,i==open_index,[]))
        else:
            block=[open_index,open_index+1]
            cells=[p for i in block for p in rooms[i]]
            choices=[level for level in sorted({r for r,c in cells}) if
                     all(any(r>=level for r,c in rooms[i]) and any(r<level for r,c in rooms[i]) for i in block) and
                     sum(r>=level for r,c in cells)<=max(len(rooms[i]) for i in block)]
            if not choices:
                continue
            level=int(rng.choice(choices))
            target=[p for p in cells if p[0]>=level]
            source_id=int(rng.choice([i for i in block if len(rooms[i])>=len(target)]))
            source=sorted(rooms[source_id],key=lambda p:p[0]+rng.random())[:len(target)]
            expected=g.copy()
            for p in target:
                expected[p]=water
            for p in source:
                g[p]=water
            isolated=next(i for i in range(3) if i not in block)
            levels=sorted({r for r,c in rooms[isolated]})[1:]
            level2=int(rng.choice(levels))
            target2=[p for p in rooms[isolated] if p[0]>=level2]
            for p in target2:
                expected[p]=water
            for p in sorted(rooms[isolated],key=lambda p:p[0]+rng.random())[:len(target2)]:
                g[p]=water
            try:
                a=analyse(g)
                actual=allocate(g,a)
                assert np.array_equal(actual,expected)
                if not matching_matters(g,actual,a):
                    continue
                if np.array_equal(actual,no_lateral_equalization(g,a)):
                    continue
                if rng.integers(2):
                    g=g[:,::-1].copy()
                    actual=actual[:,::-1].copy()
                return {'input':g,'output':actual}
            except AssertionError:
                continue
    return fallback_demo(colors)

def bends(path):
    ds=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
    return sum(a!=b for a,b in zip(ds,ds[1:]))

def used_active(grid):
    a=analyse(grid)
    used=set()
    for i,j,opened,path in a[-1]:
        if opened:
            d=(path[1][0]-path[0][0],path[1][1]-path[0][1])
            used.add(int(grid[add(path[0],(-d[1],d[0]))]))
    return used

def polyline(points):
    path=[points[0]]
    for q in points[1:]:
        p=path[-1]
        assert (p[0]==q[0]) != (p[1]==q[1])
        d=(int(np.sign(q[0]-p[0])),int(np.sign(q[1]-p[1])))
        while p!=q:
            p=add(p,d)
            path.append(p)
    return path

def paint_pipe(g,path,color,bg,wall,other=None,gap=False):
    cells=set(path)
    for p in path:
        for d in DIRS:
            z=add(p,d)
            if inside(g,z) and z not in cells and g[z]!=wall:
                assert g[z] in (bg,color)
                g[z]=color
    if other is not None:
        d=(path[1][0]-path[0][0],path[1][1]-path[0][1])
        g[add(path[0],(-d[1],d[0]))]=other
    if gap:
        ds=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
        eligible=[k for k in range(3,len(path)-3) if len(set(ds[k-2:k+2]))==1]
        if eligible:
            k=eligible[len(eligible)//2]
            d=ds[k]
            for z in (add(path[k],(-d[1],d[0])),add(path[k],(d[1],-d[0]))):
                g[z]=bg

def rectangle_room(g,r,c,h,w,bg,wall):
    g[r:r+h,c:c+w]=wall
    g[r+1:r+h-1,c+1:c+w-1]=bg
    return [(y,x) for y in range(r+1,r+h-1) for x in range(c+1,c+w-1)]

def fallback_demo(colors):
    bg,water,wall,a,b=colors[:5]
    g=np.full((15,28),bg,np.int8)
    rooms=[rectangle_room(g,3,c,7,4,bg,wall) for c in (2,14,22)]
    paint_pipe(g,polyline([(6,6),(6,13)]),a,bg,wall,gap=True)
    paint_pipe(g,polyline([(6,18),(6,21)]),a,bg,wall,other=b)
    out=g.copy()
    target=[p for i in (0,1) for p in rooms[i] if p[0]>=7]
    target2=[p for p in rooms[2] if p[0]>=5]
    for p in target+target2:
        out[p]=water
    for p in sorted(rooms[0])[:len(target)]+sorted(rooms[2])[:len(target2)]:
        g[p]=water
    assert np.array_equal(answer(g),out)
    return {'input':g,'output':out}

def fallback_pair(n,partitions,colors,mode='general'):
    """Finite spare constructions, under the same input-visible rule.

    They are reached only after 200 failed random scenes; direct self-tests
    exercise them even when the random seed suite does not need them.
    """
    bg,water,wall=colors[:3]
    palette=colors[3:]
    g=np.full((30,30),bg,np.int8)
    if mode=='elevation':
        boxes=[(2,2,7,5),(12,2,6,5),(22,2,6,5),(2,22,7,5)]
        pipes=[(0,1,[(7,7),(7,12),(14,12),(14,7)],True),
               (1,2,[(18,4),(21,4)],True),
               (0,3,[(4,7),(4,21)],False)]
        blocks=[[0,1,2],[3]]
    else:
        boxes=[(2,2,6,5),(12,2,6,5),(22,2,6,5),
               (2,23,6,5),(12,23,6,5),(22,23,6,5)][:n]
        left=[(0,1,[(8,4),(11,4)],True),(1,2,[(18,4),(21,4)],True)]
        top=(0,3,[(4,7),(4,22)],True)
        right=(3,4,[(8,25),(11,25)],True)
        bottom=(2,5,[(24,7),(24,22)],True)
        if n==4:
            blocks=[[0,1,2],[3]]
            pipes=left+[(0,3,top[2],False)]
        elif n==5 and partitions==[3,2]:
            blocks=[[0,1,2],[3,4]]
            pipes=left+[right,(0,3,top[2],False)]
        elif n==5:
            blocks=[[0,1,2,3],[4]]
            pipes=left+[top,(1,4,[(14,7),(14,22)],False)]
        elif partitions==[4,2]:
            blocks=[[0,1,2,5],[3,4]]
            pipes=left+[bottom,right,(0,3,top[2],False)]
        else:
            blocks=[[0,1,2,3,4],[5]]
            pipes=left+[top,right,(2,5,bottom[2],False)]
    rooms=[rectangle_room(g,*box,bg,wall) for box in boxes]
    for k,(i,j,points,opened) in enumerate(pipes):
        color=palette[k%len(palette)]
        other=palette[(k+1)%len(palette)] if not opened else None
        paint_pipe(g,polyline(points),color,bg,wall,other=other,gap=True)
    out=g.copy()
    for k,block in enumerate(blocks):
        cells=[p for i in block for p in rooms[i]]
        level=14 if k==0 else (max(p[0] for p in cells) if len(block)==1 else 16)
        target=[p for p in cells if p[0]>=level]
        for p in target:
            out[p]=water
        for p in sorted(cells,key=lambda p:(p[1],p[0]))[:len(target)]:
            g[p]=water
    assert np.array_equal(answer(g),out)
    return {'input':g,'output':out}

def integer(value):
    return isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_))

def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read vessels and reciprocal matched rails; conserve each group's volume."""
    g=np.asarray(input_grid)
    if g.ndim!=2 or not np.issubdtype(g.dtype,np.integer) or not all(1<=d<=30 for d in g.shape) or not np.all((0<=g)&(g<=9)):
        raise ValueError('expected a 1..30 integer ARC grid')
    return answer(g)

def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not integer(seed) or int(seed)<0:
        raise ValueError('seed must be a nonnegative integer')
    if (not integer(num_train) or not integer(num_test) or
        not 2<=int(num_train)<=10 or not 1<=int(num_test)<=4 or num_train<=num_test):
        raise ValueError('need 2..10 train, 1..4 test, with train > test')
    rng=np.random.default_rng(int(seed))
    colors=list(map(int,rng.permutation(10)))
    puzzle={'train':[],'test':[]}
    seen=set()
    for split,count in (('train',int(num_train)),('test',int(num_test))):
        for index in range(count):
            for retry in range(30):
                # Three semantic roles persist; each pair resamples its guide
                # palette and geometry. Equality is not a color dictionary.
                pair_colors=colors[:3]+list(map(int,rng.permutation(colors[3:])))[:int(rng.integers(2,6))]
                if split=='train' and index==0:
                    pair=first_demo(rng,pair_colors)
                elif split=='train' and index==1:
                    pair=make_pair(rng,4,[3,1],pair_colors,'elevation')
                else:
                    n=int(rng.integers(5,7)) if split=='test' else int(rng.integers(4,6))
                    small=int(rng.choice([1,2])) if n>=5 else 1
                    pair=make_pair(rng,n,[n-small,small],pair_colors)
                key=(pair['input'].shape,pair['input'].tobytes())
                if key in seen:
                    continue
                if split=='train' and index==1:
                    # Compare actual matched endpoint colors, not palettes.
                    if not (used_active(pair['input'])-used_active(puzzle['train'][0]['input'])):
                        continue
                puzzle[split].append(pair)
                seen.add(key)
                break
            else:
                raise RuntimeError('distinct example construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle

def inspect_pair(inp,out):
    """Check returned arrays by storage levels and graph volume balances.

    These checks do not use sampled room boxes or selected level parameters.
    They supplement, but are not independent approval of, the shared reader.
    """
    assert inp.shape==out.shape
    a=analyse(inp)
    bg,water,wall,rooms,links=a
    assert 3<=len(rooms)<=6 and any(e[2] for e in links) and any(not e[2] for e in links)
    room_cells={p for room in rooms for p in room}
    exterior=np.ones(inp.shape,bool)
    for p in room_cells:
        exterior[p]=False
    assert np.array_equal(inp[exterior],out[exterior]), 'walls/rails/exterior changed'
    assert all(int(out[p]) in (bg,water) for p in room_cells)
    assert np.count_nonzero(inp==water)==np.count_nonzero(out==water)>0
    assert not np.array_equal(inp,out)
    assert np.argwhere(out==water)[:,0].sum()>np.argwhere(inp==water)[:,0].sum(), 'gravity did not lower potential'
    group_list=groups(len(rooms),links)
    for group in group_list:
        cells=[p for i in group for p in rooms[i]]
        supplied=sum(inp[p]==water for p in cells)
        filled=[p for p in cells if out[p]==water]
        assert len(filled)==supplied, 'component volume changed'
        if filled:
            level=min(p[0] for p in filled)
            assert all((out[p]==water)==(p[0]>=level) for p in cells), 'unequal or partial level'
    for r,c in np.argwhere(out==water):
        assert r+1<out.shape[0] and out[r+1,c] in (water,wall), 'unsupported liquid'
    assert np.array_equal(solve(inp.copy()),out)
    assert np.array_equal(solve(out.copy()),out), 'not a fixed point'
    assert matching_matters(inp,out,a), 'rail equality ablation does not change gravitational allocation'
    assert not np.array_equal(no_lateral_equalization(inp,a),out), 'lateral equalization ablation does not change answer'
    assert not np.array_equal(allocate(inp,a,strict=False,local=True),out), 'communication ablation does not change answer'
    assert transfers_between_vessels(inp,out,rooms,water), 'no exchange between vessels'
    gaps=0
    turns=[]
    lengths=[]
    guide_colors=set(map(int,np.unique(inp)))-{bg,water,wall}
    for i,j,opened,path in links:
        ds=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
        turns.append(bends(path))
        lengths.append(len(path))
        for k in range(2,len(path)-2):
            if ds[k-1]==ds[k] and paired(inp,path[k],ds[k],guide_colors) is None:
                gaps+=1
    room_status=[]
    for room in rooms:
        volume=sum(out[p]==water for p in room)
        room_status.append('dry' if volume==0 else 'full' if volume==len(room) else 'partial')
    return dict(shape=inp.shape,roles=(bg,water,wall),rooms=len(rooms),
                groups=tuple(sorted(map(len,group_list))),open=sum(e[2] for e in links),
                closed=sum(not e[2] for e in links),turns=tuple(turns),
                lengths=tuple(lengths),gaps=gaps,volume=int(np.count_nonzero(inp==water)),
                status=tuple(room_status),guide_colors=tuple(sorted(guide_colors)))

def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    n,m=len(puzzle['train']),len(puzzle['test'])
    assert 2<=n<=10 and 1<=m<=4 and n>m
    seen=set()
    common_roles=None
    train_stats=[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out=np.asarray(pair['input']),np.asarray(pair['output'])
            for g in (inp,out):
                assert g.ndim==2 and np.issubdtype(g.dtype,np.integer)
                assert all(1<=d<=30 for d in g.shape) and np.all((0<=g)&(g<=9))
            key=(inp.shape,inp.tobytes())
            assert key not in seen, 'duplicate input'
            seen.add(key)
            s=inspect_pair(inp,out)
            if common_roles is None:
                common_roles=s['roles']
            assert s['roles']==common_roles, 'puzzle role conventions changed'
            if split=='train':
                train_stats.append(s)
            else:
                assert s['rooms']>max(train_stats[0]['rooms'],train_stats[1]['rooms'])
                assert max(s['groups'])>=3
    first,second=train_stats[:2]
    assert first['rooms']==3 and first['groups']==(1,2) and first['gaps']>0
    assert not any(first['turns']), 'first demonstration should isolate equality/equalization'
    first_pair=puzzle['train'][0]
    inp,out=np.asarray(first_pair['input']),np.asarray(first_pair['output'])
    _,water,_,rooms,links=analyse(inp)
    assert any(not any(inp[p]==water for p in room) and any(out[p]==water for p in room) for room in rooms), 'no initially dry recipient'
    assert second['rooms']==4 and second['groups']==(1,3)
    assert 'full' in second['status'] and 'dry' in second['status'] and 'partial' in second['status']
    assert any(second['turns'])
    assert len(used_active(inp)|used_active(np.asarray(puzzle['train'][1]['input'])))>=2
    assert first['volume']!=second['volume'] or first['shape']!=second['shape']
    return None

def puzzle_signature(p):
    return tuple((s,tuple((np.asarray(x['input']).shape,np.asarray(x['input']).tobytes(),np.asarray(x['output']).tobytes()) for x in p[s])) for s in ('train','test'))

def self_test():
    """Bounded 42-seed suite, explicit counts, ablations and spare layouts."""
    signatures=set()
    shapes=set()
    normalized=set()
    roles_seen=[set(),set(),set()]
    volumes=set()
    counts=set()
    bends_seen=set()
    lengths=set()
    statuses=set()
    multi_groups=0
    for seed in range(42):
        p=generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(puzzle_signature(p))
        assert puzzle_signature(p)==puzzle_signature(generate_puzzle(seed)), 'nondeterminism'
        for split in ('train','test'):
            for pair in p[split]:
                g=pair['input']
                s=inspect_pair(g,pair['output'])
                shapes.add(s['shape'])
                for bucket,color in zip(roles_seen,s['roles']):
                    bucket.add(color)
                volumes.add(s['volume'])
                counts.add(s['rooms'])
                bends_seen.update(s['turns'])
                lengths.update(s['lengths'])
                statuses.update(s['status'])
                multi_groups+=sum(k>1 for k in s['groups'])>1
                bg,water,wall=s['roles']
                role_grid=np.where(g==bg,0,np.where(g==water,1,np.where(g==wall,2,3))).astype(np.int8)
                normalized.add((g.shape,role_grid.tobytes()))
    assert len(signatures)==42
    assert len(normalized)>=190 and len(shapes)>=30
    assert all(len(bucket)>=8 for bucket in roles_seen)
    assert counts=={3,4,5,6} and len(volumes)>=25
    assert {0,1,2}<=bends_seen and len(lengths)>=15
    assert statuses=={'dry','full','partial'} and multi_groups>=5
    for n in range(2,11):
        for m in range(1,min(4,n-1)+1):
            p=generate_puzzle(703,n,m)
            assert len(p['train'])==n and len(p['test'])==m
    for n,m in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.5,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,n,m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    colors=[0,1,8,2,3,4,5]
    spare=[fallback_demo(colors),fallback_pair(4,[3,1],colors,'elevation'),
           fallback_pair(4,[3,1],colors),fallback_pair(5,[4,1],colors),
           fallback_pair(5,[3,2],colors),fallback_pair(6,[5,1],colors),fallback_pair(6,[4,2],colors)]
    for pair in spare:
        inspect_pair(pair['input'],pair['output'])
    bad=generate_puzzle(31,2,1)
    bad['test'][0]['output']=bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted output')
    return None
