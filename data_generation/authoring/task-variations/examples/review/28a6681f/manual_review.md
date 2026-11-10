Each puzzle instance has 3 parts: a boundary, water and a solid, abstractly shaped container(s) (which are non-moving, and can be more than one randomly placed shapes). The water in the input grids is a random shape(s) in a random. In the output grid, the container shapes remain the exact same as they were in their corresponding input grids but the water is affected by gravity, and has flowed downwards till there's no more place to move 

Note: In each puzzle instance, the color of water MUST be the same in all pairs, and is unique (no boundary or container pixel can have that color in that instance)

In each puzzle instance vary the following:
- across pairs, the colors of the container/other abstract shapes seems to be the same. Every pair in an instance should vary the colors (should be different from the background
- 

Between puzzle seeds, vary the following:
- background color (it is currently black only)
- the abstract shapes seem very similar across tasks. There should be more variation here. Go crazy

General variations to add
- all the synthetic puzzles seem to have water at the top. If you look at the original, there is  
- You can see in train 1 of the original task that water fills only one crevice and does not spill over. Such variations are missing
- the way the water is allowed to flow is that it wants to be the same level everywhere, and can move sideways and diagonally downwards to flow. It will do so till there is no more place to move either because it is blocked by a container /abstract wall or other water. This leads to it having a flat meniscus in the outputs (but diffierent water bodies in a single output grid may have different levels). 
- there is no need to have the water be a single contiguous shape, there can be multiple water objects in the input and after settling down it may lead to multiple different output grids
- the container/wall abstract shapes seem very similar. Its currently a single object with the variations being closed, one opening or 2 openings (up or down). The container/wall shapes can be as abstract and random as possible. It is easy to identify and disambiguate water and the walls because in the ttrain pairs you find out the color of the water as it is the only thing that moves
- Just go crazy with the variations
