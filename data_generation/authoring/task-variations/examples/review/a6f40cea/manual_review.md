This task has 3 layers: the background, the middle transparent rectangles, and the top layer opaque rectangles. The output is just part of the grid behind the biggest top rectangle, with the background changed to that top rectangle's fill color.

Within each puzzle instance, vary the following:
- the background color in different pairs
- the number of middle transparent rectangles
- the location of the middle transparent rectangles (as long as they don't intersect each other, anything is allowed, even one inside the other like in the test examples, and they dont need to be concentric)
- the boundary colors of the middle transparent rectangles: (a) very few colors are used in the overall puzzle and (b) within each middle square boundary, only 1 or 2 colors are used -- solid or striped. You can use a lot more colors within that boundary), as long as it differentiates with the background.
- the boundary patterns of the middle transparent rectangles. Its currently only striped and solid, it should also include more complicated patterns or even random patterns. Note that any rectangle behind the biggest top rectangle should not have a random pattern and should be regular so the hidden part can be inferred from the visible boundary
- the boundary colors of the top rectangle. Each pair can have a different color (as long as it doesn't match the boundary color or fill color). It can also have a pattern (regular or random), as long as it is obvious that this is the top rectangle (If the fill color different from the background).

Note: In the situation where the top rectangle is not covering any middle squares, and it has the same fill color as the background, it becomes ambiguous whether the square is the top one or a middle one. Then this ambiguity is resolved by looking at the biggest rectangle. Also note that there can be multiple top rectangles, as long as there is a unique largest top rectangle

Also, include cases where the top square covers none of the middle squares so its just a solid color output

Note: In 1/14 seeds, use minimal variations. In a separate 1/7 seeds use as many combinations of variations as possible. The rest can vary among this spectrum
