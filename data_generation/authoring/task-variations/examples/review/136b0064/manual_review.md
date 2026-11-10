Currently in this puzzle, the input grid is divided into 2 halves by a vertical line. The left half contains many 3x3 shapes placed in a 2xN grid. The right half contains a single pixel in the top row. The output grid is the right half of the input grid, excluding the vertical line, with a new set of shapes superimposed on top.

The shapes in the input are a code. Each shape encodes a particular shape to be put into the output. Currently the output shapes are horizontal or vertical line segments, with each line starting at one pixel below the previous one and going in a direction that is also encoded by the shape (the first line starts below the single pixel in the top row).

The shapes are read vertically downward, (go down column 1, then column 2, and so on)

General variations needed:
- All the puzzle instances have the same set of input shapes AND each shape is encoded to the exact same line segment in the output. Both of these need to vary. Each puzzle instance should have different set of shapes, and each shape can be encoded to different line segments, and the types of line segments should also different (but each of these should be consistent within an instance so that it is decodable)
- increase the number of columns of shapes (but remember the output grid can be 30x30 at max)
- Vary the size of the shapes (eg: 2x2, 4x4, 2x3, etc) but be consistent within each puzzle
- Currently the input shapes and their colors are consistent in a puzzle instance -- same shapes have the same colors. This gives the model a shortcut to learn. This need not be the case so vary them inside a puzzle instance (but then the corresponding output line segment should match the color of that shape in that pait)
- the background and vertical line currently have the same color in all the pairs. This need not be the case since it is obvious what the background and the vertical lines are

In 1/14th of the puzzle instances, dont do much variation
