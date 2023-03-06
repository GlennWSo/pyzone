import numpy as np

points = np.array(
    [
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.5, -5.0, 0.0],
        [-0.2499999999999999, -5.0, 0.43301270189221935],
        [-0.2500000000000002, -5.0, -0.4330127018922192],
        [-0.2499999999999999, 5.0, 0.43301270189221935],
        [0.5, 5.0, 0.0],
        [-0.2500000000000002, 5.0, -0.4330127018922192],
        [-0.2499999999999999, 1.0, 0.43301270189221935],
        [-0.2500000000000002, 1.0, -0.2500000000000002],
        [-0.2500000000000001, 1.0, -0.08660254037844378],
        [0.0500000000000001, 1.0, 0.2598076211353316],
        [0.1830127018922194, 1.0, 0.1830127018922194],
        [0.5, -1.0, 0.0],
        [0.2, -1.0, 0.17320508075688773],
        [0.1830127018922192, -1.0, -0.1830127018922192],
        [-0.2499999999999999, -1.0, 0.43301270189221935],
        [-0.25, -1.0, 0.25],
        [0.1999999999999999, 1.0, -0.17320508075688767],
        [-0.2500000000000002, 1.0, -0.4330127018922192],
        [0.5, 1.0, 0.0],
        [-0.2500000000000002, -1.0, -0.4330127018922192],
        [-0.25, -1.0, 0.08660254037844389],
        [0.04999999999999988, -1.0, -0.2598076211353315],
    ]
)

faces = np.array(
    [
        [0, 2, 1],
        [3, 2, 0],
        [4, 2, 3],
        [5, 4, 3],
        [6, 4, 5],
        [5, 7, 6],
        [1, 7, 5],
        [1, 2, 7],
        [8, 10, 9],
        [11, 13, 12],
        [0, 14, 3],
        [15, 16, 1],
        [3, 17, 18],
        [17, 3, 14],
        [0, 1, 16],
        [16, 14, 0],
        [19, 20, 4],
        [19, 4, 6],
        [6, 21, 19],
        [20, 22, 4],
        [2, 4, 22],
        [2, 22, 23],
        [24, 25, 5],
        [26, 24, 5],
        [26, 3, 18],
        [26, 5, 3],
        [1, 5, 25],
        [1, 25, 15],
        [27, 7, 2],
        [27, 6, 7],
        [28, 27, 2],
        [29, 6, 27],
        [6, 29, 21],
        [2, 23, 28],
        [12, 26, 18],
        [18, 11, 12],
        [8, 20, 18],
        [11, 18, 17],
        [8, 18, 19],
        [13, 25, 24],
        [12, 13, 24],
        [10, 29, 27],
        [11, 17, 14],
        [20, 8, 22],
        [9, 22, 8],
        [16, 13, 14],
        [11, 14, 13],
        [9, 28, 23],
        [9, 23, 22],
        [29, 10, 21],
        [24, 21, 12],
        [8, 21, 10],
        [26, 12, 21],
        [8, 19, 21],
        [28, 9, 27],
        [13, 16, 15],
        [15, 25, 13],
        [10, 27, 9],
    ]
)

polyfaces = np.pad(faces, ((0, 0), (1, 0)), constant_values=3).ravel()