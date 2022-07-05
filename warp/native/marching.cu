#include "warp.h"
#include "marching.h"

#include "thrust/device_ptr.h"
#include "thrust/sort.h"

namespace wp {


    //  point numbering

    //       7-----------6
    //      /|          /|
    //     / |         / |
    //    /  |        /  |
    //   4-----------5   |
    //   |   |       |   |
    //   |   3-------|---2
    //   |  /        |  /
    //   | /         | /
    //   |/          |/
    //   0-----------1

    //  edge numbering

    //       *-----6-----*
    //      /|          /|
    //     7 |         5 |
    //    /  11       /  10
    //   *-----4-----*   |
    //   |   |       |   |
    //   |   *-----2-|---*
    //   8  /        9  /
    //   | 3         | 1
    //   |/          |/
    //   *-----0-----*


    //   z
    //   |  y
    //   | /
    //   |/
    //   0---- x

    __constant__ int marchingCubeCorners[8][3] = { {0,0,0}, {1,0,0},{1,1,0},{0,1,0}, {0,0,1}, {1,0,1},{1,1,1},{0,1,1} };

    __constant__ int firstMarchingCubesId[257] = {
    0, 0, 3, 6, 12, 15, 21, 27, 36, 39, 45, 51, 60, 66, 75, 84, 90, 93, 99, 105, 114,
    120, 129, 138, 150, 156, 165, 174, 186, 195, 207, 219, 228, 231, 237, 243, 252, 258, 267, 276, 288,
    294, 303, 312, 324, 333, 345, 357, 366, 372, 381, 390, 396, 405, 417, 429, 438, 447, 459, 471, 480,
    492, 507, 522, 528, 531, 537, 543, 552, 558, 567, 576, 588, 594, 603, 612, 624, 633, 645, 657, 666,
    672, 681, 690, 702, 711, 723, 735, 750, 759, 771, 783, 798, 810, 825, 840, 852, 858, 867, 876, 888,
    897, 909, 915, 924, 933, 945, 957, 972, 984, 999, 1008, 1014, 1023, 1035, 1047, 1056, 1068, 1083, 1092, 1098,
    1110, 1125, 1140, 1152, 1167, 1173, 1185, 1188, 1191, 1197, 1203, 1212, 1218, 1227, 1236, 1248, 1254, 1263, 1272, 1284,
    1293, 1305, 1317, 1326, 1332, 1341, 1350, 1362, 1371, 1383, 1395, 1410, 1419, 1425, 1437, 1446, 1458, 1467, 1482, 1488,
    1494, 1503, 1512, 1524, 1533, 1545, 1557, 1572, 1581, 1593, 1605, 1620, 1632, 1647, 1662, 1674, 1683, 1695, 1707, 1716,
    1728, 1743, 1758, 1770, 1782, 1791, 1806, 1812, 1827, 1839, 1845, 1848, 1854, 1863, 1872, 1884, 1893, 1905, 1917, 1932,
    1941, 1953, 1965, 1980, 1986, 1995, 2004, 2010, 2019, 2031, 2043, 2058, 2070, 2085, 2100, 2106, 2118, 2127, 2142, 2154,
    2163, 2169, 2181, 2184, 2193, 2205, 2217, 2232, 2244, 2259, 2268, 2280, 2292, 2307, 2322, 2328, 2337, 2349, 2355, 2358,
    2364, 2373, 2382, 2388, 2397, 2409, 2415, 2418, 2427, 2433, 2445, 2448, 2454, 2457, 2460, 2460 };

    __constant__ int marchingCubesIds[2460] = {
    0, 8, 3, 0, 1, 9, 1, 8, 3, 9, 8, 1, 1, 2, 10, 0, 8, 3, 1, 2, 10, 9, 2, 10, 0, 2, 9, 2, 8, 3, 2,
    10, 8, 10, 9, 8, 3, 11, 2, 0, 11, 2, 8, 11, 0, 1, 9, 0, 2, 3, 11, 1, 11, 2, 1, 9, 11, 9, 8, 11, 3,
    10, 1, 11, 10, 3, 0, 10, 1, 0, 8, 10, 8, 11, 10, 3, 9, 0, 3, 11, 9, 11, 10, 9, 9, 8, 10, 10, 8, 11, 4,
    7, 8, 4, 3, 0, 7, 3, 4, 0, 1, 9, 8, 4, 7, 4, 1, 9, 4, 7, 1, 7, 3, 1, 1, 2, 10, 8, 4, 7, 3,
    4, 7, 3, 0, 4, 1, 2, 10, 9, 2, 10, 9, 0, 2, 8, 4, 7, 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, 8,
    4, 7, 3, 11, 2, 11, 4, 7, 11, 2, 4, 2, 0, 4, 9, 0, 1, 8, 4, 7, 2, 3, 11, 4, 7, 11, 9, 4, 11, 9,
    11, 2, 9, 2, 1, 3, 10, 1, 3, 11, 10, 7, 8, 4, 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, 4, 7, 8, 9,
    0, 11, 9, 11, 10, 11, 0, 3, 4, 7, 11, 4, 11, 9, 9, 11, 10, 9, 5, 4, 9, 5, 4, 0, 8, 3, 0, 5, 4, 1,
    5, 0, 8, 5, 4, 8, 3, 5, 3, 1, 5, 1, 2, 10, 9, 5, 4, 3, 0, 8, 1, 2, 10, 4, 9, 5, 5, 2, 10, 5,
    4, 2, 4, 0, 2, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, 9, 5, 4, 2, 3, 11, 0, 11, 2, 0, 8, 11, 4,
    9, 5, 0, 5, 4, 0, 1, 5, 2, 3, 11, 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, 10, 3, 11, 10, 1, 3, 9,
    5, 4, 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, 5, 4, 8, 5,
    8, 10, 10, 8, 11, 9, 7, 8, 5, 7, 9, 9, 3, 0, 9, 5, 3, 5, 7, 3, 0, 7, 8, 0, 1, 7, 1, 5, 7, 1,
    5, 3, 3, 5, 7, 9, 7, 8, 9, 5, 7, 10, 1, 2, 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, 8, 0, 2, 8,
    2, 5, 8, 5, 7, 10, 5, 2, 2, 10, 5, 2, 5, 3, 3, 5, 7, 7, 9, 5, 7, 8, 9, 3, 11, 2, 9, 5, 7, 9,
    7, 2, 9, 2, 0, 2, 7, 11, 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, 11, 2, 1, 11, 1, 7, 7, 1, 5, 9,
    5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, 11, 10, 0, 11,
    0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, 11, 10, 5, 7, 11, 5, 10, 6, 5, 0, 8, 3, 5, 10, 6, 9, 0, 1, 5,
    10, 6, 1, 8, 3, 1, 9, 8, 5, 10, 6, 1, 6, 5, 2, 6, 1, 1, 6, 5, 1, 2, 6, 3, 0, 8, 9, 6, 5, 9,
    0, 6, 0, 2, 6, 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, 2, 3, 11, 10, 6, 5, 11, 0, 8, 11, 2, 0, 10,
    6, 5, 0, 1, 9, 2, 3, 11, 5, 10, 6, 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, 6, 3, 11, 6, 5, 3, 5,
    1, 3, 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, 6, 5, 9, 6,
    9, 11, 11, 9, 8, 5, 10, 6, 4, 7, 8, 4, 3, 0, 4, 7, 3, 6, 5, 10, 1, 9, 0, 5, 10, 6, 8, 4, 7, 10,
    6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, 6, 1, 2, 6, 5, 1, 4, 7, 8, 1, 2, 5, 5, 2, 6, 3, 0, 4, 3,
    4, 7, 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, 3,
    11, 2, 7, 8, 4, 10, 6, 5, 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, 0, 1, 9, 4, 7, 8, 2, 3, 11, 5,
    10, 6, 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, 5,
    1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, 6,
    5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, 10, 4, 9, 6, 4, 10, 4, 10, 6, 4, 9, 10, 0, 8, 3, 10, 0, 1, 10,
    6, 0, 6, 4, 0, 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, 1, 4, 9, 1, 2, 4, 2, 6, 4, 3, 0, 8, 1,
    2, 9, 2, 4, 9, 2, 6, 4, 0, 2, 4, 4, 2, 6, 8, 3, 2, 8, 2, 4, 4, 2, 6, 10, 4, 9, 10, 6, 4, 11,
    2, 3, 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, 6, 4, 1, 6,
    1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, 8, 11, 1, 8, 1, 0, 11,
    6, 1, 9, 1, 4, 6, 4, 1, 3, 11, 6, 3, 6, 0, 0, 6, 4, 6, 4, 8, 11, 6, 8, 7, 10, 6, 7, 8, 10, 8,
    9, 10, 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, 10, 6, 7, 10,
    7, 1, 1, 7, 3, 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7,
    3, 9, 7, 8, 0, 7, 0, 6, 6, 0, 2, 7, 3, 2, 6, 7, 2, 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, 2,
    0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, 11,
    2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, 0, 9, 1, 11,
    6, 7, 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, 7, 11, 6, 7, 6, 11, 3, 0, 8, 11, 7, 6, 0, 1, 9, 11,
    7, 6, 8, 1, 9, 8, 3, 1, 11, 7, 6, 10, 1, 2, 6, 11, 7, 1, 2, 10, 3, 0, 8, 6, 11, 7, 2, 9, 0, 2,
    10, 9, 6, 11, 7, 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, 7, 2, 3, 6, 2, 7, 7, 0, 8, 7, 6, 0, 6,
    2, 0, 2, 7, 6, 2, 3, 7, 0, 1, 9, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, 10, 7, 6, 10, 1, 7, 1,
    3, 7, 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, 7, 6, 10, 7,
    10, 8, 8, 10, 9, 6, 8, 4, 11, 8, 6, 3, 6, 11, 3, 0, 6, 0, 4, 6, 8, 6, 11, 8, 4, 6, 9, 0, 1, 9,
    4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, 6, 8, 4, 6, 11, 8, 2, 10, 1, 1, 2, 10, 3, 0, 11, 0, 6, 11, 0,
    4, 6, 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, 8,
    2, 3, 8, 4, 2, 4, 6, 2, 0, 4, 2, 4, 6, 2, 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, 1, 9, 4, 1,
    4, 2, 2, 4, 6, 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, 10, 1, 0, 10, 0, 6, 6, 0, 4, 4, 6, 3, 4,
    3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, 10, 9, 4, 6, 10, 4, 4, 9, 5, 7, 6, 11, 0, 8, 3, 4, 9, 5, 11,
    7, 6, 5, 0, 1, 5, 4, 0, 7, 6, 11, 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, 9, 5, 4, 10, 1, 2, 7,
    6, 11, 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, 3, 4, 8, 3,
    5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, 7, 2, 3, 7, 6, 2, 5, 4, 9, 9, 5, 4, 0, 8, 6, 0, 6, 2, 6,
    8, 7, 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, 9,
    5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, 4, 0, 10, 4,
    10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, 6, 9, 5, 6, 11, 9, 11,
    8, 9, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, 6, 11, 3, 6,
    3, 5, 5, 3, 1, 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1,
    2, 10, 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, 5,
    8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, 9, 5, 6, 9, 6, 0, 0, 6, 2, 1, 5, 8, 1, 8, 0, 5, 6, 8, 3,
    8, 2, 6, 2, 8, 1, 5, 6, 2, 1, 6, 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, 10, 1, 0, 10,
    0, 6, 9, 5, 0, 5, 6, 0, 0, 3, 8, 5, 6, 10, 10, 5, 6, 11, 5, 10, 7, 5, 11, 11, 5, 10, 11, 7, 5, 8,
    3, 0, 5, 11, 7, 5, 10, 11, 1, 9, 0, 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, 11, 1, 2, 11, 7, 1, 7,
    5, 1, 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, 7, 5, 2, 7,
    2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, 2, 5, 10, 2, 3, 5, 3, 7, 5, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10,
    2, 5, 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, 1,
    3, 5, 3, 7, 5, 0, 8, 7, 0, 7, 1, 1, 7, 5, 9, 0, 3, 9, 3, 5, 5, 3, 7, 9, 8, 7, 5, 9, 7, 5,
    8, 4, 5, 10, 8, 10, 11, 8, 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, 0, 1, 9, 8, 4, 10, 8, 10, 11, 10,
    4, 5, 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, 0,
    4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, 9,
    4, 5, 2, 11, 3, 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, 5, 10, 2, 5, 2, 4, 4, 2, 0, 3, 10, 2, 3,
    5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, 8, 4, 5, 8, 5, 3, 3,
    5, 1, 0, 4, 5, 1, 0, 5, 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, 9, 4, 5, 4, 11, 7, 4, 9, 11, 9,
    10, 11, 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, 3, 1, 4, 3,
    4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, 9, 7, 4, 9, 11, 7, 9,
    1, 11, 2, 11, 1, 0, 8, 3, 11, 7, 4, 11, 4, 2, 2, 4, 0, 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, 2,
    9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, 3, 7, 10, 3,
    10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, 1, 10, 2, 8, 7, 4, 4, 9, 1, 4, 1, 7, 7, 1, 3, 4, 9, 1, 4,
    1, 7, 0, 8, 1, 8, 7, 1, 4, 0, 3, 7, 4, 3, 4, 8, 7, 9, 10, 8, 10, 11, 8, 3, 0, 9, 3, 9, 11, 11,
    9, 10, 0, 1, 10, 0, 10, 8, 8, 10, 11, 3, 1, 10, 11, 3, 10, 1, 2, 11, 1, 11, 9, 9, 11, 8, 3, 0, 9, 3,
    9, 11, 1, 2, 9, 2, 11, 9, 0, 2, 11, 8, 0, 11, 3, 2, 11, 2, 3, 8, 2, 8, 10, 10, 8, 9, 9, 10, 2, 0,
    9, 2, 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, 1, 10, 2, 1, 3, 8, 9, 1, 8, 0, 9, 1, 0, 3, 8};

    __constant__ int marchingCubesEdgeLocations[12][4] = {
        // relative cell coords, edge within cell
        {0, 0, 0,  0},
        {1, 0, 0,  1},
        {0, 1, 0,  0},
        {0, 0, 0,  1},

        {0, 0, 1,  0},
        {1, 0, 1,  1},
        {0, 1, 1,  0},
        {0, 0, 1,  1},

        {0, 0, 0,  2},
        {1, 0, 0,  2},
        {1, 1, 0,  2},
        {0, 1, 0,  2}
    };


	// ---------------------------------------------------------------------------------------
	struct MarchingCubes
	{
		MarchingCubes() 
        {
			memset(this, 0, sizeof(MarchingCubes));
		}

        __device__ __host__ int cell_index(int xi, int yi, int zi) const
        {
            return (xi * ny + yi) * nz + zi;
        }
        __device__ __host__ void cell_coord(int cell_index, int& xi, int& yi, int& zi) const
        {
            zi = cell_index % nz; cell_index /= nz;
            yi = cell_index % ny;
            xi = cell_index / ny;
        }

		// grid
        int nx;
        int ny;
        int nz;

		int* first_cell_vert;
		int* first_cell_tri;
		int* cell_verts;

        // used for computing normals
		vec3* avg_verts;
		int* avg_num;

        int num_cells;   

        // capacity limits
        int max_cells;
        int max_verts;

	};


/*	// -----------------------------------------------------------------------------
	__global__ void deviceComputeParticleDensity(vec3* deviceParticlePos, int numParticles, MarchingCubes mc)
	{
		int pNr = blockIdx.x * blockDim.x + threadIdx.x;
		if (pNr >= numParticles)
			return;

		vec3 p = deviceParticlePos[pNr] - mc.orig;

		if (p.x < 0.0f || p.y < 0.0f || p.z < 0.0f)
			return;

		int cxi = (int)floorf(p.x / mc.spacing);
		int cyi = (int)floorf(p.y / mc.spacing);
		int czi = (int)floorf(p.z / mc.spacing);

		if (cxi >= mc.nx - 1 || cyi >= mc.ny - 1 || czi >= mc.nz - 1)
			return;

		float h = 1.5f * mc.spacing;
		float h2 = h * h;
		float kernelScale = 315.0f / (64.0f * 3.14159265f * h2 * h2 * h2 * h2 * h) / mc.restDensity;

		float rho = 0.0f;
		vec3 n = vec3(Zero);

		for (int xi = cxi; xi <= cxi + 1; xi++) {
			for (int yi = cyi; yi <= cyi + 1; yi++) {
				for (int zi = czi; zi <= czi + 1; zi++) {

					vec3 cellPos = vec3(xi * mc.spacing, yi * mc.spacing, zi * mc.spacing);
					vec3 r = p - cellPos;
					float r2 = r.magnitudeSquared();
					if (r2 < h2) {
						float w = (h2 - r2);
						w = kernelScale * w * w * w;

						int cell_index = mc.cell_index(xi, yi, zi);
						atomicAdd(&density[cell_index], w);
						r.normalize();
						r *= -w;
						atomicAdd3(&densityNormal[cell_index], r);
					}
				}
			}
		}
	}
*/

	// -----------------------------------------------------------------------------------
	__global__ void count_cell_verts(MarchingCubes mc, const float* density, float threshold)
	{
		int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (cell_index >= mc.num_cells)
			return;

		int xi, yi, zi;
		mc.cell_coord(cell_index, xi, yi, zi);

		mc.first_cell_vert[cell_index] = 0;
		if (xi >= mc.nx - 1 || yi >= mc.ny - 1 || zi >= mc.nz - 1)
			return;

		float d0 = density[cell_index];
		float dx = density[mc.cell_index(xi + 1, yi, zi)];
		float dy = density[mc.cell_index(xi, yi + 1, zi)];
		float dz = density[mc.cell_index(xi, yi, zi + 1)];

		int num = 0;
		if ((d0 <= threshold && dx >= threshold) || (dx <= threshold && d0 >= threshold))
			num++;
		if ((d0 <= threshold && dy >= threshold) || (dy <= threshold && d0 >= threshold))
			num++;
		if ((d0 <= threshold && dz >= threshold) || (dz <= threshold && d0 >= threshold))
			num++;

		mc.first_cell_vert[cell_index] = num;
	}

	// -----------------------------------------------------------------------------------
	__global__ void create_cell_verts(MarchingCubes mc, vec3* __restrict__ vertices, vec3* normals, const float* __restrict__  density, float threshold)
	{
		int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (cell_index >= mc.num_cells)
			return;

		int xi, yi, zi;
		mc.cell_coord(cell_index, xi, yi, zi);
		if (xi >= mc.nx - 1 || yi >= mc.ny - 1 || zi >= mc.nz - 1)
			return;

		vec3 p = vec3(xi + 0.5f, yi + 0.5f, zi + 0.5f);

		float d0 = density[cell_index];
		float ds[3];
		ds[0] = density[mc.cell_index(xi + 1, yi, zi)];
		ds[1] = density[mc.cell_index(xi, yi + 1, zi)];
		ds[2] = density[mc.cell_index(xi, yi, zi + 1)];

		// vec3 n0 = densityNormal[cell_index];
		// vec3 ns[3];
		// ns[0] = densityNormal[mc.cell_index(xi + 1, yi, zi)];
		// ns[1] = densityNormal[mc.cell_index(xi, yi + 1, zi)];
		// ns[2] = densityNormal[mc.cell_index(xi, yi, zi + 1)];

		int first = mc.first_cell_vert[cell_index];

		for (int dim = 0; dim < 3; dim++) 
        {
			float d = ds[dim];
			mc.cell_verts[3 * cell_index + dim] = 0;

			if ((d0 <= threshold && d >= threshold) || (d <= threshold && d0 >= threshold)) 
            {
				float t = (d != d0) ? clamp((threshold - d0) / (d - d0), 0.0f, 1.0f) : 0.5f;
				int id = first++;

				vec3 off;
				off[dim] = t;
				vertices[id] = p + off;

				// vec3 n = normalize(n0 + t * (ns[dim] - n0));
				// normals[id] = -n;

				mc.cell_verts[3 * cell_index + dim] = id;
			}
		}
	}

	// -----------------------------------------------------------------------------------
	__global__ void count_cell_tris(MarchingCubes mc, const float* __restrict__ density, float threshold)
	{
		int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (cell_index >= mc.num_cells)
			return;

		int xi, yi, zi;
		mc.cell_coord(cell_index, xi, yi, zi);

		mc.first_cell_tri[cell_index] = 0;
		if (xi >= mc.nx - 2 || yi >= mc.ny - 2 || zi >= mc.nz - 2)
			return;

		int code = 0;
		for (int i = 0; i < 8; i++) {
			int cxi = xi + marchingCubeCorners[i][0];
			int cyi = yi + marchingCubeCorners[i][1];
			int czi = zi + marchingCubeCorners[i][2];

			if (density[mc.cell_index(cxi, cyi, czi)] >= threshold)
				code |= (1 << i);
		}

		mc.first_cell_tri[cell_index] = firstMarchingCubesId[code + 1] - firstMarchingCubesId[code];
	}

	// -----------------------------------------------------------------------------------
	__global__ void create_cell_tris(MarchingCubes mc, const float* __restrict__ density, int* __restrict__ triangles, float threshold)
	{
		int cell_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (cell_index >= mc.num_cells)
			return;

		int xi, yi, zi;
		mc.cell_coord(cell_index, xi, yi, zi);
		if (xi >= mc.nx - 2 || yi >= mc.ny - 2 || zi >= mc.nz - 2)
			return;

		int code = 0;
		for (int i = 0; i < 8; i++)
        {
			int cxi = xi + marchingCubeCorners[i][0];
			int cyi = yi + marchingCubeCorners[i][1];
			int czi = zi + marchingCubeCorners[i][2];

			if (density[mc.cell_index(cxi, cyi, czi)] >= threshold)
				code |= (1 << i);
		}

		int firstIn = firstMarchingCubesId[code];
		int num = firstMarchingCubesId[code + 1] - firstIn;
		int firstOut = mc.first_cell_tri[cell_index];

		for (int i = 0; i < num; i++)
        {
			int eid = marchingCubesIds[firstIn + i];

			int exi = xi + marchingCubesEdgeLocations[eid][0];
			int eyi = yi + marchingCubesEdgeLocations[eid][1];
			int ezi = zi + marchingCubesEdgeLocations[eid][2];
			int edgeNr = marchingCubesEdgeLocations[eid][3];

			int id = mc.cell_verts[3 * mc.cell_index(exi, eyi, ezi) + edgeNr];
			triangles[firstOut + i] = id;
		}
	}

	// -----------------------------------------------------------------------------------
	__global__ void smooth_verts(MarchingCubes mc, vec3* vertices, int* triangles, int num_tris)
	{
		int triNr = blockIdx.x * blockDim.x + threadIdx.x;
		if (triNr >= num_tris / 3)
			return;

		for (int i = 0; i < 3; i++)
        {
			int id0 = triangles[3 * triNr + i];
			int id1 = triangles[3 * triNr + (i + 1) % 3];
			int id2 = triangles[3 * triNr + (i + 2) % 3];

			vec3 p1 = vertices[id1];
			vec3 p2 = vertices[id2];

			atomic_add(&mc.avg_num[id0], 2);
			atomic_add(&mc.avg_verts[id0], p1);
			atomic_add(&mc.avg_verts[id0], p2);
		}
	}

	// -----------------------------------------------------------------------------------
	__global__ void average_verts(MarchingCubes mc, vec3* vertices, int num_verts)
	{
		int vNr = blockIdx.x * blockDim.x + threadIdx.x;
		if (vNr >= num_verts)
			return;
		
		int num = mc.avg_num[vNr];
		if (num > 0)
			vertices[vNr] = mc.avg_verts[vNr] / (float)num;
	}

	// -----------------------------------------------------------------------------------
	__global__ void compute_normals(MarchingCubes mc, const vec3* __restrict__ vertices, vec3* normals, const int* __restrict__ triangles, int num_verts, int num_tris)
	{
		int triNr = blockIdx.x * blockDim.x + threadIdx.x;
		if (triNr >= num_tris / 3)
			return;

		int id0 = triangles[3 * triNr];
		int id1 = triangles[3 * triNr + 1];
		int id2 = triangles[3 * triNr + 2];

		vec3 p0 = vertices[id0];
		vec3 p1 = vertices[id1];
		vec3 p2 = vertices[id2];
		vec3 n = cross(p1 - p0, p2 - p0);

		atomic_add(&normals[id0], n);
		atomic_add(&normals[id1], n);
		atomic_add(&normals[id2], n);
	}

	// -----------------------------------------------------------------------------------
	__global__ void smooth_normals(MarchingCubes mc, vec3* normals, const int* __restrict__ triangles, int num_tris)
	{
		int triNr = blockIdx.x * blockDim.x + threadIdx.x;
		if (triNr >= num_tris / 3)
			return;

		int id0 = triangles[3 * triNr];
		int id1 = triangles[3 * triNr + 1];
		int id2 = triangles[3 * triNr + 2];

		vec3 n0 = normals[id0];
		vec3 n1 = normals[id1];
		vec3 n2 = normals[id2];

		vec3 n = normalize(n0 + n1 + n2);

		atomic_add(&normals[id0], n);
		atomic_add(&normals[id1], n);
		atomic_add(&normals[id2], n);
	}

	// -----------------------------------------------------------------------------------
	__global__ void normalize_normals(vec3* normals, int num_verts)
	{
		int vNr = blockIdx.x * blockDim.x + threadIdx.x;
		if (vNr >= num_verts)
			return;

		normals[vNr] = normalize(normals[vNr]);
	}


	// // -----------------------------------------------------------------------------------
	// void create(PbSim::vec3* deviceParticlePos, int numParticles)
	// {
	// 	if (!density)
	// 		return;

	// 	// create density field

	// 	clearDeviceBuffer(density, mc.num_cells);
	// 	clearDeviceBuffer(densityNormal, mc.num_cells);

	// 	deviceComputeParticleDensity << < numParticles / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (
	// 		deviceParticlePos, numParticles, data);

	// 	if (mc.restDensity == 1.0f) {

	// 		// initialize rest density

	// 		std::vector<float> density;
	// 		getDeviceBuffer(density, density, mc.num_cells);

	// 		float sum = 0.0f;
	// 		int num = 0;
	// 		for (int i = 0; i < mc.num_cells; i++) {
	// 			if (density[i] > 0.0) {
	// 				sum += density[i];
	// 				num++;
	// 			}
	// 		}

	// 		if (num > 0)
	// 			mc.restDensity = sum / (float)num;

	// 		clearDeviceBuffer(density, mc.num_cells);
	// 		clearDeviceBuffer(densityNormal, mc.num_cells);

	// 		deviceComputeParticleDensity << < numParticles / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (
	// 			deviceParticlePos, numParticles, data);
	// 	}

	// 	// create vertices

	// 	count_cell_verts << < mc.num_cells / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

	// 	int numLast;
	// 	getDeviceObject(numLast, mc.first_cell_vert, mc.num_cells - 1);

	// 	thrust::exclusive_scan(
	// 		thrust::device_ptr<int>(mc.first_cell_vert),
	// 		thrust::device_ptr<int>(mc.first_cell_vert + mc.num_cells),
	// 		thrust::device_ptr<int>(mc.first_cell_vert));

	// 	getDeviceObject(mc.numVerts, mc.first_cell_vert, mc.num_cells - 1);
	// 	mc.numVerts += numLast;

	// 	if (mc.numVerts > mc.maxVerts) {
	// 		mc.maxVerts = 2 * mc.numVerts;
	// 		freeDeviceBuffers(mc.verts, mc.normals);
	// 		allocDeviceBuffer(mc.verts, mc.maxVerts);
	// 		allocDeviceBuffer(mc.normals, mc.maxVerts);

	// 		freeDeviceBuffers(mc.avg_verts, mc.avg_num);
	// 		allocDeviceBuffer(mc.avg_verts, mc.maxVerts);
	// 		allocDeviceBuffer(mc.avg_num, mc.maxVerts);
	// 	}

	// 	create_cell_verts << < mc.num_cells / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

	// 	// create triangles

	// 	count_cell_tris << < mc.num_cells / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

	// 	getDeviceObject(numLast, mc.first_cell_tri, mc.num_cells - 1);

	// 	thrust::exclusive_scan(
	// 		thrust::device_ptr<int>(mc.first_cell_tri),
	// 		thrust::device_ptr<int>(mc.first_cell_tri + mc.num_cells),
	// 		thrust::device_ptr<int>(mc.first_cell_tri));

	// 	getDeviceObject(mc.numTriIds, mc.first_cell_tri, mc.num_cells - 1);
	// 	mc.numTriIds += numLast;

	// 	if (mc.numTriIds > mc.maxTriIds) {
	// 		mc.maxTriIds = 2 * mc.numTriIds;
	// 		freeDeviceBuffers(mc.triIds);
	// 		allocDeviceBuffer(mc.triIds, mc.maxTriIds);
	// 	}

	// 	create_cell_tris << < mc.num_cells / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

	// 	// smooth verts

	// 	int numTris = mc.numTriIds / 3;

	// 	for (int i = 0; i < 3; i++) {
	// 		clearDeviceBuffer(mc.avg_num, mc.numVerts);
	// 		clearDeviceBuffer(mc.avg_verts, mc.numVerts);

	// 		smooth_verts << < numTris / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
	// 		average_verts << < mc.numVerts / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
	// 	}

	// 	// compute normals

	// 	clearDeviceBuffer(mc.normals, mc.numVerts);

	// 	compute_normals << < numTris / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
	// 	normalize_normals << < mc.numVerts / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

	// 	//for (int i = 0; i < 5; i++)
	// 	//	smooth_normals <<<  numTris / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

	// 	//normalize_normals << < mc.numVerts / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
	// }


    // -------------------------
    void marching_cubes_resize(MarchingCubes& mc, int nx, int ny, int nz, int max_vertices)
    {
        mc.nx = nx;
        mc.ny = ny;
        mc.nz = nz;

        mc.num_cells = nx*ny*nz;

        if (mc.num_cells > mc.max_cells)
        {
            const int num_to_alloc = mc.num_cells*3/2;

            mc.first_cell_vert = (int*)alloc_device(sizeof(int) * num_to_alloc);
            mc.first_cell_tri = (int*)alloc_device(sizeof(int) * num_to_alloc);
            mc.cell_verts = (int*)alloc_device(sizeof(int) * 3 * num_to_alloc);

            mc.max_cells = num_to_alloc;

        }
        
        if (max_vertices > mc.max_verts)
        {
            const int num_to_alloc = max_vertices*3/2;

            mc.avg_verts = (vec3*)alloc_device(sizeof(vec3) * num_to_alloc);
            mc.avg_num = (int*)alloc_device(sizeof(int) * num_to_alloc);

            mc.max_verts = num_to_alloc;
        }

    }

    // -------------------------
    void marching_cubes_free(MarchingCubes& mc)
    {
        free_device(mc.first_cell_vert);
        free_device(mc.first_cell_tri);
        free_device(mc.cell_verts);

        free_device(mc.avg_verts);
        free_device(mc.avg_num);
    }



	// // ------------------------------------------------------------------------------------
	// void readBackSurface(std::vector<PbSim::vec3>& vertices, std::vector<PbSim::vec3>& normals, std::vector<int>& triIds)
	// {
	// 	vertices.resize(mc.numVerts);
	// 	normals.resize(mc.numVerts);
	// 	triIds.resize(mc.numTriIds);

	// 	if (mc.numVerts == 0 || mc.numTriIds == 0)
	// 		return;

	// 	getDeviceBuffer(vertices, mc.verts);
	// 	getDeviceBuffer(normals, mc.normals);
	// 	getDeviceBuffer(triIds, mc.triIds);
	// }

	// // -----
	// void read_mesh(float *vertices, float* normals, int* triIds)
	// {
	// 	cudaCheck(cudaMemcpy(vertices, mc.verts, 3*mc.numVerts * sizeof(float), cudaMemcpyDeviceToHost));
	// 	cudaCheck(cudaMemcpy(normals, mc.normals, 3*mc.numVerts * sizeof(float), cudaMemcpyDeviceToHost));
	// 	cudaCheck(cudaMemcpy(triIds, mc.triIds, mc.numTriIds * sizeof(int), cudaMemcpyDeviceToHost));
	// }


// ///////////////////////////////////////////////
// Data libdata;
// Data& initlib(float ax, float ay, float az, 
// 			 float bx, float by, float bz,
// 			 float step, 
// 			 float threshold)
// {
//     Bounds3 worldBounds(vec3(ax,ay,az), vec3(bx,by,bz));
//     init(worldBounds, step, threshold);

// 	libmc.num_cells = mc.num_cells;
// 	libmc.nx = mc.nx;
// 	libmc.ny = mc.ny;
// 	libmc.nz = mc.nz;
	
// 	return libdata;
// }

// int set_sdf(std::vector<float> host_sdf){
// 	if (host_sdf.size() != mc.num_cells)
// 		return -1;

// 	setDeviceBuffer(density, host_sdf);
// 	return 0;
// }

// int send_sdf2device(float *host_data, int n)
// {
// 	if (mc.num_cells != n)
// 		return -1;
// 	cudaCheck(cudaMemcpy(density, host_data, n * sizeof(float), cudaMemcpyHostToDevice));	
// 	return 0;
// }
// void make_mesh(int *nvert, int *nnorm, int *ntris)
// {
// 	create_mesh();
// 	*nvert = mc.numVerts;
// 	*nnorm = mc.numVerts;
// 	*ntris = mc.numTriIds;

// }

}

uint64_t marching_cubes_create_device()
{
    wp::MarchingCubes* mc = new wp::MarchingCubes();
    return (uint64_t)(mc);
}

void marching_cubes_destroy_device(uint64_t id)
{   
    if (!id)
        return;

    wp::MarchingCubes* mc = (wp::MarchingCubes*)(id);
    wp::marching_cubes_free(*mc);
    delete mc;
}


WP_API int marching_cubes_surface_device(
    uint64_t id,
    const float* field, 
    int nx, 
    int ny, 
    int nz, 
    float threshold, 
    wp::vec3* verts, 
    int* triangles, 
    int max_verts, 
    int max_tris, 
    int* out_num_verts, 
    int* out_num_tris)
{

    if (!id)
        return -1;

    if (!field)
        return -1;
    
    wp::MarchingCubes& mc = *(wp::MarchingCubes*)(id);

    // reset counts
    *out_num_verts = 0;
    *out_num_tris = 0;

    // resize temporary memory
    marching_cubes_resize(mc, nx, ny, nz, max_verts);

    // create vertices
    wp_launch_device(wp::count_cell_verts, mc.num_cells, (mc, field, threshold) );

    int num_last;		
    memcpy_d2h(&num_last, &mc.first_cell_vert[mc.num_cells - 1], sizeof(int));

    thrust::exclusive_scan(
        thrust::device_ptr<int>(mc.first_cell_vert),
        thrust::device_ptr<int>(mc.first_cell_vert + mc.num_cells),
        thrust::device_ptr<int>(mc.first_cell_vert));

    int num_verts;
    memcpy_d2h(&num_verts, &mc.first_cell_vert[mc.num_cells - 1], sizeof(int));
    synchronize();
    
    num_verts += num_last;

    // check we have enough storage, if not then
    // return required vertex buffer size, let user resize
    if (num_verts > max_verts)
    {
        *out_num_verts = num_verts;
        return -1;
    }

    // if (mc.numVerts > mc.maxVerts) 
    // {
    // 	mc.maxVerts = 2 * mc.numVerts;
    // 	freeDeviceBuffers(mc.verts, mc.normals);
    // 	allocDeviceBuffer(mc.verts, mc.maxVerts);
    // 	allocDeviceBuffer(mc.normals, mc.maxVerts);

    // 	freeDeviceBuffers(mc.avg_verts, mc.avg_num);
    // 	allocDeviceBuffer(mc.avg_verts, mc.maxVerts);
    // 	allocDeviceBuffer(mc.avg_num, mc.maxVerts);
    // }

    // create vertices
    wp_launch_device(wp::create_cell_verts, mc.num_cells, (mc, verts, NULL, field, threshold));

    // create triangles
    wp_launch_device(wp::count_cell_tris, mc.num_cells, (mc, field, threshold));

    
    memcpy_d2h(&num_last, &mc.first_cell_tri[mc.num_cells - 1], sizeof(int));

    thrust::exclusive_scan(
        thrust::device_ptr<int>(mc.first_cell_tri),
        thrust::device_ptr<int>(mc.first_cell_tri + mc.num_cells),
        thrust::device_ptr<int>(mc.first_cell_tri));


    int num_indices;
    memcpy_d2h(&num_indices, &mc.first_cell_tri[mc.num_cells - 1], sizeof(int));
    synchronize();

    num_indices += num_last;

    int num_tris = num_indices/3;

    if (num_tris > max_tris)
    {
        *out_num_tris = num_tris;
        return -1;
    }

    // if (mc.numTriIds > mc.maxTriIds) {
    // 	mc.maxTriIds = 2 * mc.numTriIds;
    // 	freeDeviceBuffers(mc.triIds);
    // 	allocDeviceBuffer(mc.triIds, mc.maxTriIds);
    // }

    wp_launch_device(create_cell_tris, mc.num_cells, (mc, field, triangles, threshold));


    *out_num_verts = num_verts;
    *out_num_tris = num_tris;

    return 0;

    // // smooth verts

    // int numTris = mc.numTriIds / 3;

    // for (int i = 0; i < 3; i++) {
    // 	clearDeviceBuffer(mc.avg_num, mc.numVerts);
    // 	clearDeviceBuffer(mc.avg_verts, mc.numVerts);

    // 	smooth_verts << < numTris / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
    // 	average_verts << < mc.numVerts / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
    // }

    // compute normals

    // clearDeviceBuffer(mc.normals, mc.numVerts);

    // compute_normals << < numTris / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
    // normalize_normals << < mc.numVerts / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

    //for (int i = 0; i < 5; i++)
    //	smooth_normals <<<  numTris / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);

    //normalize_normals << < mc.numVerts / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (data);
}
