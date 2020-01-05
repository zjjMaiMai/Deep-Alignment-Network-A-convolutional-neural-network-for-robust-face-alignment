from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import cv2


def draw_points(img, points, radius=1, color=(255, 0, 0), thickness=-1):
    img_draw = img.copy()
    draw_shiftbits = 4
    draw_multiplier = 1 << 4

    for p in points:
        p_f = np.round(p * draw_multiplier).astype(np.int)
        cv2.circle(img_draw, tuple(p_f), draw_multiplier * radius, color, thickness,
                   cv2.LINE_AA, draw_shiftbits)
    return img_draw
