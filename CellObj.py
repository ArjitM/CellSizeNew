import numpy as np
import cv2
import skimage
from skimage import measure


class Cell:

    def __init__(self, num, labels, zslice):

        self._interior = labels == num
        self._area = np.sum(self._interior)
        self._perimeter = measure.perimeter(self._interior)
        self._score = 4 * 3.14 * self._area / (self._perimeter ** 2)

        self._zslice = zslice
        self._num = num

    @property
    def interior(self):
        return self._interior

    @property
    def area(self):
        return self._area

    @property
    def zslice(self):
        return self._zslice

    @property
    def num(self):
        return self._num

    def contains_or_contained(self, other):
        contains, contained = False, False

        # If one cell is contained by the other, their union (OR operation) will be exactly equal to the containing cell
        both = np.bitwise_or(self._interior, other.interior)
        diff1 = np.bitwise_xor(self._interior, both)
        contains = not np.any(diff1)
        diff2 = np.bitwise_xor(other.interior, both)
        contained = not np.any(diff2)
        # overlap = np.any(np.bitwise_and(self._interior, other.interior))
        return contains, contained

    def contours(self):
        cont, _ = cv2.findContours(self._interior.astype(np.ubyte), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return cont



