import numpy as np
import skimage
import cv2
from CellObj import Cell

MAX_MULT = 10**5


class SliceStack:
    """
    Convenience class to store all ZSlices.
    """

    def __init__(self):
        self._slices = {}

    def addZSlice(self, zslice):
        self._slices[zslice.num] = zslice

    def zslice(self, num):
        return self._slices.get(num)

    @property
    def numZSlices(self):
        return len(self._slices.keys())


class ZSlice:
    """
    A single slice along the Z-axis for a 3D image stack.
    """

    def __init__(self, labels, num):
        self._labels = labels
        self._cells = {}
        self._num = num
        self.makeCells()

    @property
    def labels(self):
        return self._labels

    @property
    def num(self):
        return self._num

    def cell(self, cellNum):
        return self._cells.get(cellNum)

    def makeCells(self):
        for i in range(1, np.max(self._labels) + 1):
            self._cells[i] = Cell(i, self._labels, self)

    def numCells(self):
        return np.max(self._labels)

    def cells(self):
        return self._cells.values()

    '''
    Find which cells in OTHER slice overlap with cell #cellNum in THIS slice.
    '''
    def getOverlaps(self, cellNum, other):
        # Mask is 1 for cell  interior, 0 otherwise
        mask = self._cells.get(cellNum).interior.astype(np.ubyte)
        interact = mask * other.labels
        overlaps = np.unique(interact)
        return overlaps[overlaps != 0]


class FinalZSlice(ZSlice):
    """
    Constructed ZSlice where only finalized cells are stored. Each cell is identifiable by slice of origin and its
    number therein.
    """
    def __init__(self):
        labels = np.zeros((512, 512)).astype(np.uint32)
        super().__init__(labels, -1)

    def addCell(self, cell):
        # Specify cell slice number and cell number in slice with a single value.
        # E.g. Cell 53 on slice 12 with MAX_MULT = 100,000 renders value 120,0053
        cellNum = cell.zslice.num * MAX_MULT + cell.num
        self._labels[cell.interior] = cellNum
        self._cells[cellNum] = cell

    def removeCell(self, cell):
        # Specify cell slice number and cell number in slice with a single value.
        # E.g. Cell 53 on slice 12 with MAX_MULT = 100,000 renders value 120,0053
        cellNum = cell.zslice.num * MAX_MULT + cell.num
        self._labels[cell.interior] = 0
        self._cells.pop(cellNum)







