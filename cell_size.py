import cv2
import numpy as np
from scipy import ndimage as ndi
import skimage.filters
import multiprocessing
from multiprocessing import Pool
from skimage import measure, io
import argparse
import logging
import traceback
import os
import sys
from skimage.util import img_as_ubyte
import pandas as pd
import os.path

from StackObj import SliceStack, FinalZSlice, ZSlice

MIN_AREA = 100  # sq pixels
MAX_AREA = 600
ROUNDNESS_THRESH = 0.3
DIST_TRANSFORM_FRAC = 0.1


def threshold(img, nucleus_mode, block_size=45):
    # See https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html for thresholding details
    if not nucleus_mode:
        img = np.bitwise_not(img)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 1)
    return thresh


def watershedBoundaries(thresh, img):
    """
    Method to detect and segment objects using watershed technique.
    :param thresh: Thresholded binary image
    :param img:    Original image
    :return:       Labeled objects detected in image
    """
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = cv2.erode(opening, kernel, iterations=3)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    closing = cv2.erode(closing, kernel, iterations=1)

    # sure background area
    sure_bg = closing

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    dt = dist_transform
    _, sure_fg = cv2.threshold(dist_transform, DIST_TRANSFORM_FRAC * dt.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = ndi.gaussian_filter(img, 0.5)
    markers = cv2.watershed(img, markers)

    labels = np.copy(markers)

    # Assign segmentation boundaries to background pixels. Retain original MARKERS.
    labels[markers == -1] = 1

    # Label 1 is bkgd, objects start at 2
    centers = np.zeros(labels.shape, dtype=np.int16)
    labelMax = np.max(labels)

    comByLabel = {}
    if labelMax >= 2:
        for i in range(2, labelMax + 1):

            # Find the center of mass (COM) for each object individually, using a single-value list of labels.
            com = ndi.measurements.center_of_mass(thresh, labels, [i])[0]

            # If COM is undefined, object is unviable. Must be excluded.
            if np.any(np.isnan(com)):
                comByLabel[i] = None
                continue

            com = tuple(np.floor(com).astype(np.int))
            comByLabel[i] = com

            # Label centers of viable "cells" by object number.
            centers[com] = i

    cenVis = np.zeros(centers.shape)
    cenVis[centers > 0] = 255

    kernel3 = np.ones((3, 3))
    cenVis = cv2.morphologyEx(cenVis, cv2.MORPH_CLOSE, kernel3, iterations=5)
    cenVis = ndi.convolve(cenVis, kernel3)

    cenVis = (cenVis > 0).astype(np.int8)
    _, newCOMLabels = cv2.connectedComponents(cenVis)

    relabeled = np.zeros(labels.shape)
    for label in comByLabel.keys():
        com = comByLabel.get(label)
        if com is None:
            continue
        assignToLabel = newCOMLabels[com]
        relabeled[labels == label] = assignToLabel

    finalLabels = np.zeros(labels.shape)
    areas = []
    counter = 1
    # Merge objects whose COMs are close.
    for i in range(int(np.max(relabeled)) + 1):
        clust = (relabeled == i).astype(np.uint8)
        closing = cv2.morphologyEx(clust, cv2.MORPH_CLOSE, kernel3, iterations=1)
        area = np.sum(closing > 0)
        perimeter = skimage.measure.perimeter(closing)
        score = 4 * 3.14 * area / (perimeter ** 2)

        if area < MIN_AREA or area > MAX_AREA or score < ROUNDNESS_THRESH:
            finalLabels[closing > 0] = 0
        else:
            finalLabels[closing > 0] = counter
            counter += 1
            areas.append(area)

    return finalLabels.astype(np.uint8), np.array(areas)


def processStack(prefix):
    """
    Method to process a 3D stack of input images. Assumes images are named ../<prefix>.tif or .jpg
    :param prefix: Directory of images + common name <prefix>
    :return: None
    """

    x = 1
    sliceStack = SliceStack()
    while True:
        try:
            ext = ".tif"
            inFile = prefix + 'piece-' + str(x).rjust(4, '0') + ext
            if not os.path.isfile(inFile):
                inFile = inFile.replace(ext, ".jpg")
                ext = '.jpg'
            nucleusMode = '-rfp-' in inFile or 'ucleus' in inFile
            virus_inject = 'RFP' in inFile
            img = cv2.imread(inFile, cv2.IMREAD_GRAYSCALE)
            if img is None:
                break

            thresh = threshold(img, nucleusMode)
            labels, areas = watershedBoundaries(thresh, img)

            labVis = skimage.color.label2rgb(labels, bg_label=0)
            cv2.imwrite(inFile.replace(ext, "_labels.tif"), img_as_ubyte(labVis))

            # Create a new slice with given labels and add it to the stack of slices]
            sliceStack.addZSlice(ZSlice(labels, x))

            cont, _ = cv2.findContours(img_as_ubyte(labels > 0), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(img, cont, -1, (255, 0, 255), 1)
            cv2.imwrite(inFile.replace(ext, "_contours.tif"), img)

        except IOError:
            if x == 1:
                print("{0} not processed. IO error.".format(prefix))
                return
        else:
            x += 1

    finalZSlice = collate(sliceStack)
    fVis = skimage.color.label2rgb(finalZSlice.labels.astype(np.uint8), bg_label=0)
    cv2.imwrite(prefix + "labels.tif", img_as_ubyte(fVis))

    overlay(finalZSlice, sliceStack, prefix)



def collate(sliceStack, segment=True):
    """
    Given a 3D stack of images whose objects are labeled, save only objects on slices where object area is maximal.
    If an object is segmented into several objects on another slice, either retain the large unsegmented object to
    prevent oversegmentation or break object into smaller objects to address undersegmentation.
    :param sliceStack: 3D stack of images composed of 2D slices
    :param segment: Break large z-axis overlapping objects into smaller objects.
    :return: A constructed 2D Slice containing only objects as seen in their maximal area states.
    """
    finalSlice = FinalZSlice()

    for sliceNum in range(1, sliceStack.numZSlices):

        zslice = sliceStack.zslice(sliceNum)
        for cellNum in range(1, zslice.numCells() + 1):

            overlaps = zslice.getOverlaps(cellNum, finalSlice)
            numOverlaps = len(overlaps)
            cell = zslice.cell(cellNum)

            if numOverlaps == 0:
                # No overlaps indicates the cell is spatially "new"
                finalSlice.addCell(cell)

            elif numOverlaps == 1:
                otherCellNum = overlaps[0]
                otherCell = finalSlice.cell(otherCellNum)
                contains, contained = cell.contains_or_contained(otherCell)

                if (not contains) and (not contained):
                    # If neither contains the other, it is possible the "finalized cell" needs to be
                    # segmented into multiple cells found on this slice.
                    overlaps = finalSlice.getOverlaps(otherCellNum, zslice)
                    if len(overlaps) > 1:
                        if segment:
                            # Remove "finalized cell" and add multiple segmented cells
                            finalSlice.removeCell(otherCell)
                            for ocell in overlaps:
                                finalSlice.addCell(zslice.cell(ocell))
                    else:
                        # Since the cells are 1:1 and neither contains the other, retain the larger of the two
                        if cell.area > otherCell.area:
                            finalSlice.removeCell(otherCell)
                            finalSlice.addCell(cell)

                if contains:
                    finalSlice.removeCell(otherCell)
                    finalSlice.addCell(cell)

                # If CELL is contained by OTHERCELL, no action is necessary since OTHERCELL is already in FINALSLICE.

            else:
                # One cell overlaps with multiple finalized cells.
                if not segment:
                    # Remove smaller "finalized cells" and add large overlapping cell.
                    for cNum in overlaps:
                        finalSlice.removeCell(finalSlice.cell(cNum))
                    finalSlice.addCell(cell)

    return finalSlice


def overlay(finalZSlice, sliceStack, prefix):

    threeDImgs = []
    x = 1
    while True:
        try:
            ext = ".tif"
            inFile = prefix + 'piece-' + str(x).rjust(4, '0') + ext
            if not os.path.isfile(inFile):
                inFile = inFile.replace(ext, ".jpg")
                ext = '.jpg'
            img = cv2.imread(inFile, cv2.IMREAD_COLOR)
            if img is None:
                break
        except IOError:
            break
        else:
            threeDImgs.append(img)
            x += 1
    threeDImgs = np.asarray(threeDImgs)

    for i in range(1, x):
        for cell in sliceStack.zslice(i).cells():
            cont = cell.contours()
            cv2.drawContours(threeDImgs[cell.zslice.num - 1], cont, -1, (255, 255, 255), 1)

    for cell in finalZSlice.cells():
        cont = cell.contours()
        cv2.drawContours(threeDImgs[cell.zslice.num - 1], cont, -1, (255, 0, 255), 1)

    io.imsave(prefix + "Largest3D.tif", threeDImgs)
    df = pd.DataFrame(data={"Areas": np.array([cell.area for cell in finalZSlice.cells()])})
    df.to_excel(prefix + "areas.xlsx")


def one_arg(prefix):
    try:
        processStack(prefix)
    except Exception as e:
        print("Error occured in processing {0}: {1}".format(prefix, e))
        logging.error(traceback.format_exc())
    else:
        print("Processed: ", prefix)



def getImageDirectories(locations):
    prefixes = []

    def recursiveDirectories(loc):
        nonlocal prefixes
        try:
            for d in next(os.walk(loc))[1]:
                if 'normal' in d or '_RFP' in d:
                    prefixes.append(loc + d + '/')
                    # print(loc + d + '/')
                else:
                    recursiveDirectories(loc + d + '/')
        except StopIteration:
            pass

    for loc in locations:
        recursiveDirectories(loc)
    return prefixes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Local or cluster")
    parser.add_argument('-l', '--local', dest="localRun", default=False, action="store_true")
    args = parser.parse_args()

    if args.localRun:

        prefixes = getImageDirectories(["/Users/arjitmisra/Documents/Kramer_Lab/RAW/RAW2/"])
        with Pool(2) as p:
            p.map(one_arg, prefixes)
        #     try:
        #         processStack(p)
        #     except Exception as e:
        #         print("Error occured in processing {0}: {1}".format(p, e))
        #         logging.error(traceback.format_exc())
        #     else:
        #         print("Processed: ", p)
        # processStack("/Users/arjitmisra/Documents/Kramer_Lab/RAW/RAW2excluded/VAF_new_cohort/expt3/nucleus/p3f2_normal/")
        #
        # locations = [
        #     '../vit A/vit_A_free/',
        #     '../Cell-Size-Project/WT/',
        #     '../Cell-Size-Project/RD1-P2X7KO/',
        #     '../Cell-Size-Project/RD1/',
        #     '../VAF_new_cohort/',
        #     '../YFP-RA-viruses/'
        # ]

    else:
        locations = [
            '/global/scratch/arjitmisra/2-14/vit_A_free/',
            '/global/scratch/arjitmisra/2-14/WT/',
            '/global/scratch/arjitmisra/2-14/RD1-P2X7KO/',
            '/global/scratch/arjitmisra/2-14/RD1/',
            '/global/scratch/arjitmisra/2-14/VAF_new_cohort/',
            '/global/scratch/arjitmisra/YFP-RA-viruses/'
        ]
        prefixes = getImageDirectories(locations)[args.startIndex:args.endIndex]
        cpus = multiprocessing.cpu_count()
        with Pool(cpus * 8) as p:
            p.map(one_arg, prefixes)





