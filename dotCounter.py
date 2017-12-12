'''
Get area of image covered by bacteria
Created by Shashank Manjunath, December 2017
'''

import os
from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter1d
from scipy.misc import imsave
import numpy as np
import matplotlib.pyplot as plt
from roipoly import roipoly
import pandas as pd


class imageStudy:
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = os.listdir(data_path)

    def getMasks(self, showMask=0):
        # Processes masks for all images in data path
        for file in self.files:
            if 'IMG' in file and 'roi' not in file and 'bw' not in file:
                self.process_img(file, showMask=showMask)

    def countArea(self, csv_fname=None):
        # Counts the area of all images in data path and returns it as a pandas DataFrame
        # csv_fname parameter is the name of a csv file to output to
        imgName = []
        area = []
        for file in self.files:
            if 'IMG' in file and 'bw' in file:
                imgName += [file[3:]]
                area += [self.countImgArea(file)]

        peakArea = {'Image': imgName,
                    '% Area': area}
        peakArea = pd.DataFrame(peakArea, columns=['Image', '% Area'])

        if csv_fname != None:
            peakArea.to_csv(self.data_path+csv_fname)
        return peakArea

    def countImgArea(self, fname):
        # Counts the area of a particular image pointed to by fname
        fpath = self.data_path + fname
        bw_file = imread(fpath)
        bw_file = bw_file > 0
        area = np.sum(bw_file.ravel())
        total = bw_file.shape[0]*bw_file.shape[1]
        pctArea = (float(area)/float(total))*100
        return pctArea

    def process_img(self, fname, showMask=0):
        # Creates a mask for the image pointed to by fname
        fpath = self.data_path + fname
        img_file = imread(fpath)
        img_file = np.sum(img_file, 2).astype(np.float64)
        img_file = normalize_image(img_file)

        roi_name = 'roi_' + fname[:-5] + '.npy'
        bw_name = 'bw_' + fname[:-5] + '.jpg'

        if roi_name not in os.listdir(self.data_path):
            print('ROI file not found. Please manually define ROI.')
            plt.imshow(img_file)
            roi = roipoly()
            img_roi = roi.getMask(img_file)

            np.save(roi_name, img_roi)
        else:
            img_roi = np.load(self.data_path + roi_name)

        maskImg = img_roi * img_file

        filtImg = np.zeros(maskImg.shape)

        for i in range(maskImg.shape[0]):
            filtImg[i, :] = processRow(maskImg[i, :], 0.5)

        bw_img = filtImg > 0.65
        imsave(self.data_path + bw_name, bw_img)

        if showMask != 0:
            plt.imshow(filtImg > 0.65, cmap='gray')
            plt.show()


def normalize_image(img): # Normalizes the image
    norm_image = img/float(400)
    return norm_image


def findFirstGreater(list, val): # Finds first value in the list greater than the value supplied
    for i in range(len(list)):
        if list[i] > val:
            return i
    return -1


def processRow(rowDat, avgVal):
    # Processes row such that the average pixel intensity of the row is equal to avgVal

    if np.sum(rowDat) < 0.1:
        return rowDat


    ydat = gaussian_filter1d(rowDat[rowDat > 0], sigma=3)
    xdat = np.arange(0, ydat.shape[0], 1)

    p = np.poly1d(np.polyfit(xdat, ydat, 2))
    predY = p(xdat)
    xOffset = findFirstGreater(rowDat, 0.25)
    filtDat = np.zeros(rowDat.shape)

    for i in range(len(predY)):
        delta = predY[i] - avgVal
        filtDat[i+xOffset] = rowDat[i+xOffset] - delta

    return filtDat
