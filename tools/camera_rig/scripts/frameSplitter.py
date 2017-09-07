# -*- coding: utf-8 -*-
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
"""
Created: Feb 2017

Undistorts and splits camera frames into: left, front and right views.
This script was only tested with camera frames with 120 degress field of view  
Uses provided intrinsic calibration (in yaml file)

@author: Nikolai Smolyanskiy
"""

#%%
import cv2
import os
import sys
import argparse
import glob
import yaml
import numpy as np
import math
from numpy import cross, eye, dot
from scipy.linalg import expm3, norm
import skimage.transform

# A yaml constructor is for loading from a yaml node.
# This is taken from @misha 's answer: http://stackoverflow.com/a/15942429
def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)

# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def opencv_matrix_representer(dumper, mat):
    mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)
yaml.add_representer(np.ndarray, opencv_matrix_representer)

def readYamlFile(yamlFilePath) :
    with open(yamlFilePath) as yamlFile:
        _ = yamlFile.readline() # skip 1st line since it may be incompatible with python yaml
        data = yaml.load(yamlFile)
    return data


def splitImages(srcPath, dstPath, calibFilePath, imageExt, convertToGrayscale) :
    horizFOVCoeff = 1.0
    vertFOVCoeff = 1.4 # 1.4 is the max coeff that allows us to avoid black boundaries at the bottom of the rotated views
    #sideViewAngleInRadians = 0.52359878 # 30 degrees in radians
    sideViewAngleInRadians = 0.436332 # 25 degrees (views are 60 deg and overlap)
    #newCameraMatrixFOVScale = 0.41 # needed to zoom in to avoid black boundaries in the views
    newCameraMatrixFOVScale = 0.5 # needed to zoom in to avoid black boundaries in the views

    if not os.path.exists(dstPath):
        print "Creating {} directory".format(dstPath)
        os.mkdir(dstPath)
    
    leftViewPath = "{}/lv".format(dstPath)
    if not os.path.exists(leftViewPath):
        print "Creating {} directory".format(leftViewPath)
        os.mkdir(leftViewPath)

    centerViewPath = "{}/cv".format(dstPath)
    if not os.path.exists(centerViewPath):
        print "Creating {} directory".format(centerViewPath)
        os.mkdir(centerViewPath)

    rightViewPath = "{}/rv".format(dstPath)
    if not os.path.exists(rightViewPath):
        print "Creating {} directory".format(rightViewPath)
        os.mkdir(rightViewPath)

    calibData = readYamlFile(calibFilePath)
    imageWidth = calibData['FrameWidth']        
    imageHeight = calibData['FrameHeight']
    K = np.array(calibData['CameraMatrix'])
    Distort = np.array(calibData['DistortionCoeffs'])
    print "Loaded intrinsics from {}. Camera matrix and distortion coeffs:".format(calibFilePath)
    print K
    print Distort
    print "Image dimensions: {},{}".format(imageWidth, imageWidth)
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, Distort, (imageWidth, imageHeight), 
        np.identity(3), fov_scale=newCameraMatrixFOVScale)
    print "Estimated new camera matrix (after undistortion) as:"
    print newK

    print "Creating mapping tables for left, center and right views"
    mapX_L, mapY_L = cv2.fisheye.initUndistortRectifyMap(K, Distort, np.array([0, sideViewAngleInRadians, 0]), 
        newK, (int(float(imageWidth*horizFOVCoeff)), int(float(imageHeight*vertFOVCoeff))), cv2.CV_32FC1)
    mapX_C, mapY_C = cv2.fisheye.initUndistortRectifyMap(K, Distort, np.identity(3), 
        newK, (int(float(imageWidth*horizFOVCoeff)), int(float(imageHeight*vertFOVCoeff))), cv2.CV_32FC1)
    mapX_R, mapY_R = cv2.fisheye.initUndistortRectifyMap(K, Distort, np.array([0, -sideViewAngleInRadians, 0]), 
        newK, (int(float(imageWidth*horizFOVCoeff)), int(float(imageHeight*vertFOVCoeff))), cv2.CV_32FC1)
    
    print "Creating left, center, right views by remapping..."
    fileSearchPattern = "{}/*.{}".format(srcPath, imageExt)
    for inFilePath in glob.glob(fileSearchPattern) :
        print "Reading {}".format(inFilePath)
        pathParts = os.path.split(inFilePath)
        imageFileName = pathParts[1]
        image = cv2.imread(inFilePath)

        if convertToGrayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # cut left view
        imageView = cv2.remap(image, mapX_L, mapY_L, cv2.INTER_LINEAR)
        outFilePath = "{}/{}".format(leftViewPath, imageFileName)
        print "Writing {} file".format(outFilePath) 
        cv2.imwrite(outFilePath, imageView)

        # cut center view
        imageView = cv2.remap(image, mapX_C, mapY_C, cv2.INTER_LINEAR)
        outFilePath = "{}/{}".format(centerViewPath, imageFileName)
        print "Writing {} file".format(outFilePath) 
        cv2.imwrite(outFilePath, imageView)

        # cut right view
        imageView = cv2.remap(image, mapX_R, mapY_R, cv2.INTER_LINEAR)
        outFilePath = "{}/{}".format(rightViewPath, imageFileName)
        print "Writing {} file".format(outFilePath) 
        cv2.imwrite(outFilePath, imageView)

    return       
    
def main(argv):
    parser = argparse.ArgumentParser(
        prog="frameSplitter",
        usage="frameSplitter.py <inputDir> <outputDir> <calibration> [-g|--grayscale] [-e|--ext IMAGE_FILE_EXTENSION]",
        description="Splits wide angle frames into left, center and right view images"
    )
    
    parser.add_argument("inputdir", type=str, help="path to an input directory")
    parser.add_argument("outputdir", type=str, help="path to an output directory")
    parser.add_argument("calibration", type=str, help="path to yml calibration file with camera intrinsics")
    parser.add_argument("-g", "--grayscale", action='store_true', default=False, help="if specified, converts images to grayscale")   
    parser.add_argument("-e", "--ext", type=str, default="jpg", help="read images of EXT type/extension")   

    args = parser.parse_args()

    print ""
    print "Reading {} images from {} and writing them out to {} directory".format(args.ext, args.inputdir, args.outputdir)    
    if args.grayscale :
        print "Converting images to grayscale..."
    print ""

    splitImages(args.inputdir, args.outputdir, args.calibration, args.ext, args.grayscale)

    print "Done!"

if __name__ == "__main__":
   main(sys.argv[1:])

