# -*- coding: utf-8 -*-
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
"""
Created on Feb 8 21:40:14 2017

@author: Nikolai Smolyanskiy
"""
# How to run (example): 
# python videoParser.py ./pathToVideoFile.mp4 ./pathToframes -p video1. -e png -c 3

import cv2
import os
import sys
import argparse
import imageio

def breakVideoIntoFrames(videoFilePath, frameDirectory, namePrefix, extension, skipCount) :
    pathParts = os.path.split(videoFilePath)
    videoFileName = pathParts[1].replace(".", "_")
    print "Input file path: {}".format(videoFilePath)
    print "Parsing {} video file".format(videoFileName)    
    print "Using OpenCV version: {}".format(cv2.__version__)    
    
    videoReader = imageio.get_reader(videoFilePath,  'ffmpeg')
    #vidcap = cv2.VideoCapture(videoFilePath)
    #frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameCount = videoReader.get_length()
    print "Reading {} frames".format(frameCount)

    baseDir = "{}".format(frameDirectory)
    if not os.path.exists(baseDir):
        print "Creating {} directory".format(baseDir)
        os.mkdir(baseDir)

    print "Reading every {} frame".format(skipCount)
    count = 0
    success = True
    for i in range(0, frameCount-1) :  # somehow ffmpeg does not like to read the very last frame!
    #for i, image in enumerate(videoReader):
        #for attemptCount in range(0, 2000) : # need to loop here to wait for the reader to be ready to read a frame
        #    success, image = vidcap.read()
        #    if success:
        #        break
        
        #image = videoReader.get_data(i)    
    
        if not success:
            print "Cannot read frame {}".format(i)
            break

        if i % skipCount == 0 :
            print "Reading frame {}".format(i)
            image = videoReader.get_data(i)
            frameFilePath = "{}/{}.{}{:04d}.{}".format(baseDir, videoFileName, namePrefix, count, extension)
            print "Writing it as {} file".format(frameFilePath) 
            # convert to opencv BGR from RGB
            image = image[:,:,::-1]
            cv2.imwrite(frameFilePath, image)     
            count += 1
      
    videoReader.close()
    
    return

def main(argv):
    parser = argparse.ArgumentParser(
        prog="videoParser",
        usage="videoParser.py <inputVideoFile> <outputDir> [-p|--prefix outFrameNamePrefix] [-c|--skipcount inputFrameSkipCount] [-e|--ext outFrameExtension]",
        description="Parses a given video file and breaks it into individual frames")
    
    parser.add_argument("video", type=str, help="path to an input video file")
    parser.add_argument("outdir", type=str, help="path to a directory for parsed output frames")
    parser.add_argument("-p", "--prefix", type=str, default="Frame.", help="prefix output frame names with PREFIX")
    parser.add_argument("-c", "--skipcount", type=int, default=1, help="write out every SKIPCOUNT frame")   
    parser.add_argument("-e", "--ext", type=str, default="jpg", help="use EXT as output frame type/extension")   

    args = parser.parse_args()

    print ""
    print "Reading video file {} and writing frames to {} directory".format(args.video, args.outdir)
    print "Each frame file has {} prefix and {} format/extension, writing out every {} frame".format(args.prefix, args.ext, args.skipcount)
    print ""

    breakVideoIntoFrames(args.video, args.outdir, args.prefix, args.ext, args.skipcount)

if __name__ == "__main__":
   main(sys.argv[1:])

