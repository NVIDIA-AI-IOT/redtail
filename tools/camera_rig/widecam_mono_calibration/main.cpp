// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

/*
 * main.cpp
 *
 * Wide field of view mono camera calibration tool. 
 * It uses a fisheye camera model (ATAN) from OpenCV
 *
 * Created: Nikolai Smolyanskiy, Feb 2017
 */

#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>

#if defined(__linux__)
#include <dirent.h>
#else
#include "dirent.h"
#endif

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
using namespace cv;

#include "utils.h"

#define MAX_PATH_LENGTH 1024
#define KEY_ESCAPE 27

static unsigned int g_calibrationObjWidth = 9;      // 9 squares in width
static unsigned int g_calibrationObjHeight = 7;     // 7 squares in height
static float g_calibrationObjSquareSize = 0.100f;    // chessboard square size in meters = 100mm

static void ShowHelp()
{
    std::cout << endl <<
        "Calibrates a mono camera\n" <<
        "Usage:\n" <<
        "   CameraCalibration -input=pathToImagesOfCalibrationTarget -results=pathToResults\n\n" <<
        "to calibrate from data in pathToImagesOfCalibrationTarget and to store results (intrinsic params as yml file and undistorted input images) in pathToResults\n";
}

enum APPMODE
{
    APPMODE_UNKNOWN = 0,
    APPMODE_CALIBRATE,
};

int Calibrate(const char* pathToImages, const char* pathToResults);
bool FindCalibrationPattern(Mat& grayImage, Size& patternSize, vector<Point2f>& corners);

// Main program
int main( int argc, char** argv )
{
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    APPMODE appMode = APPMODE_UNKNOWN;
    char* stringParam;
    char pathToImages[MAX_PATH_LENGTH];
    char pathToResults[MAX_PATH_LENGTH];

    //printf("OpenCV version: %s", cv::getBuildInformation().c_str());
    cout << "Using OpenCV version: " << CV_VERSION << endl;

    if(argc>1)
    {
        if (getCmdLineArgumentString(argc, (const char **) argv, "input", &stringParam))
        {
            strcpy(pathToImages, stringParam);

            if (getCmdLineArgumentString(argc, (const char **) argv, "results", &stringParam))
            {
                strcpy(pathToResults, stringParam);
                appMode = APPMODE_CALIBRATE;
            }
        }
    }

    int retCode = 0;
    if(appMode==APPMODE_CALIBRATE)
    {
        retCode = Calibrate(pathToImages, pathToResults);
    }
    else
    {
        ShowHelp();
        retCode = 0;
    }

    return retCode;
}

bool FindCalibrationPattern(Mat& grayImage, Size& patternSize, vector<Point2f>& corners)
{
    bool patternFound = findChessboardCorners(grayImage, patternSize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
    if(patternFound)
    {
        cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1),
                TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    }

    return patternFound;
}

int Calibrate(const char* pathToImages, const char* pathToResults)
{
    assert(pathToImages != NULL && pathToResults != NULL);

    // Enumerate input images
    printf("Calibrating based on jpg images in the folder: %s\n", pathToImages);

    // Enumerate input files
    DIR* dir;
    struct dirent* dirEntry;
    dir = opendir(pathToImages);
    if (dir == NULL)
    {
        printf("ERROR: cannot open directory %s", pathToImages);
        return -1;
    }

    vector<string> inputFileNames;
    while ((dirEntry = readdir(dir)) != NULL)
    {
        std::string fileName = std::string(dirEntry->d_name);
        if ( findSubstrIC(fileName, ".jpg") )
        {
            inputFileNames.push_back(fileName); // TODO: check if it's at the end also!
        }
    }
    closedir(dir);

    if (inputFileNames.size() == 0)
    {
        printf("ERROR: have not found any input jpg frames in directory %s", pathToImages);
        return -1;
    }

    // Go over all image pairs and find corners
    char filePath[MAX_PATH_LENGTH];

    Mat frame;
    sprintf(filePath, "%s/%s", pathToImages, inputFileNames[0].c_str());
    frame = imread(filePath);
    printf("Input image resolution: width=%d, height=%d\n", frame.cols, frame.rows);
    unsigned int frameHeight = frame.rows;
    unsigned int frameWidth = frame.cols;

    Mat grayFrame;

    Size patternSize(g_calibrationObjWidth - 1, g_calibrationObjHeight - 1); // interior number of corners is number of squares - 1
    vector<Point2f> frameCorners; // these will be filled by the detected corners

    vector<vector<Point2d> > imagePoints;
    vector<vector<Point3d> > calibObjectPoints;

    // Fill the 3D coordinates of the calibration object in the model space
    vector<Point3d> calibObject;
    for (unsigned int row = 0; row < g_calibrationObjHeight - 1; row++)
    {
        for (unsigned int col = 0; col < g_calibrationObjWidth - 1; col++)
        {
            calibObject.push_back(Point3d(float(col)*g_calibrationObjSquareSize /* x */, float(row)*g_calibrationObjSquareSize /* y */, 0 /* z */));
        }
    }
    printf("Calibration object has %d points\n", (int)calibObject.size());

    for (vector<string>::iterator it = inputFileNames.begin(); it != inputFileNames.end(); it++)
    {
        frameCorners.clear();

        const char* frameFileName = it->c_str();
        printf("Reading image: %s\n", frameFileName);

        sprintf(filePath, "%s/%s", pathToImages, frameFileName);
        frame = imread(filePath);
        if (frameHeight != frame.rows || frameWidth != frame.cols)
        {
            assert(false);
            printf("ERROR: image %s has different dimensions. All images must have the same size. Bailing...\n", frameFileName);
            return -1;
        }

        cvtColor(frame, grayFrame, CV_RGB2GRAY);
        bool frameCornersFound = FindCalibrationPattern(grayFrame, patternSize, frameCorners);
        if (frameCornersFound)
        {
            drawChessboardCorners(frame, patternSize, Mat(frameCorners), true);
        }
        sprintf(filePath, "%s/%s", pathToResults, frameFileName);
        imwrite(filePath, frame);

        if (!frameCornersFound)
        {
            printf("WARNING: Could not find corners on: %s, it will be skipped!\n", frameFileName);
        }
        else
        {
            printf("Found %d corners\n", (int)frameCorners.size());

            assert(calibObject.size() == frameCorners.size());

            // Add found frame corners to the array of image point arrays
            imagePoints.push_back(vector<Point2d>());
            vector<Point2d>& arrayOfImagePoints = imagePoints.back(); // get just added last empty array from the vector
            for (vector<Point2f>::iterator it = frameCorners.begin(); it != frameCorners.end(); ++it)
            {
                arrayOfImagePoints.push_back(Vec2d(it->x, it->y));
            }

            // Add calibration object points to the array of object point arrays
            calibObjectPoints.push_back(calibObject);
        }
    }

    // Do calibration
    printf("Calibrating camera based on %d images...\n", (int)calibObjectPoints.size());

    Matx33d K;
    Vec4d   Distortion;
    Size frameSize(frameWidth, frameHeight);

    printf("Calibrating intrinsics...\n");
    int cameraCalibFlags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_CHECK_COND | cv::fisheye::CALIB_FIX_SKEW;
    double reprojectionError = fisheye::calibrate(calibObjectPoints, imagePoints, frameSize,
        K, Distortion, cv::noArray(), cv::noArray(), cameraCalibFlags /*, cv::TermCriteria(3, 20, 1e-6)*/);

    printf("Intrinsics calibration is done, camera reprojection error=%f\n", reprojectionError);
    printf("Calibration parameters:\n");
    std::cout << "Camera matrix=" << std::endl << " " << K << endl << endl;
    std::cout << "Distortion coefficients=" << std::endl << " " << Distortion << endl << endl;

    // Store computed intrinsics and extrinsics
    printf("Storing computed intrinsics in calibration.yml...\n");
    sprintf(filePath, "%s/calibration.yml", pathToResults);
    FileStorage fs(filePath, FileStorage::WRITE);
    time_t rawTime;
    time(&rawTime);
    fs << "Date" << asctime(localtime(&rawTime));
    fs << "FrameWidth" << int(frameWidth);
    fs << "FrameHeight" << int(frameHeight);
    fs << "CameraMatrix" << Mat(K);
    fs << "DistortionCoeffs" << Distortion;
    fs.release();
    printf("Calibration parameters have been saved in %s\n\n", filePath);

    // Test computed parameters by un-distorting calibration images
    printf("Testing computed parameters by un-distorting calibration images...\n");

    // Prepare maps in X and Y for undistortion and stereo rectification
    Matx33d R = Matx33d::eye();
    Mat newCameraMatrix;
    fisheye::estimateNewCameraMatrixForUndistortRectify(K, Distortion, frameSize, Matx33d::eye(), newCameraMatrix);
    std::cout << "New camera matrix after undistort and rectify:" << std::endl << " " << newCameraMatrix << std::endl;

    Mat mapX, mapY;
    fisheye::initUndistortRectifyMap(K, Distortion, Matx33d::eye(), newCameraMatrix, frameSize, CV_32FC1, mapX, mapY);

    Mat undistortedImage;
    for(vector<string>::iterator it=inputFileNames.begin(); it!=inputFileNames.end(); it++)
    {
        const char* frameFileName = it->c_str();

        printf("Undistorting: %s\n", frameFileName);

        sprintf(filePath, "%s/%s", pathToImages, frameFileName);
        frame = imread(filePath);

        // Undistort an input frame and dump results
        cv::remap(frame, undistortedImage, mapX, mapY, cv::INTER_LINEAR);
        sprintf(filePath, "%s/undist_%s", pathToResults, frameFileName);
        imwrite(filePath, undistortedImage);
    }

    printf("Done!\n");

    return 0;
}

