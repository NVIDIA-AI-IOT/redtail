
This tool can be used to calibrate camera's intrinsic parameters according to fisheye camera model from OpenCV (ATAN model)

To calibrate do this:

Record chessboard frames for calibration:
- Display chessboard_100mm.pdf on a monitor in landscape mode
- Record a videoclip with a camera that you want to calibrate. Cover most of the camera frustrum
- Run VideoParser.py to 

Run Calibration tool:

MonoCameraCalibration -input=<path to calibration target images dir> -results=<path to results dir>

This will read frames from <path to calibration target images dir>, will calibrate and then will dump results (intrinsic params as yml file and undistorted input images) into <path to results dir>

