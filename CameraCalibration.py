import cv2
import numpy as np
from time import time
from datetime import datetime
from pandas import DataFrame
from os.path import exists
from os import remove
import xml.etree.ElementTree as ET

IMAGE_SIZE = (640, 360)

def PixelsToReal(u, v):
    inv_intrinsics = np.linalg.inv(avrg_mtx)
    inv_extrinsics = np.linalg.pinv(np.concatenate([avrg_rot, avrg_trans], axis=1))
    point = np.array((u, v, 1))
    remap = np.matmul(inv_intrinsics, point)
    remap = np.matmul(inv_extrinsics, remap)
    return remap / remap[-1]

def DistanceCalc(coord1, coord2):
    distance = coord1 - coord2
    distance = np.sqrt(distance[0]**2 + distance[1]**2 + distance[3]**2)
    return distance

def ClickEvent(event, x, y, flags, param): #callback event for mouse events
    if event == cv2.EVENT_LBUTTONDOWN: #left click on image/video

        param.append((x, y)) # appends coordinates to array
        if len(param) == 2: #calculates distance of 2 pairs of coordinates on array
            size = np.sqrt((param[0][0] - param[1][0])**2 + (param[0][1] - param[1][1])**2)
            print('Line length = %.5f'%(size), 'pixels')

            if finished:
                coord1 = PixelsToReal(param[0][0], param[0][1])
                coord2 = PixelsToReal(param[1][0], param[1][1])
                distance = DistanceCalc(coord1, coord2)
                print('Real distance = %.5f'%(distance))

        if len(param) == 3: # every third click counts as a 1st click
            param.clear()
            param.append((x, y))

'''
Following method adapted from the available on:
https://stackoverflow.com/questions/18574108/how-do-convert-a-pandas-dataframe-to-xml
Author: Andy Hayden - 2nd of september 2013
From a Pandas dataframe, creats fields from each line in it and saves them as a .xml file
'''

def to_xml(df, filename=None, mode='a+'):
    if exists(filename) and iterator == 0: #removes old files when restarting program
            remove(filename)

    if iterator == 0:
        res = '<calibration>\n\t<matrix>\n'
    else:
        res = '\t<matrix>\n'

    def row_to_xml(row):
        xml = ['\t\t<row>']
        for i, col_name in enumerate(row.index):
            xml.append('\t\t\t<value name=\"{0}\">{1}</value>'.format(col_name, row.iloc[i]))
        xml.append('\t\t</row>\n')
        return '\n'.join(xml)
    res += '\n'.join(df.apply(row_to_xml, axis=1))

    res += '\t</matrix>\n'

    if iterator == repeats - 1:
        res += '</calibration>'

    if filename is None:
        return res
    with open(filename, mode) as f:
        f.write(res)

def avg_mtx(filename=None, src='cal'):
    root = ET.parse(filename).getroot()
    data = []
    for val in root.iter('value'):
        data.append(float(val.text))

    data = np.array(data)
    if src == 'cal' or src == 'rot':
        data = np.reshape(data, (repeats, 3, 3))
    elif src == 'dist':
        data = np.reshape(data, (repeats, 5))
    elif src == 'trans':
        data = np.reshape(data, (repeats, 3))

    avg = np.average(data, axis=0)
    std_dev = np.std(data, axis=0, ddof=1)

    if src == 'trans':
        norms = np.linalg.norm(data, axis=1)
        norm_std = np.std(norms, axis=0)
        norms = norms.mean()
        return avg, std_dev, norms, norm_std

    return avg, std_dev

rawClicks = [] #coordinates of clicks on raw image
udClicks = [] #coordinates of clicks on undistorted images

'''
Following code based upon the available on Aprender/Moodle
(https://aprender.ead.unb.br/mod/resource/view.php?id=282863).
Adapted according to project's goals
'''

def calibration(WebCam, square_size, board_h, board_w, time_step, max_images):
    '''
    Method for calibrating the camera using a chessboard pattern;
    calibrates while finding intrinsic parameters, distortion, rotational and translational matrices

    Parameters:
        -WebCam: OpenCV object referencing camera
        -square_size: length (mm) of side of a single square on chessboard pattern
        -board_h: number of intersections between 4 squares counted vertically
        -board_w: number of intersections between 4 squares counted vertically
        -time_step: time (s) to wait between chessboard detection (allows to move patter to new position) 
        -max_images: number of snapshots to be taken of the chessboard pattern

    Returns:
        -mtx: 3x3 intrinsic parameters matrix
        -dist: 1x5 distortion matrix
        -rmtx: 3x3 rotational matrix (average between max_images snapshots)
        -tvecs: 1x3 translational vector (average between max_images snapshots)
    '''
    # stop criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_h*board_w,3), dtype='float32')
    objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

    # map units in mm since we have square_size
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D points in image plane.

    #Set start time for the detections
    start_time = time()

    # counter for number of detected images
    detected_images = 0

    corners_img = None

    while detected_images != max_images:
        # saves time elapsed since last chessboard image capture (time_step condition)
        elapsed = time() - start_time

        grab, img = WebCam.read()
        if not grab or img is None:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Webcam', img)

        # finds chessboard patter corners on image
        ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h), None)

        # If found, add object points, image points (after refining them)
        if ret == True and elapsed > time_step:
            detected_images += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # shows user chessboard corners found
            corners_img = cv2.drawChessboardCorners(img, (board_w,board_h), corners2, ret)
            cv2.imshow('Corners Detected',img)

            # resests star_time after finding pattern (time_step condition)
            start_time = time()

        # press 'q' to close program
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s') and corners_img is not None:
            cv2.imwrite('capture' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.jpg', corners_img)

    # destroy windows used in caliibration
    cv2.destroyAllWindows()

    # calibrates camera according to parameters found
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    rvecs = np.array(rvecs)
    rmtx = np.zeros((3,3))

    for vector in rvecs:
        aux, _ = cv2.Rodrigues(vector)
        rmtx += aux

    rmtx /= max_images
    
    tvecs = np.array(tvecs).mean(axis=0).reshape((1, 3))

    print('Intrinsic parameters matrix:\n{}'.format(mtx))
    print('Distortion parameters:\n{}'.format(dist))
    print('Rotational matrix\n{}'.format(rmtx))
    print('Translational matrix\n{}'.format(tvecs.transpose()))

    mtx_df = DataFrame({'Column1' : mtx[:, 0], 'Column2' : mtx[:, 1], 'Column3' : mtx[:, 2]})
    to_xml(mtx_df, filename='calibrationMatrix.xml')
    dist_df = DataFrame({'K1' : dist[:, 0], 'K2' : dist[:, 1], 'P1' : dist[:, 2], 'P2' : dist[:, 3], 'K3' : dist[:, 4]})
    to_xml(dist_df, filename='distortionMatrix.xml')
    rmtx_df = DataFrame({'Column1': rmtx[:, 0], 'Column2' : rmtx[:, 1], 'Column3' : rmtx[:, 2]})
    to_xml(rmtx_df, filename='rotationMatrix.xml')
    tvec_df = DataFrame({'tx': tvecs[:, 0], 'ty' : tvecs[:, 1], 'tz' : tvecs[:, 2]})
    to_xml(tvec_df, filename='translationMatrix.xml')

    return mtx, dist, rmtx, tvecs

def correct_distortion(WebCam, mtx, dist):
    '''
    Method to correct camera (webcam) distortion while showing the original and the corrected feeds

    Parameters:
        -WebCam: OpenCV object referencing camera
        -mtx: intrinsic parameters matrix found for said camera
        -dist: distortion parameters found for said camera
    '''

    # initialize 'raw' (no correction) and 'undistorted' (corrected) windows
    cv2.namedWindow('raw')
    cv2.namedWindow('undistorted')
    cv2.setMouseCallback('raw', ClickEvent, param=rawClicks)
    cv2.setMouseCallback('undistorted', ClickEvent, param=udClicks)

    grab, img = WebCam.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    h,  w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    # mapping to remove camera distortion
    mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    
    print('Press \'q\' to quit/repeat calibration')
    print('Press \'r\' to save raw image in current directory')
    print('Press \'u\' to save undistorted image in current directory')

    while True:
        grab, img = WebCam.read()
        if not grab:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        # remapping
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        if len(rawClicks) == 1:
            cv2.circle(img, rawClicks[0], 2, (0, 255, 0), thickness=2) #keeps a circle in first click for user reference
        if len(rawClicks) == 2:
            cv2.line(img, rawClicks[0], rawClicks[1], color=(0, 255, 0), thickness=2) #shows line
        cv2.imshow('raw', img)

        if len(udClicks) == 1:
            cv2.circle(dst, udClicks[0], 2, (0, 255, 0), thickness=2) #keeps a circle in first click for user reference
        if len(udClicks) == 2:
            cv2.line(dst, udClicks[0], udClicks[1], color=(0, 255, 0), thickness=2) #shows line
        cv2.imshow('undistorted', dst)

        # press 'q' to quit
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'): # press 'r' to save raw
            cv2.imwrite('raw_capture' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.jpg', img)
        elif k == ord('u'): # press 'u' to save undistorted
            cv2.imwrite('ud_capture' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.jpg', dst)

    cv2.destroyAllWindows()
    udClicks.clear()
    rawClicks.clear()

if __name__ == "__main__":
    # opens computer camera and saves reference to object
    WebCam = cv2.VideoCapture(0)

    # number of images to be used in calculating intrinsic parameters
    max_images = 5

    # number of intersections between 4 spaces on
    # chessboard pattern (horizontally and vertically)
    board_w = 6
    board_h = 8

    # size of each space on chessboard pattern (mm)
    square_size = 28

    # time to wait for user to move chessboard between captures
    time_step = 2

    # number of calibrations for averaging the matrix
    repeats = 5

    finished = False

    for iterator in range(repeats):
        mtx, dist, rmtx, tvecs = calibration(WebCam, square_size, board_h, board_w, time_step, max_images)
        correct_distortion(WebCam, mtx, dist)
    
    finished = True

    avrg_mtx, std_dev = avg_mtx('calibrationMatrix.xml')
    avrg_dist, dist_dev = avg_mtx('distortionMatrix.xml', src='dist')
    avrg_rot, rot_dev = avg_mtx('rotationMatrix.xml', src='rot')
    avrg_trans, trans_dev, avrg_norm, norm_std = avg_mtx('translationMatrix.xml', src='trans')

    avrg_trans = avrg_trans.reshape((3, 1))
    trans_dev = trans_dev.reshape((3, 1))

    print('Average intrinsic parameters matrix:\n{}\n\n'.format(avrg_mtx))
    print('Intrinsic parameters standard deviation:\n{}\n\n'.format(std_dev))
    print('Average distortion parameters:\n{}\n\n'.format(avrg_dist))
    print('Distortion parameters standard deviation:\n{}\n\n'.format(dist_dev))
    print('Average rotation matrix:\n{}\n\n'.format(avrg_rot))
    print('Rotation matrix standard deviation:\n{}\n\n'.format(rot_dev))
    print('Average translation matrix:\n{}\n\n'.format(avrg_trans))
    print('Translation matrix standard deviation:\n{}\n\n'.format(trans_dev))

    print('|t| =', np.linalg.norm(avrg_trans), '+/-', norm_std)

    correct_distortion(WebCam, avrg_mtx, avrg_dist)