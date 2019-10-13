from argparse import ArgumentParser
import numpy as np
import cv2

MAX_DISTANCE_PIANO = 19700

MAX_DISTANCE_PLAYROOM = 85000

def ReadMiddleburyTxt(path):
    f = open(path, 'r')
    text = f.readlines()
    f.close()
    newtext = []
    for line in text:
        newtext.append(line.replace('[', ' ').replace(';', '').replace(']', ' ').replace('=', ' '))

    text = list(map(lambda l: l.split(), newtext))
    intrinsics0 = np.array(text[0][1:], dtype='float').reshape((3, 3))
    intrinsics1 = np.array(text[1][1:], dtype='float').reshape((3, 3))
    doffs = float(text[2][1])
    baseline = float(text[3][1])
    width = int(text[4][1])
    height = int(text[5][1])
    ndisp = int(text[6][1])

    return intrinsics0, intrinsics1, doffs, baseline, width, height, ndisp

def DisparityCalculator(imgLeft, imgRight, minDisp, maxDisp):
    numDisp = maxDisp - minDisp


    if numDisp%16 != 0:
        print('Error! maxDisp - minDisp must be a multiple of 16')
        quit()

    windowSize = 5
    leftMatcher = cv2.StereoSGBM_create(    # from https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
        minDisparity=minDisp,
        numDisparities=numDisp,             # must be a multiple of 16
        blockSize=5,                        # must be odd. Usually from 3 to 11
        P1=(8 * 3 * windowSize ** 2),
        P2=(32 * 3 * windowSize ** 2),      # must be greater than P1
        disp12MaxDiff=1,
        uniquenessRatio=10,                 # percentage for the best value to win the second best. Usually from 5 to 15
        speckleWindowSize=0,                # 0 to disable
        speckleRange=2,                     # noise speckle filtering, 1 or 2 is usually good enough (implicitly multiplied by 16)
        preFilterCap=63,                    # truncation for prefiltered images at range [-val, val]
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)# 3-way sgbm
    

    rightMatcher = cv2.ximgproc.createRightMatcher(matcher_left=leftMatcher)

    displ = np.int16(leftMatcher.compute(imgLeft, imgRight))
    dispr = np.int16(rightMatcher.compute(imgRight, imgLeft))

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=leftMatcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.2)

    filteredImg = wls_filter.filter(displ, imgLeft, None, dispr)
    cv2.filterSpeckles(filteredImg, 0, 4000, maxDisp) 

    filteredImg = np.uint8(cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX))
    return filteredImg

def calcWorldCoordinates(width, height, focalLength, baseline, disp):
    xL = np.tile(np.arange(np.float32(width)), (height, 1))
    yL = np.tile(np.arange(np.float32(height)), (width, 1))
    yR = yL
    xR = xL + disp
    const = baseline/2
    deltaX = xL-xR
    deltaX[deltaX == 0.0] = np.inf
    X = -const*((xL + xR) / deltaX)
    Y = -const*(np.transpose(yL + yR) / deltaX)
    const = baseline * focalLength
    Z = -const / deltaX
    world_coordinates = cv2.merge((X,Y,Z))
    return world_coordinates

def normalizeDepth(worldCoordinates, whichImg):
    X,Y,Z = cv2.split(worldCoordinates)
    if whichImg ==  1:
        Z[Z > MAX_DISTANCE_PIANO] = MAX_DISTANCE_PIANO
    else:
        Z[Z > MAX_DISTANCE_PLAYROOM] = MAX_DISTANCE_PLAYROOM
    
    Z = cv2.normalize(src=Z, dst=Z, beta=254, alpha=0, norm_type=cv2.NORM_MINMAX)
    Z[Z == 0] = 255

    worldCoordinates= np.uint8(cv2.merge((X,Y,Z)))
    return worldCoordinates

def Req1():
    while(True):
        print('Choose an image set to work with:')
        print('\t1) Piano')
        print('\t2) Playroom')
        print('\t3) Quit')
        option = input('-> ')
        valid = True
        if option == '1':    
            intrinsics0, _, _, baseline, width, height, ndisp = ReadMiddleburyTxt('../data/Middlebury/Piano-perfect/calib.txt')

            img0 = cv2.imread('../data/Middlebury/Piano-perfect/im0.png')
            img1 = cv2.imread('../data/Middlebury/Piano-perfect/im1.png')
        elif option == '2':
            intrinsics0, _, _, baseline, width, height, ndisp = ReadMiddleburyTxt('../data/Middlebury/Playroom-perfect/calib.txt')

            img0 = cv2.imread('../data/Middlebury/Playroom-perfect/im0.png')
            img1 = cv2.imread('../data/Middlebury/Playroom-perfect/im1.png')
        elif option == '3':
            quit()
        else:
            print('Invalid choice!')
            valid = False

        if valid:
            disparity = DisparityCalculator(imgLeft=img0, imgRight=img1, minDisp=0, maxDisp=(ndisp - (ndisp%16)))
            
            worldCoordinates = calcWorldCoordinates(width, height, intrinsics0[0][0], baseline, disparity)
            worldCoordinates = normalizeDepth(worldCoordinates, int(option))
            _, _, Z = cv2.split(worldCoordinates)

            if option ==  '1':
                cv2.imwrite('../data/Middlebury/Piano-perfect/disparity.pgm', disparity)
                cv2.imwrite('../data/Middlebury/Piano-perfect/depth.png', Z)
            elif option == '2':
                cv2.imwrite('../data/Middlebury/Playroom-perfect/disparity.pgm', disparity)
                cv2.imwrite('../data/Middlebury/Playroom-perfect/depth.png', Z)

            Z = cv2.resize(Z, (width//4, height//4), interpolation=cv2.INTER_AREA)
            disparity = cv2.resize(disparity, (width//4, height//4), interpolation=cv2.INTER_AREA)

            cv2.imshow('Disparity', disparity)
            cv2.imshow('Depth', Z)
            print('Press any key to continue')
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def Req2():
    pass

def Req3():
    pass

def Req4():
    pass

# main()
parser = ArgumentParser(description='Choice of requirement')
parser.add_argument('--requirement', '-r', action='store', dest='req', default=1, help='Choose a requirement to see in action (1, 2, 3 or 4)')

arguments = parser.parse_args()

if arguments.req == '1':
    Req1()

elif arguments.req == '2':
    Req2()

elif arguments.req == '3':
    Req3()

elif arguments.req == '4':
    Req4()

else:
    print('Error: invalid requirement! Value must be 1, 2, 3 or 4')