import cv2
import numpy as np
from math import sqrt
from time import time
from datetime import datetime
from pandas import DataFrame
from os.path import exists
from os import remove

def ClickEvent(event, x, y, flags, param): #callback event for mouse events
    if event == cv2.EVENT_LBUTTONDOWN: #left click on image/video

        param.append((x, y)) # appends coordinates to array
        if len(param) == 2: #calculates distance of 2 pairs of coordinates on array
            size = sqrt((param[0][0] - param[1][0])**2 + (param[0][1] - param[1][1])**2)
            print('Line length = %.5f'%(size), 'pixels')

        if len(param) == 3: # every third click counts as a 1st click
            param.clear()
            param.append((x, y))

'''
Rotina a seguir adaptada a partir da disponível em:
https://stackoverflow.com/questions/18574108/how-do-convert-a-pandas-dataframe-to-xml
Autor: Andy Hayden - 2 de setembro de 2013
Recebe dataframe do módulo pandas, cria campos a partir de cada linha do dataframe e salva no arquivo
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
            xml.append('\t\t<{0}>{1}</{0}>'.format(col_name, row.iloc[i]))
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

rawClicks = [] #coordinates of clicks on raw image
udClicks = [] #coordinates of clicks on undistorted images

'''
Código a seguir baseado no disponível na plataforma Aprender
(https://aprender.ead.unb.br/mod/resource/view.php?id=282863).
Adaptado conforme interesses do Projeto.
'''

def calibration(WebCam, square_size, board_h, board_w, time_step, max_images):
    '''
    Metodo para calibrar a camera, utiliza o padrao de calibracao forncido (pattern.pdf) para 
calcular a matriz dos parametros intrinsecos da camera e os parametros de distorcao da mesma

    Parametros:
        -WebCam: Objeto do openCV que abriu a webcam do computador
        -square_size: Tamanho (mm) do quadrado no padrao impresso
        -board_h: Quantidade de intersecoes entre 4 quadrados na vertical 
        -board_w: Quantidade de intersecoes entre 4 quadrados na horizontal 
        -time_step: Tempo (s) de espera entre deteccoes para poder movimentar o padrao 
        -max_images: Numero total de fotos tiradas do padrao para fazer a calibracao

    Retorno:
        -mtx: matriz dos parametros intrinsecos da camera calculados na calibracao
        -dist: parametros de distorcao da camera calculados na calibracao
    '''
    # stop criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, square_size, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_h*board_w,3), dtype='float32')
    objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D points in image plane.

    #Set start time for the detections
    start_time = time()

    # counter for number of detected images
    detected_images = 0

    while detected_images != max_images:
        # saves time elapsed since last chessboard image capture (time_step condition)
        elapsed = time() - start_time

        grab, img = WebCam.read()
        if not grab or img is None:
            break

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        cv2.imshow('Webcam', img)

        # finds chessboard patter corners on image
        ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

        # If found, add object points, image points (after refining them)
        if ret == True and elapsed > time_step:
            detected_images += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # shows user chessboard corners found
            img = cv2.drawChessboardCorners(img, (board_w,board_h), corners2,ret)
            cv2.imshow('Corners Detected',img)

            # resests star_time after finding pattern (time_step condition)
            start_time = time()

        # pressing q to close program
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    # destroy windows used in caliibration
    cv2.destroyAllWindows()

    # calibrates camera according to parameters found
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    print('Intrinsic parameters matrix:\n{}'.format(mtx))
    print('Distortion parameters:\n{}'.format(dist))

    mtx_df = DataFrame({'Column1' : mtx[:, 0], 'Column2' : mtx[:, 1], 'Column3' : mtx[:, 2]})
    to_xml(mtx_df, filename='calibrationMatrix.xml')
    dist_df = DataFrame({'K1' : dist[:, 0], 'K2' : dist[:, 1], 'P1' : dist[:, 2], 'P2' : dist[:, 3], 'K3' : dist[:, 4]})
    to_xml(dist_df, filename='distortionMatrix.xml')


    return mtx, dist

def correct_distortion(WebCam, mtx, dist):
    '''
    Metodo para corrigir a distorcao na imagem da webcam e mostrar na tela a imagem original da camera e a 
imagem sem distorcao

    Parametros:
        -WebCam: Objeto do openCV que abriu a webcam do computador
        -mtx: matriz dos parametros intrinsecos da camera calculados na calibracao
        -dist: parametros de distorcao da camera calculados na calibracao
    '''

    #Inicializa as janelas raw e undistorted
    cv2.namedWindow('raw')
    cv2.namedWindow('undistorted')
    cv2.setMouseCallback('raw', ClickEvent, param=rawClicks)
    cv2.setMouseCallback('undistorted', ClickEvent, param=udClicks)

    grab, img = WebCam.read()
    h,  w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    # Mapeamento para retirar a distorcao da imagem
    mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    
    print('Press \'q\' to quit/repeat calibration')
    print('Press \'r\' to save raw image in current directory')
    print('Press \'u\' to save undistorted image in current directory')

    while True:
        grab, img = WebCam.read()
        if not grab:
            break

        img = cv2.flip(img, 1)
        
        #remapeamento
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

        #Aperte a tecla 'q' para encerrar o programa
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        elif k == ord('r'):
            cv2.imwrite('capture' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.jpg', img)

        elif k == ord('u'):
            cv2.imwrite('capture' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.jpg', dst)

    cv2.destroyAllWindows()
    udClicks.clear()
    rawClicks.clear()

if __name__ == "__main__":
    # opens computer camera and saves reference to object
    WebCam = cv2.VideoCapture(0)

    # number of images to be used in calculating intrinsic parameters
    max_images = 3

    # number of intersections between 4 spaces on
    # chessboard pattern (horizontally and vertically)
    board_w = 8
    board_h = 6

    # size of each space on chessboard pattern (mm)
    square_size = 28

    # time to wait for user to move chessboard between captures
    time_step = 2

    # number of calibrations for averaging the matrix
    repeats = 2

    for iterator in range(repeats):
        mtx, dist = calibration(WebCam, square_size, board_h, board_w, time_step, max_images)
        correct_distortion(WebCam, mtx, dist)