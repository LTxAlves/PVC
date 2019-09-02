import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog as fd
from sys import exit
import numpy as np
from math import sqrt

master = tk.Tk()

def quit_event(): #closing the Tkinter window event
    master.destroy()
    master.quit()

master.protocol("WM_DELETE_WINDOW", quit_event)

choice = tk.StringVar() #defining variabe
choice.set("BWImg") #initializing it

def callback(value): #callback for tkinter button click
    global choice
    choice = value

def coloredImageClick(img, b, g, r): #function to handle colored image rgb value detection
    aux = img #copies img
    value_clicked = np.array([b, g, r]) #creates array from clicked rgb values
    aux = np.square(np.subtract(aux.astype('int32'), value_clicked.astype('int32'))) #square of each difference: Bi - Bc, Gi - Gc, Ri - Rc
    aux = np.sqrt(np.sum(aux, axis=2, keepdims=True)) < 13 #square root of sum of squares made into array of bools accordint to requirement (< 13)
    newimg = (np.where(aux, (0, 0, 255), img)).astype(img.dtype) #fills according pixels with red
    cv2.imshow('New Image', newimg) #shows copy of img with red values

def grayscaleImageClick(img, px): #function to handle grayscale pixel intensity detection
    bools = (np.absolute(img.astype('int32') - px) < 13).reshape((img.shape[0], img.shape[1], 1)) #checks if absolute value of difference of pixels is < 13
    newimg = (np.reshape(img,(img.shape[0], img.shape[1], 1))).repeat(3, axis=2) #reshapes and copies so image can be colored (fill with red)
    newimg = (np.where(bools, (0, 0, 255), newimg)).astype(img.dtype) #fills according pixels with red
    cv2.imshow('New Image', newimg) #shows copy of img with red values

def ClickEvent(event, x, y, flags, param): #callback event for mouse events
    if event == cv2.EVENT_LBUTTONDOWN: #left click on image/video
        print('Coordinates:\tx (column) = ' + str(x) + '\ty (row) = ' + str(y)) #prints coordinates of click
        if colored: #routine for colored images/videos
            param.clear()
            param.append(img[y, x][0])
            param.append(img[y, x][1])
            param.append(img[y, x][2])
            print('Color values:\tR = ' + str(param[2]) + ',\tG = ' + str(param[1]) +  ',\tB = ' + str(param[0]))
            coloredImageClick(img, param[0], param[1], param[2])
        else: #routine for graysacle images/videos
            param.clear()
            param.append(img[y, x])
            print("Pixel intensity: " + str(param[0]))
            grayscaleImageClick(img, param[0])

button = tk.Radiobutton(master, text="Grayscale Image", variable=choice, command=lambda *args: callback("BWImg"))
button.pack(anchor=tk.W)
button = tk.Radiobutton(master, text="Color Image", variable=choice, command=lambda *args: callback("RGBImg"))
button.pack(anchor=tk.W)
button = tk.Radiobutton(master, text="Grayscale Video", variable=choice, command=lambda *args: callback("BWVid"))
button.pack(anchor=tk.W)
button = tk.Radiobutton(master, text="Color Video", variable=choice, command=lambda *args: callback("RGBVid"))
button.pack(anchor=tk.W)
button = tk.Radiobutton(master, text="Grayscale Webcam Video", variable=choice, command=lambda *args: callback("BWCam"))
button.pack(anchor=tk.W)
button = tk.Radiobutton(master, text="Color Webcam Video", variable=choice, command=lambda *args: callback("RGBCam"))
button.pack(anchor=tk.W)
button.wait_variable(choice) #buttons for user selection

isImg = True #initializing variable to check if is image

if "Img" in choice: #tests to check type of usage
    file_chosen = fd.askopenfilename(title="Choose a file", filetypes=[('image', ('.jpg', '.jpeg'))])
elif "Vid" in choice:
    file_chosen = fd.askopenfilename(title="Choose a file", filetypes=[('video', ('.avi', '.264'))])
    isImg = False
else:
    file_chosen = None
    isImg = False

flag = cv2.IMREAD_COLOR #initializing flag for colored images
colored = True #initializing variable to check if colored or not

param = [] #initializing empty array for mouse callback event

if "BW" in choice: #tests if grayscale instead of colored
    colored = False
    flag = cv2.IMREAD_GRAYSCALE

if isImg: #image routine
    img = cv2.imread(file_chosen, flag)
    if img is None:
        print("Error opening image")
        exit()
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', ClickEvent, param)

    print("Press 'q' to quit!")

    while cv2.waitKey(1) != ord('q'):
        if cv2.getWindowProperty('Image', 0) < 0:
            break
    
    print('Exiting program...')
    cv2.destroyAllWindows()
    exit()

elif file_chosen is not None: #video file routine
    vid = cv2.VideoCapture(file_chosen)
    frame_counter = 0
    print("Press 'q' to quit!")
    while(vid.isOpened()):
        ret, img = vid.read()
        if ret:
            frame_counter += 1
            if not colored:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Video', img)
            cv2.setMouseCallback('Video', ClickEvent, param)

            if colored and param:
                coloredImageClick(img, param[0], param[1], param[2])
            elif param:
                grayscaleImageClick(img, param[0])

            if frame_counter == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if (cv2.waitKey(25) & 0xFF == ord('q')) or (cv2.getWindowProperty('Video', 0) < 0):
                break
    vid.release()
    cv2.destroyAllWindows()
    exit()

else: #webcam video routine
    vid = cv2.VideoCapture(0)
    print("Press 'q' to quit!")
    while(vid.isOpened()):
        ret, img = vid.read()
        if ret:
            if not colored:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.imshow('Video', img)
            cv2.setMouseCallback('Video', ClickEvent, param)

            if colored and param:
                coloredImageClick(img, param[0], param[1], param[2])
            elif param:
                grayscaleImageClick(img, param[0])

            if (cv2.waitKey(25) & 0xFF == ord('q')) or (cv2.getWindowProperty('Video', 0) < 0):
                break
    vid.release()
    cv2.destroyAllWindows()
    exit()