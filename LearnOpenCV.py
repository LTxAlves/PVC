import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog as fd
from sys import exit
import numpy as np
from math import sqrt

master = tk.Tk()

def quit_event():
    master.quit()
    master.destroy()

master.protocol("WM_DELETE_WINDOW", quit_event)

choice = tk.StringVar()
choice.set("BWImg")

def callback(value):
    global choice
    choice = value

def coloredImageClick(img, b, g, r):
    print("Color values: R = " + str(r) + ", G = " + str(g) +  ", B = " + str(b))
    aux = img
    value_clicked = np.array([b, g, r])
    aux = np.square(np.subtract(aux.astype('int32'), value_clicked.astype('int32')))
    aux = np.sum(aux, axis=2, keepdims=True)
    aux = np.sqrt(aux)
    aux = aux < 13
    newimg = (np.where(aux, (0, 0, 255), img)).astype(img.dtype)
    cv2.imshow('New Image', newimg)

def grayscaleImageClick(img, px):
    print("Pixel intensity: " + str(px))
    bools = (np.absolute(img.astype('int32') - px) < 13)
    bools = np.reshape(bools,(img.shape[0], img.shape[1], 1))
    newimg = np.reshape(img,(img.shape[0], img.shape[1], 1))
    newimg = np.repeat(newimg, 3, axis=2)
    newimg = (np.where(bools, (0, 0, 255), newimg)).astype(img.dtype)
    cv2.imshow('New Image', newimg)

def imgClickEvent(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Coordinates:\tx (column) = ' + str(x) + '\ty (row) = ' + str(y))
        if colored:
            b, g, r = img[y, x]
            coloredImageClick(img, b, g, r)
        else:
            px = int(img[y, x])
            grayscaleImageClick(img, px)

b = tk.Radiobutton(master, text="Grayscale Image", variable=choice, command=lambda *args: callback("BWImg"))
b.pack(anchor=tk.W)
b = tk.Radiobutton(master, text="Color Image", variable=choice, command=lambda *args: callback("RGBImg"))
b.pack(anchor=tk.W)
b = tk.Radiobutton(master, text="Grayscale Video", variable=choice, command=lambda *args: callback("BWVid"))
b.pack(anchor=tk.W)
b = tk.Radiobutton(master, text="Color Video", variable=choice, command=lambda *args: callback("RGBVid"))
b.pack(anchor=tk.W)
b = tk.Radiobutton(master, text="Webcam Video", variable=choice, command=lambda *args: callback("CamVid"))
b.pack(anchor=tk.W)
b.wait_variable(choice)

isImg = True

if "Img" in choice:
    file_chosen = fd.askopenfilename(title="Choose a file", filetypes=[('image', ('.jpg', '.jpeg'))])
elif "Vid" in choice:
    file_chosen = fd.askopenfilename(title="Choose a file", filetypes=[('video', ('.avi', '.264'))])
    isImg = False
else:
    file_chosen = None
    isImg = False

flag = cv2.IMREAD_COLOR
colored = True

if "BW" in choice:
    colored = False
    flag = cv2.IMREAD_GRAYSCALE

if isImg:
    img = cv2.imread(file_chosen, flag)
    if img is None:
        print("Error opening image")
        exit()
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', imgClickEvent)

    print("Press 'q' to quit!")

    while cv2.waitKey(1) != ord('q'):
        if cv2.getWindowProperty('Image', 0) < 0:
            break
    
    print('Exiting program...')
    cv2.destroyAllWindows()
    exit()

elif file_chosen is not None:
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
            if frame_counter == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if (cv2.waitKey(25) & 0xFF == ord('q')) or (cv2.getWindowProperty('Video', 0) < 0):
                break
    vid.release()
    cv2.destroyAllWindows()
    exit()
