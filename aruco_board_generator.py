import cv2 as cv
import numpy as np
from PIL import Image

################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Wspolczesne techniki obrazowania geometrii obiektow przestrzennych statycznych i w ruchu
# 6607-CSZ00-DSP-23SL5
#
# Skaner powierzchni wykorzystujacy glowice pomiarowa stykowa z tablica targetow ArUco
#
# Program generujacy tablice kodow ArUco
#
# 2023-06-05
#
################################################################################################


def cm_to_px(centimeters):
    return int((300 * centimeters) / 2.54)    # pixels with resolution of 300 DPI

# wybieramy slownik ArUco
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# generowanie tablicy targetow
board = cv.aruco.GridBoard((2, 2), 1.0, 0.5, aruco_dict)

# rozmiar boku calego wygenerowanego obrazka
arucosize=cm_to_px(2*2+1)

# generowanie obrazka z tablicy
img = board.generateImage((arucosize,arucosize), marginSize=0, borderBits=1)

# skalowanie bo zla drukarka
imgn = cv.resize(img, (int(arucosize*1.01),arucosize), interpolation = cv.INTER_AREA)

# zapis obrazka do png
#cv.imwrite('board.png', imgn)

# generowanie pdf do druku z zachowaniem odpowiedniej skali
page = Image.new(mode="RGB", size=(cm_to_px(21), cm_to_px(29.7)), color=(255, 255, 255))
PIL_image = Image.fromarray(np.uint8(imgn)).convert('RGB')
# przesuniecie na kartce by nie lezal na krawedzi
x = 100   # poziomo
y = 100   # pionowo
page.paste(PIL_image,[0+x,0+y,int(arucosize*1.01)+x,arucosize+y])   # skalowanie bo zla drukarka
page.save('file.pdf', resolution=300)

# podglad jaki obrazek wygenerowal
cv.imshow("board",img)
cv.waitKey(0)



