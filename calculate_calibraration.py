import cv2 as cv
import numpy as np
import glob


################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Wspolczesne techniki obrazowania geometrii obiektow przestrzennych statycznych i w ruchu
# 6607-CSZ00-DSP-23SL5
#
# Skaner powierzchni wykorzystujacy glowice pomiarowa stykowa z tablica targetow ArUco
#
# Program obliczajacy parametry kalibracyjne kamery
#
# 2023-06-05
#
################################################################################################


# wyswietlanie przeskalowanego obrazka
def show_frame(window_name, frame, scale_output):
    width = int(frame.shape[1] * scale_output)
    height = int(frame.shape[0] * scale_output)
    dim = (width, height)
    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    cv.imshow(window_name, resized)


#============================================================
# main ------
#============================================================

# skalowanie by zmiescilo sie na ekranie
scale_output = 50. / 100.

# tyle przeciec kratki szukamy
chessboardSize = (8,6)

# rozmiar kratki w mm
size_of_chessboard_squares_mm = 30

# rozmiar obrazow z kamery
frameSize = (1920, 1200)

# kryteria dopasowania
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# przygotowanie tablicy
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * size_of_chessboard_squares_mm

# tablice do przechowywania object points i image points
objpoints0 = []  # 3d point in real world space
imgpoints0 = []  # 2d points in image plane.

# licznik dobrych obrazkow
nok = 0

# tworzenie listy obrazkow
images0 = sorted(glob.glob('images/camera0/*.png'))

for image0 in images0:

    print(f"wczytanie obrazka {image0}..")
    img0 = cv.imread(image0)

    # zamiana na skale szarosci
    imgGray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
#   kamera i tak daje mono

    # odszukanie przeciec na tablicy
    ret0, corners0 = cv.findChessboardCorners(imgGray0, chessboardSize, None)

    # jesli znalazl tablice wyznaczanie z dokladnoscia subpikselowa
    if ret0:

        objpoints0.append(objp)

        # nowa lista pozycji przeciec w tablicy
        corners0 = cv.cornerSubPix(imgGray0, corners0, (11,11), (-1,-1), criteria)
        imgpoints0.append(corners0)

        # wyswietlanie tablicy z narysowanymi kolorowymi liniami laczacymi punkty
        cv.drawChessboardCorners(img0, chessboardSize, corners0, ret0)
        show_frame("image0", img0, scale_output)

        nok += 1
        print("dobre ", nok)
        cv.waitKey(1000) # chwila przerwy by dalo sie popatrzec
    else:
        print("nic nie znalazl na obrazku...")
        show_frame("image0", img0, scale_output)
        cv.waitKey(1000)

# sprzatanie okien po wyswietlaniu obrazkow
cv.destroyAllWindows()

# kalibracja kamery

print("Kalibracja kamery...")

ret0, cameraMatrix0, dist0, rvecs0, tvecs0 = cv.calibrateCamera(objpoints0, imgpoints0, frameSize, None, None)

print('rmse:', ret0)
print('camera matrix:\n', cameraMatrix0)
print('distortion coeffs:', dist0)
print('rvecs0:', rvecs0)
print('tvecs0:', tvecs0)

if ret0:
    # zapis dopasowanych parametrow kalibracyjnych
    np.savez(
        "camera0_parameters",
        camMatrix=cameraMatrix0,
        distCoef=dist0,
        rVector=rvecs0,
        tVector=tvecs0,
    )

# wyznaczanie bledu dopasowania

imp_x = []
imp_y = []
imp_e = []
mean_error = 0
for i in range(len(objpoints0)):
    # przelicza punkty na pozycje na obrazku z wykorzystaniem znalezionych parametrow
    imgpoints2, _ = cv.projectPoints(objpoints0[i], rvecs0[i], tvecs0[i], cameraMatrix0, dist0)

    # blad dopasowania
    error = cv.norm(imgpoints0[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)

    # przeglad wszystkich punktow
    for imp0, imp2 in zip(imgpoints0[i], imgpoints2):
        imp_x.append(imp0[0][0])
        imp_y.append(imp0[0][1])
        imp_e.append(cv.norm(imp0, imp2))
        # zapis bledow do sprawdzenia
        f = open("data_calib.txt", "a")
        content = str(imp0[0][0]) + " " + str(imp0[0][1]) + " "  + str(cv.norm(imp0, imp2)) + "\n"
        f.write(content)
        f.close()

    # blad dla obrazka kalibracyjnego
    print("error", i, error)
    mean_error += error

# sredni blad na wszystkich obrazkow kalibracyjnych
print( "total error: {}".format(mean_error/len(objpoints0)) )

# rysowanie bledow na wszystkich obrazkach
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)
ax.scatter(imp_x, imp_y, s=2, c=imp_e, cmap='jet')
plt.show()

