import cv2 as cv     # opencv 4.7.x
import numpy as np
import EasyPySpin

################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Wspolczesne techniki obrazowania geometrii obiektow przestrzennych statycznych i w ruchu
# 6607-CSZ00-DSP-23SL5
#
# Skaner powierzchni wykorzystujacy glowice pomiarowa stykowa z tablica targetow ArUco
#
# Program do obslugi glowicy z tablica kodow ArUco
#
# 2023-06-05
#
################################################################################################

def open_PySpin_camera(cam_number, print_output):
    # otwieranie polaczenia z kamera
    cap = EasyPySpin.VideoCapture(cam_number)

    # wyswietlanie aktualnych parametrow pracy
    if print_output:
        print("width",cap.get(cv.CAP_PROP_FRAME_WIDTH))
        print("height",cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        print("AcquisitionResultingFrameRate",cap.get(cv.CAP_PROP_FPS))
        print("AutoExposureEVCompensation",cap.get(cv.CAP_PROP_BRIGHTNESS))
        print("Gain",cap.get(cv.CAP_PROP_GAIN))
        print("ExposureTime",cap.get(cv.CAP_PROP_EXPOSURE))
        print("Gamma",cap.get(cv.CAP_PROP_GAMMA))
        print("DeviceTemperature",cap.get(cv.CAP_PROP_TEMPERATURE))
        print("GammaEnable",cap.get_pyspin_value("GammaEnable"))
        print("DeviceModelName",cap.get_pyspin_value("DeviceModelName"))
        print("DeviceSerialNumber",cap.get_pyspin_value("DeviceSerialNumber"))

    return cap


def set_PySpin_camera_param(cap):
    # ustawianie parametrow pracy
    cap.set_pyspin_value("AcquisitionFrameRateEnable", True)
    cap.set_pyspin_value("AcquisitionFrameRate", 5.0)
    cap.set_pyspin_value("GammaEnable", True)
    cap.set_pyspin_value("Gamma", 0.4)


def show_frame(window_name, frame, scale_output):
    # wyswietlanie przeskalowanego obrazka
    width = int(frame.shape[1] * scale_output)
    height = int(frame.shape[0] * scale_output)
    dim = (width, height)
    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    cv.imshow(window_name, resized)


def corner2point(corner):
    ptemp = corner
    itemp = ptemp.astype(int)
    point1 = tuple(itemp)
    return point1


def draw_aruco_detections(frame, corners, ids, point_size, font, fontScale, color, thickness):
    # zaznaczanie kazdego wykrytego naroznika targetow ArUco

    mark = ids.shape[0]

    for i in range(mark):

        point = corner2point(corners[i][0][0])
        cv.circle(frame, point, point_size, (0, 0, 255), 2)
        # numer targetu ArUco wg slownika
        frame = cv.putText(frame, str(ids[i][0]), point, font, fontScale, color, thickness, cv.LINE_AA)

        point = corner2point(corners[i][0][1])
        cv.circle(frame, point, point_size, (0, 0, 255), 2)

        point = corner2point(corners[i][0][2])
        cv.circle(frame, point, point_size, (0, 0, 255), 2)

        point = corner2point(corners[i][0][3])
        cv.circle(frame, point, point_size, (0, 0, 255), 2)
    else:
        mark = 0

    return mark


def drawAxes(img, corners, imgpts):
    # rysuje glowice pomiarowa na obrazku
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # zakonczenie glowicy pomiarowej
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 1)
    # ramka obejmujaca tablice targetow
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 1)
    return img


# ============================================================
# main ------
# ============================================================

# skalowanie by zmiescilo sie na ekranie
scale_output = 75. / 100.

# parametry rysowanej kropki
point_size = 5

# parametry napisow
font = 0
fontScale = 1
color = (0, 0, 255)
thickness = 1

# ustawianie detektora arUco

# slownik z jakiego skladaja sie targety
arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# rozmiar pojedynczego targetu ArUco
MARKER_SIZE = 20  # centimeters

# tablica targetow jaka szukamy
board = cv.aruco.GridBoard((2, 2), MARKER_SIZE, MARKER_SIZE*0.5, arucoDict)

# inicjalizacja detektora ArUco
arucoParams = cv.aruco.DetectorParameters()
arucoDetector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

# laczenie z camera z PySpin
cap0 = open_PySpin_camera(0, True)

# ustawienie parametrow pracy kamery
set_PySpin_camera_param(cap0)

# laczenie z web camera otwierana przez cv2
# cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# czytanie pliku z parametrami kalibracyjnymi kamery
data = np.load("camera0_parameters.npz")
cameraMatrix0 = data["camMatrix"]
dist0 = data["distCoef"]

# licznik zmierzonych punktow
num = 0

interdist = 0

ttV_zero = []

# glowna petla programu
while True:
    # przechwycenie obrazu z kamer
    ret0, frame0 = cap0.read()

    if ret0:
        # konwersja na skale szarosci
        imgcolor = cv.cvtColor(frame0, cv.COLOR_GRAY2RGB)

        # wykrywanie targetow ArUco
        (corners0, ids0, rejected0) = arucoDetector.detectMarkers(frame0)

        if corners0:
            # jesli znalaz targety

            # odszukanie tablicy targetow
            objPointsB, imgPointsB = board.matchImagePoints(corners0, ids0)

            if objPointsB is not None:
                # jesli znalazl tablice

                # dopasowanie pozy tablicy
                valid, rVecB, tVecB = cv.solvePnP(objPointsB, imgPointsB, cameraMatrix0, dist0)

                # rysowanie osi dla calej tablicy
                cv.drawFrameAxes(imgcolor, cameraMatrix0, dist0, rVecB, tVecB, MARKER_SIZE*2.5)
                print(tVecB[0], tVecB[1], tVecB[2])

                # wspolrzedne zakonczenia glowicy pomiarowej we wspolrzednych tablicy znacznikow w mm
                pX = 24.2   # czerwone
                pY = 24.8   # zielone
                pZ = 60.0   # niebieskie

                # lista punktow do narysowania
                axis = np.float32([[pX, pY, pZ], [pX, pY, pZ], [pX, pY, pZ], [pX, pY, pZ],
                                   [0, 0, 0], [0, 50.0, 0], [50.0, 50.0, 0], [50.0, 0, 0]])

                # projekcja 3D konca glowicy pomiarowej na obrazku
                imgpts, jac = cv.projectPoints(axis, rVecB, tVecB, cameraMatrix0, dist0)

                # rysowanie konca glowicy pomiarowej
                imgcolor = drawAxes(imgcolor, corners0, imgpts)

                # liczenie rzerzywistych wspolrzednych konca glowicy pomiarowej
                # liczenie macierzy rotacji
                rMatB, _ = cv.Rodrigues(rVecB)
                # punkt konca glowicy
                original_point = np.matrix([[pX], [pY], [pZ]])
                # polozenie punktu konca glowicy
                rotated_point = rMatB * original_point + tVecB

        # pokazuje wynikowy obrazek ze wszystkimi dodatkami
        show_frame("video0", imgcolor, scale_output)

    key = cv.waitKey(1)
    if key == ord('q'):
        # koniec
        break

    elif key == ord('s') and ret0:
        # 's' zapis aktualnej pozycji i obrazka
        print(cv.imwrite('pomiar_' + str(num) + '.png', frame0))
        print(f"image{num}.png saved!")
        # zapis pozycji
        f = open("pomiar.txt", "a")
        f.write(str(num) + " " + format(float(rotated_point[0]), '.1f')
                + " " + format(float(rotated_point[1]), '.1f')
                + " " + format(float(rotated_point[2]), '.1f') + "\n")
        f.close()

        # licznik pomiarow
        num += 1

# czysczenie okien i polaczen
cap0.release()
cv.destroyAllWindows()
