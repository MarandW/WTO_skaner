import cv2 as cv
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
# Program program zbierajacy obrazy tablicy kalibracyjnej
#
# 2023-06-05
#
################################################################################################


def open_PySpin_camera(cam_number, print_output):
    # polaczenie z kamera
    cap = EasyPySpin.VideoCapture(cam_number)

    # podanie aktualnych parametrow pracy
    if print_output :
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
    # ustawienie parametrow pracy kamery
    cap.set_pyspin_value("AcquisitionFrameRateEnable", True)
    cap.set_pyspin_value("AcquisitionFrameRate", 5.0)
    cap.set_pyspin_value("GammaEnable", True)
    cap.set_pyspin_value("Gamma", 0.4)

def show_frame(window_name, frame, scale_output):
    # wyswietlenie przeskalowanego obrazka z kamery
    width = int(frame.shape[1] * scale_output)
    height = int(frame.shape[0] * scale_output)
    dim = (width, height)
    resized = cv.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv.imshow(window_name, resized)

#============================================================
# main ------
#============================================================

# skalowanie by zmiescilo sie na ekranie
scale_output = 50. / 100.

# laczenie z camera przez PySpin
cap0 = open_PySpin_camera(0, True)

# ustawenie parametrow pracy kamery
set_PySpin_camera_param(cap0)

# lacznie z web camera otwierana przez cv2
#cap0 = cv2.VideoCapture(0)

# licznik obrazkow kalibracyjnych
num = 0

# maksymalna liczba obrazkow kalibracyjnych
maxnum = 40

while cap0.isOpened():

    # pobranie obrazka z kamery
    ret0, frame0 = cap0.read()

    if ret0:
        # wyswietlenie obrazka z kamery
        show_frame("video0", frame0, scale_output)

    k = cv.waitKey(1)
    if k == ord('q') or num == maxnum:
        # koniec zbierania obrazkow
        break
    elif k == ord('s')  and ret0:
        # 's' zapis obrazka z kamery
        print(cv.imwrite('images/camera0/image' + str(num) + '.png', frame0))
        print(f"image{num}.png saved!")
        num += 1

# sprzatanie polaczeni i okien
cap0.release()
cv.destroyAllWindows()