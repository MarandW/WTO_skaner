import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Wspolczesne techniki obrazowania geometrii obiektow przestrzennych statycznych i w ruchu
# 6607-CSZ00-DSP-23SL5
#
# Skaner powierzchni wykorzystujacy glowice pomiarowa stykowa z tablica targetow ArUco
#
# Program dopasowujacy plaszczyzne do pomiarow stolu a nastepnie przeliczajacy pomiar profilu
# na nowe wspolrzedne wzgledem powierzchni stolu
#
# 2023-06-05
#
################################################################################################


def make_matrix(v1, v2, v3):
    # wyznaczanie macierzy rotacji do ukladu opartego o trzy punkty z miejscem zerowym w punkcie v1

    a = v2-v1   # wektor osi X
    b = v3-v1   # drugi wektor

#   od wersji zalezy znak na koniec
    c = np.cross(a, b)   # wektor osi Z
#    c = np.cross(b, a)   # wektor osi Z

    # normalizacja wektora osi X
    a2 = a / np.linalg.norm(a)

    if np.linalg.norm(c)>0:
        # normalizacja wektora osi Z
        c2 = c / np.linalg.norm(c)
    else:
        print("v1 v2 v3 sa wspolliniowe")

    # wektor osi Y
    b2 = np.cross(c2, a2)

    # normalizacja wektora osi Y
    b2 = b2 / np.linalg.norm(b2)

    # macierz rotacji
    # znormalizowane wektory X Y Z
    M0 = np.column_stack((a2, b2, c2))
    return M0


def project_point_plane(point, plane):

    base, normal = plane
    normal = np.linalg.norm(normal)
    vector = np.subtract(point, base)
    snormal = normal * np.dot(vector, normal)

    return np.subtract(point, snormal)


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax, zoom=1.):
    # ustaianie by skala plt byla rowna w kazdej osi
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0])) / zoom
    set_axes_radius(ax, origin, radius)


def perp_error(params, xyz):
    # oblicza srednie oddalenie punktow od plaszczyzny

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a, b, c, d = params

    length = np.sqrt(a**2 + b**2 + c**2)
    derr = (a * x + b * y + c * z + d) / length
#    for i in range(len(x)):
#        print(derr[i])
    return (np.abs(a * x + b * y + c * z + d) / length).mean(), derr


#============================================================
# main ------
#============================================================

# tablica z pomiarami plaszczyzny
xyz = np.array([[-90.2, 117.6, 499.0],
    [18.7, 132.5, 471.7],
    [142.0, 123.4, 488.9],
    [159.4, 79.2, 570.0],
    [106.2, 23.3, 673.7],
    [4.8, -8.6, 734.5],
    [-141.6, -36.3, 786.5],
    [-220.2, 13.3, 695.9],
    [-63.2, 55.6, 617.9],
    [-57.1, 23.5, 676.6],
    [70.3, 70.6, 590.0],
    [-124.6, 21.8, 679.7],
    [-151.8, -47.6, 806.3],
    [-210.8, 102.4, 530.2],
    [-94.1, 138.0, 465.7],
    [88.9, 130.8, 479.9],
    [184.1, 123.7, 487.7],
    [185.9, 59.5, 609.4]])

# wczytanie pomiarow profilu
profil = np.loadtxt('pomiar.txt',usecols=np.arange(1, 4))

# liczenie srodkowego punktu
centroid = np.average(xyz, axis=0)

# przeliczanie punktow wzgledem centrum
xyzR = xyz - centroid

# dopasowanie SVD
u, sigma, v = np.linalg.svd(xyzR)
normal = v[-1]

# parametry plaszczyzny
x0, y0, z0 = centroid
a, b, c = normal
d = -(a * x0 + b * y0 + c * z0)
print("ABCD:",a ,b ,c ,d)

# obliczanie sredniej odleglosci punktow od plaszczyzny
dmerr, derr = perp_error((a, b, c, d), xyz)
print("Mean err: {:.5f}\n".format(dmerr))

# obliczenia wspolrzedne rzutu punktow na plaszczyzne _p
xyz_p = xyz.copy()
norm1 = normal / np.linalg.norm(normal)
for i in range(len(derr)):
    rzut = norm1 * derr[i]
    xyz_p[i] = xyz[i]-rzut
    print(derr[i], rzut, np.linalg.norm(rzut),xyz[i],xyz_p[i])
perp_error((a, b, c, d), xyz_p)

# przygotowanie wektora normalnego do plaszczyzny do narysowania w plt
forGraphs = list()
forGraphs.append(np.array([centroid[0],centroid[1],centroid[2],normal[0]*100,normal[1]*100, normal[2]*100]))
forGraphs = np.asarray(forGraphs)
X, Y, Z, U, V, W = zip(*forGraphs)

# przygotowanie dopasowanej plaszczyzny do narysowania w plt

# dlugosc wektora tanspozycji
d = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2]

xyzT = np.transpose(xyz)
# tworzenie siatki punktow na plaszczyznie
xx, yy = np.meshgrid(np.arange(min(xyzT[0]), max(xyzT[0])),
                  np.arange(min(xyzT[1]), max(xyzT[1])))
z = (-normal[0] * xx - normal[1] * yy + d) * 1. /normal[2]


#######################################
# punkty w nowym ukladzie wspolrzednych

# przygotowanie tablicy na punktow plaszczyzny po przeliczeniu na nowe wspolrzedne _n
xyz_n = xyz_p.copy()

# macierz rotacji do nowego ukladu spolrzednych
M1 = make_matrix(xyz_p[0], xyz_p[1], xyz_p[2])

# przeliczanie punktow plaszczyzny do nowego ukladu wspolrzednych
for i in range(len(derr)):
    xyz_n[i] = np.transpose(M1).dot(xyz[i] - xyz[0] )

# przygotowanie tablicy na punktow skanowania po przeliczeniu na nowe wspolrzedne
profil_n = profil.copy()

# przeliczanie wspolrzenych dla skanu
for i in range(profil.shape[0]):
    profil_n[i] = np.transpose(M1).dot(profil[i] - xyz[0] )

# przesuwa uklad wspolrzednch plaszczyzny na srodek obszaru
xyz_n = xyz_n - np.average(xyz_n, axis=0)

# przesuwa uklad wspolrzednch profilu na srodek obszaru
profil_n = profil_n - np.average(xyz_n, axis=0)


# przygotowanie punktow plaszczyzny do wyswietlania przez plt
xyz_nT = np.transpose(xyz_n)

# przygotowanie punktow skanu do wyswietlania przez plt
profil_nT = np.transpose(profil_n)



# inicjacja wykresu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

##########################################################
# rysunek 1 : rysowanie dopasowania plaszczyzny do punktow

# rysowanie powierzchni
ax.plot_surface(xx, yy, z, alpha=0.2)

# rysowanie wektora normalnego
ax.quiver(X, Y, Z, U, V, W)

# rysowanie punktow
ax.scatter(xyzT[0],xyzT[1],xyzT[2], c = derr, cmap = 'rainbow')

##########################################################################################
# rysunek 2 : rysowanie punktow plaszczyzny i punktow skanu w nowym ukladzie wspolrzednych

# pukty plaszczyzny
#ax.scatter(xyz_nT[0],xyz_nT[1],xyz_nT[2], c = derr, cmap = 'rainbow')

# linia skanowania
#ax.plot(profil_nT[0],profil_nT[1],profil_nT[2])

# ustawianie osi
set_axes_equal(ax)
ax.set_box_aspect([1, 1, 1])
plt.show()
