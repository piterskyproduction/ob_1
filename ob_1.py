import copy
from math import comb
from tkinter import *
import numpy as np
#я сдам лабыыыыыы
root = Tk()
root.geometry('800x660+1+1')
root.config(bg="#0f0f0f")
root.title("Геометрическое моделирование")
canv = Canvas(bg='white')
canv.pack(fill=BOTH,expand=1)
points = []
def click(event):
    print("X:" , event.x , "Y:", event.y)
    canv.create_oval(event.x, event.y, event.x+6, event.y+6, width=0, fill='red')
    points.append([event.x, event.y])

def clear():
    canv.delete('all')
    points.clear()

def lagrange():
    # x = np.array([0, 1, 2, 3, 4])
    # y = np.array([0, -3, 0, 0, 0])
    x = []
    y = []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    x = np.array(x)
    y = np.array(y)

    def lagrange_poli(x, y, t):
        p = 0
        for j in range(len(y)):
            p1 = 1
            p2 = 1
            for i in range(len(x)):
                if i == j:
                    p1 = p1 * 1
                    p2 = p2 * 1
                else:
                    p1 = p1 * (t - x[i])
                    p2 = p2 * (x[j] - x[i])
            p = p + y[j] * p1 / p2
        return p

    x_solve = np.linspace(np.min(x), np.max(x), 100)
    y_solve = [lagrange_poli(x, y, i) for i in x_solve]

    for i in range(len(y_solve)):
        if i != len(x_solve) - 1:
            canv.create_line(x_solve[i], y_solve[i], x_solve[i + 1], y_solve[i + 1])
def beze():
    def beze_poly(i, n, t):
        return comb(n, i) * (t ** i) * (1 - t) ** (n - i)

    def beze_curve(points, nTimes=1000):
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([beze_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals



    xvals, yvals = beze_curve(points, nTimes=1000)
    for i in range(len(xvals)):
        if i != len(xvals) -1:
            canv.create_line(xvals[i],yvals[i],xvals[i+1],yvals[i+1])

def ermit():
    dt = 0.02
    X = []
    Y = []
    # Input.
    string_xv = list(str(xeditv.get()).split(" "))
    int_xv = [int(i) for i in string_xv]
    xpointsv = np.array(int_xv)

    string_yv = list(str(yeditv.get()).split(" "))
    int_yv = [int(i) for i in string_yv]
    ypointsv = np.array(int_yv)
    n = 0
    lev = []
    l1v = []
    while n < len(xpointsv):
        l1v.append(xpointsv[n])
        l1v.append(ypointsv[n])
        l2v = copy.deepcopy(l1v)
        lev.append(l2v)
        l1v.clear()
        n += 1
    vectors = lev
    for i in range(len(points) - 1):
        t = 0
        while t <= 1:
            if i != len(points) - 1:
                t += dt
                x = points[i][0] * (2 * t ** 3 - 3 * t ** 2 + 1) + points[i + 1][0] * (-2 * t ** 3 + 3 * t ** 2) + \
                    (vectors[i][0]-points[i][0]) * (
                            t ** 3 - 2 * t ** 2 + t) + (vectors[i + 1][0]-points[i + 1][0]) * (t ** 3 - t ** 2)
                X.append(x)
                y = points[i][1] * (2 * t ** 3 - 3 * t ** 2 + 1) + points[i + 1][1] * (-2 * t ** 3 + 3 * t ** 2) + \
                    (vectors[i][1]-points[i][1]) * (
                            t ** 3 - 2 * t ** 2 + t) + (vectors[i + 1][1]-points[i + 1][1]) * (t ** 3 - t ** 2)
                Y.append(y)
            if i == len(points) - 2:
                t += dt
                x = points[i][0] * (2 * t ** 3 - 3 * t ** 2 + 1) + points[-1][0] * (-2 * t ** 3 + 3 * t ** 2) + \
                    (vectors[i][0]-points[i][0]) * (
                            t ** 3 - 2 * t ** 2 + t) + (vectors[-1][0]-points[-1][0]) * (t ** 3 - t ** 2)
                X.append(x)
                y = points[i][1] * (2 * t ** 3 - 3 * t ** 2 + 1) + points[-1][1] * (-2 * t ** 3 + 3 * t ** 2) + \
                    (vectors[i][1]-points[i][1]) * (
                            t ** 3 - 2 * t ** 2 + t) + (vectors[-1][1]-points[-1][1]) * (t ** 3 - t ** 2)
                Y.append(y)
    for i in range(len(X)):
        if i != len(X) - 1:
            canv.create_line(X[i], Y[i], X[i + 1], Y[i + 1])
def b_spline():
    global x, y
    import scipy.interpolate as si
    P = [points[0]]
    P.extend(points)
    P.append(points[-1])
    P.append(points[-1])
    t = 0
    degree = int(sedit.get())

    def N_1(t):
        tmin = 0
        tmax = 1
        dt = 0.1
        t_i = []
        while tmin <= tmax:
            t_i.append(tmin)
            tmin += dt
        for i in range(len(t_i) - 2):
            if t >= t_i[i] and t <= t_i[i + 1]:
                return (t - t_i[i]) / (t_i[i + 2] - t_i[i])
            if t >= t_i[i + 1] and t <= t_i[i + 2]:
                return (t_i[i + 2] - t) / (t_i[i + 2] - t_i[i + 1])

    def N_2(t):
        tmin = 0
        tmax = 1
        dt = 0.1
        t_i = []
        while tmin <= tmax:
            t_i.append(tmin)
            tmin += dt
        for i in range(len(t_i) - 3):
            if t >= t_i[i] and t <= t_i[i + 1]:
                # print(((t-t_i[i])**2/(t_i[i+2]-t_i[i])*(t_i[i+1]-t_i[i])))
                return ((t - t_i[i]) ** 2 / (t_i[i + 2] - t_i[i]) * (t_i[i + 1] - t_i[i]))
            if t >= t_i[i + 1] and t <= t_i[i + 2]:
                # print(((t-t_i[i])*(t_i[i+2]-t) / (t_i[i + 2] - t_i[i+1])*(t_i[i + 2] - t_i[i])) + ((t-t_i[i+1])*(t_i[i+3]-t) / (t_i[i + 3] - t_i[i+1])*(t_i[i + 2] - t_i[i + 1])))
                return ((t - t_i[i]) * (t_i[i + 2] - t) / (t_i[i + 2] - t_i[i + 1]) * (t_i[i + 2] - t_i[i])) + (
                            (t - t_i[i + 1]) * (t_i[i + 3] - t) / (t_i[i + 3] - t_i[i + 1]) * (t_i[i + 2] - t_i[i + 1]))

    def N_3(t):
        tmin = 0
        tmax = 1
        dt = 0.1
        t_i = []
        while tmin <= tmax:
            t_i.append(tmin)
            tmin += dt
        for i in range(len(t_i) - 3):
            if t >= t_i[i] and t <= t_i[i + 1]:
                # print(((t-t_i[i])**2/(t_i[i+2]-t_i[i])*(t_i[i+1]-t_i[i])))
                return ((t - t_i[i])  / (t_i[i + 2] - t_i[i]) * (t_i[i + 1] - t_i[i]))*N_2(t)
            if t >= t_i[i + 1] and t <= t_i[i + 2]:
                # print(((t-t_i[i])*(t_i[i+2]-t) / (t_i[i + 2] - t_i[i+1])*(t_i[i + 2] - t_i[i])) + ((t-t_i[i+1])*(t_i[i+3]-t) / (t_i[i + 3] - t_i[i+1])*(t_i[i + 2] - t_i[i + 1])))
                return ((t - t_i[i]) / (t_i[i + 2] - t_i[i]) *N_2(t) + (
                        (t_i[i + 2 +1] - t) / (t_i[i + 2 +1] - t_i[i + 1])*N_2(t)))

    X = []
    Y = []
    x = 0
    y = 0
    for i in range(len(points)):
        # set_point(points[i][0],points[i][1])
        while t <= 0.9:
            if int(sedit.get()) == 1:
                x += points[i][0] * N_1(t)
                X.append(x)
                y += points[i][1] * N_1(t)
                Y.append(y)
                t += 0.1
            if int(sedit.get()) == 2:
                x += points[i][0] * N_2(t)
                X.append(x)
                y += points[i][1] * N_2(t)
                Y.append(y)
                t += 0.1
            if int(sedit.get()) == 3:
                x += points[i][0] * N_3(t)
                X.append(x)
                y += points[i][1] * N_3(t)
                Y.append(y)
                t += 0.1

    def bspline_add_points(cv, n=100, degree=3, periodic=False):
        cv = np.asarray(cv)
        count = len(cv)

        if periodic:
            factor, fraction = divmod(count + degree + 1, count)
            cv = np.concatenate((cv,) * factor + (cv[:fraction],))
            count = len(cv)
            degree = np.clip(degree, 1, degree)


        else:
            degree = np.clip(degree, 1, count - 1)

        kv = None
        if periodic:
            kv = np.arange(0 - degree, count + degree + degree - 1)
        else:
            kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)

        u = np.linspace(periodic, (count - degree), n)

        return np.array(si.splev(u, (kv, cv.T, degree))).T


    p = bspline_add_points(P, n=100, degree=degree, periodic=False)
    X, Y = p.T
    for i in range(len(X)):
        if i != len(X) - 1:
            canv.create_line(X[i], Y[i], X[i + 1], Y[i + 1])
def catmull_rom():
   # P = [[1, 0], [1, 0], [1, 0], [2, 0], [3, 0], [4, 3], [5, 0], [5, 0], [5, 0]]
    P = [points[0]]
    P.extend(points)
    P.append(points[-1])
    P.append(points[-1])
    X = []
    Y = []
    for i in range(len(P) - 1):
        t = 0
        while t <= 1:
            if i != len(P) - 4:
                t += 0.01
                x = 1 / 2 * (-t * (1 - t) ** 2 * P[i][0] + (2 - 5 * t ** 2 + 3 * t ** 3) * P[i + 1][0] + t * (
                            1 + 4 * t - 3 * t ** 2) * P[i + 2][0] - t ** 2 * (1 - t) * P[i + 3][0])
                X.append(x)
                y = 1 / 2 * (-t * (1 - t) ** 2 * P[i][1] + (2 - 5 * t ** 2 + 3 * t ** 3) * P[i + 1][1] + t * (
                            1 + 4 * t - 3 * t ** 2) * P[i + 2][1] - t ** 2 * (1 - t) * P[i + 3][1])
                Y.append(y)
            if i == len(P) - 4:
                t += 0.01
                x = 1 / 2 * (-t * (1 - t) ** 2 * P[i][0] + (2 - 5 * t ** 2 + 3 * t ** 3) * P[i + 1][0] + t * (
                        1 + 4 * t - 3 * t ** 2) * P[i + 2][0] - t ** 2 * (1 - t) * P[i + 3][0])
                X.append(x)
                y = 1 / 2 * (-t * (1 - t) ** 2 * P[i][1] + (2 - 5 * t ** 2 + 3 * t ** 3) * P[i + 1][1] + t * (
                        1 + 4 * t - 3 * t ** 2) * P[i + 2][1] - t ** 2 * (1 - t) * P[i + 3][1])
                Y.append(y)
                break
        if i == len(P) - 4:
            break
    for i in range(len(X)):
        if i != len(X) -1:
            canv.create_line(X[i],Y[i],X[i+1],Y[i+1])


btn = Button(root, text="Очистить полотно", command=clear, width=75, bg="#0f0f0f", fg="red")
btn.pack(side=TOP)
btn1 = Button(root, text="Интерполирующая кривая Лагранжа", command=lagrange, width=75, bg="#0f0f0f", fg="red")
btn1.pack(side=TOP)
btn1 = Button(root, text="Аппроксимирующая кривая Безье", command=beze, width=75, bg="#0f0f0f", fg="red")
btn1.pack(side=TOP)
btn2 = Button(root, text="Интерполирующая кривая Эрмита", command=ermit, width=75, bg="#0f0f0f", fg="red")
btn2.pack(side=TOP)
btn1 = Button(root, text="Аппроксимирующая кривая B-сплайн", command=b_spline, width=75, bg="#0f0f0f", fg="red")
btn1.pack(side=TOP)
btn1 = Button(root, text="Интерполирующая кривая Катмулл-Ром", command=catmull_rom, width=75, bg="#0f0f0f", fg="red")
btn1.pack(side=TOP)
lbl = Label(root, text="Введите степень B-сплайна", bg="#0f0f0f", fg="red")
lbl.pack(side=TOP)
sedit = Entry(root, width=90)
sedit.pack(side=TOP)

lbl = Label(root, text="Введите координаты векторов (для Эрмита)", bg="#0f0f0f", fg="red")
lbl.pack(side=TOP)

xeditv = Entry(root, width=90)
xeditv.pack(side=TOP)

yeditv = Entry(root, width=90)
yeditv.pack(side=TOP)

canv.bind('<1>',click)                   # <---
root.mainloop()
