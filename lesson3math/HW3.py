from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# код, реализующий расчет длины вектора, заданного его координатами
x = 3
y = 6
len_vec = np.sqrt(x ** 2 + y ** 2)

print('Length = ', len_vec)

# код, реализующий построение графика окружности
t = np.arange(0, 2*np.pi, 0.01)
r = 4
plt.plot(r*np.sin(t), r*np.cos(t), lw=3)
plt.axis('equal')

plt.show()

# код, реализующий построение графика эллипса
t = np.arange(0, 2*np.pi, 0.01)
r = 4
plt.plot(r*np.sin(t), r*np.cos(t), lw=3)
plt.axis

plt.show()

# код, реализующий построение графика параболы
x = []
x2 = []
y = []
y2 = []

for i in range(1000):
    a = 400
    b = 200
    x.append(-i)
    x2.append(i)
    y.append(np.sqrt(b ** 2 + (b ** 2 * (i ** 2 / a ** 2))))
    y2.append(-np.sqrt(b ** 2 + (b ** 2 * (i ** 2 / a ** 2))))

plt.plot(x, y, color='b')
plt.plot(x, y2, color='b')
plt.plot(x2, y2, color='b')
plt.plot(x2, y, color='b')
plt.axis('scaled')
plt.show()

#Нарисуйте трехмерный график двух параллельных плоскостей.
#Нарисуйте трехмерный график двух любых поверхностей второго порядка.

#
x = np.linspace(-2*np.pi, 3*np.pi, 201)

k, a, b = np.linspace(2, 10, num = 3)
k1, a1, b1 = np.linspace(11, 20, num = 3)
k2, a2, b2 = np.linspace(21, 30, num = 3)

print(f'k={k}, a={a}, b={b}')
print(f'k1={k1}, a1={a1}, b1={b1}')
print(f'k2={k2}, a2={a2}, b2={b2}')

plt.figure(figsize = (10, 5))
plt.plot(x, k * np.cos(x - a) + b, label='k, a, b')
plt.plot(x, k1 * np.cos(x - a1) + b1, color='g', label='k1, a1, b1')
plt.plot(x, k2 * np.cos(x - a2) + b2, color='r', label='k2, a2, b2')

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(frameon=False)
plt.show()

# Напишите код, который будет переводить полярные координаты в декартовы.

def polar2cart(r,phi):
    x = r*math.cos(phi)
    y = r*math.sin(phi)
    return x,y

polar2cart(2,45)

# Напишите код, который будет рисовать график окружности в полярных координатах.

phi = np.arange(0., 2., 1./180.)*np.pi
plt.polar(phi, [10]*len(phi))

plt.show()

# Напишите код, который будет рисовать график отрезка прямой линии в полярных координатах.

phi = np.arange(4, 8, 2)
print(phi)
rho = np.arange(4, 8, 2)
print(rho)
plt.polar(phi, rho)

plt.show()


#
