import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

with open('cuda_result') as fp:
    line = fp.readline()
    ny, nx = line.split()
    ny = int(ny)
    nx = int(nx)
    u = numpy.zeros(ny*nx)
    v = numpy.zeros(ny*nx)
    p = numpy.zeros(ny*nx) 
    line = fp.readline()
    c = 0
    for i in line.split():
        u[c] = float(i)
        c += 1
    line = fp.readline()
    c = 0
    for i in line.split():
        v[c] = float(i)
        c += 1
    line = fp.readline()
    c = 0
    for i in line.split():
        p[c] = float(i)
        c += 1

u = u.reshape((ny, nx))
v = v.reshape((ny, nx))
p = p.reshape((ny, nx))

fstr = "cuda_result.png"
print("Plot is saved as", fstr)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)
fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, p, cmap=cm.viridis)
pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.savefig(fstr)
