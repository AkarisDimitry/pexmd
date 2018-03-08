import matplotlib
matplotlib.interactive(True)
import threading
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import axes3d

data = np.random.rand(4,3,15)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection= '3d')
ax1.hold(False)
for i in range(data.shape[2]):
	print i
	td = data[:, :, i]
        ax1.plot(td[:, 0], td[:, 1],td[:,1],'yo')
	ax1.set(title='Dinamica de particulas', xlabel='Eje X', ylabel='Eje Y', zlabel='Eje Z')
	ax1.set(xlim=[-0.1,1.1], ylim=[-0.1,1.1],zlim=[-0.1,1.1])
	fig.canvas.draw()
	time.sleep(0.5)

