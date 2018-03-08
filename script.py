from pexmd import particles, box, integrator, interaction

import numpy as np
import matplotlib.pyplot as plt

# Weird way to initialize the particles (see #2)
positions = np.array([[-0.6, -0.6, 0.0], [0.6, -0.5, 0.0],
                      [-0.6, 0.6, 0.0], [0.6, 0.6, 0.0]])
part = particles.PointParticles(len(positions))
part.x = positions
part.t = 1
part.mass = np.array([3.0, 3.0, 2.0, 3.0])
dt = 0.005
evol = integrator.Andersen(dt,100,0.5)

# We should initalize this in a much better way (see #3)
x0 = np.array([-0.7]*3)
xf = np.array([0.7]*3)
b = box.Box(x0, xf, t='Fixed')

#lj = interaction.LennardJones([1, 1], 5.4, 1.0, 1.0, "None")
lj = interaction.Morse([1, 1], 5.4, 1.0, 1.0, "None")
pp = []
kk = []
tt = []
for t in np.arange(0, 2, dt):
  part.x, part.v = evol.first_step(part.x, part.v, part.a)
  part.x, part.v = b.wrap_boundary(part.x, part.v)
  part.f, e = lj.forces(part.x, part.v, part.t)
  part.x, part.v = evol.last_step(part.x, part.v, part.a)
  tt.append(t)
  pp.append(e)
  k = 0
  for vv, m in zip(part.v, part.mass):
    k += np.dot(vv, vv)*m
  k /= 2
  kk.append(k)
np.savetxt('pp', pp)
np.savetxt('kk', kk)
np.savetxt('tt', tt)

plt.plot(tt,kk,'ro')
plt.plot(tt,pp,'ro')
plt.show()
