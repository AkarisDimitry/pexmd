"""
A proof of concept on how to call everything and perform a *very* small simulation.

This example will change and retrofit the library design.

Problems exposed by using this example (that runs as is):

(Some of these problems should be first set as unit tests)

2 - The "Particles" constructor should have everything as optional and, if needed, get the number or particles from the length of the array
3 - The box constructor should accept lists or tuples and convert them internally to np arrays
4 - Watch where we use 32bit and 64bit floats in the np arrays
5 - This should evolve (eventually) to a regression test
"""

from pexmd import particles, box, integrator, interaction

import numpy as np
import math
import matplotlib.pyplot as plt

nparticles = 70
dt = 0.1
t_end = 4
temperature_ini = 10.0
scale = 100.

# Initializing particles:
positions_ini = np.random.rand(nparticles, 3)
positions_ini = scale * positions_ini
#velocities_ini = np.random.normal(0.,math.sqrt(temperature_ini),size=(nparticles, 3))
velocities_ini = np.zeros((nparticles, 3), dtype=np.float32)
masses_ini = np.full((nparticles), 1.)


part = particles.PointParticles(nparticles)
part.x = positions_ini
part.v = velocities_ini
part.t = 1
part.mass = masses_ini

evol = integrator.VelVerlet(dt)
#evol = integrator.Andersen(dt, temperature_ini, 10.)

# We should initalize this in a much better way (see #3)
x0 = np.array([-scale]*3)
xf = np.array([scale]*3)
b = box.Box(x0, xf, t='Fixed')

#lj = interaction.LennardJones([1, 1], 5.4, 1.0, 1.0, "Displace")
lj = interaction.Morse([1, 1], 5.4, 1.0, 1.0, 1.0, "Displace")
pp = []
kk = []
tt = []
for t in np.arange(0, t_end, dt):
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
  plt.plot(t, k, 'bo')
  plt.plot(t, -e + k, 'go')
  plt.plot(t, -e, 'ro')
  print(t)
  kk.append(k)
np.savetxt('pp', pp)
np.savetxt('kk', kk)
np.savetxt('tt', tt)

plt.show()
