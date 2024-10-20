from ctypes import CDLL, POINTER, c_int, c_double
import taichi as ti
import numpy as np
from numpy.ctypeslib import ndpointer

nd_array_dim2 = ndpointer(dtype=np.double, ndim=2, flags='C')
nd_array_dim3 = ndpointer(dtype=np.double, ndim=3, flags='C')

fluid_euler_dll = CDLL("./build/Release/fluid_euler.dll")
fluid_euler_dll.initialize.argtypes = [nd_array_dim3, nd_array_dim3, nd_array_dim2, c_int, c_int, c_double, c_double, c_int]
fluid_euler_dll.clear.argtypes = []
fluid_euler_dll.step.argtypes = []
fluid_euler_dll.step.restype = c_int
fluid_euler_dll.calculate_vorticity.argtypes = []

RES_X = 512
RES_Y = 512
dt = 0.03
decay = 0.99
compression_iters = 40
velocities = np.zeros((RES_X, RES_Y, 2), dtype=np.double)
new_velocities = np.zeros((RES_X, RES_Y, 2), dtype=np.double)
velocity_curls = np.zeros((RES_X, RES_Y), dtype=np.double)

fluid_euler_dll.initialize(
    velocities,
    new_velocities,
    velocity_curls,
    RES_X,
    RES_Y,
    dt,
    decay,
    compression_iters
)

def reset():
    fluid_euler_dll.clear()
    velocities.fill(0)
    new_velocities.fill(0)
    velocity_curls.fill(0)

def main():
    visualize_v = False  # visualize velocity
    visualize_c = True  # visualize curl
    select = 0
    paused = False

    gui = ti.GUI("Fluid Euler", (RES_X, RES_Y))

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                reset()
            elif e.key == "v":
                visualize_v = True
                visualize_c = False
            elif e.key == "c":
                visualize_c = True
                visualize_v = False
            elif e.key == "p":
                paused = not paused

        if not paused:
            select = fluid_euler_dll.step()
        if visualize_c:
            fluid_euler_dll.calculate_vorticity()
            gui.set_image(velocity_curls * 0.03 + 0.5)
        elif visualize_v:
            gui.set_image((velocities if select == 0 else new_velocities) * 0.01 + 0.5)
        gui.show()


if __name__ == "__main__":
    main()
