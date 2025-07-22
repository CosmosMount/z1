import numpy as np
from pytransform3d import rotations
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from simulator import z1Simulator
from tracker import VRTracker
from processor import VuerPreprocessor
from utils import link06_init_pose


class VuerTeleop:
    def __init__(self):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.processor = VuerPreprocessor()
        self.simulator = z1Simulator()
        self.tracker = VRTracker(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)

    def step(self):
        if self.tracker.connected.value:
            self.processor.connected = True
            right_controller_mat, left_controller_mat = self.processor.process(self.tracker)
            print(f"Right Controller Matrix: {right_controller_mat}")
            target = np.concatenate([right_controller_mat[:3,3], rotations.quaternion_from_matrix(right_controller_mat[:3, :3])[[1, 2, 3, 0]]])
            print(f"Target Pose: {target}")
        else:
            target = link06_init_pose
        self.simulator.step(target)

if __name__ == "__main__":
    teleop = VuerTeleop()
    try:
        while True:
            teleop.step()
    except KeyboardInterrupt:
        print("Exiting Vuer Teleop")
        teleop.simulator.end()
        exit(0)
