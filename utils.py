#!/usr/bin/env python

import sys
import array

import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from inputs import get_gamepad
import math
import threading


def resize_image(img):
    im = resize(img, (Sample.IMG_H, Sample.IMG_W, Sample.IMG_D))
    im_arr = im.reshape((Sample.IMG_H, Sample.IMG_W, Sample.IMG_D))
    return im_arr


class Screenshot(object):
    SRC_W = 640
    SRC_H = 480
    SRC_D = 3

    OFFSET_X = 0
    OFFSET_Y = 0


class Sample:
    IMG_W = 200
    IMG_H = 66
    IMG_D = 3


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = 255.0
    # MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.Z = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self):
        x = self.LeftJoystickX
        y = self.LeftJoystickY
        a = self.A
        b = self.B
        z = self.Z
        #rb = self.RightBumper
        return [x, y, a, b, z]  # removed right bumper for now


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = (2.* event.state / XboxController.MAX_JOY_VAL) - 1. # normalize and recenter between -1 and 1
                    # print("y", self.LeftJoystickY)
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = (2.* event.state / XboxController.MAX_JOY_VAL) - 1. # normalize between -1 and 1
                    # print("x", self.LeftJoystickX)
                elif event.code == 'ABS_Z':  #ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RZ':  #'ABS_RX'
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'BTN_TOP2':#'ABS_Z':
                    pass
                elif event.code == 'BTN_PINKIE':#'ABS_RZ':
                    self.RightBumper = event.state
                    print("rb", event.state)
                elif event.code == 'BTN_BASE':#'BTN_TL':
                    self.Z = event.state
                    # print("Z", event.state)
                elif event.code == 'BTN_BASE2':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL  # normalize between 0 and 1
                elif event.code == 'BTN_THUMB':
                    self.A = event.state
                    # print("A", event.state)
                elif event.code == 'BTN_TOP':
                    self.Y = event.state
                elif event.code == 'BTN_TRIGGER':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMB2':
                    # print(event.state)
                    self.B = event.state
                    # print("B", event.state)
                elif event.code == 'BTN_THUMBL':  #thumb l
                    self.LeftThumb = event.state
                elif event.code == 'BTN_BASE5':  # thumb r
                    self.RightThumb = event.state
                elif event.code == 'BTN_BASE6':  #select
                    self.Back = event.state
                elif event.code == 'BTN_BASE4':  #start
                    self.Start = event.state
                elif event.code == 'ABS_HAT0X':
                    self.LeftDPad = event.state / XboxController.MAX_JOY_VAL # normalize
                #elif event.code == 'ABS_HAT0X':
                 #   self.RightDPad = event.state / XboxController.MAX_JOY_VAL # normalize
                elif event.code == 'ABS_HAT0Y':
                    self.UpDPad = event.state / XboxController.MAX_JOY_VAL # normalize
                #elif event.code == 'BTN_TRIGGER_HAPPY4':
                 #   self.DownDPad = event.state


class Data(object):
    def __init__(self):
        self._X = np.load("data/X.npy")
        self._y = np.load("data/y.npy")
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._X.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


def load_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5))
    return image_files, joystick_values


# training data viewer
def viewer(sample):
    image_files, joystick_values = load_sample(sample)

    plotData = []

    plt.ion()
    plt.figure('viewer', figsize=(16, 6))

    for i in range(len(image_files)):

        # joystick
        print(i, " ", joystick_values[i,:])

        # format data
        plotData.append( joystick_values[i,:] )
        if len(plotData) > 30:
            plotData.pop(0)
        x = np.asarray(plotData)

        # image (every 3rd)
        if (i % 3 == 0):
            plt.subplot(121)
            image_file = image_files[i]
            img = mpimg.imread(image_file)
            plt.imshow(img)

        # plot
        plt.subplot(122)
        plt.plot(range(i,i+len(plotData)), x[:,0], 'r')
        plt.hold(True)
        plt.plot(range(i,i+len(plotData)), x[:,1], 'b')
        plt.plot(range(i,i+len(plotData)), x[:,2], 'g')
        plt.plot(range(i,i+len(plotData)), x[:,3], 'k')
        plt.plot(range(i,i+len(plotData)), x[:,4], 'y')
        plt.draw()
        plt.hold(False)

        plt.pause(0.0001) # seconds
        i += 1


# prepare training data
def prepare(samples):
    print("Preparing data")

    X = []
    y = []

    for sample in samples:
        print(sample)

        # load sample
        image_files, joystick_values = load_sample(sample)

        # add joystick values to y
        y.append(joystick_values)

        # load, prepare and add images to X
        for image_file in image_files:
            image = imread(image_file)
            vec = resize_image(image)
            X.append(vec)

    print("Saving to file...")
    X = np.asarray(X)
    y = np.concatenate(y)

    np.save("data/X", X)
    np.save("data/y", y)

    print("Done!")
    return


if __name__ == '__main__':
    if sys.argv[1] == 'viewer':
        viewer(sys.argv[2])
    elif sys.argv[1] == 'prepare':
        prepare(sys.argv[2:])
