import time
from controller import *
import math
import cv2
import os
import numpy as np
from gym import spaces
import gym
import torchvision.transforms as transforms
from PIL import Image as PILImage
import random
from scipy.spatial import distance
import torch


def preprocess(frame):
    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=PILImage.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    obs = 0
    if frame is not None:
        obs = transform(PILImage.fromarray(frame))
        # print('transform completed')
    else:
        pass
        # print('missed frame!')

    # time.sleep(0.1)
    return obs


class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Env, self).__init__()
        # yolov5
        model_ = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # ----------------------------- WEBOTS STUFF ----------------------------- #

        self.robot = Supervisor()
        self.left_wheel = self.robot.getDevice('wheel_left_joint')
        self.right_wheel = self.robot.getDevice('wheel_right_joint')
        self.timestep = int(self.robot.getBasicTimeStep())
        # get robot devices
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.robot.keyboard.enable(self.timestep)
        self.display = self.robot.getDevice('display')
        self.touch_sensor = self.robot.getTouchSensor('base_cover_link')
        self.touch_sensor.enable(self.timestep)
        self.display_width = self.display.getWidth()
        self.display_height = self.display.getHeight()
        self.display.attachCamera(self.camera)
        self.display.setColor(0x7CFC00)
        self.display.setFont(font='Arial', size=12, antiAliasing=True)
        self.max_speed = 12
        self.cruising_speed = 1.8
        self.left_wheel.setPosition(float('inf'))
        self.right_wheel.setPosition(float('inf'))
        self.fov = 0.45
        self.camera.setFov(self.fov)
        self.tiago = self.robot.getSelf()
        self.position = self.tiago.getField('translation')
        self.rotation = self.tiago.getField('rotation')
        self.human_counter = 1
        self.root = self.robot.getRoot()
        self.children = self.root.getField('children')
        # self.human_model = '/media/dimitris/data_linux/Deep Learning Assignment/RL/human.wbo'
        # self.children.importMFNode(2, self.human_model)
        self.yolo = model_
        self.episode = 0
        self.previous_position = (0, 0)

        # ----------------------------- GYM STUFF ----------------------------- #

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
        self.step_counter = 0
        self.totalsteps = 0

    def move_forward(self):
        speed_right = self.max_speed
        speed_left = self.max_speed
        self.left_wheel.setVelocity(speed_left)
        self.right_wheel.setVelocity(speed_right)

    def move_backwards(self):
        speed_right = -self.max_speed
        speed_left = -self.max_speed
        self.left_wheel.setVelocity(speed_left)
        self.right_wheel.setVelocity(speed_right)

    def rotate_left(self):
        speed_right = self.cruising_speed
        speed_left = 0.0
        self.left_wheel.setVelocity(speed_left)
        self.right_wheel.setVelocity(speed_right)

    def rotate_right(self):
        speed_left = self.cruising_speed
        speed_right = 0.0
        self.left_wheel.setVelocity(speed_left)
        self.right_wheel.setVelocity(speed_right)

    def stand_still(self):
        speed_left = 0.0
        speed_right = 0.0
        self.left_wheel.setVelocity(speed_left)
        self.right_wheel.setVelocity(speed_right)

    def reset(self):
        # print("reset called")
        self.position.setSFVec3f([random.randint(-2, 2), 0.1, random.randint(-11, -8)])
        # self.position.setSFVec3f([0.033, 0.092, -8.56])
        self.rotation.setSFRotation([-0.579, -0.578, -0.575, 2.1])
        self.totalsteps += self.step_counter
        self.step_counter = 0

        obs = self.get_obs()
        obs = preprocess(obs)
        return obs

    def step(self, action):
        self.step_counter += 1
        done = False
        reward = 0
        # obs = None
        while self.robot.step(self.timestep) != -1:
            # straight forward!
            if action == 0:
                self.move_forward()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            # turn right
            if action == 1:
                self.rotate_right()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            # turn left
            if action == 2:
                self.rotate_left()
                obs = self.get_obs()
                reward = self.get_reward(obs)

            if reward == 1000 or reward == -50 or self.step_counter == 4500:
                done = True
                self.episode += 1
            obs = preprocess(obs)
            self.previous_position = (float("{:.5f}".format(np.float32(self.position.getSFVec3f()[0]))),
                                      float("{:.5f}".format(np.float32(self.position.getSFVec3f()[2]))))
            if done:
                print('episode ' + str(self.episode) + ' finished after ' + str(self.step_counter) + ' steps, reward '
                                                                                                     ':: ' + str(
                    reward))
                print('totalsteps ' + str(self.totalsteps) + '/' + str(300000))
            return obs, reward, done, {}

    def get_obs(self):
        # Get camera input and pass it through the network
        cameradata = self.camera.getImage()
        if cameradata:
            # Resize image and pass it through yolo classifier
            frame = np.frombuffer(cameradata, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
            frame = frame[:, :, :3]
            return frame

    def get_reward(self, frame):
        reward = 0

        current_position = (float("{:.5f}".format(np.float32(self.position.getSFVec3f()[0]))),
                            float("{:.5f}".format(np.float32(self.position.getSFVec3f()[2]))))
        p2 = (0, -7)
        d = distance.euclidean(current_position, p2)
        # print(current_position, p2, d)
        if self.previous_position == current_position:
            print('stuck')
            return -50

        if frame is not None:
            results = self.yolo(frame)
            res = results.pandas().xyxy[0]  # img1 predictions (pandas)
            if (res['name'] == 'person').any():
                person_bbox = res.iloc[0]
                if d <= 1.5:
                    print('reached')
                    return 1000

                if person_bbox['xmin'] >= ((frame.shape[1]) / 2):
                    reward = 5
                    # person in right.
                elif person_bbox['xmax'] <= ((frame.shape[1]) / 2):
                    reward = 5
                    # person in left.
                else:
                    reward = 15
                    # center? not actually. but in a first attempt the person is in center.
            else:
                return -15
        return reward
