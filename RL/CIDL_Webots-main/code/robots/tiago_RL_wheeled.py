from abc import ABC

from controller import *
import math

import os
import numpy as np
import random
from gym import spaces
import gym
import torchvision.transforms as transforms
from PIL import Image as PILImage

def preprocess(frame):
    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=PILImage.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    obs = transform(PILImage.fromarray(frame))
    return obs


class Env(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Env, self).__init__()

        # setup Webots environment
        self.robot = Supervisor()
        self.tiago = self.robot.getSelf()
        self.left_wheel = self.robot.getDevice('wheel_left_joint')
        self.right_wheel = self.robot.getDevice('wheel_right_joint')
        self.timestep = int(self.robot.getBasicTimeStep())
        # get robot devices
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.max_speed = 6.4
        self.cruising_speed = 5.5
        self.left_wheel.setPosition(float('inf'))
        self.right_wheel.setPosition(float('inf'))
        self.left_wheel.setVelocity(0.0)
        self.right_wheel.setVelocity(0.0)
        self.left_wheel.setAcceleration(-1)
        self.right_wheel.setAcceleration(-1)
        self.tiago_position = self.tiago.getField('translation')
        self.tiago_rotation = self.tiago.getField('rotation')
        self.root = self.robot.getRoot()
        self.children = self.root.getField('children')
        # Import into world
        # self.human_model = 'human.wbo'
        # self.children.importMFNode(-1, self.human_model)

        # ----------------------------- GYM STUFF ----------------------------- #
        # Discrete action-space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
        self.reward_range = [-1000, 1000]
        self.metadata = None
        self.step_counter = 0
        self.num_envs = 1

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

    def reset_simulation(self):
        self.robot.simulationReset()
        self.robot.step(self.timestep)
        self.left_wheel.setPosition(float('inf'))
        self.right_wheel.setPosition(float('inf'))
        self.left_wheel.setVelocity(0.0)
        self.right_wheel.setVelocity(0.0)
        # Wait one second
        while self.robot.step(self.timestep) != -1:
            if self.robot.getTime() > 1:
                break

    def reset(self):
        self.reset_simulation()
        obs = self.get_obs()
        if obs is not None:
            obs = preprocess(obs)
        return obs

    def step(self, action):
        self.step_counter += 1
        done = False
        # Action translation
        while self.robot.step(self.timestep) != -1:
            if action == 0:
                self.stand_still()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 1:
                self.move_forward()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 2:
                self.move_backwards()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 3:
                self.rotate_right()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 4:
                self.rotate_left()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if self.step_counter == 1000 or reward == 5:
                done = True
            obs = preprocess(obs)
            return obs, reward, done, {}

    def get_obs(self):
        # Get camera input and pass it through the network
        cameradata = self.camera.getImage()
        if cameradata:
            # Resize image and pass it through opencv's face detector
            frame = np.frombuffer(cameradata, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
            frame = frame[:, :, :3]
            return frame

    # Implement reward return based on task
    def get_reward(self, frame):
        if frame:
            pass
        reward = 1
        return reward
