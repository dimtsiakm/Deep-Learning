from abc import ABC

from controller import *
import math
import numpy as np
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

        self.device = 'cuda'
        # setup Webots environment
        self.robot = Supervisor()
        self.mavic = self.robot.getSelf()
        self.mavic_rotation = self.mavic.getField('rotation')
        self.mavic_position = self.mavic.getField('translation')
        self.timestep = int(self.robot.getBasicTimeStep())

        # get robot devices
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.robot.keyboard.enable(self.timestep)
        self.front_left_led = self.robot.getDevice('front left led')
        self.front_right_led = self.robot.getDevice('front right led')
        self.imu = self.robot.getDevice('inertial unit')
        self.imu.enable(self.timestep)
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.camera_roll_motor = self.robot.getDevice('camera roll')
        self.camera_pitch_motor = self.robot.getDevice('camera pitch')

        self.front_left_motor = self.robot.getDevice("front left propeller")
        self.front_right_motor = self.robot.getDevice("front right propeller")
        self.rear_left_motor = self.robot.getDevice("rear left propeller")
        self.rear_right_motor = self.robot.getDevice("rear right propeller")
        self.motors = [self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor]
        for i in range(len(self.motors)):
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(1.0)

        self.k_vertical_thrust = 68.5
        self.k_vertical_offset = 0.6
        self.k_vertical_p = 3.0
        # Default values crash the Drone
        # self.k_roll_p = 50.0
        # self.k_pitch_p = 30.0
        self.k_roll_p = 10.0
        self.k_pitch_p = 10.0
        self.target_altitude = 1.5
        self.robot.keyboard.enable(self.timestep)
        self.root = self.robot.getRoot()
        self.children = self.root.getField('children')
        # Import function
        # self.human_model = 'human.wbo'
        # self.children.importMFNode(-1, self.human_model)

        # ----------------------------- GYM STUFF ----------------------------- #
        # Discrete action-space - Change to desired actions
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
        self.reward_range = [-1000, 1000]
        self.metadata = None
        self.step_counter = 0
        self.num_envs = 1

    def move_forward(self):
        pitch_disturbance = 2.0
        yaw_disturbance = 0.0
        return pitch_disturbance, yaw_disturbance

    def move_backwards(self):
        pitch_disturbance = -2.0
        yaw_disturbance = 0.0
        return pitch_disturbance, yaw_disturbance

    def rotate_left(self):
        pitch_disturbance = 0.0
        yaw_disturbance = -1.3
        return pitch_disturbance, yaw_disturbance

    def rotate_right(self):
        pitch_disturbance = 0.0
        yaw_disturbance = 1.3
        return pitch_disturbance, yaw_disturbance

    def stand_still(self):
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0
        return pitch_disturbance, yaw_disturbance

    def reset_simulation(self):
        self.robot.simulationReset()
        self.robot.step(self.timestep)
        for i in range(len(self.motors)):
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(1.0)
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
        done = False
        # Action translation
        while self.robot.step(self.timestep) != -1:
            self.step_counter += 1
            roll = self.imu.getRollPitchYaw()[0] + (math.pi / 2.0)
            pitch = self.imu.getRollPitchYaw()[1]
            altitude = self.gps.getValues()[1]
            rotation = self.compass.getValues()
            roll_acceleration = self.gyro.getValues()[0]
            pitch_acceleration = self.gyro.getValues()[1]
            self.camera_roll_motor.setPosition(-0.115 * roll_acceleration)
            self.camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)
            roll_disturbance = 0.0
            pitch_disturbance = 0.0
            yaw_disturbance = 0.0
            if action == 0:
                pitch_disturbance, yaw_disturbance = self.stand_still()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 1:
                pitch_disturbance, yaw_disturbance = self.move_forward()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 2:
                pitch_disturbance, yaw_disturbance = self.move_backwards()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 3:
                pitch_disturbance, yaw_disturbance = self.rotate_right()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            if action == 4:
                pitch_disturbance, yaw_disturbance = self.rotate_left()
                obs = self.get_obs()
                reward = self.get_reward(obs)
            # Done condition - 1000 steps
            if self.step_counter == 1000:
                self.step_counter = 0
                done = True
            # Compute the roll, pitch, yaw and vertical inputs.
            roll_input = self.k_roll_p * max(-1.0, min(roll, 1.0)) + roll_acceleration + roll_disturbance
            pitch_input = self.k_pitch_p * max(-1.0, min(pitch, 1.0)) - pitch_acceleration + pitch_disturbance

            yaw_input = yaw_disturbance
            clamped_difference_altitude = max(-1.0, min(self.target_altitude - altitude + self.k_vertical_offset, 1.0))
            vertical_input = self.k_vertical_p * math.pow(clamped_difference_altitude, 3.0)

            # Actuate the motors taking into consideration all the computed inputs.
            front_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
            front_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
            rear_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
            rear_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)
            if obs is not None:
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
        else:
            frame = None
            return frame

    # Implement reward return based on task
    def get_reward(self, frame):
            if frame:
                pass
            reward = 1
            return reward
