from controller import *
import time
import threading
import math
from scipy.spatial.transform import Rotation
from pathlib import Path
import cv2
from torchvision import transforms
from Model import Model
import torch
import torch.nn.functional as F
import numpy as np
from controller import Lidar

import torchvision.models as models

labels = {'right': 0, 'center': 1, 'left': 2, 'noposition': 3}
def get_key(val):
    for key, value in labels.items():
        if val == value:
            return key
    return "key doesn't exist"

if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)

    #model = Model(channels=3).to(device)
    #model = models.mobilenet_v3_large()
    # num_ftrs = model.classifier[3].in_features
    # model.classifier[3] = torch.nn.Linear(num_ftrs, 4)

    model.to(device)
    PATH = '/media/dimitris/data_linux/Deep Learning Assignment/CNN/dataset_human_position/logs/resnet18/reduced_plateau/resnet18ckpt.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # setup Webots environment
    robot = Supervisor()
    # viewpoint in environment
    view = robot.getFromDef('view')
    # viewfield = view.getField('position')
    # vieworifield = view.getField('orientation')

    tiago = robot.getSelf()
    left_wheel = robot.getDevice('wheel_left_joint')
    right_wheel = robot.getDevice('wheel_right_joint')

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())//2

    # human = robot.getFromDef('human0')
    # human_position = human.getField('translation').getSFVec3f()

    # get robot devices
    camera = robot.getDevice('camera')
    camera.enable(timestep)
    robot.keyboard.enable(timestep)

    display = robot.getDevice('display')
    display_width = display.getWidth()
    display_height = display.getHeight()
    display.attachCamera(camera)
    display.setColor(0x7CFC00)
    display.setFont(font='Arial', size=12, antiAliasing=True)
    max_speed = 3.0
    speed_left = 0.0
    speed_right = 0.0
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))

    img_number = 0

    starttime = time.time()
    endtime = starttime

    while robot.step(timestep) != -1:
        robot_time = robot.getTime()
        #key = robot.keyboard.getKey()

        tiago_position = tiago.getField('translation').getSFVec3f()
        tiago_rotation = tiago.getField('rotation')
        tiago_rotation = tiago_rotation.getSFRotation()

        image = camera.getImage()
        frame = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        frame = frame[:, :, :3]

        image = data_transforms(frame).to(device)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            out = model(image)
            softmax = F.softmax(out)
            predicted_label = torch.argmax(softmax).cpu().detach().numpy()
            #predicted_label = get_key(predicted_label)
            print(get_key(predicted_label), predicted_label)

        if predicted_label == 2:  # KB LEFT
            speed_left = 0#-max_speed
            speed_right = max_speed
        if predicted_label == 1:  # KB UP
            speed_left = max_speed
            speed_right = max_speed
        if predicted_label == 0:  # KB RIGHT
            speed_left = max_speed/2
            speed_right = 0#-max_speed
        if predicted_label == 3:  # KB DOWN
            speed_left = max_speed/2
            speed_right = 0#max_speed/4

        left_wheel.setVelocity(float(speed_left))
        right_wheel.setVelocity(float(speed_right))










    # path = '/media/dimitris/data_linux/human10/noposition/'
    # Path(path).mkdir(parents=True, exist_ok=True)
    #
    # if(endtime - starttime) > 0.25:
    #     cv2.imwrite(os.path.join(path, 'hmn_'+str(img_number)+'.jpg'), frame)
    #     print('image saved!, img number is :: ' + str(img_number))
    #     img_number +=1
    #     starttime = endtime
    # else:
    #     endtime = time.time()
