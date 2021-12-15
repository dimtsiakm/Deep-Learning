from controller import *

# setup Webots environment
robot = Supervisor()
# viewpoint in environment
view = robot.getFromDef('view')

tiago = robot.getSelf()
left_wheel = robot.getDevice('wheel_left_joint')
right_wheel = robot.getDevice('wheel_right_joint')

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

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

while robot.step(timestep) != -1:
    time = robot.getTime()
    key = robot.keyboard.getKey()
    if key == 314:  # KB LEFT
        speed_left = -max_speed
        speed_right = max_speed
    if key == 315:  # KB UP
        speed_left = max_speed
        speed_right = max_speed
    if key == 316:  # KB RIGHT
        speed_left = max_speed
        speed_right = -max_speed
    if key == 317:  # KB DOWN
        speed_left = -max_speed
        speed_right = -max_speed
    if key == -1:  # None
        speed_left = 0.0
        speed_right = 0.0

    left_wheel.setVelocity(speed_left)
    right_wheel.setVelocity(speed_right)
