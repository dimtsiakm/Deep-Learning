"""mavic2 controller."""

from controller import *
import math

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
# get robot devices
camera = robot.getDevice('camera')
camera.enable(timestep)
front_left_led = robot.getDevice('front left led')
front_right_led = robot.getDevice('front right led')
imu = robot.getDevice('inertial unit')
imu.enable(timestep)
gps = robot.getDevice('gps')
gps.enable(timestep)
compass = robot.getDevice('compass')
compass.enable(timestep)
gyro = robot.getDevice('gyro')
gyro.enable(timestep)
robot.keyboard.enable(timestep)

camera_roll_motor = robot.getDevice('camera roll')
camera_pitch_motor = robot.getDevice('camera pitch')

front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")


motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

for i in range(len(motors)):
    motors[i].setPosition(float('inf'))
    motors[i].setVelocity(1.0)


# Wait one second
while robot.step(timestep) != -1:
    if robot.getTime() > 1:
        break


k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
# Default values crash the drone
# k_roll_p = 50.0
# k_pitch_p = 30.0
k_roll_p = 10.0
k_pitch_p = 10.0

target_altitude = 1.5

while robot.step(timestep) != -1:
    time = robot.getTime()
    roll = imu.getRollPitchYaw()[0] + (math.pi / 2.0)
    pitch = imu.getRollPitchYaw()[1]
    altitude = gps.getValues()[1]
    roll_acceleration = gyro.getValues()[0]
    pitch_acceleration = gyro.getValues()[1]

    # Blink the front LEDs alternatively with a 1 second rate
    led_state = int(time) % 2
    front_left_led.set(led_state)
    front_right_led.set(not led_state)

    # Stabilize the Camera by actuating the camera motors according to the gyro feedback.
    camera_roll_motor.setPosition(-0.115 * roll_acceleration)
    camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)

    # Transform the keyboard input to disturbances on the stabilization algorithm.
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0
    key = robot.keyboard.getKey()
    while key > 0:
        if key == 315:  # KB UP
            pitch_disturbance = 2.0
            break
        if key == 317:  # KB DOWN
            pitch_disturbance = -2.0
            break
        if key == 316:  # KB RIGHT
            yaw_disturbance = 1.3
            break
        if key == 314:  # KB LEFT
            yaw_disturbance = -1.3
            break
        if key == 65852:  # SHIFT + KB RIGHT
            roll_disturbance = -1.0
            break
        if key == 65850:  # SHIFT + KB LEFT
            roll_disturbance = 1.0
            break
        if key == 65851:  # SHIFT + KB UP
            target_altitude += 0.05
            break
        if key == 65853:  # SHIFT + KB DOWN
            target_altitude -= 0.05
            break

    # Compute the roll, pitch, yaw and vertical inputs.
    roll_input = k_roll_p * max(-1.0, min(roll, 1.0)) + roll_acceleration + roll_disturbance
    pitch_input = k_pitch_p * max(-1.0, min(pitch, 1.0)) - pitch_acceleration + pitch_disturbance

    yaw_input = yaw_disturbance
    clamped_difference_altitude = max(-1.0, min(target_altitude - altitude + k_vertical_offset, 1.0))
    vertical_input = k_vertical_p * math.pow(clamped_difference_altitude, 3.0)

    # Actuate the motors taking into consideration all the computed inputs.
    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)


