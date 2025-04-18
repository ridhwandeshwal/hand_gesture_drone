from djitellopy import Tello
import time

tello = Tello()

tello.connect()
tello.takeoff()
print("taking off")
print(tello.get_battery())
time.sleep(5)
# tello.move_left(100)
# print(tello.get_battery())
# time.sleep(5)
# tello.rotate_counter_clockwise(90)
# print(tello.get_battery())
# time.sleep(5)
print('forward')
tello.move_forward(40)
print(tello.get_battery())
time.sleep(5)
print('land')
tello.land()
print(tello.get_battery())