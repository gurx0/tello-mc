from djitellopy import Tello
import time

# Создаем объект дрона
tello = Tello()

# Подключаемся к дрону
tello.connect()

# Проверяем уровень заряда батареи
print(f"Battery: {tello.get_battery()}%")

# Взлетаем
tello.takeoff()

print(f"Battery: {tello.get_battery()}%")

tello.flip_forward()
# Ожидаем 5 секунд, чтобы дрон стабилизировался в воздухе
time.sleep(2)

print(f"Battery: {tello.get_battery()}%")

# Садимся
tello.land()

# Отключаемся от дрона
tello.end()
