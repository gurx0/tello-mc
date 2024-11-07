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

# Ожидаем 5 секунд, чтобы дрон стабилизировался в воздухе
time.sleep(5)

# Садимся
tello.land()

# Отключаемся от дрона
tello.end()
