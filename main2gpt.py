import socket
import threading
import time
from warnings import catch_warnings

import cv2
from queue import Queue
import mediapipe as mp

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Можно настроить количество рук для отслеживания
mp_drawing = mp.solutions.drawing_utils

# Настройки сокета
host = ''
port = 9000
locaddr = (host, port)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tello_address = ('192.168.10.1', 8889)
s.bind(locaddr)

# Видеопоток
video_host = '192.168.10.1'
video_port = 11111
video_url = f'udp://@{video_host}:{video_port}'

# Каскад для распознавания лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Очередь для команд дрону
command_queue = Queue()

# Объект для синхронизации видеопотока
frame_lock = threading.Lock()
current_frame = None

def send_commands():
    """Поток для отправки команд дрону"""
    while True:
        command = command_queue.get()  # Получаем команду из очереди
        if command:
            s.sendto(command.encode(encoding='utf-8'), tello_address)

def is_palm_close(landmarks):
    fingertip_ids = [8, 12, 16, 20]
    for tip_id in fingertip_ids:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            return False
    wrist = landmarks[0]
    thumb = landmarks[4]
    if thumb.y < wrist.y:
        return True  # Это кулак
    return False

def palm_detection(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            if is_palm_close(landmarks):
                s.sendto('flip l'.encode(encoding='utf-8'), tello_address)
                print('flip successful')
                time.sleep(0.1)

def face_detection(frame):
    """Функция для распознавания лиц с плавной корректировкой поворота"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Выбираем самое большое лицо
        t_face = faces[0]
        max_area = 0
        for face in faces:
            x, y, w, h = face
            if w * h > max_area:
                t_face = face
                max_area = w * h
        (x, y, w, h) = t_face

        # Определяем центры лица и кадра
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Расчёт ошибок по осям
        error_x = face_center_x - frame_center_x  # ошибка по горизонтали
        error_y = face_center_y - frame_center_y  # ошибка по вертикали

        # Пропорциональное управление для поворота (yaw)
        dead_zone = 20  # порог мёртвой зоны по горизонтали
        if abs(error_x) > dead_zone:
            K_yaw = 0.2  # коэффициент усиления для поворота
            yaw_speed = int(K_yaw * error_x)
            # Ограничиваем значение от -100 до 100
            yaw_speed = max(min(yaw_speed, 100), -100)
        else:
            yaw_speed = 0

        # Управление движением вперёд/назад по размеру лица
        z = w * h
        if z < 32500:
            fb_speed = 40  # двигаемся вперёд
        elif z > 40000:
            fb_speed = -40  # двигаемся назад
        else:
            fb_speed = 0

        # Управление движением по вертикали
        vertical_dead_zone = 20
        if error_y < -vertical_dead_zone:
            ud_speed = 20  # поднимаемся
        elif error_y > vertical_dead_zone:
            ud_speed = -20  # опускаемся
        else:
            ud_speed = 0

        lr_speed = 0  # горизонтальное смещение влево/вправо не используется

        # Отправляем комбинированную команду управления
        command_queue.put('rc 0 0 0 0')
        command_queue.put(f'rc {lr_speed} {fb_speed} {ud_speed} {yaw_speed}')
    else:
        # Если лицо не обнаружено – дрон удерживает позицию и медленно вращается
        command_queue.put('rc 0 0 0 0')
        # При отсутствии лица можно задать медленный поворот (например, для сканирования)
        command_queue.put('rc 0 0 0 25')
    return frame

def stream():
    """Поток для захвата и отображения видеопотока"""
    global current_frame
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Error: Cannot open video stream.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        # Сохраняем текущий кадр с блокировкой
        with frame_lock:
            current_frame = frame.copy()

        cv2.imshow('Tello Video Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    s.sendto('land'.encode(encoding='utf-8'), tello_address)
    cap.release()
    cv2.destroyAllWindows()

def detect_faces():
    """Поток для распознавания лиц и запуска детекции рук"""
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = face_detection(current_frame)
                cv2.imshow('Tello Video Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                palm_detection(current_frame)
        time.sleep(0.1)

def recv():
    """Поток для получения ответов от дрона"""
    while True:
        try:
            data, server = s.recvfrom(1518)
            executed = data.decode(encoding='utf-8')
            print(executed)
        except Exception as ex:
            print(ex)
            command_queue.put('rc 0 0 -30 0')
            break

# Запуск потоков
get_data = threading.Thread(target=recv)
get_data.start()

# Перевод дрона в режим команд
s.sendto('command'.encode(encoding='utf-8'), tello_address)

video_thread = threading.Thread(target=stream)
video_thread.start()

face_thread = threading.Thread(target=detect_faces)
face_thread.start()

command_thread = threading.Thread(target=send_commands)
command_thread.start()

# Запрос состояния батареи и запуск видеопотока, затем взлёт
s.sendto('battery?'.encode(encoding='utf-8'), tello_address)
s.sendto('streamon'.encode(encoding='utf-8'), tello_address)  # Включаем видеопоток
s.sendto('takeoff'.encode(encoding='utf-8'), tello_address)
time.sleep(2)
