import socket
import threading
import time
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
def is_palm_open(landmarks):

    thumb_dist = landmarks[4].x - landmarks[0].x  # Расстояние между большим пальцем и запястьем
    index_dist = landmarks[8].y - landmarks[5].y  # Расстояние между указательным пальцем и ладонью
    middle_dist = landmarks[12].y - landmarks[9].y  # Средний палец
    ring_dist = landmarks[16].y - landmarks[13].y  # Безымянный палец
    pinky_dist = landmarks[20].y - landmarks[17].y  # Мизинец

    # Если пальцы раздвинуты, то расстояния должны быть достаточными
    if thumb_dist > 0.1 and index_dist > 0.1 and middle_dist > 0.1 and ring_dist > 0.1 and pinky_dist > 0.1:
        return True
    return True
def palm_detection(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            landmarks = landmarks.landmark
            is_palm = is_palm_open(landmarks)
            print(True if is_palm else '')
            if is_palm:
                command_queue.put('flip w')  # Добавляем команду поворота влево


def face_detection(frame):
    """Функция для распознавания лиц в кадре"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        t_face = faces[0]
        prev = 0
        for face in faces:
            x, y, w, h = face
            if w * h > prev:
                t_face = face
            prev = w * h
        (x, y, w, h) = t_face

        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        print(f"XY: {x}, {y}")

        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow('Tello Video Stream', frame)

        cv2.waitKey(1)

        if face_center_x < frame_center_x - 110:
            command_queue.put('ccw 20')  # Добавляем команду поворота влево
        elif face_center_x > frame_center_x + 110:
            command_queue.put('cw 20')  # Добавляем команду поворота вправо

        print(frame_center_y, face_center_y)
        if face_center_y < frame_center_y - 20:
            command_queue.put('up 20')
            print('up')
        elif face_center_y > frame_center_y + 20:
            command_queue.put('down 20')
            print('down')

        z = w * h
        print(z)
        if int(z) < 30000:
            command_queue.put('rc 0 20 0 0')  # Дрон двигается вперед
            # command_queue.put('forward 30')
        elif int(z) > 42500:
            command_queue.put('rc 0 -20 0 0')  # Дрон двигается назад
            # command_queue.put('back 30')  # Дрон двигается назад

        if face_center_x < frame_center_x - 110:
            command_queue.put('ccw 20')  # Добавляем команду поворота влево
        elif face_center_x > frame_center_x + 110:
            command_queue.put('cw 20')  # Добавляем команду поворота вправо
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

        # Блокируем доступ к кадру при его обновлении
        with frame_lock:
            current_frame = frame.copy()


        # cv2.imshow('Tello Video Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

def detect_faces():
    """Поток для распознавания лиц"""
    while True:
        with frame_lock:
            if current_frame is not None:
                face_detection(current_frame)
                palm_detection(current_frame)
        time.sleep(0.1)  # Задержка, чтобы не загружать процессор

def recv():
    """Поток для получения ответов от дронов"""
    while True:
        try:
            data, server = s.recvfrom(1518)
            executed = data.decode(encoding='utf-8')
            print(executed)
        except Exception as ex:
            print(ex)
            break

# Поток для получения команд от дрона
get_data = threading.Thread(target=recv)
get_data.start()

# Отправка команды дрону для перехода в режим команд
s.sendto('command'.encode(encoding='utf-8'), tello_address)

# Поток для видеопотока
video = threading.Thread(target=stream)
video.start()

# Поток для распознавания лиц
face_thread = threading.Thread(target=detect_faces)
face_thread.start()

# Поток для отправки команд
command_thread = threading.Thread(target=send_commands)
command_thread.start()

# Запуск дрона
s.sendto('battery?'.encode(encoding='utf-8'), tello_address)
s.sendto('streamon'.encode(encoding='utf-8'), tello_address)  # Включаем видеопоток
# s.sendto('takeoff'.encode(encoding='utf-8'), tello_address)  # Взлет
time.sleep(3)


# Получение данных от пользователя
while True:
    data = input()
    s.sendto(data.encode(encoding='utf-8'), tello_address)
