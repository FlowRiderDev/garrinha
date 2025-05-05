import cv2
import mediapipe as mp
import time
import math
import serial

# Configurações
TIME_INTERVAL = 0.1       # segundos entre leituras
SERIAL_PORT = 'COM3'      # ajuste conforme sua porta
BAUD_RATE = 115200

# Ângulos mínimo e máximo para cada servo
MIN_X, MAX_X       = 30, 150   # eixo X (pino 3)
MIN_Y, MAX_Y       = 20, 110   # eixo Y (pino 5)
MIN_Z, MAX_Z       = 20, 60    # eixo Z (pino 6)
MIN_GRIP, MAX_GRIP = 0, 50     # garra (pino 9)

# Threshold de mudança mínima para enviar atualização (graus)
THRESH_X    = 2
THRESH_Y    = 2
THRESH_Z    = 2
THRESH_GRIP = 5

# Inicializa Serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Últimos valores enviados (inicializa no meio/faixa aberta)
last_x = (MIN_X + MAX_X) // 2
last_y = (MIN_Y + MAX_Y) // 2
last_z = (MIN_Z + MAX_Z) // 2
last_grip = MIN_GRIP

# Mapeia valor normalizado para ângulo entre out_min e out_max
def map_angle(norm, in_min, in_max, out_min, out_max):
    val = (norm - in_min) / (in_max - in_min)
    angle = int(val * (out_max - out_min) + out_min)
    return max(out_min, min(out_max, angle))

# Detecta se a mão está aberta (polegar x mindinho)
def is_hand_open(lm, threshold=0.7):
    width = math.hypot(lm[5].x - lm[17].x, lm[5].y - lm[17].y)
    dist = math.hypot(lm[4].x - lm[20].x, lm[4].y - lm[20].y)
    return dist > threshold * width

# Calcula profundidade via distância 3D punho-base do mindinho
def compute_z(lm):
    return math.sqrt(
        (lm[0].x - lm[17].x)**2 +
        (lm[0].y - lm[17].y)**2 +
        (lm[0].z - lm[17].z)**2
    )

# Loop principal
cap = cv2.VideoCapture(0)
prev_time = time.time()
print("Pressione 'q' para sair.")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Detecta presença de mão esquerda e landmarks da direita
    left_present = False
    right_lm = None
    if results.multi_handedness and results.multi_hand_landmarks:
        for hd, lm in zip(results.multi_handedness, results.multi_hand_landmarks):
            if hd.classification[0].label == 'Left':
                left_present = True
            else:
                right_lm = lm.landmark

    # Se só a mão direita estiver, processa a cada intervalo
    if not left_present and right_lm:
        now = time.time()
        if now - prev_time >= TIME_INTERVAL:
            prev_time = now
            # Normaliza coordenadas
            x_norm = right_lm[0].x - 0.5
            y_norm = 0.5 - right_lm[0].y
            z_norm = compute_z(right_lm)
            open_flag = is_hand_open(right_lm)

            # Mapeia para ângulos e clamp
            x_ang    = map_angle(x_norm, -0.5, 0.5, MIN_X, MAX_X)
            y_ang    = map_angle(y_norm, -0.5, 0.5, MIN_Y, MAX_Y)
            z_ang    = map_angle(z_norm,   0.0, 0.5, MIN_Z, MAX_Z)
            grip_ang = MIN_GRIP if open_flag else MAX_GRIP

            # Envia apenas se mudança > threshold
            if (abs(x_ang - last_x) > THRESH_X or
                abs(y_ang - last_y) > THRESH_Y or
                abs(z_ang - last_z) > THRESH_Z or
                abs(grip_ang - last_grip) > THRESH_GRIP):
                last_x, last_y, last_z, last_grip = x_ang, y_ang, z_ang, grip_ang
                packet = f"{x_ang},{y_ang},{z_ang},{grip_ang}\n"
                ser.write(packet.encode())
                print(f"=> {packet.strip()}")

    # Desenha marcações e exibe frame
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Hands', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()