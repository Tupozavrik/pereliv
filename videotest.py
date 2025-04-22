import torch
import cv2
import pyautogui
import numpy as np
import time
import keyboard
import threading

pyautogui.FAILSAFE = False  # отключить аварийку
# Загружаем модель
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/FP/yolov5/runs/train/dota_yolo960_finetune_final3/weights/best.pt')
model.conf = 0.5

screen_w, screen_h = pyautogui.size()
paused = False
last_attack_time = 0
last_crip_time = 0
last_button_check = 0
check_interval = 10  # Проверка igrat/prinat каждые 10 секунд

# === пауза ===
def toggle_pause():
    global paused
    paused = not paused
    print("Пауза включена" if paused else "Возобновлено")

def monitor_hotkey():
    while True:
        keyboard.wait('ctrl+shift+q')
        toggle_pause()
        time.sleep(0.5)

def screenshot_rgb():
    # PyAutoGUI возвращает RGB изображение
    screenshot = pyautogui.screenshot()
    # Просто преобразуем в numpy массив без изменения порядка каналов
    return np.array(screenshot)

def click_on_box(results, label_name):
    df = results.pandas().xyxy[0]
    matches = df[df['name'] == label_name]
    
    for _, row in matches.iterrows():
        if row['confidence'] < 0.5:
            continue
        x_center = int((row['xmin'] + row['xmax']) / 2)
        y_center = int((row['ymin'] + row['ymax']) / 2)
        print(f"🔘 Клик по {label_name} в точку ({x_center}, {y_center})")
        pyautogui.moveTo(x_center, y_center, duration=0.2)
        pyautogui.click()
        return True
    return False

# Запускаем поток для мониторинга горячих клавиш
threading.Thread(target=monitor_hotkey, daemon=True).start()

# Создаем окно для отображения результатов
cv2.namedWindow("Отладка YOLO", cv2.WINDOW_NORMAL)

while True:
    if paused:
        time.sleep(0.1)
        continue

    now = time.time()
    # Получаем RGB изображение
    frame = screenshot_rgb()
    # YOLOv5 ожидает RGB, так что всё правильно
    results = model(frame)

    # Рендеринг результатов (также в RGB)
    annotated_frame = results.render()[0]  
    # Конвертируем в BGR для отображения через OpenCV
    annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Отладка YOLO", annotated_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # === Проверка igrat / prinat каждые 10 секунд ===
    if now - last_button_check >= check_interval:
        print("🕵️ Проверка на igrat / prinat")
        found = False
        found |= click_on_box(results, 'igrat')
        found |= click_on_box(results, 'prinat')
        if not found:
            print("❌ Кнопки igrat / prinat не найдены.")
        last_button_check = now

    # === детект крипов ===
    found_crips = False
    for *xyxy, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        if label == 'crips' and conf > 0.5:
            found_crips = True
            break

    if found_crips:
        last_crip_time = time.time()
        print("Крипы найдены! Атакуем.")
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label == 'crips' and conf > 0.5 and time.time() - last_attack_time >= 1:
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                pyautogui.moveTo(cx, cy)
                pyautogui.click(button='right')
                last_attack_time = time.time()
        time.sleep(0.5)
        continue

    print("Ожидание следующего цикла...")
    time.sleep(1)

cv2.destroyAllWindows()
