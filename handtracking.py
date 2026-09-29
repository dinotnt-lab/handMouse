import cv2
import mediapipe as mp
import time
import math
from dataclasses import dataclass
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Listener, Key
import tkinter as tk


# ---------- Config ----------
@dataclass
class Config:
    # Toggle
    toggle_pause_key: str = "x"
    quit_key: str = "q"

    # Movement mode
    trackpad_mode: bool = False  # True = relative (trackpad), False = absolute mapping

    # ROI (active camera area) in pixels (drawn on the camera feed)
    roi_min_x: int = 20
    roi_max_x: int = 300
    roi_min_y: int = 20
    roi_max_y: int = 250

    # Movement tuning
    sensitivity: float = 1.5      # >1 faster, <1 slower
    deadzone_px: float = 5.0         # ignore tiny movements
    smoothing: float = 0.3     # 0..1 (higher = smoother but laggier)

    # Click tuning
    click_cooldown_s: float = 0
    pinch_click_dist_px: float = 22.0     # thumb-index pinch distance (pixels in camera image)
    pinch_right_dist_px: float = 22.0     # thumb-middle pinch distance (pixels in camera image)


CFG = Config()
pinch_left_down = False
pinch_right_down = False

# hysteresis (release distance should be bigger than press distance)
PINCH_LEFT_RELEASE_PX = CFG.pinch_click_dist_px
PINCH_RIGHT_RELEASE_PX = CFG.pinch_right_dist_px

# ---------- Helpers ----------
def get_screen_size():
    # Built-in, no extra deps
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


# ---------- State ----------
mouse = MouseController()
screen_w, screen_h = get_screen_size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

paused = False
prev_hand_xy = None            # for trackpad mode
abs_mouse_xy = None            # smoothed absolute target for absolute mode
last_click_t = 0.0
last_rclick_t = 0.0


# ---------- Keyboard toggle ----------
def on_press(key):
    global paused, prev_hand_xy, abs_mouse_xy
    try:
        if key.char == CFG.toggle_pause_key:
            paused = not paused
            prev_hand_xy = None
            abs_mouse_xy = None
            print("Paused" if paused else "Resumed")
    except AttributeError:
        # ignore non-char keys
        pass


listener = Listener(on_press=on_press)
listener.daemon = True
listener.start()


# ---------- Core logic ----------
def in_roi(x, y):
    return (CFG.roi_min_x <= x <= CFG.roi_max_x) and (CFG.roi_min_y <= y <= CFG.roi_max_y)


def roi_normalize(x, y):
    # map ROI -> [0,1]
    nx = (x - CFG.roi_min_x) / max(1, (CFG.roi_max_x - CFG.roi_min_x))
    ny = (y - CFG.roi_min_y) / max(1, (CFG.roi_max_y - CFG.roi_min_y))
    nx = clamp(nx, 0.0, 1.0)
    ny = clamp(ny, 0.0, 1.0)
    return nx, ny


def move_mouse_from_middle_finger(mid_xy):
    global prev_hand_xy, abs_mouse_xy

    if paused:
        return

    x, y = mid_xy
    if not in_roi(x, y):
        prev_hand_xy = None
        abs_mouse_xy = None
        return

    if CFG.trackpad_mode:
        # relative deltas
        if prev_hand_xy is None:
            prev_hand_xy = (x, y)
            return

        dx = (x - prev_hand_xy[0])
        dy = (y - prev_hand_xy[1])

        # Convert camera delta -> screen delta
        roi_w = max(1, CFG.roi_max_x - CFG.roi_min_x)
        roi_h = max(1, CFG.roi_max_y - CFG.roi_min_y)

        move_x = (dx / roi_w) * screen_w * CFG.sensitivity
        move_y = (dy / roi_h) * screen_h * CFG.sensitivity

        # invert Y if you want "natural" trackpad feel; keep as-is here
        # apply deadzone
        if abs(move_x) >= CFG.deadzone_px or abs(move_y) >= CFG.deadzone_px:
            # EMA smoothing on delta (simple: scale delta)
            move_x *= (1.0 - (1.0 - CFG.smoothing))
            move_y *= (1.0 - (1.0 - CFG.smoothing))
            mouse.move(move_x, move_y)

        prev_hand_xy = (x, y)
        return

    # absolute mapping: ROI -> screen coordinates with smoothing
    nx, ny = roi_normalize(x, y)

    # flip axes to match mirrored webcam if desired:
    nx = 1 - nx
    ny = 1 - ny  # typical: raise hand = cursor up

    target_x = nx * (screen_w - 1)
    target_y = ny * (screen_h - 1)

    if abs_mouse_xy is None:
        abs_mouse_xy = (target_x, target_y)
    else:
        ax, ay = abs_mouse_xy
        ax = ax + (target_x - ax) * CFG.smoothing
        ay = ay + (target_y - ay) * CFG.smoothing
        abs_mouse_xy = (ax, ay)

    # pynput can't set absolute reliably cross-platform; emulate via move delta
    # read current position indirectly by keeping our own target; this is imperfect but workable
    # better: use pyautogui.position() + moveTo(); not using extra deps here
    # so: move by delta between previous abs target and new abs target
    # store previous target locally
    if not hasattr(move_mouse_from_middle_finger, "_prev_abs"):
        move_mouse_from_middle_finger._prev_abs = abs_mouse_xy
        return

    px, py = move_mouse_from_middle_finger._prev_abs
    ax, ay = abs_mouse_xy
    dx = (ax - px) * CFG.sensitivity
    dy = (ay - py) * CFG.sensitivity

    max_step = 25  # pixels per frame
    dx = clamp(dx, -max_step, max_step)
    dy = clamp(dy, -max_step, max_step)

    if abs(dx) >= CFG.deadzone_px or abs(dy) >= CFG.deadzone_px:
        mouse.move(dx, dy)

    move_mouse_from_middle_finger._prev_abs = abs_mouse_xy


def handle_clicks(thumb_xy, index_xy, middle_xy):
    global pinch_left_down, pinch_right_down, last_click_t, last_rclick_t

    if paused:
        return

    now = time.time()

    d_left2 = dist2(thumb_xy, index_xy)
    d_right2 = dist2(thumb_xy, middle_xy)

    press_left2 = CFG.pinch_click_dist_px ** 2
    release_left2 = (PINCH_LEFT_RELEASE_PX ** 2)

    press_right2 = CFG.pinch_right_dist_px ** 2
    release_right2 = (PINCH_RIGHT_RELEASE_PX ** 2)

    # ---- Left click (thumb-index) edge triggered ----


    thumb_tip = pts[4]
    index_tip = pts[8]
    middle_tip = pts[12]
    ring_tip = pts[13]
    pinky_tip = pts[17]

    distance = thumb_tip[0] - index_tip[0] + thumb_tip[1] - index_tip[1] + \
               thumb_tip[0] - middle_tip[0] + thumb_tip[1] - middle_tip[1] + \
               thumb_tip[0] - ring_tip[0] + thumb_tip[1] - ring_tip[1] + \
               thumb_tip[0] - pinky_tip[0] + thumb_tip[1] - pinky_tip[1]
    
    if distance > 300 and time.time() > (last_click_t + CFG.click_cooldown_s):
        mouse.press(Button.right)
        last_click_t = time.time()
        print("right down")
    else:
        mouse.release(Button.right)

    # ---- Right click (thumb-middle) edge triggered ----
    a = math.sqrt((thumb_xy[0] - index_xy[0])**2 + (thumb_xy[1] - index_xy[1])**2)
    if a < CFG.pinch_right_dist_px and time.time() > (last_click_t + CFG.click_cooldown_s):
        mouse.press(Button.left)
        print("left down")
        last_click_t = time.time()
    else:
        mouse.release(Button.left)

    # if not pinch_right_down:
    #     if d_right2 <= press_right2 and (now - last_rclick_t) >= CFG.click_cooldown_s:
    #         #mouse.click(Button.right)
    #         print("right click")
    #         last_rclick_t = now
    #         pinch_right_down = True
    # else:
    #     if d_right2 >= release_right2:
    #         pinch_right_down = False


# ---------- Main loop ----------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    #frame = cv2.flip(frame, 1)  # mirror for usability
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # draw ROI
    cv2.rectangle(
        frame,
        (CFG.roi_min_x, CFG.roi_min_y),
        (CFG.roi_max_x, CFG.roi_max_y),
        (255, 0, 0),
        2,
    )

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]

        h, w, _ = frame.shape
        pts = []
        for lm in hand.landmark:
            pts.append((int(lm.x * w), int(lm.y * h)))

        # Mediapipe indices
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        MIDDLE_MCP = 9
        

        thumb = pts[THUMB_TIP]
        index = pts[INDEX_TIP]
        middle = pts[MIDDLE_TIP]
        middle_mcp = pts[MIDDLE_MCP]
        ring = pts[13]
        pinky = pts[17]

        # visuals
        cv2.circle(frame, thumb, 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, index, 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, middle, 8, (0, 255, 255), cv2.FILLED)

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        move_mouse_from_middle_finger(middle_mcp)
        handle_clicks(thumb, index, middle)

    cv2.putText(
        frame,
        "PAUSED" if paused else "RUNNING",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255) if paused else (0, 255, 0),
        2,
    )

    cv2.imshow("Hand Mouse", frame)

    if (cv2.waitKey(1) & 0xFF) == ord(CFG.quit_key):
        break

cap.release()
cv2.destroyAllWindows()