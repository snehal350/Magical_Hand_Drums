# magical_hand_drums.py
import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math
import time

# --- Constantsa ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BACKGROUND_COLOR = (12, 16, 28)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

# --- Drum Colors ---
PAD_COLORS = [
    (255, 99, 71),    # Kick (tomato)
    (255, 206, 84),   # Snare (amber)
    (169, 219, 114),  # Hi-hat (mint)
    (135, 206, 250),  # Tom (sky)
    (203, 153, 201)   # Crash (lavender)
]
PAD_SHADOW = (30, 30, 40)

# --- Gameplay ---
DEBOUNCE_FRAMES = 3

# --- Particles ---
PARTICLES = []

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-5, -1)
        self.lifespan = random.randint(18, 45)
        self.radius = random.randint(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15  # gravity
        self.lifespan -= 1
        self.radius -= 0.08
        if self.radius < 0: self.radius = 0

    def draw(self, screen):
        alpha = max(0, int(255 * (self.lifespan / 45)))
        if alpha <= 0: return
        surface = pygame.Surface((int(self.radius*2)+2, int(self.radius*2)+2), pygame.SRCALPHA)
        pygame.draw.circle(surface, self.color + (alpha,), (int(self.radius), int(self.radius)), int(self.radius))
        screen.blit(surface, (self.x - self.radius, self.y - self.radius))

# --- Sound Generation (drum-ish) ---
def make_stereo(sound_mono):
    stereo = np.ascontiguousarray(np.vstack((sound_mono, sound_mono)).T)
    return stereo

def generate_kick(duration=0.6, sample_rate=44100, amplitude=0.9):
    # pitch drop sine (kick)
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    # exponential drop from ~120Hz down to ~40Hz
    start_hz = 120.0
    end_hz = 40.0
    freqs = start_hz * (end_hz / start_hz) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
    wave = np.sin(phase)
    env = np.exp(-8 * t)  # short decay
    wave = wave * env
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    sound = (wave * (2**15 - 1) * amplitude).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(sound))

def generate_snare(duration=0.35, sample_rate=44100, amplitude=0.7):
    # snare = filtered noise + short body
    n = int(duration * sample_rate)
    noise = np.random.normal(0, 1, n)
    # high-pass-ish by subtracting a slow-moving average
    kernel = int(0.005 * sample_rate)
    if kernel < 1: kernel = 1
    smooth = np.convolve(noise, np.ones(kernel)/kernel, mode='same')
    hp = noise - smooth
    env = np.exp(-12 * np.linspace(0, duration, n))
    wave = hp * env
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    sound = (wave * (2**15 - 1) * amplitude).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(sound))

def generate_hihat(duration=0.12, sample_rate=44100, amplitude=0.45):
    n = int(duration * sample_rate)
    noise = np.random.normal(0, 1, n)
    # metallic by high frequency emphasis
    t = np.linspace(0, duration, n, endpoint=False)
    metallic = noise * (1.0 - t/duration) * (1 + 2 * np.sin(2 * np.pi * 12000 * t))
    env = np.exp(-40 * t)
    wave = metallic * env
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    sound = (wave * (2**15 - 1) * amplitude).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(sound))

def generate_tom(duration=0.45, sample_rate=44100, amplitude=0.7, base_hz=220):
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    freqs = base_hz * (0.95 ** (t * 10))  # slight drop
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
    body = np.sin(phase)
    env = np.exp(-6 * t)
    wave = body * env
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    sound = (wave * (2**15 - 1) * amplitude).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(sound))

def generate_crash(duration=1.2, sample_rate=44100, amplitude=0.6):
    n = int(duration * sample_rate)
    noise = np.random.normal(0, 1, n)
    t = np.linspace(0, duration, n, endpoint=False)
    env = np.exp(-3 * t)  # long decay
    wave = noise * env
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    sound = (wave * (2**15 - 1) * amplitude).astype(np.int16)
    return pygame.sndarray.make_sound(make_stereo(sound))

# --- Finger Counting Logic (same as your) ---
def count_fingers(hand_landmarks, hand_label):
    if not hand_landmarks: return 0
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0
    try:
        # Thumb
        if hand_label == 'Right':
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x:
                fingers_up += 1
        else:
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 2].x:
                fingers_up += 1
        # Other fingers
        for i in range(1, 5):
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                fingers_up += 1
    except Exception:
        return 0
    return fingers_up

# --- Drawing helpers ---
STARS = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(1, 2)) for _ in range(120)]
def draw_background(screen):
    screen.fill(BACKGROUND_COLOR)
    for x, y, r in STARS:
        brightness = random.randint(120, 255)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), r)

def draw_circular_webcam(screen, frame):
    frame_height, frame_width, _ = frame.shape
    cam_diameter = 300
    
    # Scale webcam frame and convert to Pygame surface
    frame = cv2.resize(frame, (cam_diameter, cam_diameter))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(np.rot90(frame))
    
    # Create circular mask
    mask_surface = pygame.Surface((cam_diameter, cam_diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (cam_diameter // 2, cam_diameter // 2), cam_diameter // 2)
    
    # Apply mask to the webcam surface
    cam_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    # Position and draw the "magic mirror"
    pos_x = SCREEN_WIDTH // 2 - cam_diameter // 2
    pos_y = 150
    pygame.draw.circle(screen, PAD_COLORS[4], (pos_x + cam_diameter//2, pos_y + cam_diameter//2), cam_diameter//2 + 8)
    screen.blit(cam_surface, (pos_x, pos_y))



def draw_drum_pads(screen, pad_names, active_pads):
    # Layout five circular pads across center-bottom
    pad_count = len(pad_names)
    pad_radius = 70
    spacing = (SCREEN_WIDTH - 200) // pad_count
    center_y = SCREEN_HEIGHT - 220
    start_x = 100 + spacing // 2
    font = pygame.font.Font(None, 28)
    for i, name in enumerate(pad_names):
        cx = start_x + i * spacing
        cy = center_y
        shadow_pos = (cx + 6, cy + 8)
        pygame.draw.circle(screen, PAD_SHADOW, shadow_pos, pad_radius + 8)
        color = PAD_COLORS[i % len(PAD_COLORS)]
        is_active = i in active_pads
        draw_r = pad_radius + (8 if is_active else 0)
        pygame.draw.circle(screen, color, (cx, cy), draw_r)
        pygame.draw.circle(screen, BLACK_COLOR, (cx, cy), draw_r, 4)
        text = font.render(name, True, BLACK_COLOR)
        screen.blit(text, (cx - text.get_width() // 2, cy - text.get_height() // 2))

# --- Main Application ---
def run_drum_app():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=4, buffer=512)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Hand Drums")
    title_font = pygame.font.Font(None, 72)
    info_font = pygame.font.Font(None, 36)

    # Mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Drum pads and sounds
    pad_names = ["Kick", "Snare", "HiHat", "Tom", "Crash"]
    # generate sounds
    drum_sounds = [
        generate_kick(),
        generate_snare(),
        generate_hihat(),
        generate_tom(base_hz=180),
        generate_crash()
    ]

    # state
    stable_fingers = {'Left': 0, 'Right': 0}
    pending_fingers = {'Left': 0, 'Right': 0}
    debounce_counters = {'Left': 0, 'Right': 0}

    global PARTICLES

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        ret, frame = cap.read()
        print("Frame OK")
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        raw_fingers = {'Left': 0, 'Right': 0}
        active_pads = set()

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_fingers[hand_label] = count_fingers(hand_landmarks, hand_label)

        for hand_label in ['Left', 'Right']:
            if raw_fingers[hand_label] == pending_fingers[hand_label]:
                debounce_counters[hand_label] += 1
            else:
                pending_fingers[hand_label] = raw_fingers[hand_label]
                debounce_counters[hand_label] = 0

            if debounce_counters[hand_label] >= DEBOUNCE_FRAMES:
                if stable_fingers[hand_label] != pending_fingers[hand_label]:
                    stable_fingers[hand_label] = pending_fingers[hand_label]
                    # play sound if >0 fingers
                    if stable_fingers[hand_label] > 0:
                        # map finger count to pad index:
                        # Left hand controls pads 0-2 (Kick, Snare, HiHat)
                        # Right hand controls pads 2-4 (HiHat, Tom, Crash)
                        if hand_label == 'Left':
                            offset = 0
                        else:
                            offset = 2
                        pad_idx = offset + (stable_fingers[hand_label] - 1)
                        if pad_idx < 0: pad_idx = 0
                        if pad_idx >= len(drum_sounds):
                            pad_idx = len(drum_sounds) - 1
                        # play
                        try:
                            drum_sounds[pad_idx].play()
                        except Exception:
                            pass
                        # spawn particles at pad position
                        spacing = (SCREEN_WIDTH - 200) // len(pad_names)
                        pad_x = 100 + spacing // 2 + pad_idx * spacing
                        pad_y = SCREEN_HEIGHT - 220
                        for _ in range(28):
                            PARTICLES.append(Particle(pad_x + random.uniform(-20,20), pad_y + random.uniform(-10,40), PAD_COLORS[pad_idx % len(PAD_COLORS)]))

            # Visual active pad highlight based on stable gesture
            if stable_fingers[hand_label] > 0:
                if hand_label == 'Left':
                    offset = 0
                else:
                    offset = 2
                pad_idx = offset + (stable_fingers[hand_label] - 1)
                if 0 <= pad_idx < len(pad_names):
                    active_pads.add(pad_idx)

        # --- Drawing ---
        draw_background(screen)
        draw_drum_pads(screen, pad_names, active_pads)

        # particles update + draw
        for p in PARTICLES:
            p.update()
        PARTICLES = [p for p in PARTICLES if p.lifespan > 0]
        for p in PARTICLES:
            p.draw(screen)

        draw_circular_webcam(screen, frame)

        # UI text
        title = title_font.render("Magical Hand Drums", True, WHITE_COLOR)
        info_l = info_font.render(f"Left fingers: {stable_fingers['Left']}", True, WHITE_COLOR)
        info_r = info_font.render(f"Right fingers: {stable_fingers['Right']}", True, WHITE_COLOR)
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 30))
        screen.blit(info_l, (40, 60))
        screen.blit(info_r, (SCREEN_WIDTH - 40 - info_r.get_width(), 60))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    print("Application closed.")

if __name__ == '__main__':
    run_drum_app()