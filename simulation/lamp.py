"""
Lamp Simulation Module
=======================
Renders an animated desk lamp via Pygame that responds to engagement state
with expressive 6-DOF behaviors — tilt, pan, brightness, color, particles.

Design decision: Pygame over matplotlib/tkinter because it provides
hardware-accelerated rendering, smooth 30fps animation, and audio support
all in one lightweight package.
"""

import pygame
import math
import time
import random
from enum import Enum
from dataclasses import dataclass, field


class LampState(Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    ATTENTION_SEEKING = "attention_seeking"
    OBSERVING = "observing"
    LISTENING = "listening"


@dataclass
class LampPose:
    """Simplified 6-DOF lamp pose for expressive animation."""
    head_tilt: float = 0.0          # Nod (-45 to 45 degrees)
    head_pan: float = 0.0           # Turn (-30 to 30)
    neck_angle: float = -30.0       # Arm angle
    brightness: float = 0.5         # 0.0 to 1.0
    color: tuple = (255, 220, 100)  # Light color RGB


class LampSimulation:
    """Pygame-based desk lamp with state-driven expressive animation."""

    def __init__(self, width=1024, height=700):
        self.width = width
        self.height = height
        self.state = LampState.IDLE
        self.pose = LampPose()
        self.target_pose = LampPose()

        # Animation
        self.state_enter_time = time.time()
        self.anim_speed = 0.08
        self.seek_phase = 0.0
        self.particles = []
        self.status_text = ""

        # Head position cache (set during draw)
        self._bulb_pos = (width // 2, height // 2)
        self._head_dir = math.pi

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("LeLamp — Expressive Lamp Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        self.title_font = pygame.font.SysFont("Consolas", 22, bold=True)

        self.base_x = width // 2
        self.base_y = height - 80

        # Set initial target
        self._update_target_for_state()

    # ── Public API ──────────────────────────────────────────

    def set_state(self, new_state: LampState):
        """Transition the lamp to a new behavioral state."""
        if new_state != self.state:
            self.state = new_state
            self.state_enter_time = time.time()
            self._update_target_for_state()

    def update(self, dt: float):
        """Advance animation by dt seconds."""
        s = self.anim_speed

        # Smooth lerp toward targets
        self.pose.head_tilt = _lerp(self.pose.head_tilt, self.target_pose.head_tilt, s)
        self.pose.neck_angle = _lerp(self.pose.neck_angle, self.target_pose.neck_angle, s)
        self.pose.brightness = _lerp(self.pose.brightness, self.target_pose.brightness, s)
        self.pose.color = tuple(
            int(_lerp(c, t, s))
            for c, t in zip(self.pose.color, self.target_pose.color)
        )

        # Per-state overlays
        t = time.time() - self.state_enter_time
        if self.state == LampState.IDLE:
            self.pose.brightness = self.target_pose.brightness + 0.05 * math.sin(t * 1.5)
        elif self.state == LampState.ATTENTION_SEEKING:
            self.seek_phase += dt * 3.0
            self.pose.head_tilt = 5.0 * math.sin(self.seek_phase * 2.0)
            self.pose.head_pan = 10.0 * math.sin(self.seek_phase * 1.3)
            self.pose.brightness = 0.6 + 0.3 * abs(math.sin(self.seek_phase))
        elif self.state == LampState.ENGAGED:
            self.pose.head_tilt = self.target_pose.head_tilt + 2.0 * math.sin(t * 2.0)

        self._update_particles()

    def render(self, webcam_surface=None):
        """Draw everything to the screen."""
        self.screen.fill((20, 20, 30))
        self._draw_floor_glow()
        self._draw_lamp()
        self._draw_light_cone()
        self._draw_particles()
        self._draw_hud()

        if webcam_surface:
            pip_rect = webcam_surface.get_rect()
            pip_rect.topright = (self.width - 15, 15)
            pygame.draw.rect(self.screen, (60, 60, 80), pip_rect.inflate(4, 4), border_radius=4)
            self.screen.blit(webcam_surface, pip_rect)

        pygame.display.flip()

    def frame_to_surface(self, frame) -> pygame.Surface:
        """Convert an OpenCV BGR frame to a small Pygame surface for PiP."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (240, 180))
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def handle_events(self) -> bool:
        """Process Pygame events. Returns False if user wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def close(self):
        pygame.quit()

    # ── State targets ───────────────────────────────────────

    def _update_target_for_state(self):
        targets = {
            LampState.IDLE: LampPose(
                head_tilt=10, neck_angle=-20, brightness=0.3,
                color=(80, 80, 120)),
            LampState.ENGAGED: LampPose(
                head_tilt=-15, neck_angle=-40, brightness=1.0,
                color=(255, 220, 100)),
            LampState.ATTENTION_SEEKING: LampPose(
                head_tilt=0, neck_angle=-30, brightness=0.7,
                color=(255, 140, 60)),
            LampState.OBSERVING: LampPose(
                head_tilt=-20, neck_angle=-45, brightness=0.9,
                color=(100, 200, 255)),
            LampState.LISTENING: LampPose(
                head_tilt=-10, neck_angle=-35, brightness=0.8,
                color=(180, 255, 180)),
        }
        self.target_pose = targets.get(self.state, LampPose())

    # ── Drawing helpers ─────────────────────────────────────

    def _draw_lamp(self):
        bx, by = self.base_x, self.base_y

        # Base
        base_pts = [(bx-60, by), (bx+60, by), (bx+45, by-15), (bx-45, by-15)]
        pygame.draw.polygon(self.screen, (60, 60, 70), base_pts)
        pygame.draw.polygon(self.screen, (80, 80, 90), base_pts, 2)

        # Lower arm
        arm_len = 160
        neck_rad = math.radians(self.pose.neck_angle)
        jx, jy = bx + 10, by - 15
        ex = jx + arm_len * math.sin(neck_rad * 0.5)
        ey = jy - arm_len * math.cos(neck_rad * 0.5)
        pygame.draw.line(self.screen, (90, 90, 100), (jx, jy), (int(ex), int(ey)), 6)

        # Upper arm
        upper_len = 130
        head_angle = neck_rad + math.radians(self.pose.head_tilt)
        hx = ex + upper_len * math.sin(head_angle)
        hy = ey - upper_len * math.cos(head_angle)
        pygame.draw.line(self.screen, (100, 100, 110), (int(ex), int(ey)), (int(hx), int(hy)), 5)

        # Joints
        pygame.draw.circle(self.screen, (120, 120, 130), (jx, jy), 7)
        pygame.draw.circle(self.screen, (120, 120, 130), (int(ex), int(ey)), 6)

        # Lamp head / shade
        shade_dir = head_angle + math.pi
        perp = shade_dir + math.pi / 2
        sz = 35
        pan_off = math.radians(self.pose.head_pan) * 0.3

        tip = (hx + sz * 0.5 * math.sin(shade_dir), hy + sz * 0.5 * math.cos(shade_dir))
        left = (hx + sz * math.sin(perp + pan_off), hy + sz * math.cos(perp + pan_off))
        right = (hx - sz * math.sin(perp - pan_off), hy - sz * math.cos(perp - pan_off))
        shade_pts = [_int2(tip), _int2(left), _int2(right)]

        br = self.pose.brightness
        shade_col = tuple(min(255, int(c * br * 0.4 + 40)) for c in self.pose.color)
        pygame.draw.polygon(self.screen, shade_col, shade_pts)
        pygame.draw.polygon(self.screen, (150, 150, 160), shade_pts, 2)

        # Bulb glow
        bulb_x = int((left[0] + right[0]) / 2)
        bulb_y = int((left[1] + right[1]) / 2)
        gr = int(12 * br)
        if gr > 0:
            glow = pygame.Surface((gr * 4, gr * 4), pygame.SRCALPHA)
            for r in range(gr, 0, -1):
                a = int(100 * (r / gr) * br)
                pygame.draw.circle(glow, (*self.pose.color, a), (gr * 2, gr * 2), r)
            self.screen.blit(glow, (bulb_x - gr * 2, bulb_y - gr * 2))

        self._bulb_pos = (bulb_x, bulb_y)
        self._head_dir = shade_dir

    def _draw_light_cone(self):
        if self.pose.brightness < 0.1:
            return
        bx, by = self._bulb_pos
        length = int(300 * self.pose.brightness)
        spread = 0.4
        d = self._head_dir
        lx = bx + length * math.sin(d - spread)
        ly = by + length * math.cos(d - spread)
        rx = bx + length * math.sin(d + spread)
        ry = by + length * math.cos(d + spread)

        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        alpha = int(40 * self.pose.brightness)
        pygame.draw.polygon(surf, (*self.pose.color, alpha),
                            [(bx, by), (int(lx), int(ly)), (int(rx), int(ry))])
        self.screen.blit(surf, (0, 0))

    def _draw_floor_glow(self):
        surf = pygame.Surface((self.width, 60), pygame.SRCALPHA)
        a = int(30 * self.pose.brightness)
        pygame.draw.ellipse(surf, (*self.pose.color, a),
                            (self.base_x - 150, 0, 300, 40))
        self.screen.blit(surf, (0, self.base_y - 10))

    def _draw_hud(self):
        colors = {
            LampState.IDLE: (128, 128, 160),
            LampState.ENGAGED: (100, 255, 100),
            LampState.ATTENTION_SEEKING: (255, 160, 60),
            LampState.OBSERVING: (100, 200, 255),
            LampState.LISTENING: (180, 255, 180),
        }
        c = colors.get(self.state, (200, 200, 200))
        label = self.state.value.replace('_', ' ').title()
        self.screen.blit(self.title_font.render(f"● {label}", True, c), (15, 15))

        # Brightness bar
        self.screen.blit(self.font.render("Brightness", True, (120, 120, 140)), (15, 50))
        pygame.draw.rect(self.screen, (40, 40, 50), (15, 72, 150, 8), border_radius=4)
        pygame.draw.rect(self.screen, c, (15, 72, int(150 * self.pose.brightness), 8),
                         border_radius=4)

        if self.status_text:
            self.screen.blit(
                self.font.render(self.status_text, True, (160, 160, 180)),
                (15, self.height - 35))

    def _update_particles(self):
        if self.pose.brightness > 0.5:
            if random.random() < self.pose.brightness * 0.3:
                bx, by = self._bulb_pos
                self.particles.append({
                    'x': bx + random.uniform(-20, 20),
                    'y': by + random.uniform(-10, 30),
                    'vx': random.uniform(-0.5, 0.5),
                    'vy': random.uniform(-1.5, -0.3),
                    'life': 1.0,
                    'decay': random.uniform(0.01, 0.03),
                    'size': random.uniform(1, 3),
                })
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= p['decay']
        self.particles = [p for p in self.particles if p['life'] > 0][:50]

    def _draw_particles(self):
        for p in self.particles:
            r = max(1, int(p['size'] * p['life']))
            c = tuple(min(255, int(v * p['life'])) for v in self.pose.color)
            pygame.draw.circle(self.screen, c, (int(p['x']), int(p['y'])), r)


# ── Utility ─────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _int2(pair):
    return (int(pair[0]), int(pair[1]))
