"""
Lunar Lander PPO simulation backend.

Runs the Gymnasium LunarLander-v3 environment in a background thread with a
trained PPO model, stepping at a fixed rate. Exposes a tiny HTTP API that the
3D front-end (Blender) polls each frame.

Endpoints (all JSON):
    GET  /state          -> current observation + action + episode info
    POST /reset          -> reset the environment, start a new episode
    POST /pause          -> pause stepping
    POST /resume         -> resume stepping
    GET  /health         -> {"ok": true, "model": "<path>"}

Run:
    python server.py --model ../.venv/models/master_best.zip --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


# -------------------- SIM STATE --------------------

class SimState:
    """Thread-safe snapshot of the current simulation tick."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.episode = 0
        self.step = 0
        self.obs = [0.0] * 8
        self.action = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.paused = False
        self.model_path = ""

    def snapshot(self) -> dict:
        with self._lock:
            x, y, vx, vy, angle, angv, leg1, leg2 = self.obs
            return {
                "episode": self.episode,
                "step": self.step,
                "x": x, "y": y,
                "vx": vx, "vy": vy,
                "angle": angle,
                "angular_velocity": angv,
                "leg1_contact": bool(leg1 > 0.5),
                "leg2_contact": bool(leg2 > 0.5),
                "action": int(self.action),
                "reward": self.reward,
                "total_reward": self.total_reward,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "paused": self.paused,
            }

    def update(self, obs, action, reward, terminated, truncated) -> None:
        with self._lock:
            self.obs = [float(v) for v in obs]
            self.action = int(action)
            self.reward = float(reward)
            self.total_reward += float(reward)
            self.terminated = bool(terminated)
            self.truncated = bool(truncated)
            self.step += 1

    def new_episode(self, obs) -> None:
        with self._lock:
            self.episode += 1
            self.step = 0
            self.obs = [float(v) for v in obs]
            self.action = 0
            self.reward = 0.0
            self.total_reward = 0.0
            self.terminated = False
            self.truncated = False


STATE = SimState()
RESET_EVENT = threading.Event()
PAUSE_EVENT = threading.Event()


# -------------------- SIM LOOP --------------------

def sim_loop(model_path: str, hz: float = 50.0, deterministic: bool = True) -> None:
    import gymnasium as gym
    from stable_baselines3 import PPO

    print(f"[server] loading model: {model_path}")
    model = PPO.load(model_path)
    STATE.model_path = model_path

    env = gym.make("LunarLander-v3")
    obs, _ = env.reset()
    STATE.new_episode(obs)

    dt = 1.0 / hz
    next_tick = time.perf_counter()

    while True:
        # Check pause
        if PAUSE_EVENT.is_set():
            time.sleep(0.05)
            next_tick = time.perf_counter()
            continue

        # Check reset
        if RESET_EVENT.is_set():
            obs, _ = env.reset()
            STATE.new_episode(obs)
            RESET_EVENT.clear()

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _info = env.step(action)
        STATE.update(obs, action, reward, terminated, truncated)

        if terminated or truncated:
            time.sleep(1.2)  # brief pause so the viewer can show the landing/crash
            obs, _ = env.reset()
            STATE.new_episode(obs)

        # pace
        next_tick += dt
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()


# -------------------- HTTP --------------------

class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence console noise
        return

    def do_GET(self):
        if self.path.startswith("/state"):
            self._json(200, STATE.snapshot())
        elif self.path.startswith("/health"):
            self._json(200, {"ok": True, "model": STATE.model_path})
        else:
            self._json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if self.path.startswith("/reset"):
            RESET_EVENT.set()
            self._json(200, {"ok": True, "action": "reset"})
        elif self.path.startswith("/pause"):
            PAUSE_EVENT.set()
            STATE.paused = True
            self._json(200, {"ok": True, "paused": True})
        elif self.path.startswith("/resume"):
            PAUSE_EVENT.clear()
            STATE.paused = False
            self._json(200, {"ok": True, "paused": False})
        else:
            self._json(404, {"error": "unknown endpoint"})


def main():
    ap = argparse.ArgumentParser()
    default_model = str(Path(__file__).resolve().parent.parent / "models" / "master_best.zip")
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    t = threading.Thread(target=sim_loop, args=(args.model, args.hz), daemon=True)
    t.start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] listening on http://{args.host}:{args.port}")
    print(f"[server] GET /state  POST /reset  POST /pause  POST /resume  GET /health")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[server] shutting down")


if __name__ == "__main__":
    main()
