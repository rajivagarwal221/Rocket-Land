"""Watch the 3D lander fly, or record it.

    python src/watch_3d.py --baseline                          live window, PID baseline
    python src/watch_3d.py --model models/master_v28_best.zip  live window, trained policy
    python src/watch_3d.py --baseline --video logs/flight.mp4  headless recording

The live window is MuJoCo's own viewer: drag to orbit, scroll to zoom,
Space to pause. Recording needs `pip install imageio imageio-ffmpeg`.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from sim3d.baseline import PIDController
from sim3d.rocket_env import OBS_DIM
from sim3d.wrappers import base_env, make_vec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--target-radius", type=float, default=0.5)
    ap.add_argument("--video", type=Path, help="write an MP4 instead of opening a window")
    ap.add_argument("--camera", choices=("wide", "chase"), default="wide", help="video camera")
    args = ap.parse_args()
    if args.baseline == (args.model is not None):
        ap.error("pass exactly one of --model or --baseline")

    model, n_stack = None, 1
    if args.model is not None:
        if not args.model.exists():
            ap.error(f"checkpoint not found: {args.model}")
        from stable_baselines3 import SAC

        model = SAC.load(str(args.model), device="cpu")
        n_stack = model.observation_space.shape[0] // OBS_DIM

    vec = make_vec(1, args.seed, args.difficulty, args.target_radius, n_stack)
    env = base_env(vec)
    pid = PIDController()

    def fly(on_frame):
        for ep in range(args.episodes):
            vec.seed(args.seed + ep)
            obs = vec.reset()
            pid.reset()
            while True:
                if model is None:
                    action = pid(env)[None, :]
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = vec.step(action)
                if done[0]:
                    info = infos[0]
                    print(
                        f"episode {ep + 1}: {info['outcome']:<13} "
                        f"error {info['landing_error']:.2f} m, "
                        f"touchdown {info['impact_speed']:.2f} m/s, "
                        f"wind {info['wind_speed']:.1f} m/s"
                    )
                    break
                # After `done` the vec env has already reset, so only draw live steps.
                if not on_frame():
                    return

    if args.video:
        import imageio
        import mujoco

        renderer = mujoco.Renderer(env.model, height=720, width=960)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(args.video), fps=50) as writer:

            def on_frame():
                env.sync_visuals()
                renderer.update_scene(env.data, camera=args.camera)
                writer.append_data(renderer.render())
                return True

            fly(on_frame)
        renderer.close()
        print(f"video written: {args.video.resolve()}")
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 12.0
            viewer.cam.elevation = -15.0
            viewer.cam.lookat[:] = (0.0, 0.0, 3.0)

            def on_frame():
                env.sync_visuals()
                viewer.sync()
                time.sleep(env.dt_ctrl)
                return viewer.is_running()

            fly(on_frame)
    vec.close()


if __name__ == "__main__":
    main()
