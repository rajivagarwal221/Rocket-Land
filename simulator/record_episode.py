"""
Collect a PPO episode trajectory from the LunarLander-v3 env, bake it
as keyframes on the Rocket rig in combined.blend, then render to MP4.

Run (headless):
    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" ^
      -b blender/combined.blend ^
      --python simulator/record_episode.py ^
      -- --frames 450 --out blender/sim_episode.mp4

The `--` separates Blender's args from this script's args.
"""
from __future__ import annotations
import argparse, math, os, sys, subprocess
from pathlib import Path


# ---------------- 1. collect a trajectory ----------------

def collect_trajectory(model_path: Path, max_steps: int, venv_python: Path):
    """Run the PPO env in a subprocess using the project's venv (which has
    torch / gymnasium installed) and return a list of state dicts."""
    collector = r"""
import sys, json
import gymnasium as gym
from stable_baselines3 import PPO
model_path, max_steps = sys.argv[1], int(sys.argv[2])
model = PPO.load(model_path)
env = gym.make("LunarLander-v3")
obs, _ = env.reset(seed=11)
traj = []
total = 0.0
for i in range(max_steps):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(a)
    total += float(r)
    traj.append({
        "x": float(obs[0]), "y": float(obs[1]),
        "vx": float(obs[2]), "vy": float(obs[3]),
        "angle": float(obs[4]), "av": float(obs[5]),
        "leg1": bool(obs[6] > 0.5), "leg2": bool(obs[7] > 0.5),
        "action": int(a), "reward": float(r), "total": total,
        "term": bool(term), "trunc": bool(trunc),
    })
    if term or trunc:
        break
env.close()
print("__TRAJ_START__")
print(json.dumps(traj))
print("__TRAJ_END__")
"""
    print(f"[collect] launching venv python: {venv_python}")
    out = subprocess.check_output(
        [str(venv_python), "-c", collector, str(model_path), str(max_steps)],
        cwd=str(model_path.parent.parent),  # project root (models/<file>.zip → parent.parent = project root)
        text=True,
    )
    s = out.index("__TRAJ_START__") + len("__TRAJ_START__")
    e = out.index("__TRAJ_END__")
    import json
    traj = json.loads(out[s:e].strip())
    print(f"[collect] got {len(traj)} steps, final reward={traj[-1]['total']:.1f}")
    return traj


# ---------------- 2. bake trajectory as keyframes ----------------

def bake_trajectory(traj, pad_x=0.0, pad_y=5.5, pad_z=0.03, scale_x=6.0, scale_z=6.0):
    import bpy
    rocket_coll = bpy.data.collections.get("Rocket")
    if rocket_coll is None:
        raise RuntimeError("combined.blend must have a 'Rocket' collection")

    # ensure the rig empty exists and all rocket parts parent to it
    rig = bpy.data.objects.get("RocketRig")
    if rig is None:
        parts = [o for o in rocket_coll.objects if o.type == "MESH"]
        cx = sum(o.location.x for o in parts) / max(1, len(parts))
        cy = sum(o.location.y for o in parts) / max(1, len(parts))
        cz = sum(o.location.z for o in parts) / max(1, len(parts))
        rig = bpy.data.objects.new("RocketRig", None)
        rig.empty_display_type = "PLAIN_AXES"; rig.empty_display_size = 0.6
        rig.location = (cx, cy, cz)
        rocket_coll.objects.link(rig)
        for o in list(rocket_coll.objects):
            if o is rig or o.parent is not None:
                continue
            mw = o.matrix_world.copy()
            o.parent = rig
            o.matrix_parent_inverse = rig.matrix_world.inverted()
            o.matrix_world = mw
    # rename Flame/FlameInner -> MainFlame/MainFlameInner
    for src, dst in (("Flame", "MainFlame"), ("FlameInner", "MainFlameInner")):
        o = bpy.data.objects.get(src)
        if o is not None:
            o.name = dst

    scn = bpy.context.scene
    scn.frame_start = 1
    scn.frame_end = len(traj)

    # animate rig + flame visibility per step (1 traj step == 1 frame)
    main_f = bpy.data.objects.get("MainFlame")
    inner_f = bpy.data.objects.get("MainFlameInner")
    for i, s in enumerate(traj, start=1):
        scn.frame_set(i)
        rig.location = (pad_x + s["x"] * scale_x,
                        pad_y,
                        pad_z + max(0.05, s["y"] * scale_z))
        rig.rotation_euler = (rig.rotation_euler.x, -s["angle"], rig.rotation_euler.z)
        rig.keyframe_insert("location")
        rig.keyframe_insert("rotation_euler")
        if main_f is not None:
            main_f.hide_viewport = (s["action"] != 2)
            main_f.hide_render = (s["action"] != 2)
            main_f.keyframe_insert("hide_viewport")
            main_f.keyframe_insert("hide_render")
        if inner_f is not None:
            inner_f.hide_viewport = (s["action"] != 2)
            inner_f.hide_render = (s["action"] != 2)
            inner_f.keyframe_insert("hide_viewport")
            inner_f.keyframe_insert("hide_render")
    # constant interpolation on visibility, linear on transform
    def _iter_fcurves(action):
        """Blender 5.x splits fcurves across action layers/channelbags."""
        if hasattr(action, "fcurves") and action.fcurves:
            yield from action.fcurves
            return
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for cb in getattr(strip, "channelbags", []):
                    yield from cb.fcurves
    for obj in (rig, main_f, inner_f):
        if obj is None or not obj.animation_data or not obj.animation_data.action:
            continue
        for fc in _iter_fcurves(obj.animation_data.action):
            mode = "CONSTANT" if "hide_" in fc.data_path else "LINEAR"
            for kp in fc.keyframe_points:
                kp.interpolation = mode
    print(f"[bake] baked {len(traj)} frames")


# ---------------- 3. render to video ----------------

def configure_render(out_path):
    import bpy
    scn = bpy.context.scene
    scn.render.fps = 50  # env steps per second
    scn.render.resolution_x = 720
    scn.render.resolution_y = 1080  # portrait, matches combined.blend framing
    scn.render.resolution_percentage = 60  # faster preview
    try:
        scn.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scn.render.engine = "BLENDER_EEVEE"
    # This Blender build has no FFMPEG writer — render PNG sequence into
    # a sibling folder, then stitch to MP4 with opencv afterwards.
    scn.render.image_settings.file_format = "PNG"
    frames_dir = out_path.parent / (out_path.stem + "_frames")
    frames_dir.mkdir(exist_ok=True)
    scn.render.filepath = str(frames_dir / "f_")
    # stash for the stitch step
    scn["__frames_dir__"] = str(frames_dir)
    # lighter sampling for a speedy preview
    if hasattr(scn.eevee, "taa_render_samples"):
        scn.eevee.taa_render_samples = 24


def main():
    # argv after `--`
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"models\master_best.zip")
    ap.add_argument("--venv-python", default=r".venv\Scripts\python.exe")
    ap.add_argument("--frames", type=int, default=450)
    ap.add_argument("--out", default=r"blender\sim_episode.mp4")
    args = ap.parse_args(argv)

    proj = Path(__file__).resolve().parent.parent
    model_path = (proj / args.model).resolve()
    venv_py    = (proj / args.venv_python).resolve()
    out_path   = (proj / args.out).resolve()

    traj = collect_trajectory(model_path, args.frames, venv_py)
    bake_trajectory(traj)
    configure_render(out_path)

    import bpy
    print(f"[render] writing {out_path}")
    bpy.ops.render.render(animation=True)
    print(f"[render] done: {out_path}")


main()
