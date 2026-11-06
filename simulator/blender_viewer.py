"""
Blender live viewer for the Lunar Lander simulator backend.

Open combined.blend, open this file in the Scripting workspace, and press Run.
The script will:
  * ensure there is a single "Rocket" Empty that owns all parts of the
    polished rocket (so one transform drives the whole thing),
  * ensure there are MainFlame / LeftFlame / RightFlame objects that
    represent the engine plumes (toggled by action),
  * start a background poll of the backend (default http://127.0.0.1:8765)
    and smoothly drive the Rocket transform from the live Gym state.

Press ESC in the viewport, or use the "Lunar" N-panel's Stop button, to end.
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request

import bpy
from mathutils import Vector

# ---------------- CONFIG ----------------
HOST = "127.0.0.1"
PORT = 8765
POLL_HZ = 40.0            # times/sec we poll the backend
LERP_RATE = 18.0          # smoothing: higher = snappier
# World calibration for combined.blend (pad centered at world Y=5.5)
PAD_WORLD_X = 0.0
PAD_WORLD_Y = 5.5
PAD_WORLD_Z = 0.03
SCALE_X = 6.0             # Blender units per env x unit
SCALE_Z = 6.0             # Blender units per env y unit (env y -> blender z)
HOVER_MIN_Z = 0.05        # don't let the rocket phase into the floor

# ---------------- SHARED STATE ----------------
_LATEST = {"ok": False}
_LOCK = threading.Lock()
_STOP = threading.Event()


def _poller() -> None:
    url = f"http://{HOST}:{PORT}/state"
    dt = 1.0 / POLL_HZ
    next_t = time.perf_counter()
    while not _STOP.is_set():
        try:
            with urllib.request.urlopen(url, timeout=0.25) as r:
                data = json.loads(r.read().decode())
            data["ok"] = True
            with _LOCK:
                _LATEST.clear(); _LATEST.update(data)
        except Exception:
            with _LOCK:
                _LATEST["ok"] = False
        next_t += dt
        sleep = next_t - time.perf_counter()
        if sleep > 0: time.sleep(sleep)
        else: next_t = time.perf_counter()


# ---------------- ONE-SHOT SCENE SETUP ----------------

def _ensure_scene_setup():
    """
    Create a single 'Rocket' Empty parented over the existing rocket parts,
    and ensure MainFlame/LeftFlame/RightFlame exist. Idempotent.
    """
    # 1. Rocket empty + parenting -----------------------------------
    rocket_coll = bpy.data.collections.get("Rocket")
    if rocket_coll is None:
        # no "Rocket" collection -> nothing to adapt
        return None

    rocket_empty = bpy.data.objects.get("RocketRig")
    if rocket_empty is None:
        # find current centroid of the rocket parts in world space
        parts = [o for o in rocket_coll.objects if o.type == "MESH"]
        if parts:
            cx = sum(o.location.x for o in parts) / len(parts)
            cy = sum(o.location.y for o in parts) / len(parts)
            cz = sum(o.location.z for o in parts) / len(parts)
        else:
            cx, cy, cz = PAD_WORLD_X, PAD_WORLD_Y, 3.0

        rocket_empty = bpy.data.objects.new("RocketRig", None)
        rocket_empty.empty_display_type = "PLAIN_AXES"
        rocket_empty.empty_display_size = 0.5
        rocket_empty.location = (cx, cy, cz)
        rocket_coll.objects.link(rocket_empty)

        # parent all rocket parts to the empty, keeping their world transforms
        for o in list(rocket_coll.objects):
            if o is rocket_empty: continue
            if o.parent is None:
                # emulate "keep transform" parenting
                mw = o.matrix_world.copy()
                o.parent = rocket_empty
                o.matrix_parent_inverse = rocket_empty.matrix_world.inverted()
                o.matrix_world = mw

    # 2. Flame objects ------------------------------------------------
    main_f = bpy.data.objects.get("MainFlame")
    if main_f is None:
        # rename existing "Flame" / "FlameInner" from combined.blend, if present
        for src, dst in (("Flame", "MainFlame"), ("FlameInner", "MainFlameInner")):
            o = bpy.data.objects.get(src)
            if o is not None:
                o.name = dst
        main_f = bpy.data.objects.get("MainFlame")

    if main_f is None:
        # create one from scratch, under the empty
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.35, radius2=0.03, depth=1.0)
        main_f = bpy.context.view_layer.objects.active
        main_f.name = "MainFlame"
        # orange emissive material
        mat = bpy.data.materials.new("MF_Mat"); mat.use_nodes = True
        b = mat.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (1, 0.6, 0.1, 1)
        b.inputs["Emission Color"].default_value = (1, 0.6, 0.1, 1)
        b.inputs["Emission Strength"].default_value = 12
        b.inputs["Alpha"].default_value = 0.85
        mat.blend_method = "BLEND"
        main_f.data.materials.append(mat)
        rocket_coll.objects.link(main_f)
        for c in list(main_f.users_collection):
            if c is not rocket_coll:
                try: c.objects.unlink(main_f)
                except: pass
        # position below nozzle in rig-local space (rig is at rocket centroid ~3m)
        main_f.location = (rocket_empty.location.x,
                           rocket_empty.location.y,
                           rocket_empty.location.z - 1.4)
        main_f.parent = rocket_empty
        main_f.matrix_parent_inverse = rocket_empty.matrix_world.inverted()

    for name, offset in (("LeftFlame", (-0.55, 0, 0.2)), ("RightFlame", (0.55, 0, 0.2))):
        f = bpy.data.objects.get(name)
        if f is None:
            rot_y = math.radians(-90 if "Left" in name else 90)
            bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.12, radius2=0.01, depth=0.45, rotation=(0, rot_y, 0))
            f = bpy.context.view_layer.objects.active
            f.name = name
            mat = bpy.data.materials.new(f"{name}_Mat"); mat.use_nodes = True
            b = mat.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (0.5, 0.85, 1, 1)
            b.inputs["Emission Color"].default_value = (0.5, 0.85, 1, 1)
            b.inputs["Emission Strength"].default_value = 10
            b.inputs["Alpha"].default_value = 0.8
            mat.blend_method = "BLEND"
            f.data.materials.append(mat)
            rocket_coll.objects.link(f)
            for c in list(f.users_collection):
                if c is not rocket_coll:
                    try: c.objects.unlink(f)
                    except: pass
            f.location = (rocket_empty.location.x + offset[0],
                          rocket_empty.location.y + offset[1],
                          rocket_empty.location.z + offset[2])
            f.parent = rocket_empty
            f.matrix_parent_inverse = rocket_empty.matrix_world.inverted()

    # start with flames hidden
    for n in ("MainFlame", "MainFlameInner", "LeftFlame", "RightFlame"):
        o = bpy.data.objects.get(n)
        if o is not None:
            o.hide_viewport = True; o.hide_render = True

    # 3. HUD --------------------------------------------------------
    hud = bpy.data.objects.get("HUD")
    if hud is None:
        bpy.ops.object.text_add(location=(PAD_WORLD_X, PAD_WORLD_Y + 2.5, 6.0))
        hud = bpy.context.view_layer.objects.active
        hud.name = "HUD"
        hud.data.body = "EP 00   STEP 0000   SCORE +000.0   READY"
        hud.data.size = 0.5
        hud.data.align_x = "CENTER"
        hud.rotation_euler = (math.radians(90), 0, 0)
        mat = bpy.data.materials.new("HUD_Mat"); mat.use_nodes = True
        b = mat.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.9, 0.95, 1, 1)
        b.inputs["Emission Color"].default_value = (0.6, 0.9, 1, 1)
        b.inputs["Emission Strength"].default_value = 3.0
        hud.data.materials.append(mat)
        env = bpy.data.collections.get("Environment") or bpy.context.scene.collection
        env.objects.link(hud)
        for c in list(hud.users_collection):
            if c is not env:
                try: c.objects.unlink(hud)
                except: pass

    return rocket_empty


# ---------------- APPLY STATE ----------------

def _apply_state(rig, state, interp):
    if rig is None:
        return

    tx = PAD_WORLD_X + state["x"] * SCALE_X
    tz = PAD_WORLD_Z + max(HOVER_MIN_Z, state["y"] * SCALE_Z)
    ty = PAD_WORLD_Y
    tang = -state["angle"]  # Blender Y-rotation matches env tilt sign

    cx, cy, cz = rig.location
    cang_y = rig.rotation_euler.y
    k = max(0.0, min(1.0, interp))
    rig.location.x = cx + (tx - cx) * k
    rig.location.y = cy + (ty - cy) * k
    rig.location.z = cz + (tz - cz) * k
    rig.rotation_euler.y = cang_y + (tang - cang_y) * k

    a = state["action"]
    for name, hit in (("MainFlame", a == 2), ("LeftFlame", a == 1), ("RightFlame", a == 3)):
        o = bpy.data.objects.get(name)
        if o is not None:
            o.hide_viewport = not hit
            o.hide_render = not hit
    inner = bpy.data.objects.get("MainFlameInner")
    if inner is not None:
        inner.hide_viewport = not (a == 2); inner.hide_render = not (a == 2)

    hud = bpy.data.objects.get("HUD")
    if hud is not None and hud.type == "FONT":
        status = ("LANDED" if state["terminated"] and state["leg1_contact"] and state["leg2_contact"]
                  else "CRASH" if state["terminated"] else "FLIGHT")
        hud.data.body = (
            f"EP {state['episode']:02d}   STEP {state['step']:04d}   "
            f"SCORE {state['total_reward']:+07.1f}   {status}"
        )


# ---------------- MODAL OPERATOR ----------------

class LunarSimOperator(bpy.types.Operator):
    bl_idname = "wm.lunar_sim"
    bl_label = "Lunar Lander Live Sim"
    bl_options = {"REGISTER"}

    _timer = None
    _last_time = 0.0
    _thread = None
    _rig = None

    def modal(self, context, event):
        if event.type == "ESC" or _STOP.is_set():
            self.cancel(context)
            return {"CANCELLED"}

        if event.type == "TIMER":
            now = time.perf_counter()
            dt = max(1e-4, now - self._last_time)
            self._last_time = now
            with _LOCK:
                data = dict(_LATEST)
            if data.get("ok"):
                interp = 1.0 - math.exp(-LERP_RATE * dt)
                _apply_state(self._rig, data, interp)
        return {"PASS_THROUGH"}

    def execute(self, context):
        self._rig = _ensure_scene_setup()
        if self._rig is None:
            self.report({"ERROR"}, "No 'Rocket' collection found. Open combined.blend first.")
            return {"CANCELLED"}
        _STOP.clear()
        self._thread = threading.Thread(target=_poller, daemon=True)
        self._thread.start()
        self._last_time = time.perf_counter()
        self._timer = context.window_manager.event_timer_add(1.0 / 60.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, f"Lunar sim running. Backend: http://{HOST}:{PORT}")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        _STOP.set()
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class LunarSimStopOperator(bpy.types.Operator):
    bl_idname = "wm.lunar_sim_stop"
    bl_label = "Stop Lunar Sim"

    def execute(self, context):
        _STOP.set()
        return {"FINISHED"}


class LunarSimResetOperator(bpy.types.Operator):
    bl_idname = "wm.lunar_sim_reset"
    bl_label = "Reset Episode"

    def execute(self, context):
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"http://{HOST}:{PORT}/reset", method="POST"),
                timeout=0.3,
            )
            self.report({"INFO"}, "Episode reset requested")
        except Exception as e:
            self.report({"WARNING"}, f"Backend not reachable: {e}")
        return {"FINISHED"}


class LunarSimPanel(bpy.types.Panel):
    bl_label = "Lunar Sim"
    bl_idname = "VIEW3D_PT_lunar_sim"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lunar"

    def draw(self, context):
        c = self.layout.column()
        c.operator("wm.lunar_sim", icon="PLAY")
        c.operator("wm.lunar_sim_reset", icon="FILE_REFRESH")
        c.operator("wm.lunar_sim_stop", icon="PAUSE")
        with _LOCK:
            ok = _LATEST.get("ok", False)
            d = dict(_LATEST)
        c.separator()
        c.label(text=f"Backend: {'ONLINE' if ok else 'OFFLINE'}")
        if ok:
            c.label(text=f"Ep {d.get('episode', '-')}  Step {d.get('step', '-')}")
            c.label(text=f"Score {d.get('total_reward', 0):+.1f}")
            c.label(text=f"Action {d.get('action', '-')}")


_classes = (LunarSimOperator, LunarSimStopOperator, LunarSimResetOperator, LunarSimPanel)


def register():
    for c in _classes:
        try: bpy.utils.register_class(c)
        except Exception: pass


def unregister():
    _STOP.set()
    for c in reversed(_classes):
        try: bpy.utils.unregister_class(c)
        except Exception: pass


if __name__ == "__main__":
    try: unregister()
    except Exception: pass
    register()
    bpy.ops.wm.lunar_sim()
