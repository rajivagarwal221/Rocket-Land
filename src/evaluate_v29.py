"""Seeded V29 classical baseline audit. Never passes simulator truth to control."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluate_3d import wilson
from sim3d.guidance import LandingController
from sim3d.mission import MissionEnv
from sim3d.mission_wrappers import mission_sensors


def audit(episodes=200, seed=30000, difficulty=0.0, delay=0, noise=False, out=None, traces=None):
    if episodes <= 0 or delay < 0 or not 0 <= difficulty <= 1:
        raise ValueError('Positive episodes, nonnegative delay, difficulty in [0,1] required')
    raw = MissionEnv(difficulty=difficulty)
    env = mission_sensors(raw, delay=delay, noise_scale=1.0 if noise else 0.0)
    controller = LandingController(raw.params)
    rows = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed+ep)
            controller.reset()
            total, phases, trace = 0., Counter(), []
            while True:
                action = controller(obs)
                if traces:
                    k = raw.kinematics()
                    trace.append(dict(time=raw._steps*raw.dt_ctrl, observation=obs.tolist(),
                                      action=action.tolist(), phase=controller.phase,
                                      position=k.pos.tolist(), velocity=k.v_world.tolist(),
                                      tilt=k.tilt))
                phases[controller.phase] += 1
                obs, reward, te, tr, info = env.step(action)
                total += reward
                if te or tr:
                    row = dict(episode=ep, seed=seed+ep, steps=sum(phases.values()),
                               return_=total, **info)
                    row.update({f'phase_{k}': phases[k] for k in ('ACQUIRE','DESCEND','SETTLE')})
                    rows.append(row)
                    if traces and info['outcome'] != 'PRECISE':
                        folder = Path(traces)
                        folder.mkdir(parents=True, exist_ok=True)
                        trace.append(dict(terminal=info))
                        (folder / f'failure_{seed+ep}.jsonl').write_text(
                            '\n'.join(json.dumps(t) for t in trace)+'\n', encoding='utf-8')
                    break
            if (ep+1) % 25 == 0:
                print(f'{ep+1}/{episodes}: {dict(Counter(r["outcome"] for r in rows))}', flush=True)
    finally:
        env.close()
    counts = Counter(r['outcome'] for r in rows)
    rate = counts['PRECISE']/episodes
    summary = dict(controller='observation-driven classical v29', episodes=episodes,
                   seed=seed, difficulty=difficulty, delay_steps=delay, noise=noise,
                   outcomes=dict(counts), precise_rate=rate,
                   precise_wilson95=wilson(counts['PRECISE'], episodes),
                   clean_gate_passed=(difficulty == 0 and delay == 0 and not noise
                                      and episodes >= 200 and rate >= .99),
                   params=asdict(raw.params), config=asdict(raw.cfg),
                   xml_sha256=hashlib.sha256(Path(__file__).with_name('sim3d').joinpath('rocket.xml').read_bytes()).hexdigest())
    print(json.dumps(summary, indent=2))
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        out.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=30000)
    p.add_argument('--difficulty', type=float, default=0)
    p.add_argument('--delay', type=int, default=0)
    p.add_argument('--noise', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--traces', type=Path, help='save per-step telemetry for failed development episodes')
    a = p.parse_args()
    audit(a.episodes, a.seed, a.difficulty, a.delay, a.noise, a.out, a.traces)
