"""Seeded V29 audit: classical guidance, or guidance plus a trained residual
(--model). Never passes simulator truth to control.

--wind-limit is a pre-flight GO/NO-GO, like a launch-commit weather rule: an
episode whose mean wind exceeds the limit is SCRUBBED (not flown) and the
precise rate is reported over flown episodes. The gate reads the mean wind
before flight, standing in for a ground anemometer; gusts still fly.
Every episode keeps its own seed, so flown episodes are identical to the
unlimited audit."""
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
from sim3d.residual import ResidualLandingEnv


def audit(episodes=200, seed=30000, difficulty=0.0, delay=0, noise=False, out=None, traces=None,
          dropout=0.0, model=None, wind_limit=None, nav='gps'):
    if episodes <= 0 or delay < 0 or not 0 <= difficulty <= 1:
        raise ValueError('Positive episodes, nonnegative delay, difficulty in [0,1] required')
    if wind_limit is not None and wind_limit < 0:
        raise ValueError('wind_limit must be nonnegative (m/s)')
    raw = MissionEnv(difficulty=difficulty)
    env = mission_sensors(raw, delay=delay, noise_scale=1.0 if noise else 0.0, dropout=dropout,
                          nav=nav)
    policy = None
    if model is not None:
        from stable_baselines3 import SAC
        policy = SAC.load(str(model), device='cpu')
        env = ResidualLandingEnv(env)
        controller = env.controller
    else:
        controller = LandingController(raw.params, nav=nav)
    rows = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed+ep)
            if wind_limit is not None and raw.episode_meta['wind_speed'] > wind_limit:
                rows.append(dict(episode=ep, seed=seed+ep, steps=0, return_=0.0,
                                 **raw.episode_meta, outcome='SCRUBBED'))
                continue
            if policy is None:
                controller.reset()
            total, phases, trace = 0., Counter(), []
            while True:
                phase = controller.phase
                action = (policy.predict(obs, deterministic=True)[0] if policy is not None
                          else controller(obs))
                if traces:
                    k = raw.kinematics()
                    trace.append(dict(time=raw._steps*raw.dt_ctrl, observation=obs.tolist(),
                                      action=action.tolist(), phase=phase,
                                      position=k.pos.tolist(), velocity=k.v_world.tolist(),
                                      tilt=k.tilt))
                phases[phase] += 1
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
            if (ep+1) % 25 == 0 and rows[-1]['outcome'] != 'SCRUBBED':
                print(f'{ep+1}/{episodes}: {dict(Counter(r["outcome"] for r in rows))}', flush=True)
    finally:
        env.close()
    counts = Counter(r['outcome'] for r in rows)
    scrubbed = counts.pop('SCRUBBED', 0)
    flown = episodes - scrubbed
    if flown == 0:
        raise ValueError(f'wind limit {wind_limit} m/s scrubbed every episode')
    rate = counts['PRECISE']/flown
    name = f'classical v29 + residual {Path(model).name}' if model else 'observation-driven classical v29'
    summary = dict(controller=name, episodes=episodes,
                   seed=seed, difficulty=difficulty, delay_steps=delay, noise=noise,
                   nav_dropout_rate=dropout, nav_profile=nav, wind_limit_mps=wind_limit,
                   flown=flown, scrubbed=scrubbed, scrub_rate=scrubbed/episodes,
                   outcomes=dict(counts), precise_rate=rate,
                   precise_wilson95=wilson(counts['PRECISE'], flown),
                   clean_gate_passed=(difficulty == 0 and delay == 0 and not noise and not dropout
                                      and flown >= 200 and rate >= .99),
                   params=asdict(raw.params), config=asdict(raw.cfg),
                   xml_sha256=hashlib.sha256(Path(__file__).with_name('sim3d').joinpath('rocket.xml').read_bytes()).hexdigest())
    print(json.dumps(summary, indent=2))
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='', encoding='utf-8') as f:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(f, fieldnames=fields, restval='')
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
    p.add_argument('--dropout', type=float, default=0.0,
                   help='per-step chance a 20-200 ms position-fix dropout starts (e.g. 0.005)')
    p.add_argument('--out', type=Path)
    p.add_argument('--traces', type=Path, help='save per-step telemetry for failed development episodes')
    p.add_argument('--model', type=Path, help='trained residual SAC checkpoint (.zip)')
    p.add_argument('--wind-limit', type=float,
                   help='scrub (do not fly) episodes whose mean wind exceeds this, m/s (e.g. 4)')
    p.add_argument('--nav', choices=['gps', 'rtk'], default='gps',
                   help='position-fix quality: gps (0.10 m offset) or rtk (0.02 m)')
    a = p.parse_args()
    if a.model is not None and not a.model.is_file():
        p.error(f'checkpoint not found: {a.model}')
    audit(a.episodes, a.seed, a.difficulty, a.delay, a.noise, a.out, a.traces, a.dropout, a.model,
          a.wind_limit, a.nav)
