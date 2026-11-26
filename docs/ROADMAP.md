# Rocket-Land — Status and Roadmap

Updated 2026-11-26. Numbers are seeded audits; "held-out" = seeds 90000–90999, never used for tuning.

## Phases
1. **Perfect the policy in simulation** — 2D done; 3D in progress.
2. Small real rocket.
3. 10-foot rocket.

---

## ✅ Executed

### Phase 1 — 2D (LunarLander)
| Version | What | Result (1000 eps, shield off) |
|---|---|---|
| V20–V23 | SAC + reward recipe, noise curriculum | V23: 96.9% |
| V24, V25 | Fuel penalty / glideslope reward changes | Regressed (79.9%, 91.5%) — reward changes hurt |
| **V27** | Colab: 8-frame stack, [512,512], 8M steps | **99.2% precise — Phase-1 champion** |

### Phase 2 — 3D (MuJoCo, `src/sim3d/`)
| Step | What | Result |
|---|---|---|
| V28 | 6-DOF env, gimballed engine, from-scratch SAC | 0% — learned to hover (reward loopholes); fixed, abandoned |
| V29 baseline | Strict mission rules + observation-only classical guidance | Clean 100%, full envelope 80.5% (200 eps) |
| Wave 1 physics | Fuel burn (mass/inertia), 30% throttle floor, pad tilt ±3°, site air density ±10% | Committed `6f5f15b`; full envelope 78.5% |
| Wind guidance | Gimbal-aware disturbance observer + feedforward, position integral, gust-lull touchdown hold, 8° final-drop lean cap, looser hover gate | **Held-out 91.7%** (old 82.0%) |
| Wave 2 (part) | Position-fix dropouts (20–200 ms bursts) | No measurable cost; robust at 4× rate |
| Diagnosis | Perfect sensors only +3 pts; no gusts → 96.8%; wind ≤4 m/s → 98.2% | **Limit = gusty strong wind**, not sensing |

Held-out breakdown (every factor on): calm wind 98.8%, wind ≥3 m/s 78.6%; failures 31 crash, 22 off-pad, 29 timeout.

---

## 🔄 Executing now

| Item | State |
|---|---|
| `src/train_v29_residual.py`: SAC correction (±10–15%) on top of classical guidance; auto-resume; best saved only if it beats baseline | Committed and pushed. 23 unit tests in `tests/test_v29.py` pass |
| Residual run 1 (gamma 0.99, random-start actor) | **Regressed** 93.5% → 75% on selection seeds over 2M steps. Cause: gamma 0.99 discounted the landing reward to ~0.7% across 400–900-step episodes, and a random actor perturbed a controller that already lands 91.7% |
| Fix (`d4d0677`) | gamma 0.999, zero-start actor (untrained policy = classical exactly), lower learning rate; `scripts\train_v29.bat fresh` option |
| Residual run 2 | **Next to start:** `scripts\train_v29.bat fresh` on the local RTX 3050, or `notebooks/v29_colab.ipynb` on Colab, then `scripts\audit_v29.bat` |

---

## ⏭️ Next

1. **Run residual training afresh** (local GPU or Colab) and audit it on held-out seeds against classical guidance.
2. **Decide on the residual:** ship it only if its held-out score beats 91.7% with non-overlapping confidence intervals.
3. **If residual stalls below ~95%:** set an operating wind limit (e.g. ≤4 m/s mean already gives 98.2%) or increase gimbal authority.
4. **Remaining sim factors:** ground effect (needs measured data), abort envelope, fuel-reserve rule.
5. **Bench-measure the real rocket** (thrust curve, motor lag, gimbal servo, IMU noise, sensor latency) and replace placeholders in `RocketParams` and `IMU_NOISE_STD`. Highest-value step before Phase 2 flight.
6. Phase 2: first small-rocket flights.

## Housekeeping
- Model checkpoints (`models/*.zip`, `*.pkl`) are no longer tracked; they live locally or on Drive. Run manifests (`*_manifest.json`) stay in the repo. Old checkpoints remain in git history (`.git` ≈150 MB) unless history is rewritten.
- Seed rules: develop on 30000+, select checkpoints on 50000+, audit on 90000+, train on 1,000,000+.
