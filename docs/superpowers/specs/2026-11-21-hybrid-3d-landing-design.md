# Hybrid 3D landing: V29 design and V28 audit

Status: proposed design for review. No V29 training results exist.

## Objective and scope

Build a software-only reference for autonomous powered descent in MuJoCo.
The agreed direction is trajectory guidance, feedback control and learned
disturbance correction. Vehicle parameters remain a declared reference envelope,
not measurements from a future vehicle. V27 remains the separate 2D baseline;
its weights and success rate do not transfer to the 3D task.

Target: at least 99% precise landings inside a 0.5 m horizontal radius on a
frozen independent scenario distribution. All attempts count. A record-breaking
claim requires a separately named benchmark with matching rules.

## Evidence from V28

The supplied 500-episode audit at full difficulty, seeds starting at 20000,
reports 0 precise, 0 safe-outside, 62 crash, 72 out-of-bounds and 366 timeout.
The following code findings are verified by inspection, but causal attribution
to the saved policy still needs trajectory replay:

1. `Rocket3DEnv._potential` clips combined position distance at 5 m. Position
   shaping cannot distinguish states beyond that boundary. At an upright,
   stationary state beyond it, position potential is -50 and its stationary
   shaping contribution at gamma=0.99 is +0.5 per step, before other terms.
   This alone is not proof of a shaping bug: proper potential shaping telescopes.
2. At 50 Hz, gamma=0.99 discounts a consequence at 750 steps by 0.0005326.
   The configured 800 timeout penalty has an initial discounted magnitude of
   approximately 0.426. Raw episodic returns do not represent the SAC objective.
3. A timeout is returned as truncation; SB3 may bootstrap through time limits.
   Decide explicitly whether the deadline is a mission failure terminal state
   or an external rollout cutoff, and test the resulting learning target.
4. Potential is not explicitly zeroed at absorbing terminal states. Test the
   terminal boundary before claiming policy-invariant shaping.
5. A post-contact deadline awards PRECISE/SAFE_OUTSIDE solely by distance,
   even if the vehicle has not settled. This is a measurement defect.
6. Settling has no separate upright tilt threshold; the broad tip threshold
   is about 60 degrees. The configured settle check requires only two legs
   for five frames. These are weaker than the agreed settled-upright objective.
7. Curriculum difficulty is a function of elapsed steps, not demonstrated skill.
8. Best-model ranking currently prioritizes safe rate before precise rate.
9. The existing PID baseline reads simulator truth. It cannot be used as the
   deployed component of a fair noisy-observation hybrid comparison unchanged.
10. Model and replay buffer are saved at different intervals. Resume may combine
    different generations; manifests should report their actual provenance.

## Architecture and approaches

Accepted direction: hybrid guidance and learning. Pure end-to-end SAC is the
failed V28 baseline. A classical-only controller is the necessary reference.
The proposed implementation uses an analytic constrained descent reference
first, cascaded feedback tracking, then bounded residual SAC. A convex planner
is a later controlled experiment, after the baseline is validated; it must not
be described as implemented MPC or successive convexification beforehand.

Flow: delayed/noisy observation -> state processing -> descent guidance ->
feedback tracking -> bounded learned correction -> actuator limits -> physics.
Rewards and success classification use simulator truth for scoring only.
The controller and residual policy receive the same available observation;
randomized true mass, wind and simulator kinematics are not controller inputs.

Guidance phases: arrest unsafe descent, acquire pad, descend, touchdown/settle.
Transitions use hysteresis and explicit exit conditions. Guidance must not
indefinitely gate descent on an unreachable horizontal-error threshold.
Tracking uses consistent frames, anti-windup and achievable thrust/tilt limits.
Residual zero must exactly reproduce the classical controller. Residual bounds
and authority must be logged and included in the evaluation configuration.

## Success contract (proposed numeric defaults)

- Radius <=0.5 m, touchdown total speed <=1.0 m/s, tilt <=10 degrees.
- At least three legs in contact, total speed <=0.1 m/s and angular speed
  <=0.1 rad/s continuously for 0.5 s, before a 20 s mission deadline.
- Hull contact, excessive impact, departure and deadline failure cannot become
  success through a contact timer. Settling failure counts as failure.
- Detect impact during physics substeps; log first contact and maximum impact.
- Report horizontal and vertical impact speed separately, final tilt, dwell,
  minimum altitude, final position, actuator saturation and phase dwell times.

These numeric limits are engineering proposals, not user-provided hardware specs.

## Reward and episode semantics

Before choosing coefficients, compare discounted rewards of successful descent,
hover, slow ascent, oscillation, off-pad landing and deliberate crash.
Use bounded state costs and terminal outcome terms; if potential shaping is
retained, test terminal handling explicitly. Remove repeatable descent bonuses
unless tests demonstrate that cycles cannot profit from them. Match the
environment and learner discount. Choose a physically meaningful time horizon
and test sensitivity; do not simply increase timeout penalties by guesswork.
Expose reward components and discounted/undiscounted totals in diagnostics.

## Training stages and gates

1. Physics/unit checks: force and torque signs, yaw rotations, hover equilibrium,
   actuator lag, seeded reset, impact detection and settled-state classification.
2. Classical-only: 200 clean episodes, at least 99% precise; no RL training until
   this gate passes. Record failure trajectories rather than only returns.
3. Observation realism: compare truth-state diagnostic control against the same
   controller using delayed/noisy observations. Identify latency failures first.
4. Residual learning: start around the working controller with bounded authority.
   Test zero residual and random residual; use a new buffer and versioned outputs.
5. Adaptive curriculum: advance one disturbance axis only after >=98% precise
   on 200 validation episodes twice consecutively; preserve prior-stage sampling.
   Stop and diagnose if a gate fails rather than advance on a step schedule.
6. Full-envelope comparison: classical and hybrid on paired scenarios; accept
   the learned residual only if it improves performance without worse failure
   tails. Train at least three seeds before choosing a final candidate.

Use the current declared V28 ranges as an initial envelope for feasibility
analysis, not an unconditional promise that every sampled state is recoverable.
Publish unreachable-state criteria independently of the controller; never
silently discard failed test attempts.

## Evaluation and reproducibility

Keep training, curriculum validation and final audit seeds separate. Freeze
scenario generation, physics configuration and success rules before final
evaluation. Final audit: 10000 episodes, with a Wilson 95% interval and slices
by wind, mass/thrust ratio, offset, tilt and delay. Report both the observed
99% target and whether its 95% lower confidence bound reaches 99%; they are
different claims. Do not select checkpoints on the final audit set.

Save paired checkpoint generations (policy, optimizer, replay, curriculum and
RNG state where supported), configuration hashes, package versions, training
seed and code revision. A SHA-256 hash is integrity metadata, not a signature.
Changing environment count or reward definition requires explicit compatibility
checks and must not silently reuse an incompatible replay buffer.

## Implementation sequence and deliverables

First write physics and success-contract regression tests plus a diagnostic
trajectory runner. Then implement a new versioned environment configuration,
observation-driven guidance/controller and baseline evaluation. Add a residual
wrapper and V29 trainer only after the classical clean gate passes. Deliver a
Colab notebook with dependency/GPU checks, short correctness run, checkpoint
discovery and resume checks; use measured throughput to estimate run time.
Keep V28 evaluation reproducible rather than silently changing its historical
environment semantics.

## Public research anchors

- NASA G-FOLD project: https://techport.nasa.gov/projects/12272
- NASA 6-DOF successive convexification research:
  https://ntrs.nasa.gov/citations/20230018515
- JPL terrain-relative navigation:
  https://robotics.jpl.nasa.gov/what-we-do/flight-projects/mars-2020-rover/terrain-relative-navigation/
- SpaceX publicly documented mission phases:
  https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf

These motivate research directions. They do not establish that SpaceX uses RL,
that the proposed hybrid is either organization's internal architecture, or
that a simplified simulator predicts real-flight success.
