# Reinforcement Learning for Longitudinal Planning

This repository contains a reinforcement-learning prototype for autonomous-driving longitudinal planning.  
The current implementation uses **Stable-Baselines3 SAC** to learn a continuous **jerk** controller under:

- speed-limit constraints,
- hard / soft ST obstacles,
- smoothness requirements,
- finite-horizon longitudinal motion dynamics.

The codebase has been refactored toward **stable reward shaping**, **single-machine training efficiency**, and **stage-wise curriculum training**.

---

## 1. Problem Setup

### Environment

The environment models longitudinal planning on a fixed path over a planning horizon.

The CLI `--init_state s v a` now acts as the **reset-state anchor**:

- the environment starts around the provided `(s, v, a)`
- small random perturbations are still applied during reset for exploration
- this keeps runs reproducible while avoiding a fully deterministic initial-state distribution

At each step, the agent observes:

1. **ST occupancy image**
   - hard obstacles encoded as `1.0`
   - soft obstacles encoded as `0.5`
   - free space encoded as `0.0`
2. **Vector features**
   - current longitudinal state `(s, v, a)`
   - current time
   - reference speed
   - sampled speed-limit profile along the path

### Action

The action is a single continuous scalar:

- **jerk** `j`

The environment integrates jerk into acceleration, velocity, and station.

### Observation

The policy receives a Gymnasium `Dict` observation:

- `st_image`: `(1, H, W)` float image
- `array`: normalized float vector

The feature extractor combines:

- a lightweight CNN branch for the ST image
- an MLP branch for the vector observation

---

## 2. Reward Design

The current reward is **dense and continuous**, designed to avoid Q-value / reward explosion and to align better with longitudinal driving behavior.

### Reward structure

At each step, the implementation in `env/dp_vt_env.py` computes:

```text
r_t = r_progress
    - c_st
    - c_speed
    - c_acc
    - c_jerk
    - c_djerk
    - c_terminal
    + r_completion
```

with:

```text
Delta_s_t = s_t - s_(t-1)
```

and:

```text
r_progress = w_p * clip(Delta_s_t / (v_max * dt), -1.0, 1.5) * alpha_progress
```

where `alpha_progress` is exactly `progress_scale` in the code.

### Components

#### 2.1 Progress reward

The forward-progress term is:

```text
r_progress = w_p * clip((s_t - s_(t-1)) / (v_max * dt), -1.0, 1.5) * alpha_progress
```

`alpha_progress` starts from `1.0` and is reduced during ST interaction:

- hard-ST risk -> `hard_progress_discount`
- soft-ST interaction with likely overtaking -> `interaction_progress_discount`
- soft-ST interaction with likely yielding -> `yield_progress_discount`

This is how the code reduces the incentive to keep pushing forward when the safer behavior is to slow down or yield.

#### 2.2 Speed reward

The active reference speed is:

```text
v_ref = min(v_limit(s_t), v_cruise)
```

The code uses a deadband operator:

```text
phi(x; delta) = sign(x) * max(abs(x) - delta, 0)
```

Then the speed cost is:

```text
c_speed = w_v * (phi(v_t - v_ref; delta_v) / v_max)^2 * alpha_speed
        + w_over * max(0, v_t - v_limit(s_t) - delta_over)^2
```

where:

- `delta_v` = `speed_tracking_tolerance`
- `delta_over` = `overspeed_tolerance`
- `alpha_speed` = `interaction_speed_discount` inside any ST interaction region, otherwise `1.0`

This matches `_speed_cost(...)` in the code.

#### 2.3 Comfort reward

The comfort terms all use deadbands and quadratic penalties.

Acceleration:

```text
c_acc = w_a * (phi(a_t; delta_a) / a_comfort)^2
      + w_a_limit * (max(0, a_t - a_max) + max(0, a_min - a_t))^2
```

Jerk:

```text
c_jerk = w_j * (phi(j_t; delta_j) / j_comfort)^2
```

Jerk-rate:

```text
djerk_t = (j_t - j_(t-1)) / dt
c_djerk = w_djerk * (phi(djerk_t; delta_djerk) / djerk_comfort)^2
```

The corresponding code parameters are:

- `acc_deadband`, `acc_comfort`, `acc_limit_weight`
- `jerk_deadband`, `jerk_comfort`
- `djerk_deadband`, `djerk_comfort`

#### 2.4 ST interaction reward

For each active ST polygon at time `t`, the environment gets the current longitudinal interval:

```text
[s_lower(t), s_upper(t)]
```

and computes a forward safe gap:

```text
d_safe = d0 + tau_v * max(v_t, 0) + 0.1 * max(ds_lower(t), 0)
```

This matches `get_yield_distance(...)`.

##### Hard ST

If the ego is before the obstacle:

```text
g_front = s_lower - s_t
c_hard = w_hard * max(0, d_safe - g_front)^2
```

If the ego is already after the obstacle:

```text
g_back = s_t - s_upper
c_hard = 0.5 * w_hard * max(0, d_back - g_back)^2
```

If the ego is inside the hard ST interval, the code sets `cross_hard_st = True` and uses:

```text
rho = (s_t - s_lower) / (s_upper - s_lower + eps)
center_ratio = 1 - abs(2 * rho - 1)
c_hard = w_hard * (1 + center_ratio)^2
```

This means:

- approaching a hard ST too aggressively is penalized,
- staying near the center of a hard forbidden band is more costly,
- entering hard ST triggers an early terminal event.

##### Soft ST

The environment first checks whether overtaking still looks dynamically feasible:

```text
T_rem = max(T - t, 0)
s_reachable = s_t + max(v_t, 0) * T_rem + 0.5 * max(a_max, 0) * T_rem^2
can_overtake = (s_reachable >= s_upper + d_back)
```

If the ego is before the soft obstacle:

```text
g_front = s_lower - s_t
c_soft = w_front * max(0, d_safe - g_front)^2
```

with:

- `w_front = soft_margin_weight` if `can_overtake`
- `w_front = yield_weight` otherwise

If the ego is after the soft obstacle:

```text
c_soft = 0.5 * soft_margin_weight * max(0, d_back - (s_t - s_upper))^2
```

If the ego is inside the soft ST interval, the code defines:

```text
rho = (s_t - s_lower) / (s_upper - s_lower + eps)
center_ratio = 1 - abs(2 * rho - 1)
c_center = soft_margin_weight * center_ratio^2
```

Then it adds an edge-preference occupancy term:

```text
if can_overtake:
    d_edge = s_upper - s_t
else:
    d_edge = s_t - s_lower

rho_occ = d_edge / (s_upper - s_lower + eps)
c_occ = w_occ * rho_occ^2
```

where `w_occ` is `soft_occupancy_weight`, and `soft_overtake_discount` is applied to it in the overtaking-feasible case.

The total in-band soft cost is:

```text
c_soft = c_center + c_occ
```

This is the key difference from a plain occupancy penalty: the reward softly prefers the correct side of the soft interaction region depending on whether yielding or overtaking is more plausible.

#### 2.5 Terminal events

The terminal penalty is:

```text
c_terminal = I_hard_st * P_collision
           + I_severe_overspeed * P_over
           + I_reverse * P_reverse
           + I_out_of_bounds * P_oob
```

The episode is truncated early if any of the following is true:

- hard obstacle penetration
- severe overspeed
- reverse speed
- numerical / state out-of-bounds

This replaces the previous keep-punishing-for-the-rest-of-the-episode design.

#### 2.6 Completion bonus

At the terminal horizon `t = T`, if the episode did not end early, the code adds:

```text
r_completion = b_completion - (w_Tv * (v_T - v_ref_T)^2 + w_Ta * a_T^2)
```

where:

```text
v_ref_T = min(v_limit(s_T), v_cruise)
```

### Reward diagnostics

Each environment step exposes:

- `info["reward_terms"]`
- `info["events"]`

Important reward terms now include:

- `progress`
- `progress_scale`
- `hard_st_cost`
- `soft_st_cost`
- `speed_cost`
- `acc_cost`
- `jerk_cost`
- `djerk_cost`
- `terminal_cost`

This is used by the TensorBoard callback for debugging training behavior.

---

## 3. Training Strategy

### Algorithm

- **Soft Actor-Critic (SAC)**
- **MultiInputPolicy**
- custom `MixedFeaturesExtractor`

### Current default training mode

The default training path is now:

- **single-environment**
- **SB3-native SAC**
- **stage-wise curriculum**

This is intentional. The current priority is training stability and usable wall-clock efficiency on one machine, not full parallel rollout.

### Curriculum

The default config trains in three stages:

1. **speed**
   - no ST obstacles
   - learn speed tracking and comfort first
2. **soft-st**
   - introduce light ST interaction
3. **full**
   - full obstacle / speed-limit scenario

The same model is reused across stages.

### SB3 efficiency optimizations used

Without changing frameworks or requiring heavy parallelism, the code now improves efficiency by:

- using a smaller / cheaper CNN extractor
- avoiding large reward magnitudes
- reducing per-step observation-copy overhead
- using `train_freq=4`, `gradient_steps=4` for fewer Python-side transitions
- lowering expensive logging frequency for Q-statistics
- using curriculum to reduce wasted samples on hard scenes too early

---

## 4. Project Structure

```text
.
├── config/
│   └── config.json
├── env/
│   ├── dp_vt_env.py
│   └── env_visualiser.py
├── data_generator/
│   ├── speed_limit_generator.py
│   └── st_polygon_generator.py
├── train/
│   ├── model_components.py
│   ├── model_eval.py
│   ├── train_model.py
│   └── train_utils.py
└── main.py
```

Key files:

- `env/dp_vt_env.py`
  - environment dynamics, reward, termination, diagnostics
- `train/train_model.py`
  - SAC build / train / resume logic
  - curriculum stage orchestration
- `train/train_utils.py`
  - LR schedule, TensorBoard logging, evaluation callback
- `train/model_components.py`
  - multi-input feature extractor

---

## 5. Configuration

Main configuration lives in `config/config.json`.

Important sections:

- `dp_vt_config`
  - environment parameters
  - reward weights
- `model_config`
  - SAC hyperparameters
  - training timesteps
  - learning-rate schedule
- `curriculum_config`
  - curriculum stages and per-stage overrides
- `eval_config`
  - evaluation frequency / number of episodes
- `custom_log_config`
  - TensorBoard diagnostic logging frequency

### Example: disable curriculum

Set:

```json
"curriculum_config": {
  "enabled": false
}
```

Then the training pipeline will use the full base environment directly.

---

## 6. Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Typical required packages include:

- `numpy`
- `matplotlib`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `tensorboard`

---

## 7. Training

### Start a new training run

```bash
python3 main.py \
  --config_file config/config.json \
  --output_model models/sac_longitudinal \
  --init_state 0 0 0
```

### Resume from an existing model

```bash
python3 main.py \
  --config_file config/config.json \
  --train_from_local_model \
  --input_model models/sac_longitudinal.zip \
  --output_model models/sac_longitudinal_resume \
  --init_state 0 0 0
```

### Enable TensorBoard diagnostics

```bash
python3 main.py \
  --config_file config/config.json \
  --output_model models/sac_longitudinal \
  --init_state 0 0 0 \
  --use_custom_log
```

### Run with visualization

```bash
python3 main.py \
  --config_file config/config.json \
  --output_model models/sac_longitudinal \
  --init_state 0 0 0 \
  --show_plot
```

---

## 8. Evaluation and Diagnosis

Use `train/model_eval.py` helpers to:

- evaluate deterministic policy reward
- inspect trajectory plots
- diagnose reward decomposition

The diagnosis view plots:

- `s-t`
- `v-t`
- `a-t`
- `j-t`
- reward decomposition
- speed-limit tracking

The main evaluation entrypoint is `evaluate_model(...)`.
The legacy misspelled helper `envaluate(...)` is still kept as a compatibility alias.

---

## 9. Current Limitations

- training is still focused on **single-machine / mostly single-env SAC**
- no distributed rollout / learner architecture yet
- curriculum and reward weights still need empirical tuning
- runtime verification depends on local installation of Gymnasium / SB3 / Torch

---

## 10. Recommended Next Steps

If you continue iterating on this project, the next improvements should be:

1. tune reward weights using TensorBoard diagnostics
2. add fixed benchmark scenarios for regression evaluation
3. optimize environment generation / sampling cost
4. add light-weight vectorized rollout only after reward behavior is stable