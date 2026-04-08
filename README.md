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

The current reward is **dense and continuous**, designed to avoid Q-value / reward explosion.

### Reward structure

At each step, the implementation in `env/dp_vt_env.py` computes:

\[
r_t
= r_{\text{progress}}
- c_{\text{st}}
- c_{\text{speed}}
- c_{\text{acc}}
- c_{\text{jerk}}
- c_{\text{djerk}}
- c_{\text{terminal}}
+ r_{\text{completion}}
\]

with

\[
\Delta s_t = s_t - s_{t-1}
\]

\[
r_{\text{progress}}
= w_p \cdot
\mathrm{clip}\!\left(
\frac{\Delta s_t}{v_{\max}\,dt},
-1.0,\,
1.5
\right)
\cdot \alpha_{\text{progress}}
\]

where `progress_scale` in the code is exactly \(\alpha_{\text{progress}}\).

### Components

#### 2.1 Progress reward

The forward-progress term is proportional to normalized station increment:

\[
r_{\text{progress}}
= w_p \cdot
\mathrm{clip}\!\left(
\frac{s_t - s_{t-1}}{v_{\max}\,dt},
-1.0,\,
1.5
\right)
\cdot \alpha_{\text{progress}}
\]

The scale \(\alpha_{\text{progress}}\) starts from \(1.0\) and is reduced in ST interaction cases:

- hard-ST risk: `hard_progress_discount`
- soft-ST interaction with likely overtaking: `interaction_progress_discount`
- soft-ST interaction with likely yielding: `yield_progress_discount`

This is how the code reduces the incentive to "keep pushing forward" when the safer behavior is to slow down or yield.

#### 2.2 Speed reward

The active reference speed is:

\[
v_{\text{ref}} = \min(v_{\text{limit}}(s_t),\, v_{\text{cruise}})
\]

The code uses a deadband operator:

\[
\phi(x;\delta)=
\operatorname{sign}(x)\,\max(|x|-\delta,\,0)
\]

Then the speed cost is:

\[
c_{\text{speed}}
=
w_v
\left(
\frac{\phi(v_t - v_{\text{ref}};\,\delta_v)}{v_{\max}}
\right)^2
\cdot \alpha_{\text{speed}}
+
w_{\text{over}}
\max(0,\,
v_t - v_{\text{limit}}(s_t) - \delta_{\text{over}}
)^2
\]

where:

- \(\delta_v\) = `speed_tracking_tolerance`
- \(\delta_{\text{over}}\) = `overspeed_tolerance`
- \(\alpha_{\text{speed}} = \)`interaction_speed_discount` when inside any ST interaction region, otherwise \(1.0\)

This matches `_speed_cost(...)` exactly.

#### 2.3 Comfort reward

The comfort terms all use deadbands and quadratic penalties.

Acceleration:

\[
c_{\text{acc}}
=
w_a
\left(
\frac{\phi(a_t;\,\delta_a)}{a_{\text{comfort}}}
\right)^2
+
w_{a,\text{limit}}
\left(
\max(0, a_t-a_{\max})
+
\max(0, a_{\min}-a_t)
\right)^2
\]

Jerk:

\[
c_{\text{jerk}}
=
w_j
\left(
\frac{\phi(j_t;\,\delta_j)}{j_{\text{comfort}}}
\right)^2
\]

Jerk-rate:

\[
\dot{j}_t = \frac{j_t - j_{t-1}}{dt}
\]

\[
c_{\text{djerk}}
=
w_{\dot{j}}
\left(
\frac{\phi(\dot{j}_t;\,\delta_{\dot{j}})}{\dot{j}_{\text{comfort}}}
\right)^2
\]

where the corresponding code parameters are:

- `acc_deadband`, `acc_comfort`, `acc_limit_weight`
- `jerk_deadband`, `jerk_comfort`
- `djerk_deadband`, `djerk_comfort`

#### 2.4 ST interaction reward

For each active ST polygon at time \(t\), the environment gets the current longitudinal interval:

\[
[s_{\text{lower}}(t),\, s_{\text{upper}}(t)]
\]

and computes a forward safe gap:

\[
d_{\text{safe}}
= d_0 + \tau_v \max(v_t, 0)
+ 0.1 \max(\dot{s}_{\text{lower}}(t), 0)
\]

matching `get_yield_distance(...)`.

##### Hard ST

If the ego is before the obstacle:

\[
g_{\text{front}} = s_{\text{lower}} - s_t
\]

\[
c_{\text{hard}}
=
w_{\text{hard}}
\max(0,\,
d_{\text{safe}} - g_{\text{front}}
)^2
\]

If the ego is already after the obstacle:

\[
g_{\text{back}} = s_t - s_{\text{upper}}
\]

\[
c_{\text{hard}}
=
0.5\,w_{\text{hard}}
\max(0,\,
d_{\text{back}} - g_{\text{back}}
)^2
\]

If the ego is inside the hard ST interval, the code marks `cross_hard_st = True` and uses:

\[
\rho = \frac{s_t - s_{\text{lower}}}{s_{\text{upper}} - s_{\text{lower}} + \epsilon}
\]

\[
c_{\text{hard}}
=
w_{\text{hard}}
\left(
1 + \left[1 - |2\rho - 1|\right]
\right)^2
\]

This means:

- approaching a hard ST too aggressively is penalized,
- staying near the center of a hard forbidden band is more costly,
- entering hard ST triggers an early terminal event.

##### Soft ST

The environment first checks whether overtaking still looks dynamically feasible:

\[
s_{\text{reachable}}
=
s_t + \max(v_t,0)\,T_{\text{rem}}
+ \frac{1}{2}\max(a_{\max},0)\,T_{\text{rem}}^2
\]

\[
\text{can\_overtake}
\iff
s_{\text{reachable}}
\ge
s_{\text{upper}} + d_{\text{back}}
\]

where \(T_{\text{rem}} = \max(T - t, 0)\).

If the ego is before the soft obstacle:

\[
g_{\text{front}} = s_{\text{lower}} - s_t
\]

\[
c_{\text{soft}}
=
w_{\text{front}}
\max(0,\,
d_{\text{safe}} - g_{\text{front}}
)^2
\]

with

- \(w_{\text{front}} = w_{\text{soft-margin}}\) if overtaking is feasible
- \(w_{\text{front}} = w_{\text{yield}}\) otherwise

If the ego is after the soft obstacle:

\[
c_{\text{soft}}
=
0.5\,w_{\text{soft-margin}}
\max(0,\,
d_{\text{back}} - (s_t - s_{\text{upper}})
)^2
\]

If the ego is inside the soft ST interval, the code defines:

\[
\rho = \frac{s_t - s_{\text{lower}}}{s_{\text{upper}} - s_{\text{lower}} + \epsilon}
\]

\[
c_{\text{center}} =
w_{\text{soft-margin}}
\left(
1 - |2\rho - 1|
\right)^2
\]

and an edge-preference occupancy term:

\[
d_{\text{edge}} =
\begin{cases}
s_{\text{upper}} - s_t, & \text{if overtaking is feasible} \\
s_t - s_{\text{lower}}, & \text{otherwise}
\end{cases}
\]

\[
\rho_{\text{occ}} =
\frac{d_{\text{edge}}}{s_{\text{upper}} - s_{\text{lower}} + \epsilon}
\]

\[
c_{\text{occ}}
=
w_{\text{occ}}
\rho_{\text{occ}}^2
\]

where `soft_overtake_discount` is applied to \(w_{\text{occ}}\) in the overtaking-feasible case.

The total in-band soft cost is:

\[
c_{\text{soft}} = c_{\text{center}} + c_{\text{occ}}
\]

This is the key difference from a plain occupancy penalty: the reward now softly prefers the "correct side" of the soft interaction region depending on whether yielding or overtaking is more plausible.

#### 2.5 Terminal events

The terminal penalty is:

\[
c_{\text{terminal}}
=
\mathbb{1}_{\text{hard-ST}}\,P_{\text{collision}}
+
\mathbb{1}_{\text{severe-overspeed}}\,P_{\text{over}}
+
\mathbb{1}_{\text{reverse}}\,P_{\text{reverse}}
+
\mathbb{1}_{\text{out-of-bounds}}\,P_{\text{oob}}
\]

The episode is truncated early if any of the following is true:

- hard obstacle penetration
- severe overspeed
- reverse speed
- numerical / state out-of-bounds

This replaces the previous "keep punishing for the rest of the episode" design.

#### 2.6 Completion bonus

At the terminal horizon \(t=T\), if the episode did not end early, the code adds:

\[
r_{\text{completion}}
=
b_{\text{completion}}
-
\left(
w_{Tv}(v_T - v_{\text{ref},T})^2
+
w_{Ta}a_T^2
\right)
\]

where \(v_{\text{ref},T} = \min(v_{\text{limit}}(s_T), v_{\text{cruise}})\).

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