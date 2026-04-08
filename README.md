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

At each step:

```text
reward =
    progress_reward
  - st_cost
  - speed_cost
  - acc_cost
  - jerk_cost
  - djerk_cost
  - terminal_penalty
  + completion_bonus
```

### Components

#### 2.1 Progress reward

Encourages the ego vehicle to move forward:

- positive reward proportional to `delta_s`

#### 2.2 Speed reward

Encourages:

- tracking the active speed reference
- respecting speed limits

The implementation uses:

- quadratic tracking error
- explicit overspeed penalty
- a small deadband around the reference speed
- reduced tracking pressure while interacting with ST obstacles

#### 2.3 Comfort reward

Penalizes:

- acceleration magnitude
- jerk magnitude
- jerk rate change

All comfort terms are quadratic and include deadbands to avoid punishing harmless small motions.

#### 2.4 ST interaction reward

Hard and soft ST obstacles are handled differently:

- **hard ST**
  - penalize unsafe approach margin
  - entering the hard obstacle triggers early truncation

- **soft ST**
  - penalize unsafe following / yielding margin
  - being inside a soft region is costly but not necessarily terminal

#### 2.5 Terminal events

The episode is truncated early on:

- hard obstacle penetration
- severe overspeed
- reverse speed
- numerical / state out-of-bounds

This replaces the previous "keep punishing for the rest of the episode" design.

### Reward diagnostics

Each environment step exposes:

- `info["reward_terms"]`
- `info["events"]`

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