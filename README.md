# Reinforcement Learning Project (DQN / DDQN / PPO)

This repository contains two independent RL pipelines:

- **Value-based methods** in the project root (`DQN` / `DDQN`)
- **Policy-based method** in `ppo_code/` (`PPO`)

The code is set up for classic control tasks such as `CartPole-v0` and `Acrobot-v1`.

---

## 1) Project Structure

```text
project_1_code/
  main.py                      # Single-run DQN/DDQN entry
  learn.py                     # DQN/DDQN trainer and losses
  model.py                     # Q-network architecture
  schedule.py                  # Exploration / LR schedule utilities
  run_experiments.py           # Batch DQN/DDQN runs by env + seeds
  summarize_results.py         # Aggregate DQN/DDQN metrics and plots
  utils/
    general.py                 # CSV loading and plotting helpers

  ppo_code/
    ppo_train.py               # Single-run PPO training
    run_ppo_experiments.py     # Batch PPO runs by env + seeds
    summarize_ppo_results.py   # Aggregate PPO metrics and plots
    requirements.txt
```

---

## 2) Environment Setup

Use Python 3.10+ (tested with 3.12). Install core dependencies:

```bash
pip install torch numpy matplotlib gymnasium[classic-control]
```

If you run PPO from `ppo_code/`, you can also install from its local requirements:

```bash
cd ppo_code
pip install -r requirements.txt
```

---

## 3) DQN / DDQN (Root Directory)

### 3.1 Single run

```bash
cd /root/project/project_1_code
python main.py
```

Notes:
- In `main.py`, set environment with `gym.make(...)`
- Set `double = False` for DQN, `double = True` for DDQN

### 3.2 Batch runs (recommended)

```bash
cd /root/project/project_1_code
python run_experiments.py --env CartPole-v0 --seeds 0 1 2 --results-dir results_dqn
python run_experiments.py --env CartPole-v0 --seeds 0 1 2 --results-dir results_ddqn --double
python run_experiments.py --env Acrobot-v1 --seeds 0 1 2 --results-dir results_ddqn --double
```

### 3.3 Aggregate plots

```bash
cd /root/project/project_1_code
python summarize_results.py --results-dir results_ddqn --envs CartPole-v0 Acrobot-v1
```

Generated metrics:
- `Training Rewards`
- `Eval Rewards`
- `Max Q`
- `Loss`

---

## 4) PPO (`ppo_code/`)

### 4.1 Single run

```bash
cd /root/project/project_1_code/ppo_code
python ppo_train.py --env CartPole-v0 --seed 0 --results-dir ppo_results
python ppo_train.py --env Acrobot-v1 --seed 0 --results-dir ppo_results
```

### 4.2 Batch runs

```bash
cd /root/project/project_1_code/ppo_code
python run_ppo_experiments.py --env CartPole-v0 --seeds 0 1 2 --results-dir ppo_comp_result
python run_ppo_experiments.py --env Acrobot-v1 --seeds 0 1 2 --results-dir ppo_comp_result
```

### 4.3 Aggregate plots

```bash
cd /root/project/project_1_code/ppo_code
python summarize_ppo_results.py --results-dir ppo_comp_result --envs CartPole-v0 Acrobot-v1
```

PPO logs include:
- `Training Rewards`
- `Eval Rewards`
- `Max Q`
- `Loss`

---

## 5) Output Format

For each run (`<env>_seed<seed>`), outputs are written to:

- `log.csv`
- model weights (`model.weights` for DQN/DDQN, `model.pt` for PPO)
- metric plots (`training_rewards`, `eval_rewards`, `max_q`, `loss`)

Summary scripts also create aggregated plots under:

- `<results-dir>/summary/`

---

## 6) Reproducibility

- Both pipelines support explicit seeds.
- Prefer multi-seed runs (`--seeds 0 1 2 ...`) to evaluate stability.
- Compare methods using the same environment, seeds, and total timesteps.

---

## 7) Quick Example (DDQN + PPO)

```bash
# DDQN
cd /root/project/project_1_code
python run_experiments.py --env CartPole-v0 --seeds 0 1 2 --results-dir ddqn_results --double
python summarize_results.py --results-dir ddqn_results --envs CartPole-v0

# PPO
cd /root/project/project_1_code/ppo_code
python run_ppo_experiments.py --env CartPole-v0 --seeds 0 1 2 --results-dir ppo_results
python summarize_ppo_results.py --results-dir ppo_results --envs CartPole-v0
```
