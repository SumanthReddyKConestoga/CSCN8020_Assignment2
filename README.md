# CSCN8020 — Assignment 2: Q-Learning on the Taxi Gymnasium Environment

| | |
|---|---|
| **Student Name** | Sumanth Reddy Konannagari |
| **Student ID** | 9040660 |
| **Course** | CSCN8020 — Reinforcement Learning |
| **Repository** | https://github.com/SumanthReddyKConestoga/CSCN8020_Assignment2.git |

---

## Project Summary

This assignment implements the **Q-Learning algorithm** on the **Taxi-v4 Gymnasium environment** — a classic reinforcement learning problem where a taxi agent must learn, entirely from scratch, to navigate a 5×5 grid, pick up a passenger from one of four coloured locations, and deliver them to the correct destination. The agent starts with zero knowledge and learns purely through trial and error using rewards and penalties as feedback.

The solution is built using a **fully object-oriented architecture** across six dedicated classes, each with a single clear responsibility. The notebook runs seven experiments in total — one baseline and six hyperparameter variants — and presents all results through high-quality plots, detailed metric tables, and written interpretations directly connected to the Q-Learning pseudocode from Sutton & Barto (2018), Chapter 6.5. Five structured talking points explain the reinforcement learning theory, implementation challenges, and why Q-Learning is fundamentally different from supervised or unsupervised machine learning.

---

## The Taxi Problem — What the Agent Learns

The environment is a 5×5 city grid. A taxi starts at a random position. A passenger is waiting at one of four coloured pickup spots (Red, Green, Yellow, Blue). The passenger wants to go to one of those same four locations. The taxi must figure out — with no instructions, no map, and no pre-programmed routes — how to:

1. Drive to the passenger's location
2. Pick up the passenger
3. Drive to the correct destination
4. Drop off the passenger

**Reward structure that drives learning:**

| Event | Reward | Purpose |
|---|---|---|
| Every step taken | **−1** | Teaches the taxi to be fast |
| Correct passenger delivery | **+20** | Rewards the actual goal |
| Illegal pickup or dropoff | **−10** | Penalises wrong actions |

**State space:** 500 discrete states (25 taxi positions × 5 passenger locations × 4 destinations)

**State encoding formula:** `state = ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination`

**Action space:** 6 discrete actions (Move South=0, North=1, East=2, West=3, Pickup=4, Dropoff=5)

---

## Q-Learning — How the Agent Learns

Q-Learning is an off-policy reinforcement learning algorithm. It maintains a **Q-table** — a 500×6 array where each cell stores the estimated future reward for taking a specific action in a specific state. The table starts at zero and is updated after every single step using the **temporal difference update rule**:

```
Q(S, A) ← Q(S, A) + α [ R + γ · max_a Q(S', a) − Q(S, A) ]
```

| Symbol | Name | Role |
|---|---|---|
| α (alpha) | Learning rate | Controls how fast Q-values are updated |
| ε (epsilon) | Exploration factor | Controls random vs. greedy action selection |
| γ (gamma) | Discount factor | Controls how much future rewards are valued |
| R | Reward | Immediate feedback after taking an action |
| max Q(S', a) | Best future value | Best known value from the next state |

**Action selection — Epsilon-Greedy:**
- With probability ε → pick a **random** action (explore)
- With probability 1−ε → pick the **best known** action (exploit)

**Q-Learning is off-policy:** it always updates using the best possible next action (`max`), even if the agent took a random exploratory action. This allows it to learn the optimal policy while still exploring — the key difference from SARSA.

---

## Baseline Hyperparameters

| Hyperparameter | Symbol | Value | Meaning |
|---|---|---|---|
| Learning Rate | α | **0.1** | 10% of the TD error used per update |
| Exploration Factor | ε | **0.1** | 10% of actions are random |
| Discount Factor | γ | **0.9** | Future rewards worth 90% of face value |
| Episodes | — | **10,000** | Number of full training episodes |
| Max Steps per Episode | — | **200** | Safety cap to prevent infinite loops |

> **Note:** γ = 0.9 is fixed and **never changed** across any experiment. Only α and ε are varied.

---

## Object-Oriented Architecture

The solution uses **six classes**, each with a single clear responsibility. No core logic lives outside these classes.

| Class | Responsibility |
|---|---|
| `TaxiEnvironmentManager` | Creates, resets, steps through, and closes the Taxi-v4 Gymnasium environment. All `gym` calls are centralised here. |
| `QLearningAgent` | Owns the Q-table (500×6 NumPy array). Implements epsilon-greedy action selection and the Q-value update rule. This is the brain of the system. |
| `MetricsLogger` | Records per-episode rewards and steps. Writes a structured log file using Python's `logging` module with timestamps and log levels (INFO / WARNING / ERROR). |
| `QLearningTrainer` | Runs the full training loop (episode loop + step loop). Connects the agent, environment, and logger. Every line maps directly to the Sutton & Barto pseudocode. |
| `PlotManager` | Generates all plots with titles, axis labels, legends, raw + smoothed trend lines, and value annotations. No matplotlib code exists outside this class. |
| `ExperimentRunner` | Orchestrates all 7 experiments sequentially. Each experiment gets a fresh agent (Q-table reset to zero) and fresh environment. Stores all results for comparison. |

---

## Tasks — What Each One Does

### Task 1 — Q-Learning Implementation (Baseline)

Implements the complete Q-Learning algorithm and trains the agent using the baseline hyperparameters (α=0.1, ε=0.1, γ=0.9) for 10,000 episodes. This is the reference run — every other experiment is compared against it. The code is explicitly mapped line by line to the Sutton & Barto (2018) Chapter 6.5 pseudocode in a table inside the notebook, showing exactly which Python code corresponds to each pseudocode step: Q-table initialisation, state reset, epsilon-greedy action selection, TD update rule, state transition, and terminal check.

### Task 2 — Training Metrics and Plots

After the baseline training run, three required metrics are reported: (1) total episodes trained, (2) average steps per episode, and (3) average return per episode — both overall and for the final 100 episodes. Three plots are generated: reward per episode (raw + smoothed) showing performance improving over time, steps per episode showing the agent becoming more efficient, and rolling average return showing the convergence curve from deeply negative values to positive. Each plot has axis labels, a title, a legend, and a written interpretation below it connecting the visual trend directly to Q-Learning theory.

### Task 3A — Learning Rate (α) Hyperparameter Experiments

Tests three alternative learning rates (α = 0.001, 0.01, 0.2) one at a time while keeping ε = 0.1 and γ = 0.9 fixed. Each run uses a completely fresh Q-table and produces the same metrics as the baseline. Results are compared side by side in a metrics table and a dual-panel plot (full training view + early-training zoom to the first 3,000 episodes). The written interpretation explains why very small learning rates (0.001, 0.01) converge too slowly within 10,000 episodes, and why α = 0.2 is safe in the fully deterministic Taxi-v4 environment.

### Task 3B — Exploration Factor (ε) Hyperparameter Experiments

Tests two higher exploration factors (ε = 0.2, 0.3) one at a time while keeping α = 0.1 and γ = 0.9 fixed. Results show that higher exploration consistently hurts final performance in Taxi-v4 because even after the Q-table has converged, the agent continues taking random suboptimal actions 20–30% of the time — producing unnecessary −1 step penalties and −10 illegal-action penalties. The interpretation explains the exploration-exploitation trade-off and why lower ε is better at convergence in a deterministic environment with a known optimal policy.

### Task 4 — Best Combination Experiment

Selects the best combination of α and ε based on direct Task 3 evidence: α = 0.2 (Task 3A showed faster convergence with no instability in a deterministic environment) paired with ε = 0.1 (Task 3B showed the baseline epsilon gave the best final performance). Training is re-run with this combination and compared against the baseline using a three-panel plot (rolling average reward, smoothed steps per episode, and a bar chart of final metrics) plus a full numeric comparison table. Every choice in the justification is explicitly traced back to a specific Task 3 finding.

---

## Five Talking Points

Each talking point addresses three things: (1) a key feature of Q-Learning as an RL method, (2) a specific challenge found during implementation or testing of this assignment, and (3) why Q-Learning is reinforcement learning and not supervised or unsupervised learning. Each is also explicitly connected to Sutton & Barto (2018) Chapter 6.5 pseudocode with specific Python code line references.

| # | Topic | Key Concept Covered |
|---|---|---|
| TP 1 | **The Q-Table as a Complete Policy** | Zero initialisation, why RL needs no labelled dataset, the cold-start learning curve |
| TP 2 | **The Temporal Difference Update Rule** | Bootstrapping, TD error (Bellman error), how Q-values stabilise over episodes |
| TP 3 | **Epsilon-Greedy Exploration** | Explore-exploit dilemma, why ε=0.3 hurt performance in Task 3B, epsilon decay concept |
| TP 4 | **Off-Policy Learning** | How Q-Learning learns optimal policy while exploring, the `max` operator, SARSA comparison |
| TP 5 | **Episode Structure and Cumulative Return** | How episodic RL differs from supervised learning, why max_steps=200 was chosen, return accumulation |

---

## Repository File Structure

```
CSCN8020_Assignment2/
│
├── CSCN8020_Assignment2.ipynb         ← Main notebook (all code + theory + plots)
│
├── training.log                        ← Structured log file (all 7 experiments)
│                                         Timestamps | INFO/WARNING/ERROR levels
│                                         Episode progress every 500 episodes
│                                         Final metrics at end of each experiment
│
├── plot_baseline_metrics.png           ← 3-panel: reward/ep, steps/ep, rolling avg
├── plot_alpha_comparison.png           ← α experiments vs baseline (full + zoom)
├── plot_epsilon_comparison.png         ← ε experiments vs baseline (full + zoom)
├── plot_best_combo.png                 ← Best combo vs baseline (3 panels + bar chart)
├── plot_all_experiments_summary.png    ← All 7 experiments final bar chart summary
│
├── requirements.txt                    ← Pinned package versions (Python 3.11)
├── .flake8                             ← Flake8 config (excludes notebooks, max-line 120)
├── .pylintrc                           ← Pylint config (data science rules)
├── .gitignore                          ← Excludes venv, __pycache__, checkpoints
└── README.md                           ← This file
```

---

## How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/SumanthReddyKConestoga/CSCN8020_Assignment2.git
cd CSCN8020_Assignment2
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install gymnasium==1.3.0 numpy==2.4.4 matplotlib==3.10.8 pandas==3.0.2 notebook==7.4.4 ipykernel==6.29.5 nbformat==5.10.4 --no-cache-dir
```

Or using the requirements file:

```bash
pip install -r requirements.txt --no-cache-dir
```

### Step 4 — Launch Jupyter Notebook

```bash
jupyter notebook CSCN8020_Assignment2.ipynb
```

### Step 5 — Run the notebook

In Jupyter: **Kernel → Restart & Run All**

The notebook runs top to bottom without errors. All 7 experiments execute automatically from a single cell. All 5 plots are saved as PNG files in the project folder and also displayed inline. The training log is written to `training.log`.

> **Expected runtime:** approximately 3–6 minutes on a standard laptop.

---

## All Experiments at a Glance

| Experiment | α | ε | γ | Purpose |
|---|---|---|---|---|
| Baseline | 0.1 | 0.1 | 0.9 | Reference run — all others compared against this |
| Alpha-0.001 | 0.001 | 0.1 | 0.9 | Task 3A — test very slow learning rate |
| Alpha-0.01 | 0.01 | 0.1 | 0.9 | Task 3A — test slow learning rate |
| Alpha-0.2 | 0.2 | 0.1 | 0.9 | Task 3A — test faster learning rate |
| Epsilon-0.2 | 0.1 | 0.2 | 0.9 | Task 3B — test moderate exploration |
| Epsilon-0.3 | 0.1 | 0.3 | 0.9 | Task 3B — test high exploration |
| **Best Combo** | **0.2** | **0.1** | **0.9** | Task 4 — best α from 3A + best ε from 3B |

---

## Key Findings

- **α = 0.001 and α = 0.01** are too slow — after 10,000 episodes the Q-values have barely converged. These learning rates need 50,000+ episodes to reach what the baseline achieves in 10,000.
- **α = 0.2** is safe and faster in Taxi-v4 because the environment is fully deterministic — the TD error for a given (state, action) pair is consistent across visits, so a larger update step does not cause instability.
- **ε = 0.2 and ε = 0.3** consistently hurt final performance — after the Q-table converges, 20–30% random actions waste steps and accumulate −1 and −10 penalties, dragging down the average reward.
- **Best combination (α = 0.2, ε = 0.1)** delivers the fastest convergence while maintaining the tightest exploitation of the learned policy — the best result across all 7 experiments.

---
## Academic Integrity
- AI tools were used to support development and formatting. All code, explanations, experiment designs, and conclusions represent the student's own understanding of the Q-Learning algorithm and reinforcement learning theory. The student takes full responsibility for the correctness of the implementation and the accuracy of all written explanations.
---
