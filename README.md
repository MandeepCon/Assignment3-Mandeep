
# DQN for PongDeterministic-v4 — CSCN 8020 Assignment 3

Train a Deep Q-Network (DQN) to play **PongDeterministic-v4** using stacked grayscale frames as input, a CNN Q-network, a target network, and experience replay.  
This repository implements the three experiments required by the assignment and produces the required plots and CSV logs.

---

## ✅ What this project includes

- **Environment & Preprocessing**
  - Uses `PongDeterministic-v4` (Gym 0.26.x Atari).
  - Preprocess to **84×80 grayscale** using the provided `assignment3_utils.py` (crop → grayscale → resize).
  - **Stack the latest 4 frames** (no blending) → final state shape **(4, 84, 80)**.

- **DQN (PyTorch)**
  - CNN architecture (Atari-style):  
    `Conv(4→32, k=8, s=4) → ReLU → Conv(32→64, k=4, s=2) → ReLU → Conv(64→64, k=3, s=1) → ReLU → Flatten → Dense(512) → ReLU → Dense(6)`
  - Huber loss, Adam optimizer, gradient clipping (max-norm=10).
  - Target network synchronized episodically.

- **Experiments (run separately)**
  1. **Default:** batch=8, target-update=10
  2. **Batch sweep:** batch=16, target-update=10
  3. **Target sweep:** batch=8, target-update=3

- **Metrics & Artifacts (per experiment)**
  - Console per-episode print:  
    `Episode: i/N | Steps: <cumulative> | Score: r | Avg (5): a | Epsilon: e | Time: t s`
  - CSV with columns: `episode, reward, steps, cum_steps, avg5, ep_time_sec`
  - Plots saved:
    - `*_score_vs_steps.png` (Score per Episode vs **Total Steps**)
    - `*_avg5_vs_steps.png` (Avg(5) vs **Total Steps**)

- **Final Comparison**
  - A combined **summary table** and an **overlay plot**: `comparison_avg5_vs_episode.png`
  - Recommendation based on **Final Avg(5)** (tie-breakers: Best Score, median episode time).

---

## 📁 Project Structure

```
.
├── assignment3_utils.py          # Provided preprocessing utilities (crop→84×80, grayscale)
├── main.ipynb                    # Notebook with full training + plots + comparison
├── results_pong_dqn/             # Auto-created: CSVs and PNGs from each experiment
│   ├── exp1_default.csv
│   ├── exp1_default_score_vs_steps.png
│   ├── exp1_default_avg5_vs_steps.png
│   ├── exp2_batch16.csv
│   ├── exp2_batch16_score_vs_steps.png
│   ├── exp2_batch16_avg5_vs_steps.png
│   ├── exp3_target3.csv
│   ├── exp3_target3_score_vs_steps.png
│   ├── exp3_target3_avg5_vs_steps.png
│   └── comparison_avg5_vs_episode.png  #Comparison final plot
├── README.md
└── requirements.txt              # Create via 'python -m pip freeze > requirements.txt'
```

> The notebook creates `results_pong_dqn/` automatically and fills it with logs/plots.

---

## 🧰 Requirements

- **Python 3.10–3.11** recommended
- **PyTorch** (CPU or CUDA)
- **Gym 0.26.2** with Atari extras (this assignment targets the older Gym API)
- `ale-py`, `autorom` (for ROMs)
- NumPy, Pandas, Matplotlib

> ⚠️ The assignment uses **`PongDeterministic-v4`**, which belongs to **Gym 0.26.x** + Atari.  
> Do **not** mix with `gymnasium` unless you remap the environment name and API.

---

## 🔧 Environment Setup

Create and activate a virtual environment (choose one):

**venv (cross-platform):**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

**conda:**
```bash
conda create -n rl8020 python=3.11 -y
conda activate rl8020
```

Install the exact Gym/Atari stack and ROMs:
```bash
pip install "gym==0.26.2" "gym[atari]" ale-py autorom

# Install PyTorch (CPU-only example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# (CUDA users: choose the wheel that matches your CUDA version from https://pytorch.org)

# Install Atari ROMs (accept license)
python -m autorom --accept-license -y
```

> If `autorom` command isn’t found, the `python -m autorom --accept-license -y` form works inside the environment.

---

## ▶️ How to Run

1. Open `main.ipynb` in Jupyter or VS Code.
2. Run the cells. You’ll see per-episode logs like:
   ```text
   Episode: 1/100 | Steps: 912 | Score: -21.0 | Avg (5): -21.00 | Epsilon: 0.9950 | Time: 1.59s
   ```
3. After each experiment, the notebook saves:
   - CSV: `results_pong_dqn/exp*.csv`
   - Plots: `*_score_vs_steps.png`, `*_avg5_vs_steps.png`
4. After all three experiments, it saves `comparison_avg5_vs_episode.png` and prints a recommendation.

> For stronger improvement on Pong, increase `EPISODES` (e.g., 300–500+).

---

## 🧪 Reproducibility

- Seed = **42** for Python, NumPy, and PyTorch (CUDA if available).
- Atari determinism can still vary slightly due to drivers and backend differences.

---

## 🛠️ Troubleshooting

- **`No module named autorom`**  
  Install and run:  
  ```bash
  pip install autorom
  python -m autorom --accept-license -y
  ```

- **`Environment PongDeterministic doesn’t exist`**  
  Confirm **Gym 0.26.2** + **Atari extras** and ROMs are installed:
  ```bash
  pip install "gym==0.26.2" "gym[atari]" ale-py autorom
  python -m autorom --accept-license -y
  ```

- **CUDA not used**  
  Check `torch.cuda.is_available()`. Install proper CUDA wheels from https://pytorch.org.

---

## 📦 Export Dependencies

To capture the exact versions from your active environment:
```bash
python -m pip freeze > requirements.txt
```

*(Conda alternative)*
```bash
conda env export --from-history > environment.yml
```

*(Starter requirements, if you want a minimal list to start from)*
```
gym==0.26.2
gym[atari]==0.26.2
ale-py
autorom
numpy
pandas
matplotlib
torch
torchvision
torchaudio
```

---

## 📜 License

For academic use in **CSCN 8020**. Atari ROMs are installed via AutoROM under their respective licenses.
