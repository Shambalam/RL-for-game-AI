# Style-Conditioned NPC Behavior
### Deep Reinforcement Learning · Tactical Grid-World · PPO

One policy network. Four distinct tactical personalities. No retraining to switch between them.

This project trains a single PPO agent to play four different NPC combat styles in a procedurally generated 2D grid-world. The style — **Aggressive**, **Defensive**, **Flanker**, or **Sniper** — is assigned at runtime via a learned embedding vector. Each style has its own independent reward function, so the behaviors are genuinely distinct rather than cosmetically different.

Built as a deep learning course project, now being developed further toward real game engine integration.

---

## Results

Evaluated over 30 deterministic episodes per style after 300k training steps on a T4 GPU (~8 min per style).

| Style | Mean Reward | Win Rate | Avg Distance | Dmg Dealt | Dmg Taken |
|---|---|---|---|---|---|
| Aggressive | 87.48 | **33%** | 7.14 | 39.67 | 17.33 |
| Defensive | 24.69 | 30% | 7.61 | 39.00 | 33.67 |
| Flanker | **93.01** | 20% | **6.27** | 28.67 | 22.33 |
| Sniper | 67.56 | 17% | **7.79** | 22.67 | **18.00** |
| Vanilla PPO (baseline) | 31.20 | 22% | 7.35 | 28.40 | 24.10 |

The distance ordering (Flanker < Aggressive < Defensive < Sniper) confirms the style embedding is producing real spatial behavioral differences, not just different reward accumulation. The vanilla PPO baseline without style conditioning converges to a fixed 7.2–7.5 average distance with no runtime controllability.

---

## Environment

- **Grid:** 16×16, procedurally regenerated every episode
- **Obstacles:** ~15% of tiles — block movement and line-of-sight
- **Cover:** ~10% of tiles — passable, used as reward signal for defensive style only
- **Opponent:** scripted player that chases and shoots (deterministic, same in train/eval)
- **Hit probability:** decays linearly with distance, floor at 15%
- **Episode length:** 300 steps max

**NPC actions (Discrete 6):**
`0` move up · `1` move down · `2` move left · `3` move right · `4` stay · `5` shoot

---

## Style Reward Design

Each style has completely independent reward functions. Styles share a `-0.01` step cost, `+20` win bonus, and `-20` death penalty.

**Aggressive** — close in and deal damage
- `+4.0 × distance_closed` per step
- `+6.0` on hit · `-0.8` idle

**Defensive** — peek-shoot-cover sequence
- `+0.05` survival per step · `+0.4` in cover · `-2.0` damage taken
- `+1.5` for leaving cover to shoot · `+1.0` for returning to cover next step
- Tracks `was_in_cover` boolean between steps

**Flanker** — approach from outside player FOV, attack from the side
- `+1.5 × lateral_score` when within engagement range
- `lateral = 1 - |Δx / ||Δp|||` (1.0 = directly to the side)
- `+0.5` per step outside player's 180° movement-based FOV cone
- `+3.0` bonus for hitting from outside FOV

**Sniper** — maintain distance, acquire LOS, take long shots
- `+0.5` at normalized distance > 0.6 · `+0.2` at > 0.45
- `+10.0` for ranged hit · `+4.0 × distance_retreated` when player closes in
- Uses positive tiers only — per-step penalties accumulated badly over long episodes

---

## Setup

```bash
git clone https://github.com/yourusername/style-conditioned-npc
cd style-conditioned-npc
pip install -r requirements.txt
```

**Requirements:**

gymnasium>=0.29.0
stable-baselines3>=2.2.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0

---

## Training

Train all four styles (sequential, ~30 min total on GPU):
```bash
python train.py
```

Train a single style:
```bash
python train.py --style aggressive
python train.py --style flanker --timesteps 1000000
```

Models save to `models/<style_name>_final.zip`. TensorBoard logs go to `logs/`.

```bash
tensorboard --logdir logs/
```

---

## Evaluation

Evaluate all styles and generate comparison charts:
```bash
python evaluate.py --plot --episodes 30
```

Watch a single style play (text render):
```bash
python evaluate.py --style sniper --render --episodes 3
```

---

## Google Colab

The entire project runs in a single Colab notebook with a free T4 GPU. Open `npc_project_colab.ipynb` in Colab, set runtime to T4, and run all cells. Training takes about 8 minutes per style. The last cell downloads all trained models as a zip.

---

## Known Limitations

- **300k steps is low** for flanker and sniper — both styles would benefit from 1M+ steps to fully converge
- **Scripted deterministic opponent** — the player never adapts, which limits how hard the agent has to work to generalize
- **Cover is a reward signal only** — no mechanical damage reduction; a real game would need proper cover mechanics
- **FOV model is approximate** — player facing updates from movement deltas which has limited variance against a chasing opponent
- **No quantitative diversity metric** — behavioral separation is measured through distance/damage proxies rather than trajectory entropy or pairwise rollout distance

---

## Roadmap

- [ ] Train flanker and sniper to 1M steps
- [ ] Style vs style 1v1 evaluation (round-robin tournament)
- [ ] 2v2 team compositions with shared team reward
- [ ] Self-play opponent (replace scripted player)
- [ ] ONNX export for game engine integration
- [ ] Godot/Unity inference wrapper
- [ ] Style interpolation experiments (blend two style embeddings at runtime)
- [ ] Curriculum learning for flanker

---

## Background

This started as a deep learning course project at Kennesaw State University. The core idea is that current DRL game AI (DQN, AlphaStar, OpenAI Five) produces agents that are very good at winning but have no concept of behavioral style. Conditioning a single policy on a discrete style embedding with independent reward functions per style is a simple and practical approach to producing diverse, controllable NPC behavior — the kind that actually makes games more interesting to play.

The long-term goal is integrating this into a 3D extraction shooter, where the four styles map naturally to game roles: aggressive = rusher, defensive = objective holder, flanker = rotator, sniper = overwatch.

---

## Citation

If you use this project:

```bibtex
@misc{cararo2025styleconditioned,
  title={Style-Conditioned NPC Behavior via Deep Reinforcement Learning in a Tactical Grid-World Environment},
  author={Cararo, Daniel},
  year={2025},
  url={https://github.com/Shambalam/RL-for-game-ai}
}
```
