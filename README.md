
# Biped Robot with Deep Reinforcement Learning

This project explores how to teach a **low-cost biped robot** to walk using **deep reinforcement learning algorithms**. The entire pipeline is designed to be **accessible, reproducible, and modular**, combining simple hardware (12 DOF servos, Jetson Nano) with simulation in **MuJoCo** and training using **PyTorch**.

We compare and evaluate four state-of-the-art RL algorithms for continuous control:
- DDPG (Deep Deterministic Policy Gradient)
- D4PG (Distributed Distributional DDPG)
- SAC (Soft Actor-Critic)
- MPO (Maximum a Posteriori Policy Optimization)

The project aims to **democratize humanoid robotics** by making it feasible to train bipedal locomotion using affordable tools and open-source code.
<p align="center">
  <img src="media\robot.jpg" width="30%" alt="Robot"/>
</p>

---

## 📂 Project Structure

```bash
.
├── algorithms/         # Reinforcement Learning algorithms (SAC, DDPG, D4PG, MPO)
├── envs/               # Modular MuJoCo environments
│   ├── mujoco_env.py   # Unified environment class
│   ├── rewards/        # Modular reward functions (Walk, Target, etc.)
│   └── utils/          # Environment utilities (Randomizer)
├── src/                # Training & evaluation scripts
├── config/             # YAML configurations
├── media/              # Visuals and demos
├── requirements.txt    # Project dependencies
└── README.md
```

---

## ⚙️ How to Run

### 1. **Set up the environment**
The project requires Python 3.11+ 
```bash
# Recommended: create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. **Train an agent**
Training is fully configuration-driven. You can specify the model, environment objective, and randomization settings in a YAML file.
```bash
python src/train.py --config config/final/train_config_sac.yaml
```

---

## 🔧 Configuration Options

The system uses Pydantic for robust configuration. Key options include:

### **Environment Config (`env_config`)**
- `objective`: `walk`, `target`, or `balance`.
- `enable_mirroring`: Boolean toggle for observation mirroring.
- `reward_weights`: Dictionary to customize reward term importance.

### **Randomization Config (`randomization`)**
- `randomize_dynamics`: Toggles physics parameter randomization (friction, mass, etc.).
- `randomize_sensors`: Toggles IMU and velocity noise.
- `friction`, `mass`, `joint_damping`: Ranges for randomization.

---

## 🧠 Algorithms Summary

| Algorithm | Type       | Strengths                      | Weaknesses                      |
|-----------|------------|--------------------------------|---------------------------------|
| DDPG      | Off-policy | Fast, simple                   | Unstable, outdated              |
| D4PG      | Off-policy | Fast, stable, great for tuning | Not powerful enough             |
| SAC       | Off-policy | Robust, best final performance | Slow training                   |
| MPO       | Off-policy | Theoretically grounded         | Sensitive, complex hyperparams  |

---

## 🧪 Future Work

- Sim-to-real transfer with domain randomization  
- Use of temporal models (LSTM, 1D CNN)  
- Add direction, rotation, and velocity control

---

## 📜 License

MIT License. Feel free to fork, modify, and contribute!

---

## 🤖 Contact

Project by **Pablo Gómez Martínez**  
Contact: [pablodiegogomez@gmail.com](mailto:)
