# AGENTS.md

## Project

This repository reproduces the SAC-RCBF experiments for **Safe Reinforcement Learning Using Robust Control Barrier Functions**.

Core idea:

- SAC actor outputs `u_RL`.
- RCBF-QP safety layer computes a correction `u_S`.
- The environment receives the final safe action `u = u_RL + u_S`.
- `cbf_mode` controls how the safety layer is used:
  - `off`: no safety layer.
  - `baseline`: safety layer is used, but losses are not differentiated through the QP.
  - `full`: differentiable safety layer, losses use the safe action.
  - `mod`: constraint-agnostic task learning.

## Current workstation status

The Linux workstation environment has been configured using **方案 A: Miniforge/Conda**.

Current assumptions:

```text
Project directory: /workspace/Mod-RL-RCBF
Miniforge path:    /workspace/tools/miniforge3
Conda env name:    mod-rl-rcbf
Python version:    3.10
Install route:     Miniforge/Conda, not venv
```

Before running code in this repository, activate the environment:

```bash
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate mod-rl-rcbf
cd /workspace/Mod-RL-RCBF
```

If `conda` is already initialized in the shell, this shorter form is also acceptable:

```bash
conda activate mod-rl-rcbf
cd /workspace/Mod-RL-RCBF
```

Do **not** use the old directory name:

```text
/workspace/Mod-RL-RCBF-main
```

The correct project directory is:

```text
/workspace/Mod-RL-RCBF
```

## Repository layout

Expected files:

```text
.
├── main.py
├── build_env.py
├── plot_utils.py
├── envs/
│   ├── unicycle_env.py
│   ├── simulated_cars_env.py
│   ├── pvtol_env.py
│   ├── pyglet_rendering.py
│   └── utils.py
└── rcbf_sac/
    ├── sac_cbf.py
    ├── cbf_qp.py
    ├── diff_cbf_qp.py
    ├── dynamics.py
    ├── gp_model.py
    ├── generate_rollouts.py
    ├── model.py
    ├── compensator.py
    ├── replay_memory.py
    └── utils.py
```

## Linux workstation rules

Use the Linux workstation, not a Windows path.

Recommended working location:

```bash
cd /workspace/Mod-RL-RCBF
```

If `/workspace` does not exist on a different target workstation, use a persistent project directory such as:

```bash
mkdir -p ~/projects
cd ~/projects/Mod-RL-RCBF
```

Do not put important project files in:

```text
/tmp
/
system directories
```

Do not install project packages globally with:

```bash
sudo pip install ...
pip install ...   # when no virtual environment is active
```

All `pip install` commands for this repository must be run after:

```bash
conda activate mod-rl-rcbf
```

## Environment policy

Use one isolated Python environment for this repository.

Preferred environment for the current workstation:

```text
Conda/Miniforge environment: mod-rl-rcbf
```

Use `venv` only if the user explicitly requests a pure `venv` setup or if Conda/Miniforge is unavailable on a different machine.

Do not commit any environment directory:

```text
.venv/
env/
conda-env/
/workspace/tools/miniforge3/
```

## Miniforge installation record

Miniforge was installed under `/workspace/tools/miniforge3` using the following route:

```bash
cd /workspace
mkdir -p tools
cd tools

curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh -b -p /workspace/tools/miniforge3

source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda init bash
conda config --set auto_activate_base false

source ~/.bashrc
```

Do not reinstall Miniforge unless `/workspace/tools/miniforge3` is missing or broken.

To verify Miniforge/Conda:

```bash
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda --version
which conda
conda env list
```

## Conda environment setup record

The project environment was created as:

```bash
cd /workspace/Mod-RL-RCBF

conda create -n mod-rl-rcbf python=3.10 pip -y
conda activate mod-rl-rcbf

python -m pip install --upgrade pip setuptools wheel
```

If the environment already exists, do not recreate it. Activate it instead:

```bash
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate mod-rl-rcbf
cd /workspace/Mod-RL-RCBF
```

To verify the active environment:

```bash
which python
which pip
python -V
pip -V
conda info --envs
```

Expected signs:

```text
python should come from: /workspace/tools/miniforge3/envs/mod-rl-rcbf/...
Python should be:       3.10.x
Active env should be:   mod-rl-rcbf
```

## Dependencies

The repository imports at least the following external packages:

```text
torch
numpy
gym
matplotlib
tqdm
gpytorch
qpth
quadprog
pyglet
comet_ml
```

The repository may not include a complete `requirements.txt`. If missing, create one inside `/workspace/Mod-RL-RCBF`.

Current suggested `requirements.txt` contents:

```text
numpy
gym==0.23.1
matplotlib
tqdm
gpytorch
qpth
quadprog
pyglet==1.5.27
comet-ml
```

Install PyTorch separately before the rest of the dependencies.

## PyTorch installation policy

Always check the GPU first:

```bash
nvidia-smi
```

For the current RTX PRO 6000 workstation, prefer a CUDA-enabled PyTorch wheel.

First try:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If that fails because of driver or network constraints, try:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

After installing PyTorch, verify CUDA from Python:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

If `torch.cuda.is_available()` is `False`, do not assume the code is broken. First check the driver, CUDA wheel, proxy/network, and whether the job is running inside a container with GPU access.

## Project dependency installation record

Dependencies were installed with:

```bash
cd /workspace/Mod-RL-RCBF
conda activate mod-rl-rcbf

cat > requirements.txt <<'EOF_REQ'
numpy
gym==0.23.1
matplotlib
tqdm
gpytorch
qpth
quadprog
pyglet==1.5.27
comet-ml
EOF_REQ

python -m pip install -r requirements.txt
```

Verify imports:

```bash
python - <<'PY'
import torch
import numpy
import gym
import gpytorch
import qpth
import quadprog
import matplotlib
import tqdm
import pyglet
import comet_ml
print("python ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

## Setup commands: current Conda/Miniforge route

Use this route on the current workstation.

```bash
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate mod-rl-rcbf
cd /workspace/Mod-RL-RCBF

which python
python -V
```

If the environment does not exist, create it:

```bash
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
cd /workspace/Mod-RL-RCBF

conda create -n mod-rl-rcbf python=3.10 pip -y
conda activate mod-rl-rcbf

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

If the CUDA 12.8 wheel fails:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Setup commands: venv route

Use this route only if the user explicitly requests `venv` or Conda/Miniforge is unavailable.

```bash
cd /workspace/Mod-RL-RCBF

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio

cat > requirements.txt <<'EOF_REQ'
numpy
gym==0.23.1
matplotlib
tqdm
gpytorch
qpth
quadprog
pyglet==1.5.27
comet-ml
EOF_REQ

python -m pip install -r requirements.txt
```

Verify:

```bash
which python
python -V
python - <<'PY'
import torch, numpy, gym, gpytorch, qpth, quadprog
print("python ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

## Correct command-line arguments

Use `--env_name`, not `--env`.

The parser in `main.py` defines:

```text
--env_name
--cbf_mode
--max_episodes
--batch_size
--start_steps
--seed
--cuda
--device_num
--model_based
--updates_per_step
--rollout_batch_size
--real_ratio
--gp_max_episodes
```

The exact supported arguments should still be checked against the local `main.py` before making changes:

```bash
python main.py --help
```

## Smoke-test commands

Run these before long training.

CPU smoke test:

```bash
cd /workspace/Mod-RL-RCBF
conda activate mod-rl-rcbf

python main.py \
  --env_name Unicycle \
  --cbf_mode off \
  --max_episodes 1 \
  --batch_size 256 \
  --start_steps 5000 \
  --seed 12345
```

GPU smoke test:

```bash
cd /workspace/Mod-RL-RCBF
conda activate mod-rl-rcbf

python main.py \
  --cuda \
  --device_num 0 \
  --env_name Unicycle \
  --cbf_mode off \
  --max_episodes 1 \
  --batch_size 256 \
  --start_steps 5000 \
  --seed 12345
```

If CUDA fails, check:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

## Experiment commands

Use these as templates. Do not launch long experiments without user confirmation.

### Unicycle, no CBF

```bash
python main.py --cuda --env_name Unicycle --cbf_mode off --max_episodes 200 --seed 12345
```

### Unicycle, baseline

```bash
python main.py --cuda --env_name Unicycle --cbf_mode baseline --max_episodes 200 --seed 12345
```

### Unicycle, full differentiable RCBF

```bash
python main.py --cuda --env_name Unicycle --cbf_mode full --max_episodes 200 --seed 12345
```

### Unicycle, model-based SAC-RCBF

```bash
python main.py \
  --cuda \
  --env_name Unicycle \
  --model_based \
  --updates_per_step 2 \
  --batch_size 512 \
  --rollout_batch_size 5 \
  --real_ratio 0.3 \
  --gp_max_episodes 70 \
  --cbf_mode full \
  --max_episodes 200 \
  --seed 12345
```

### SimulatedCars, baseline

```bash
python main.py --cuda --env_name SimulatedCars --max_episodes 300 --cbf_mode baseline --seed 12345
```

### SimulatedCars, full differentiable RCBF

```bash
python main.py --cuda --env_name SimulatedCars --max_episodes 300 --cbf_mode full --seed 12345
```

### Constraint-agnostic Unicycle

```bash
python main.py --cuda --env_name Unicycle --cbf_mode mod --rand_init True --seed 12345
```

### Zero-shot transfer test

```bash
python main.py \
  --mode test \
  --validate_episodes 200 \
  --resume RUN_NUMBER \
  --cbf_mode baseline \
  --env_name Unicycle \
  --obs_config random \
  --seed 12345
```

Replace `RUN_NUMBER` with the saved run number under `output/`.

## Long-running jobs

Use `tmux` for any run longer than a quick smoke test.

```bash
tmux new -s rcbf
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate mod-rl-rcbf
cd /workspace/Mod-RL-RCBF
python main.py --cuda --env_name Unicycle --cbf_mode full --max_episodes 200 --seed 12345
```

Detach:

```text
Ctrl+B, then D
```

Resume:

```bash
tmux attach -t rcbf
```

## Coding and modification rules

- Keep the original algorithm structure unless the user asks for a research modification.
- Do not rename core variables like `uRL`, `uS`, `cbf_mode`, `DynamicsModel`, or `RCBF_SAC` without a specific reason.
- Do not silently replace the differentiable QP layer with a non-differentiable solver.
- Do not remove safety-layer logic to make a script run faster unless the user explicitly asks for a non-CBF baseline.
- Avoid changing experiment defaults unless the change is documented in the response.
- Prefer small, testable patches over broad rewrites.
- After code changes, run at least the CPU smoke test unless dependencies are missing.
- Do not commit `output/`, checkpoints, logs, virtual environments, caches, or generated plots unless the user explicitly asks.
- Do not commit `/workspace/tools/miniforge3` or any Conda package cache.

## Git and repository hygiene

Before committing changes, check:

```bash
cd /workspace/Mod-RL-RCBF
git status
```

Recommended `.gitignore` entries if they are missing:

```text
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/

# environments
.venv/
env/
conda-env/

# outputs and logs
output/
outputs/
logs/
*.log
checkpoints/
*.pt
*.pth
*.ckpt

# caches
.cache/

# local tools
/workspace/tools/miniforge3/
```

Do not commit generated training outputs unless the user explicitly asks.

## Common failure modes

### `conda: command not found`

Load Conda manually:

```bash
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate mod-rl-rcbf
```

If this works, the shell initialization is incomplete. Check `~/.bashrc` and `conda init bash`.

### Wrong project path

The correct current path is:

```bash
cd /workspace/Mod-RL-RCBF
```

Do not use:

```bash
cd /workspace/Mod-RL-RCBF-main
```

### `ModuleNotFoundError`

The environment is not activated or dependencies are missing.

Check:

```bash
which python
which pip
python -V
pip -V
```

Expected path should include:

```text
/workspace/tools/miniforge3/envs/mod-rl-rcbf/
```

Then reinstall inside the active environment:

```bash
conda activate mod-rl-rcbf
cd /workspace/Mod-RL-RCBF
python -m pip install -r requirements.txt
```

### `error: command 'gcc' failed`

System build tools are missing. On Ubuntu/Debian, ask an administrator to install:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev
```

If the user does not have sudo access, try Conda compilers:

```bash
conda install -c conda-forge compilers -y
```

### `torch.cuda.is_available()` is `False`

Check:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

If `nvidia-smi` works but PyTorch CUDA is false, reinstall PyTorch with a CUDA wheel:

```bash
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If that fails:

```bash
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Rendering or display errors

Most training commands do not require rendering. Avoid `--visualize` on a headless server unless X11 forwarding or a virtual display is configured.

### README command mismatch

If README examples use `--env`, convert them to `--env_name`, because this repository's `main.py` parser uses `--env_name`.

### Accidental global pip install

If a package was installed without the Conda environment active, do not rely on it. Activate the environment and reinstall inside it:

```bash
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate mod-rl-rcbf
python -m pip install <package-name>
```

## Response expectations for agents

When helping the user:

1. Identify whether the issue is environment, dependency, command-line argument, CUDA, Git, or algorithm logic.
2. Ask for the exact command and full traceback only when the current error text is insufficient.
3. Prefer direct Linux commands over Windows/PowerShell commands.
4. Keep explanations tied to this repository and the SAC-RCBF paper.
5. For environment setup, assume the current workstation uses **Conda/Miniforge** unless the user says otherwise.
6. Always use `/workspace/Mod-RL-RCBF` as the project directory unless the user provides a different confirmed path.
7. Do not propose reinstalling Miniforge or recreating the Conda environment unless verification shows it is missing or broken.
