# perspective-agent-p3  

## Vast.ai instance setup 

Generate the SSH key:  

```bash
git config --global user.name "hjpae"
git config --global user.email "hnjpae@gmail.com"

ssh-keygen -t ed25519 -C "hnjpae@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add SSH key and then authorize connection to GitHub:  

```bash
ssh -T git@github.com
```

Clone the repository:  

```bash
git clone git@github.com:hjpae/perspective-agent-p3.git
cd perspective-agent-p3
git remote -v
```

If overwriting the existing repo, use this:  

```bash
git remote remove origin
git remote add origin git@github.com:hjpae/perspective-agent-p3.git
```


## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate cearlab-phase3
```

Install PyTorch separately if needed:  
(The code has been tested with the cu128 wheel and is expected to be compatible with newer CUDA setups)  

```bash
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify the installation with:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```


## Repository layout

```text
cear_pilot/
  envs/          Environment definitions for Phase 1 and Phase 2
  models/        Encoder, world latent, state head, policy, decoder, and agent
  training/      Training entry points for Phase 1 and Phase 2
  analysis/      Probe analysis and final paper figure generation
  experiments/   Optional rollout collection utilities

run_phase2.sh    Main script for Phase 2 experiment sweeps
environment.yml  Conda environment specification
README.md
```


## Requirements

TBD  


## Running Phase 2

TBD


## Generating paper figures

TBD


## Data availability

TBD

