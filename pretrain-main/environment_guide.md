# Environment Setup and Update Guide

## pyenv and Mamba Installation

### pyenv Installation

To set up pyenv on your system, execute the following commands:

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo '' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
source ~/.bashrc
```

### Mamba Installation through pyenv

To install Mamba using pyenv, run these commands:

```bash
pyenv install mambaforge-22.9.0-3
pyenv global mambaforge-22.9.0-3
```

## Initial Environment Setup

To create a new environment with Mamba, run the following command:

```bash
mamba env create -n <YOUR_ENV_NAME> -f environment.yml
```

Replace `<YOUR_ENV_NAME>` with the desired name for your environment.

After creating your environment, activate it using this command:

```bash
mamba activate <YOUR_ENV_NAME>
```

Set the following environment variables to pre-build the DeepSpeed related extensions "ops":

```bash
export DS_BUILD_OPS=1
export DS_BUILD_EVOFORMER_ATTN=0
export DS_BUILD_SPARSE_ATTN=0
```

Install DeepSpeed with the specified version:

```bash
pip install deepspeed>=0.14
```

## Updating the Environment

When you add new libraries to your environment, update the `environment.yml` file to reflect these changes with the following command:

```bash
mamba env export --no-builds | grep -vE '^\s*name:' | grep -vE '^\s*prefix:' | grep -vE '^\s*- deepspeed==[^ ]+$' | awk '/- pip:/{print "  - pip:\n    - --extra-index-url https://download.pytorch.org/whl/cu118"; next}1' > environment.yml
```
