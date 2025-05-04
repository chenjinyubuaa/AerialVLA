# Implementation of AerialVLA: A Vision-Language-Action Model for Aerial Navigation with Online Dialogue
## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/chenjinyubuaa/AerialVLA
cd AerialVLA
```

#### 2. **Install the inference package:**
```bash
conda create -n aerialVLA python=3.10 -y
conda activate aerialVLA
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

#### 2. **train AerialVLA**

```
bash scripts\archived\aerialchat\finetune_lora.sh
```


#### 2. **eval AerialVLA**

```
bash scripts\archived\aerialchat\eval_batch.sh
```

