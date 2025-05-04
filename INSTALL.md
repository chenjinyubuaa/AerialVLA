conda create -p PATH/TO/ENV python=3.10
conda activate PATH/TO/ENV
pip install --upgrade pip  # Enable PEP 660 support.
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[train]"
