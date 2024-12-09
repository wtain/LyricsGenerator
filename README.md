# LyricsGenerator
Generate music lyrics

## Setup

1. Add LyricsGenius submodule:
```commandline
git submodule update --init --recursive
```

Currently (9 Dec 2024), I am using my branch ``wtain/fix-artist-search-pagination-1`` which contains some changes that I rely on and which were not yet merged:
- https://github.com/johnwmillr/LyricsGenius/pull/278
   
2. Install local submodule into the environment
```commandline
pip install -e Lyricsgenius
```

3. Setup conda environment
```commandline
conda create -n LyricsGenius python=3.9 -y
conda activate LyricsGenius
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

4. Test the setup (make sure CUDA is available)
```commandline
import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check CUDA version
print("CUDA Version:", torch.version.cuda)

# Check GPU availability
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
```