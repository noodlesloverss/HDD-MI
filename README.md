# High-Dimensional Distributional Model Inversion Attacks

This is a PyTorch implementation of the paper "High-Dimensional Distributional Model Inversion Attacks"

### 1.Setup Environment
This code has been tested with Python 3.7.9, PyTorch 1.10.1 and Cuda 11.1.

### 2.Dowloads Checkpoints
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

- FFHQ: https://github.com/NVlabs/ffhq-dataset

- We use the same target models and GAN as previous papers. You can download target models and generator at: https://drive.google.com/drive/folders/1jzu_6l9klW2IBKhHGfOKn3bpx0wUrUbr?usp=drive_link

- You can also train a GAN and a Target Classifier on your own dataset.

### Attack
- Modify the configuration.
  - change the paths for the GAN and Classifier.
  - change the paths for save attack images.

- Run:   `python main.py`

### Acknowledgements
This repository contains code snippets and some model weights from repositories mentioned below.

https://github.com/openai/maddpg

https://github.com/HanGyojin/RLB-MI

https://github.com/MKariya1998/GMI-Attack

https://github.com/SCccc21/Knowledge-Enriched-DMI