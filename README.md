# OOD Dynamic detection for a safer use of reinforcement learning

This repository implements a method to perform ood dynamic detection on a gym environment.

The main functions and classes are stored in rl_ood.py.

the notebook RL_OOD_detection.ipynb proposes examples and illustations of the method.

We test our method on the benchmark proposed in 'Benchmark for Out-of-Distribution Detection in Deep Reinforcement Learning' (available at https://arxiv.org/abs/2112.02694)

## Installation

```
conda create -n ood_env python=3.8 jupyter  -y
conda activate ood_env
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```


TODO:
- investigate the correlation between the ood score and the decreasing of the reward
- propose a gym wrapper to do online ood detection
- Complete the comparison between the benchmark and our results.
