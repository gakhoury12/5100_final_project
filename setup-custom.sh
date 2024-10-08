#!/bin/bash
conda env create -f flappy_env_3.yml
conda activate flappy_env_3
cd /home/gabe/Desktop/5100/flappy-2/flappy-bird-gym
/home/gabe/miniconda3/envs/flappy_env_3/bin/python flappy_bird_gym/cli.py --mode random