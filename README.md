# image-domain-transfer
This repository contains a Discord bot that can take input images\
and output images with the domain attributes changed.

This was done for my high school computer science 12 final project.\
Model is an implementation of the StarGAN model in PyTorch Lightning\
from the paper https://arxiv.org/abs/1711.09020.

# Discord Bot
The model weights used in the discord bot are the pretrained weights from the StarGAN paper.\
While I attempted to train my own weights, both from scratch and from pretrained,\
I didn't achieve any strong results. Read about my training process [here](https://wandb.ai/stevan-zhuang/Image%20Domain%20Transfer%20GAN/reports/Computer-Science-12-Final-Project-StarGAN-Training--Vmlldzo1NTQ2MzY?accessToken=8x8r4lqay36gg8zmlz9zgd1k0awrx7lix0okl78re04wwvpadhn8d1trbi4za1a0).

## Discord bot showcase:
![](https://github.com/Stevan-Zhuang/image-domain-transfer/blob/main/showcase/discord_bot.gif)
