# Model saving and loading
import torch

# Type hinting documentation
from argparse import Namespace
from pytorch_lightning import LightningModule

def run(model: LightningModule, config: Namespace) -> None:
    """Formats pretrained StarGAN weights to be used."""
    # StarGAN was originally trained on gpu, switch to cpu
    g_state_dict = torch.load(config.stargan_g, map_location=torch.device('cpu'))
    d_state_dict = torch.load(config.stargan_d, map_location=torch.device('cpu'))
    
    # StarGAN models use different naming
    # Fix naming so weights are correctly loaded
    g_state_dict = {key.replace('main', 'net'): value
                    for key, value in g_state_dict.items()}

    d_state_dict = {key.replace('main', 'net'): value
                    for key, value in d_state_dict.items()}

    d_state_dict['patch_fc.weight'] = d_state_dict.pop('conv1.weight')
    d_state_dict['class_fc.weight'] = d_state_dict.pop('conv2.weight')

    model.generator.load_state_dict(g_state_dict)
    model.discriminator.load_state_dict(d_state_dict)

    # Save model weights
    torch.save(model.generator.state_dict(), config.g_checkpoint)
    torch.save(model.discriminator.state_dict(), config.d_checkpoint)