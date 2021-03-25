# Reproducibility
from pytorch_lightning import seed_everything

# Model training
from model import DomainTransferGAN
from datamodule import CelebADataModule
import stargan_weights_conversion
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger

# Model deployment
import discord_bot

# Checking file path
import os

# Command line arguments and project configuration
import argparse

if __name__ =='__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Trainer arguments
    parser.add_argument('--n_epochs', type=int, default=1, help="number of epochs the model will train for")
    parser.add_argument('--log_steps', type=int, default=5, help="duration of epochs between logging")
    parser.add_argument('--train_percent', type=float, default=0.1, help="how much of the training dataset is used")

    # Model hyperparameters
    parser.add_argument('--pretrained', type=bool, default=True, help="if training from pretrained weights")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate for the model")
    parser.add_argument('--beta1', type=float, default=0.5, help="beta1 for model optimizer")
    parser.add_argument('--beta2', type=float, default=0.999, help="beta2 for model optimizer")
    parser.add_argument('--n_critic', type=int, default=5, help="how many epochs for the generator to learn once")
    parser.add_argument('--g_hidden', type=int, default=64, help="hidden layers of the generator")
    parser.add_argument('--g_n_blocks', type=int, default=6, help="number of residual blocks in the generator")
    parser.add_argument('--d_hidden', type=int, default=64, help="hidden layers of the discriminator")
    parser.add_argument('--d_n_blocks', type=int, default=5, help="number of downsample blocks in the discriminator")
    parser.add_argument('--class_weight', type=int, default=1, help="how much weight classification has on loss")
    parser.add_argument('--grad_weight', type=int, default=10, help="how much weight gradient penalty has on loss")
    parser.add_argument('--recon_weight', type=int, default=10, help="how much weight reconstruction has on loss")

    # Dataset
    parser.add_argument('--dataset_size', type=int, default=21000, help="number of examples in dataset")
    parser.add_argument('--train_size', type=int, default=20000, help="number of examples in training set")
    parser.add_argument('--test_size', type=int, default=1000, help="number of examples in test set")
    parser.add_argument('--batch_size', type=int, default=16, help="number of examples in one batch")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers in dataloader")
    parser.add_argument('--n_labels', type=int, default=5, help="number of image attributes")
    parser.add_argument('--image_start', type=list, default=[218, 178, 3], help="start height, width, channels of image")
    parser.add_argument('--image_end', type=list, default=[3, 128, 128], help="final channels, height, width of image")
    parser.add_argument('--image_mean', type=list, default=[0.5, 0.5, 0.5], help="mean for normalization")
    parser.add_argument('--image_std', type=list, default=[0.5, 0.5, 0.5], help="std for normalization")
    parser.add_argument('--choice_labels', type=list, help="chosen attributes from the dataset",
                        default=['Eyeglasses', 'Blond_Hair', 'Smiling', 'Male', 'Young'])
    parser.add_argument('--alt_label_names', type=list, help="Alternate names or labels used in results",
                        default=['Glasses', 'Hair Colour', 'expression', 'Gender', 'Age'])
    parser.add_argument('--org_labels', type=list, help="original labels of pretrained weights",
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # File paths
    parser.add_argument('--image_folder', type=str, default='celeba_data/images', help="file path to images")
    parser.add_argument('--attr_csv', type=str, default='celeba_data/labels/list_attr_celeba.csv', help="file path to attribute csv")
    parser.add_argument('--stargan_g', type=str, default='models/200000-G.ckpt', help="file path to pretrained generator")
    parser.add_argument('--stargan_d', type=str, default='models/200000-D.ckpt', help="file path to pretrained discriminator")
    parser.add_argument('--g_checkpoint', type=str, default='models/G_pretrained.ckpt', help="file path to generator checkpoint")
    parser.add_argument('--d_checkpoint', type=str, default='models/D_pretrained.ckpt', help="file path to discriminator checkpoint")
    parser.add_argument('--result_path', type=str, default='results', help="file path to save image results")

    # Program
    parser.add_argument('--train', type=bool, default=False, help="whether training model or predicting")
    parser.add_argument('--gen_images', type=int, default=4, help="how many images to save in one result")

    config = parser.parse_args()
    print(config)

    # Set random seed for reproducibility
    seed_everything(6)
    
    model = DomainTransferGAN(config)
    
    # Don't overwrite pre-existing checkpoints
    if config.pretrained and not model.checkpoints_exist():
        print("Running conversion on pretrained weights...")
        stargan_weights_conversion.run(model, config)
    
    if config.pretrained:
        print("Using pretrained weights.")

    if model.checkpoints_exist():
        print("Loading checkpoint...")
        model.load_checkpoint()

    if config.train:
        celeba_dm = CelebADataModule(config)
        
        # Log to Weights and Biases
        wandb_logger = WandbLogger(
            name='training', version='11',
            project='Image Domain Transfer GAN'
        )
        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=config.n_epochs,
            log_every_n_steps=config.log_steps,
            limit_train_batches=config.train_percent,
        )
        print("Starting training...")
        trainer.fit(model, datamodule=celeba_dm)

        # Save model to resume training later or deployment
        model.save_checkpoint()

        # Save examples of generated images
        trainer.test()
        
    if not config.train:
        # Predicting, setup and launch discord bot
        bot = discord_bot.setup(model, config)
        discord_bot.run(bot)