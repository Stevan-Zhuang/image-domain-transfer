# Discord bot
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

# Handling image data
from PIL import Image
import io
import requests
from utils import image_to_tensor, tensor_to_image

# Type hinting documentation
from argparse import Namespace
from pytorch_lightning import LightningModule

bot = commands.Bot(command_prefix='!', case_insensitive=True)

def setup(model: LightningModule, config: Namespace) -> commands.Bot:
    """Prepares the bot."""
    bot.config = config
    bot.id = 762394671745597490

    bot.generator = model.generator
    bot.discriminator = model.discriminator

    # Reaction gui
    bot.label_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣']
    bot.go_emoji = '➡️'
    bot.selected = ['🔴','🟢']

    # Keep track of received and sent messages
    bot.image_data = None
    bot.labels = None
    bot.sent_panel = None
    return bot

def run(bot: commands.Bot) -> None:
    """Brings the bot online."""
    load_dotenv()
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    bot.run(DISCORD_TOKEN)

@bot.event
async def on_ready():
    """Verify the bot has successfully come online."""
    print(f"{bot.user.name} is online")

error_message = "Please upload a valid image!"
help_message = "Upload an image of a face to translate attributes."

@bot.command(name='face', help=help_message)
async def face(ctx: discord.ext.commands.Context) -> None:
    """Creates the reaction panel for attribute selection."""
    try:
        # Fetch the sent image
        url = ctx.message.attachments[0].url
        image = Image.open(requests.get(url, stream=True).raw)
    except:
        # Image could not be found
        await ctx.send(error_message)
        return

    # If the image is a PNG file, it will have 4 channels
    # The model cannot handle 4 channels, so convert it to 3
    image = image.convert('RGB')
    
    bot.image_data = image_to_tensor(image, bot.config)

    # Guess labels with discriminator
    _, labels = bot.discriminator(bot.image_data)
    
    # Round labels to 0 and 1
    bot.labels = (labels > 0.5).float()

    bot.sent_panel = await ctx.send(embed=get_info_panel())

    # Add reactions
    for emoji in bot.label_emojis:
        await bot.sent_panel.add_reaction(emoji)

    await bot.sent_panel.add_reaction(bot.go_emoji)

@bot.event
async def on_reaction_add(reaction: discord.Reaction,
                          user: discord.User) -> None:
    """Handle user reactions to select attributes."""
    # The reaction should not come from the bot and
    # should be on the bots message
    if user.id != bot.id and reaction.message.id == bot.sent_panel.id:
        await bot.sent_panel.remove_reaction(reaction, user)
        emoji = str(reaction)
        
        if emoji in bot.label_emojis:
            label_idx = bot.label_emojis.index(emoji)
            # Change the label to its opposite
            bot.labels[0][label_idx] = float(not bot.labels[0][label_idx])
            # Edit panel
            await bot.sent_panel.edit(embed=get_info_panel())

        if emoji == bot.go_emoji:
            # Generate image
            generated_data = bot.generator(bot.image_data, bot.labels)
            generated_image = tensor_to_image(generated_data, bot.config)
            
            # To upload the image, convert to bytes
            # https://stackoverflow.com/questions/59868527
            with io.BytesIO() as image_binary:
                generated_image.save(image_binary, 'png')
                image_binary.seek(0)
                image_file = discord.File(
                    fp=image_binary, filename='gen_img.png'
                )
                ctx = reaction.message.channel
                bot.sent_image = await ctx.send(file=image_file)

def get_info_panel() -> discord.Embed:
    """Build and return the panel used to select attributes."""
    info_panel = discord.Embed(title="", color=0xf8f8ff)

    title = "Select the target attributes you would like for your image by reacting!"
    body = ["Labels are initialized at the guess of a discriminator model."]

    for label_idx in range(bot.config.n_labels):
        line = []
        # Corresponding number reaction
        line.append(bot.label_emojis[label_idx])
        # Label name, remove underscores
        line.append(bot.config.org_labels[label_idx].replace('_', ' '))
        # Red or green circle depending if label is selected
        line.append(bot.selected[int(bot.labels[0][label_idx])])

        body.append(" ".join(line))

    info_panel.add_field(name=title, value="\n".join(body))
    return info_panel