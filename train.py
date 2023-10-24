import os
import torch
import torchvision
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty

# hyperparameter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
Z_DIM = 100
NUM_EPOCHS = 1
CHANNEL_IMG = 3
FEATURE_D = 16
FEATURE_G = 16
CRITIC_ITERATIONS = 10
FIXED_NOISE = torch.rand((BATCH_SIZE, Z_DIM, 1, 1))
LAMBDA_GP = 10


# setup transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# load data
def load_data():
    return os.listdir("images")


class Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        data_idx = self.data[idx]
        image_path = f"./images/{data_idx}"
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data)


dataset = Dataset(load_data(), transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)


# initialize models
generator = Generator(Z_DIM, CHANNEL_IMG, FEATURE_G).to(DEVICE)
discriminator = Discriminator(CHANNEL_IMG, FEATURE_G).to(DEVICE)
initialize_weights(generator)
initialize_weights(discriminator)


# initialize optimizers
disc_opt = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.0, 0.99))
gen_opt = optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.99))


# initialize tensorboard
real_writer = SummaryWriter("./logs/real")
fake_writer = SummaryWriter("./logs/fake")
step = 0

generator.train()
discriminator.train()


for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(tqdm(dataloader)):
        real = real.to(DEVICE)
        cur_batch = real.size(0)

        # train discriminator
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.rand((cur_batch, Z_DIM, 1, 1)).to(DEVICE)
            fake = generator(noise)
            disc_real = discriminator(real).reshape(-1)
            disc_fake = discriminator(fake).reshape(-1)
            gp = gradient_penalty(discriminator, real, fake, DEVICE)
            loss_disc = (
                -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp
            )
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            disc_opt.step()

        # train generator
        output = discriminator(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        generator.zero_grad()
        loss_gen.backward(retain_graph=True)
        gen_opt.step()

        if batch_idx % 100 == 0 and batch_idx > 0:
            generator.eval()
            discriminator.eval()
            print(
                f"Epoch [{epoch}/{epoch}] Batch {batch_idx}/{len(dataloader)} \
                          Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

        with torch.no_grad():
            fake = generator(noise)

            real_grid = torchvision.utils.make_grid(real[:32], 8)
            fake_grid = torchvision.utils.make_grid(fake[:32], 8)

            real_images = real_writer.add_image("real_images", real_grid, global_step=step)
            fake_images = real_writer.add_image("fake_images", fake_grid, global_step=step)

        step += 1
        discriminator.train()
        generator.train()


