import os

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusions.models import AttnDownBlock, AttnUpBlock, DownBlock, UNet, UpBlock
from diffusions.pipelines import DDPMPipeline
from diffusions.schedulers import DDPM
from diffusions.utils import EMAModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm import tqdm, trange

model = UNet(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 64, 128, 128, 256, 256),
    down_block_types=(
        DownBlock,
        AttnDownBlock,
        AttnDownBlock,
        AttnDownBlock,
        AttnDownBlock,
        DownBlock,
    ),
    up_block_types=(
        UpBlock,
        AttnUpBlock,
        AttnUpBlock,
        AttnUpBlock,
        AttnUpBlock,
        UpBlock,
    ),
)

noise_scheduler = DDPM(num_train_timesteps=1000)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer)

ema_model = EMAModel(model=model)

transforms = Compose(
    [
        Resize(64, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(64),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.5], [0.5]),
    ]
)


class SlothDataset(Dataset):
    def __init__(self, transforms=None) -> None:
        super(SlothDataset, self).__init__()

        for top, _, filenames in os.walk("images"):
            pass
        self.files = [os.path.join(top, filename) for filename in filenames]

        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        item = Image.open(filename)
        item = item.convert("RGB")

        if self.transforms is not None:
            item = self.transforms(item)

        return item


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    dataset = SlothDataset(transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    for epoch in trange(30):
        model.train()
        print(f"EPOCH {epoch} STARTS")
        loss_epoch = 0.0
        for step, batch in tqdm(enumerate(dataloader), total=len(dataset) // 4):
            batch = batch.to(device)
            noise = torch.randn(batch.shape).to(batch.device)
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_steps,
                (batch.size(0),),
                device=batch.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

            noise_pred = model(noisy_images, timesteps)["sample"]
            loss = F.mse_loss(noise_pred, noise)
            loss_epoch += loss.item()
            loss.backward()

            optimizer.step()
            # lr_scheduler.step()

            # EMA
            # ema_model.step(model.cpu())
            optimizer.zero_grad()

        # save
        os.makedirs("weights", exist_ok=True)
        torch.save(
            model.to("cpu").state_dict(), f"weights/epoch_{epoch}-loss_{loss_epoch}.pt"
        )
        print(f"EPOCH {epoch} ENDS >> loss = {loss_epoch}")

        generator = torch.manual_seed(0)
        images = pipeline(batch_size=8, generator=generator)["sample"]
        images_processed = (
            einops.rearrange((images * 255).round(), "s g b r -> s r g b ")
            .numpy()
            .as_type("uint8")
        )

        os.makedirs(f"results/epoch_{epoch}")
        for i, image in enumerate(images_processed):
            img = Image.fromarray(image, mode="RGB")
            img.save(os.path.join("results", f"epoch_{epoch}", f"image_{i}.jpg"))
