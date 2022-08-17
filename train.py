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
from tqdm.notebook import tqdm, trange

model = UNet(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
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
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, T_max=1000, eta_min=1e-6
)

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

    for epoch in trange(1000):
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

            if step + 1 % 4 == 0:
                optimizer.step()
                lr_scheduler.step()

                # EMA
                # ema_model.step(model.cpu())
                optimizer.zero_grad()

        print(f"EPOCH {epoch} ENDS >> loss = {loss_epoch}")

        if epoch + 1 % 50 == 0:
            # save
            os.makedirs("weights", exist_ok=True)
            torch.save(
                model.to("cpu").state_dict(),
                f"weights/epoch_{epoch}-loss_{loss_epoch}.pt",
            )

            generator = torch.manual_seed(0)
            os.makedirs(f"results/epoch_{epoch}", exist_ok=True)
            for i in range(4):
                images = pipeline(batch_size=1, generator=generator)["sample"]
                images_processed = (
                    einops.rearrange((images * 255).round(), "b c h w -> b h w c")
                    .numpy()
                    .astype("uint8")
                )
                image = images_processed[0]
                img = Image.fromarray(image, mode="RGB")
                img.save(os.path.join("results", f"epoch_{epoch}", f"image_{i}.jpg"))
