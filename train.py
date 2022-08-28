import math
import os

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusions.models import AttnDownBlock, AttnUpBlock, DownBlock, UNet, UpBlock

# from diffusions.models import AttnDownBlock, AttnUpBlock, DownBlock, UNet, UpBlock
from diffusions.models.imagen import UnconditionalEfficientUnet, UnconditionalImagen
from diffusions.pipelines import DDPMPipeline
from diffusions.schedulers import DDPM
from diffusions.utils import EMAModel  # , resize_image_to
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm, trange

transforms = Compose(
    [
        Resize(64, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(64),
        RandomHorizontalFlip(),
        ToTensor(),
        # Normalize([0.5], [0.5]),
    ]
)


class SlothDataset(Dataset):
    def __init__(self, transforms=None) -> None:
        super(SlothDataset, self).__init__()

        for top, _, filenames in os.walk("images"):
            pass
        self.files = [
            os.path.join(top, filename)
            for filename in filenames
            if filename[-4:] == ".jpg"
        ]

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
    bsz = 128
    acc = 1

    dataset = SlothDataset(transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True)
    iters = math.floor(len(dataset) // (bsz * acc))

    dim = 32
    # model = UnconditionalEfficientUnet(
    #     sample_size=64,
    #     in_channels=3,
    #     out_channels=3,
    #     block_out_channels=(dim, dim * 2, dim * 3, dim * 4),
    #     layers_per_block=3,
    #     num_heads=(None, 4, 8, 8),
    # )
    model = UNet(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        down_block_types=(DownBlock, AttnDownBlock, AttnDownBlock, AttnDownBlock),
        up_block_types=(AttnUpBlock, AttnUpBlock, AttnUpBlock, UpBlock),
        layers_per_block=3,
        block_out_channels=(dim, dim * 2, dim * 3, dim * 4),
        mid_block_scale_factor=2**-0.5,
        groups=32,
        use_checkpoint=True,
    )
    print(model)

    noise_scheduler = DDPM(num_train_timesteps=1000)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=iters,
        T_mult=2,
        eta_min=1e-7,
    )

    ema_model = EMAModel(model=model)

    model = model.to(device)
    # sr_model = sr_model.cpu()

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    for epoch in trange(1000):
        model.train()
        # sr_model.train()
        print(f"EPOCH {epoch} STARTS")
        loss_epoch = 0.0
        for step, batch in tqdm(enumerate(dataloader), total=iters * acc):
            # org = batch["org"]
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
            # noise_pred = resized_image_to(noise_pred, 256)
            # noise_pred = sr_model(noise_pred, timesteps)["sample"]
            loss = F.mse_loss(noise_pred, noise)
            loss_epoch += loss.item()
            loss.backward()

            if (step + 1) % acc == 0:
                optimizer.step()
                lr_scheduler.step()

                # EMA
                # ema_model.step(model.cpu())
                optimizer.zero_grad()

        loss_epoch = loss_epoch / (step + 1)
        print(f"EPOCH {epoch} ENDS >> loss = {loss_epoch}")

        if (epoch + 1) % 20 == 0:
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
