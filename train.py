import os
import sys
from typing import List

import einops
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusions.models import AttnDownBlock, AttnUpBlock, DownBlock, UNet, UpBlock

# from diffusions.models import AttnDownBlock, AttnUpBlock, DownBlock, UNet, UpBlock
# from diffusions.models.imagen import UnconditionalEfficientUnet, UnconditionalImagen
from diffusions.pipelines import DDPMPipeline
from diffusions.schedulers import DDPM

# from diffusions.utils import EMAModel  # , resize_image_to
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

ckpt = None
if len(sys.argv) > 1:
    ckpt = sys.argv[1]


def get_transforms(phase: str = "train"):
    if phase == "train":
        return Compose(
            [
                Resize(64, interpolation=InterpolationMode.BILINEAR),
                CenterCrop(64),
                RandomHorizontalFlip(),
                ToTensor(),
                # Normalize([0.5], [0.5]),
            ]
        )
    return Compose(
        [
            Resize(64, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(64),
            ToTensor(),
            # Normalize([0.5], [0.5]),
        ]
    )


class SlothDataset(Dataset):
    def __init__(self, files: List[str], transforms=None) -> None:
        super(SlothDataset, self).__init__()
        self.files = files

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


class LightningModel(pl.LightningModule):
    def __init__(
        self, unet: UNet, iters_per_epoch: int, noise_scheduler: DDPM, lr: float = 1e-4
    ) -> None:
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.iters_per_epoch = iters_per_epoch
        self.criterion = F.mse_loss
        self.lr = lr

        self.pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.unet.parameters(), lr=self.lr, eps=1e-8, weight_decay=0
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.iters_per_epoch, T_mult=2
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def forward(self, sample, timesteps):
        sample = self.unet(sample, timesteps)
        return sample

    def training_step(self, batch, batch_idx):
        noise = torch.randn(batch.shape).to(batch.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_steps,
            (batch.size(0),),
            device=batch.device,
        ).long()

        noisy_images = self.noise_scheduler.add_noise(batch, noise, timesteps)
        noise_pred = self(noisy_images, timesteps)["sample"]
        loss = self.criterion(noise_pred, noise)

        self.log("train/loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        noise = torch.randn(batch.shape).to(batch.device)
        loss = 0.0
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_steps,
            (batch.size(0),),
            device=batch.device,
        ).long()

        noisy_images = self.noise_scheduler.add_noise(batch, noise, timesteps)
        noise_pred = self(noisy_images, timesteps)["sample"]
        loss = loss + self.criterion(noise_pred, noise)

        self.log("valid/loss", loss)

        return {"loss": loss.detach()}

    def test_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs) -> None:
        loss = 0.0
        for i, out in enumerate(outputs, 1):
            loss = loss + out["loss"]
        loss = loss / i

        self.log("train/epoch_loss", loss)

    def validation_epoch_end(self, outputs) -> None:
        loss = 0.0
        for i, out in enumerate(outputs, 1):
            loss = loss + out["loss"]
        loss = loss / i

        generator = torch.manual_seed(0)
        os.makedirs(f"results/epoch_{self.current_epoch}", exist_ok=True)
        for i in range(4):
            images = self.pipeline(batch_size=1, generator=generator)["sample"]
            images_processed = (
                einops.rearrange((images * 255).round(), "b c h w -> b h w c")
                .numpy()
                .astype("uint8")
            )
            image = images_processed[0]
            img = Image.fromarray(image, mode="RGB")
            img.save(
                os.path.join("results", f"epoch_{self.current_epoch}", f"image_{i}.jpg")
            )

        self.log("valid/epoch_loss", loss)


class SlothRetriever(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        for top, _, filenames in os.walk("images"):
            pass
        self.files = np.array(
            [
                os.path.join(top, filename)
                for filename in filenames
                if filename[-4:] == ".jpg"
            ]
        )

        val_idx = np.random.choice(len(self.files), size=len(self.files) // 10)
        trn_idx = np.ones(len(self.files), dtype=bool)
        trn_idx[val_idx] = False
        self.validset = SlothDataset(
            files=self.files[val_idx], transforms=get_transforms("valid")
        )
        self.trainset = SlothDataset(
            files=self.files[trn_idx], transforms=get_transforms("train")
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() if os.cpu_count() is not None else 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() if os.cpu_count() is not None else 0,
        )


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bsz = 32
    acc = 1
    iters = 2000
    lr = 1e-4

    dm = SlothRetriever(batch_size=bsz)

    dim = 32
    model = UNet(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        down_block_types=(DownBlock, AttnDownBlock, AttnDownBlock),
        up_block_types=(AttnUpBlock, AttnUpBlock, UpBlock),
        layers_per_block=3,
        block_out_channels=(dim, dim * 2, dim * 4),
        mid_block_scale_factor=2**-0.5,
        groups=32,
        use_checkpoint=True,
    )
    # model = UNet(
    #     sample_size=64,
    #     in_channels=3,
    #     out_channels=3,
    #     layers_per_block=2,
    #     block_out_channels=(128, 128, 256, 256, 512, 512),
    #     down_block_types=(
    #         DownBlock,
    #         DownBlock,
    #         DownBlock,
    #         DownBlock,
    #         AttnDownBlock,
    #         DownBlock,
    #     ),
    #     up_block_types=(
    #         UpBlock,
    #         AttnUpBlock,
    #         UpBlock,
    #         UpBlock,
    #         UpBlock,
    #         UpBlock,
    #     ),
    # )

    noise_scheduler = DDPM(num_train_timesteps=1000)

    model = LightningModel(
        unet=model,
        iters_per_epoch=iters,
        noise_scheduler=noise_scheduler,
        lr=lr,
    )

    # ema_model = EMAModel(model=model)

    # model = model.to(device)
    # sr_model = sr_model.cpu()

    logger = pl_loggers.WandbLogger(
        name="sloth-diffusion",
        project="sloth-diffusion",
    )
    checkpoint_dir = "weights"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch}-loss={valid/epoch_loss:.3f}",
        save_top_k=-1,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=checkpoint,
        max_epochs=-1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=acc,
        detect_anomaly=True,
        deterministic=False,
        benchmark=True,
    )

    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt)
