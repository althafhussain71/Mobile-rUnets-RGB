from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from dataset import DirDataset


# Referred from: https://github.com/hiepph/unet-lightning
class Unet(pl.LightningModule):
    # 4. Unet call from train for building a model
    # Referred from: https://github.com/hiepph/unet-lightning
    def __init__(self, hparams):
        super(Unet, self).__init__()
        self.hparams = hparams

        self.n_channels = hparams.n_channels
        self.n_classes = hparams.n_classes
        self.bilinear = True


    # 5. Call from below self.inc to set conv_layers
        # Referred from: https://github.com/hiepph/unet-lightning
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )


    # 6. Down sampling
        # Referred from: https://github.com/hiepph/unet-lightning
        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )


        # Referred from: https://github.com/hiepph/unet-lightning
        class up(nn.Module):
            # Referred from: https://github.com/hiepph/unet-lightning
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranpose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)


            # Referred from: https://github.com/hiepph/unet-lightning
            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1)  ## why 1?
                return self.conv(x)


        # Referred from: https://github.com/hiepph/unet-lightning
        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)


    # Referred from: https://github.com/hiepph/unet-lightning
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


    # Referred from: https://github.com/hiepph/unet-lightning
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.MSELoss()
        op_loss = loss(y_hat, y)
        tensorboard_logs = {'train_loss': op_loss}
        return {'loss': op_loss, 'log': tensorboard_logs}


    # Referred from: https://github.com/hiepph/unet-lightning
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.MSELoss()
        op_loss = loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
        #     F.binary_cross_entropy_with_logits(y_hat, y)
        return {'val_loss': op_loss}


    # Referred from: https://github.com/hiepph/unet-lightning
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    # Referred from: https://github.com/hiepph/unet-lightning
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.MSELoss()
        op_loss = loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
        #     F.binary_cross_entropy_with_logits(y_hat, y)
        print('test_loss', loss)
        return {'test_loss': op_loss}


    # Referred from: https://github.com/hiepph/unet-lightning
    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)


    # Referred from: https://github.com/hiepph/unet-lightning
    def __dataloader(self):
        dataset = self.hparams.dataset

        # download the dataset from the below link but we need to generate mask images by using filters of Pil library
        # https://data.mendeley.com/datasets/t395bwcvbw/1
        dataset = DirDataset(f'./dataset/{dataset}/train', f'./dataset/{dataset}/train_blur_masks')

        # 10% data to validation, 20% to test and remaining sample for training
        val_sample_size = int(len(dataset) * 0.1)
        train_sample_size = len(dataset) - val_sample_size
        test_sample_size = int(train_sample_size * 0.2)

        train_ds, val_ds = random_split(dataset, [train_sample_size, val_sample_size])
        train_sample_size = train_sample_size - test_sample_size

        train_ds, test_ds = random_split(train_ds, [train_sample_size, test_sample_size])

        train_loader = DataLoader(train_ds, batch_size=1, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, pin_memory=True, shuffle=False)

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
        }


    @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']


    @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']


    @pl.data_loader
    def test_dataloader(self):
        return self.__dataloader()['test']


    # Referred from: https://github.com/hiepph/unet-lightning
    @staticmethod
    def add_model_specific_args(parent_parser):
        # 2. parent_parser has dataset and lightning_logs details called from train main method
        parser = ArgumentParser(parents=[parent_parser])

        # 2. setting channels and classes then returning it back to train.py program
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=3)
        return parser
