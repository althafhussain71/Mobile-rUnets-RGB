import os
from argparse import ArgumentParser
from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 3. Calling Unet(hparams) to build a model in Unet.py
# Referred from: https://github.com/hiepph/unet-lightning
def main(hparams):
    model = Unet(hparams)
    os.makedirs(hparams.log_dir, exist_ok=True)

    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(log_dir, 'checkpoints'),
        verbose=True,
    )
    # save_best_only=False,
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=True,
    )
    trainer = Trainer(
        gpus=1,
        max_epochs=1,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=stop_callback,
    )
    trainer.fit(model)
    trainer.test(model)

# 1. Here we are sending the dataset and log directory details to Unet method add_model_specific_args
# Referred from: https://github.com/hiepph/unet-lightning
if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='lightning_logs')


    parser = Unet.add_model_specific_args(parent_parser)
    # 1. now hparams has all details dataset, log_dir, channels, classes etc
    hparams = parser.parse_args()

    main(hparams)
