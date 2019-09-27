import argparse
import yaml
import torch

from torch.utils.data import DataLoader

from src.utils import AttributeDict
from src.dataset import collate_fn
from src.trainer import Trainer
from src.losses import LootLoss
from src.dataset import LootDataset
from src.metrics import precision
from src.metrics import recall
from src.metrics import nonzero_preds
from src.metrics import shifts_squared_loss

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/unet_loot.yaml')
ARGS = parser.parse_args()

def main():
    configfile = open(ARGS.config)
    config = AttributeDict(yaml.load(configfile, Loader=yaml.FullLoader))
    
    if config.model.type == 'unet':
        from src.unet import UnetLoot as Loot
    else:
        from src.models import Loot

    print('Load model')
    loot = Loot(config.detector.n_stations, config.model.use_radius)
    print(loot)

    print('Prepare dataset for training')
    # similar for training and validation
    dataset_args = (config.detector.x_max,
                    config.detector.x_min,
                    config.detector.y_max,
                    config.detector.y_min,
                    config.detector.x_res,
                    config.detector.y_res,
                    config.detector.n_stations,
                    config.data.shifts_rescale)

    dataloader_kwargs = {
        'batch_size': config.data.batch_size,
        'num_workers': config.data.num_workers,
        'shuffle': config.data.shuffle,
        'collate_fn': collate_fn}

    train_dataset = LootDataset(config.data.train_dataset, *dataset_args)
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)

    val_dataset = LootDataset(config.data.val_dataset, *dataset_args)
    val_dataloader = DataLoader(val_dataset, **dataloader_kwargs)

    print('Train events: %d' % len(train_dataset))
    print('Validation events: %d' % len(val_dataset))

    print('Build trainer')
    optimizer = torch.optim.SGD(loot.parameters(), 
                                lr=config.optimizer.lr, 
                                momentum=config.optimizer.momentum)
    criterion = LootLoss(config.criterion.lambda1, config.criterion.lambda2)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                     base_lr=config.optimizer.lr, 
                                                     max_lr=0.1)
    trainer = Trainer(model=loot,
                      optimizer=optimizer,
                      criterion=criterion,
                      train_dataloader=train_dataloader,
                      epochs=config.training.epochs,
                      metrics=[recall, precision, nonzero_preds, shifts_squared_loss],
                      val_dataloader=val_dataloader,
                      lr_scheduler=lr_scheduler,
                      ckpt_frequency=config.training.checkpoint_freq,
                      random_seed=config.training.random_seed,
                      is_ReduceLRonPlateau=config.training.reduceLRonPlateau)

    print('Training')
    trainer.train()


if __name__ == "__main__":
    main()