import torch.nn as nn
import torch
import time
import os

from tensorboardX import SummaryWriter
from src.utils import Progbar


class Trainer(object):
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion,
                 train_dataloader,
                 epochs,
                 metrics=None,
                 val_dataloader=None,
                 lr_scheduler=None,
                 ckpt_frequency=None,
                 checkpoint_dir='ckpt', 
                 is_ReduceLRonPlateau=False,
                 start_epoch=0,
                 max_iter=1e99):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.train_batches = len(self.train_dataloader)
        self.batch_size = self.train_dataloader.batch_size
        self.epochs = epochs
        self.metrics = metrics
        self.val_dataloader = val_dataloader

        if self.val_dataloader is not None:
            self.val_batches = len(self.val_dataloader)
            # only support equal batch sizes
            assert self.batch_size == self.val_dataloader.batch_size

        self.lr_scheduler = lr_scheduler
        self.ckpt_frequency = ckpt_frequency
        self.checkpoint_dir = checkpoint_dir
        self.is_ReduceLRonPlateau = is_ReduceLRonPlateau,
        self.start_epoch = start_epoch
        self.max_iter = max_iter

        self.writer = SummaryWriter()
        
        if self.ckpt_frequency is not None:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(checkpoint_dir)


    def init_training(self):
        self._metrics_names = ['train_loss']

        if self.metrics is not None:
            self._metrics_names.extend(['train_%s' % x.__name__ for x in self.metrics])

        if self.val_dataloader is not None:
            self._metrics_names.append('val_loss')
            
            if self.metrics is not None:
                self._metrics_names.extend(['val_%s' % x.__name__ for x in self.metrics])

        self.progbar = Progbar(
            target=len(self.train_dataloader.dataset), 
            stateful_metrics=self._metrics_names)


    def refresh_metrics(self):
        for metric_name in self._metrics_names:
            setattr(self, metric_name, 0.0)

    
    def train(self):
        self.init_training()
        for self.epoch in range(self.start_epoch, self.max_epochs):
            if not self.is_ReduceLRonPlateau:
                self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_lr()
            print('Epoch: %d, lr - %.4f' % (self.epoch+1, current_lr[-1]))
            should_terminate = self.training_phase(self.train_dataloader)

            if self.val_dataloader is not None:
                self.validating_phase(self.val_dataloader)

            if (self.iter % self.ckpt_frequency) == 0:
                checkpoint_name = os.path.join(self.checkpoint_dir, 'iter_' + str(self.iter) + '.pth')
                torch.save(self.model, checkpoint_name)

            if should_terminate:
                print('Maximum number of iterations %d exceeded. Finishing training...' % self.max_iter)
                break

            if self.is_ReduceLRonPlateau:
                self.lr_scheduler.step(self.val_loss)

        self.writer.close()


    def training_phase(self, dataloader):
        self.model.train()
        self.refresh_metrics()
        train_iter = 0
        phase = 'train'

        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            self.iter += 1
            train_iter += 1

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # calculate loss and do backward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # calculate metrics
            self.train_loss += loss.item()
            self.calculate_metrics(outputs, targets, phase)  
            self.update_summary_and_progbar(train_iter, phase)

        if self.iter >= self.max_iter:
            return True 
        return False
        

    def validating_phase(self, dataloader):
        self.model.eval()
        phase = 'val'
        val_iter = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                val_iter += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.val_loss += loss.item() 
                self.calculate_metrics(outputs, targets, phase)  
                self.update_summary_and_progbar(val_iter, phase) 


    def calculate_metrics(self, outputs, targets, phase):
        assert phase in ['train', 'val']

        for metric in self.metrics:
            # calculate metric
            metric_val = metric(outputs, targets)
            metric_name = '_'.join([phase, metric.__name__])
            current_metric_val = getattr(self, metric_name)
            # accumulate metrics
            setattr(self, metric_name, current_metric_val + metric_val)


    def update_summary_and_progbar(self, current_step, phase):
        assert phase in ['train', 'val']

        progbar_update_vals = []
        for metric_name in self._metrics_names:
            if phase in metric_name:
                # normalized metric value
                metric_val = getattr(self, metric_name) / current_step
                progbar_update_vals.append((metric_name, metric_val))
                # update summary writer
                scalar_name = '%s/%s' % (phase.title(), metric_name[len(phase)+1:].title())
                self.writer.add_scalar(scalar_name, metric_val, self.iter)
        # update progbar
        self.progbar.update(current_step*self.batch_size, values=progbar_update_vals)