import torch.nn as nn
import torch
import random
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
                 random_seed=None,
                 start_epoch=0,
                 max_iter=1e99):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
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
        self.is_ReduceLRonPlateau = is_ReduceLRonPlateau
        
        if random_seed is None:
            random_seed = random.randint(0, 1000)
            
        self.random_seed = random_seed      
        self.start_epoch = start_epoch
        self.max_iter = max_iter

        self.writer = SummaryWriter()
        
        if self.ckpt_frequency is not None:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(checkpoint_dir)


    def init_training(self):
        torch.manual_seed(self.random_seed)
        self.iter = 0
        self._metrics_names = ['train_loss']

        if self.metrics is not None:
            self._metrics_names.extend(['train_%s' % x.__name__ for x in self.metrics])

        if self.val_dataloader is not None:
            self._metrics_names.append('val_loss')
            
            if self.metrics is not None:
                self._metrics_names.extend(['val_%s' % x.__name__ for x in self.metrics])


    def refresh_metrics(self):
        for metric_name in self._metrics_names:
            setattr(self, metric_name, 0.0)

    
    def train(self):
        self.init_training()
        
        for self.epoch in range(self.start_epoch, self.epochs):
            print('\nEpoch: {}'.format(self.epoch + 1))
            self.progbar = Progbar(
                target=len(self.train_dataloader.dataset), 
                stateful_metrics=self._metrics_names)

            if self.lr_scheduler is not None:
                if not self.is_ReduceLRonPlateau:
                    self.lr_scheduler.step()
                current_lr = self.lr_scheduler.get_lr()
                print('Current learning rate: %.6f' % current_lr[-1])

            self.refresh_metrics()
            should_terminate = self.training_phase()

            if self.val_dataloader is not None:
                self.validating_phase()

            if (self.iter % self.ckpt_frequency) == 0:
                self.save_checkpoint()

            if should_terminate:
                print('Maximum number of iterations %d exceeded. Finishing training...' % self.max_iter)
                break
            
            if self.is_ReduceLRonPlateau and self.lr_scheduler is not None:
                self.lr_scheduler.step(self.val_loss)

        self.writer.close()


    def training_phase(self):
        self.model.train()
        train_iter = 0
        phase = 'train'

        for inputs, targets in self.train_dataloader:
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
        

    def validating_phase(self):
        self.model.eval()
        phase = 'val'
        val_iter = 0
        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
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
        
        
    def generate_model_name(self):
        model_name = 'rs[%d]_iter[%d]_bz[%d]' % (self.random_seed, self.iter, self.batch_size)
        
        for metric_name in self._metrics_names:
            if metric_name.startswith('val_'):
                metric_val = getattr(self, metric_name) / self.val_batches
                str_to_add = '%s{%.4f}' % (metric_name[4:], metric_val)
                model_name = '_'.join([model_name, str_to_add])
        
        model_name = '.'.join([model_name, 'pth'])
        return model_name
        
        
    def save_checkpoint(self):
        checkpoint_name = os.path.join(self.checkpoint_dir, self.generate_model_name())
        torch.save(self.model, checkpoint_name) 