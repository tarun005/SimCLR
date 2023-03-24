import logging
from nt_xent import NT_Xent
import os
from pickletools import optimize
import sys
import math

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from ddp.misc import SmoothedValue, all_reduce_mean, save_model, MetricLogger

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.model.train(True)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter()
        # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # self.args.output_dir = self.writer.log_dir
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.scaler = kwargs['scaler']

        self.nt_xent = NT_Xent(self.args.batch_size, self.args.temperature, self.args.world_size)

    # def info_nce_loss(self, features):

    #     labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.args.device)

    #     features = F.normalize(features, dim=1)

    #     similarity_matrix = torch.matmul(features, features.T)
    #     # assert similarity_matrix.shape == (
    #     #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #     # assert similarity_matrix.shape == labels.shape

    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    #     logits = logits / self.args.temperature
    #     return logits, labels

    def train(self, train_loader):
        
        scaler = self.scaler
        accum_iter = self.args.accum_iter

        # save config file
        save_config_file(self.args.output_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.start_epoch, self.args.epochs):

            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}]'.format(epoch_counter)
            print_freq = 200

            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch_counter)
            for data_iter_step, (images, _) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    features_i, features_j = torch.split(features, [len(features)//2]*2, dim=0)
                    loss = self.nt_xent(features_i, features_j)
                    # logits, labels = self.info_nce_loss(features)
                    # loss = self.criterion(logits, labels)
                
                loss /= accum_iter

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopped training".format(loss_value))
                    sys.exit(1)

                if (data_iter_step + 1) % accum_iter == 0:
                    self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                if (data_iter_step + 1) % accum_iter == 0:
                    scaler.step(self.optimizer)
                    scaler.update()

                torch.cuda.synchronize()

                metric_logger.update(loss=loss_value)
                lr = self.optimizer.param_groups[0]['lr']
                metric_logger.update(lr=lr)

                loss_value_reduce = all_reduce_mean(loss_value)

                # if n_iter % self.args.log_every_n_steps == 0:
                #     top1, top5 = accuracy(logits, labels, topk=(1, 5))
                #     self.writer.add_scalar('loss', loss, global_step=n_iter)
                #     self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                #     self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                #     self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= self.args.warmup_epochs:
                self.scheduler.step()

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)

            # logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")

            if (epoch_counter % 20 == 0 or epoch_counter + 1 == self.args.epochs):
                save_model(args=self.args, epoch=epoch_counter, model=self.model, model_without_ddp=self.model.module,
                                            optimizer=self.optimizer, loss_scaler=scaler)

        logging.info("Training has finished.")
        # save model checkpoints
        # checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        # save_checkpoint({
        #     'epoch': self.args.epochs,
        #     'arch': self.args.arch,
        #     'state_dict': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        # }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        # logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
