from numpy import inf
import numpy as np

import torch
from torch import nn

import time 

class Trainer:
    def __init__(self, model, config, total_batches, src_data_loader, tgt_data_loadder):
        self.config = config
        self.logger = config.get_logger('train')

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.optimizer = self.set_optimizer(config["optimizer"]["type"])

        self.src_data_loader = src_data_loader
        self.tgt_data_loadder = tgt_data_loadder
        
        self.total_batches = total_batches

        self.lr_rate = config["optimizer"]["args"]["lr"]
        
        self.epochs = config['epochs']
        self.save_period = config['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        self.cumloss_old = np.inf 
        self.cumloss_new = np.inf 

        ##################################################
        self.log_step = int(np.sqrt(config["batch_size"]))

        ###################################################

    def set_optimizer(self, optimizer):
        if opt_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr_rate)
        elif opt_type == "ASGD":
            self.optimizer = optim.ASGD(self.model.parameters(), lr=lr_rate)


    def _train_epoch(self, epoch):
            """
            Training logic for an epoch
            :param epoch: Integer, current training epoch.
            :return: A log that contains average loss and metric in this epoch.
            """
            self.model.train()

            cumloss = 0 

            for batch_idx, (src, tgt) in enumerate(zip(src_batches, tgt_batches)):

                self.optimizer.zero_grad()

                src_fwd_inputs, src_fwd_outputs, src_bwd_inputs, src_bwd_outputs = src
                tgt_fwd_inputs, tgt_fwd_outputs, tgt_bwd_inputs, tgt_bwd_outputs = tgt
            
                # Send data to the GPU
                src_fwd_inputs, src_fwd_outputs = src_fwd_inputs.to(self.device), src_fwd_outputs.to(self.device)
                src_bwd_inputs, src_bwd_outputs = src_bwd_inputs.to(self.device), src_bwd_outputs.to(self.device)
                tgt_fwd_inputs, tgt_fwd_outputs = tgt_fwd_inputs.to(self.device), tgt_fwd_outputs.to(self.device)
                tgt_bwd_inputs, tgt_bwd_outputs = tgt_bwd_inputs.to(self.device), tgt_bwd_outputs.to(self.device)
            
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                cumloss+= loss

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()/batch_idx))
                    
            return cumloss 


    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        saved_model = False 

        for epoch in range(self.start_epoch, self.epochs + 1):

            self.cumloss_old = self.cumloss_new 

            start = time.time()

            cumloss = self._train_epoch(epoch)

            elapsed_time = time.time() - start
            self.cumloss_new = cumloss/self.total_batches

            self.logger.debug("Train elapsed_time:{0}".format(elapsed_time) + "[sec]")
            self.logger.info("Train Epoch: {}/{} Total loss: {:.6f} :".format(epoch, self.epochs+1 ,cumloss))

            improvement_rate = self.cumloss_new / self.cumloss_old
            self.logger.debug("loss improvement rate:", improvement_rate)

            new_model_name = "epoch" + str(epoch) +'.model'
            new_file = config.get_save_path() / new_model_name

            if (improvement_rate > stop_threshold):
                torch.save(self.model.state_dict(), new_file)
                saved_model = True 
                break
        
        if not saved_model:
            torch.save(self.model.state_dict(), new_file)

            # # save logged informations into log dict
            # log = {'epoch': epoch}
            # log.update(result)

            # # print logged informations to the screen
            # for key, value in log.items():
            #     self.logger.info('    {:15s}: {}'.format(str(key), value))

            # # evaluate model performance according to configured metric, save best checkpoint as model_best
            # best = False
            # if self.mnt_mode != 'off':
            #     try:
            #         # check whether model performance improved or not, according to specified metric(mnt_metric)
            #         improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
            #                    (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            #     except KeyError:
            #         self.logger.warning("Warning: Metric '{}' is not found. "
            #                             "Model performance monitoring is disabled.".format(self.mnt_metric))
            #         self.mnt_mode = 'off'
            #         improved = False

            #     if improved:
            #         self.mnt_best = log[self.mnt_metric]
            #         not_improved_count = 0
            #         best = True
            #     else:
            #         not_improved_count += 1

            #     if not_improved_count > self.early_stop:
            #         self.logger.info("Validation performance didn\'t improve for {} epochs. "
            #                          "Training stops.".format(self.early_stop))
            #         break

            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device

        input:
            n_gpu_use: (int) number of GPU to be used. 

        returns: 
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("-- Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("-- Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        
        logger.debug("-- Total of %d GPU is used for the training --", n_gpu_use)

        list_ids = list(range(n_gpu_use))

        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def loss_fn(pred, correct):
        """
            Computer the corss entropy loss given outputs from the model and labels for all tokens
            Exclude loss terms for PADding tokens.
        
        input:  
            pred: (Variable) dimension - log softmax output of the model
            correct: (Variable) dimension

        returns:

        """
        pass