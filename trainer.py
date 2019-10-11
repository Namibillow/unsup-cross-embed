import pathlib 
import time 

from numpy import inf
import numpy as np
import torch
from torch import nn
from torch import optim

class Trainer:
    def __init__(self, model, config, total_batches, src_data_loader, tgt_data_loader, src_vocab, tgt_vocab):
        self.config = config
        self.logger = config.get_logger('train')

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        
        self.weight_init_range = config["weight_init_range"]

        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab

        self.model = model
        self.init_model() #initialize model 
        
        self.model = self.model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids) # train with multiple GPUs

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.src_vocab.special_tokens["PAD"], reduction='none')
        self.optimizer = self.set_optimizer(config["optimizer"]["type"], config["optimizer"]["args"]["lr"])

        self.src_data_loader = src_data_loader
        self.tgt_data_loadder = tgt_data_loader

        
        self.total_batches = total_batches
        self.batch_size = config["batch_size"]
        
        self.epochs = config['epoch']
        self.start_epoch = 1

        self.save_period = config['save_period']
        self.checkpoint_dir = config.save_dir
        self.remove_models = config["remove_models"]

        self.cumloss_old = np.inf 
        self.cumloss_new = np.inf 

        self.log_step = int(np.sqrt(self.batch_size))

    def init_model(self):
        """
        initialize all of the model parameters using an uniform distribution
        """
        for param in self.model.parameters():
            param.data.uniform_(self.weight_init_range[0], self.weight_init_range[1])
       
        # 0 out PAD and UNK
        self.model.embedding_src.weight.data[self.src_vocab.special_tokens["PAD"]] = 0
        self.model.embedding_tgt.weight.data[self.src_vocab.special_tokens["PAD"]] = 0

        self.model.embedding_src.weight.data[self.src_vocab.special_tokens["UNK"]] = 0
        self.model.embedding_tgt.weight.data[self.tgt_vocab.special_tokens["UNK"]] = 0

    def set_optimizer(self, opt_type="SGD", lr_rate=0.5):
        """
        set optimizer specified 

        inputs:
            opt_type: (str) optimizer type
            lr_rate: (float) learning rate

        """
        if opt_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr_rate)
        elif opt_type == "ASGD":
            self.optimizer = optim.ASGD(self.model.parameters(), lr=lr_rate)

    def calc_loss(self, pred, correct):
        """
            Computer the corss entropy loss given outputs from the model and labels for all tokens
            Exclude loss terms for PADding tokens.
        
        inputs:  
            pred: (Variable) dimension - log softmax output of the model. soft maxed scores of shape (batch_size, sent_len, vocabs)
            correct: (Variable) dimension - (batch_size, sent_len)

        returns:
            loss: (float) a total loss for the batch 
        """
        batch_size, sent_len, vocabs = pred.size()

        # # only consider non-zero targets since we are not considering PAD for loss (warning: PAD has id of 0)
        # mask = correct.ge(1).type(torch.LongTensor).to(self.device) # (batch_size, sent_len)
        # mask.view(-1) # (batch_size * sent_len)
        # loss = self.cross_entropy(pred, correct) * mask # (batch_size * sent_len)

        pred = pred.view(batch_size * sent_len, vocabs) # (batch_size * sent_len, vocabs)
        correct = target.view(-1) # (batch_size * sent_len)

        loss = self.cross_entropy(pred, correct) 
        loss = torch.sum(loss) / self.batch_size 
        return loss

    def _train_batch(self, inputs, targets):
        """
        train logic for a batch
        
        inputs:
            inputs: a batch of inputs
            targets: a batch of targets for the inputs

        return: 
        """
        self.optimizer.zero_grad() 

        softmax_score = self.model(inputs)
        loss = self.calc_loss(softmax_score, targets)
        # Calculate loss 
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        self.optimizer.step()

        return loss.data.tolist()

    def _train_epoch(self, epoch):
        """
        training logic for an epoch
        
        input:
            epoch: (int) current training epoch.
            
        return:    
            cumloss: 
        """
        self.model.train()

        cumloss = 0 

        for batch_idx, (src, tgt) in enumerate(zip(src_batches, tgt_batches)):
            loss = 0 

            for curr, (fwd_inputs, fwd_outputs. bwd_inputs, bwd_outputs) in enumerate([src, tgt]):
                
                self.model.swich_lang(curr)

                fwd_inputs, fwd_outputs = fwd_inputs.to(self.device), fwd_outputs.to(self.device)
                bwd_inputs, bwd_outputs = bwd_inputs.to(self.device), bwd_outputs.to(self.device)

                self.model.switch_lstm("fwd")
                loss += self._train_batch(fwd_inputs, fwd_outputs)

                self.model.switch_lstm("bwd")
                loss += self._train_batch(bwd_inputs, bwd_outputs)
            
            cumloss+= loss

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}/{} Loss: {:.6f}'.format(
                    batch_idx,
                    self.batch_size,
                    loss.item()/batch_idx))
                
        return cumloss 


    def train(self):
        """
        full training logic

        return:
            model
        """
        not_improved_count = 0
        saved_model = False 

        for epoch in range(self.start_epoch, self.epochs + 1):

            self.cumloss_old = self.cumloss_new 

            start = time.time()

            cumloss = self._train_epoch(epoch)

            elapsed_time = time.time() - start
            self.cumloss_new = cumloss/self.total_batches

            self.logger.debug('Time taken for 1 epoch {} sec\n'.format(elapsed_time))
            self.logger.info("Train Epoch: {}/{} Total loss: {:.6f} :".format(epoch, self.epochs+1 ,self.cumloss_new))

            improvement_rate = self.cumloss_new / self.cumloss_old
            self.logger.debug("loss improvement rate:", improvement_rate)


            if (improvement_rate > stop_threshold):
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(epochs))
                break
        
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, remove_models)

        return self.model 

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device

        input:
            n_gpu_use: (int) number of GPU to be used. 

        returns: 
            device:  an object representing the device on which a torch.Tensor is or will be allocated
            list_ids: (list(int)) GPU ids to be used 
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
        
        self.logger.debug("-- Total of %d GPU is used for the training --", n_gpu_use)

        list_ids = list(range(n_gpu_use))

        return device, list_ids

    def _save_checkpoint(self, epoch, remove_models=True):
        """
        saving checkpoints
        
        inputs:
            epoch: (int) current epoch number
            remove_models: (Boolean) if True, remove the previous checkpoint
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.model'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if remove_models and epoch != 1:
            filename = Path(self.checkpoint_dir / 'checkpoint-epoch{}.model'.format(epoch-1))
            filename.unlink()
            self.logger.debug("-- removed previous model --")

   