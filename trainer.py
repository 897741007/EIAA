from numpy import recarray
from GAT_prediction import GAT_predictor
from Smi2Graph import SMI_grapher
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import os
import time
import pandas as pd
import numpy as np
    
class trainer():
    # deep learning model training module
    def __init__(self, task_name='0', model=None, epoch=100, batch_size=16, lr=1e-4, log_record=10, valid=100,
    device='cuda', show_process=False, warm_up=0, lower_aromatic=False, specify_bond=True, shuffle_data=False):
        self.task_name = task_name
        self.model = model
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        assert type(log_record) == int
        self.log_record = log_record
        self.valid = valid
        self.device = device
        self.show_process = show_process
        self.warm_up = warm_up
        self.lower_aromatic = lower_aromatic
        self.specify_bond = specify_bond
        self.shuffle_data = shuffle_data
        self.memory_record = []
        self.data_shape_record = []
        if not os.path.exists('log/'):
            os.mkdir('log/')
        self.log_file = 'log/job.log.{0}'.format(self.task_name)

    def load_file(self, file):
        data = pd.read_csv(file)
        smis = data[data.columns[0]]
        labels = data[data.columns[1]]
        return [[s, labels[idx]] for idx, s in enumerate(smis)]

    def load_data(self, training_data_file, validation_data_file=None, test_data_file=None):
        # training_data : [[smi0, label0], [smi1, label1], [smi2, label2], ... ]
        print('data loading')
        self.training_data_file = training_data_file
        self.validation_data_file = validation_data_file
        self.test_data_file = test_data_file
        self.training_data = self.load_file(training_data_file)
        self.validation_data = self.load_file(validation_data_file)
        self.test_data = self.load_file(test_data_file)
        # calculate the step number of each epoch
        self.step_per_epoch = int(len(self.training_data)/self.batch_size)
        if len(self.training_data)%self.batch_size != 0:
            self.step_per_epoch += 1
        self.warmup_step = self.warm_up*self.step_per_epoch
        self.total_step = self.epoch*self.step_per_epoch
        self.n_v_step = int(len(self.validation_data)/self.batch_size)
        if len(self.validation_data)%self.batch_size != 0:
            self.n_v_step += 1
        self.n_tt_step = int(len(self.test_data)/self.batch_size)
        if len(self.test_data)%self.batch_size != 0:
            self.n_tt_step += 1
        # define data provider
        print('data preprocessing')
        self.data_provider = SMI_grapher(for_predictor=True, device=self.device, lower_aromatic=self.lower_aromatic,
        specify_bond=self.specify_bond)
        whole_smis = [i[0] for i in self.training_data] + [i[0] for i in self.validation_data] + [i[0] for i in self.test_data]
        self.data_provider.fit_new(whole_smis)
        label_class = type(self.training_data[0][1])
        if label_class == np.int64:
            mode = 'cls'
        else:
            mode = 'reg'
        self.training_batch_provider = self.data_provider.data_provider(self.training_data, self.batch_size,
        mode=mode, do_random=self.shuffle_data)
        self.valid_batch_provider = self.data_provider.data_provider(self.validation_data, self.batch_size, mode=mode)
        self.test_batch_provider = self.data_provider.data_provider(self.test_data, self.batch_size, mode=mode)
    
    def validation(self, tag=''):
        self.model.eval()
        if self.model.prediction_class > 1:
            mode = 'cls'
            sum_acc = 0
        else:
            mode = 'reg'
            ae = 0
        sum_loss = 0
        sum_amount = 0
        for v_step in range(self.n_v_step):
            v_batch_mol, v_batch_labels = next(self.valid_batch_provider)
            batch_amount = v_batch_labels.shape[0]
            v_batch_adjm, v_batch_atoms, v_batch_ions = v_batch_mol
            valid_y = self.model.forward(v_batch_atoms, v_batch_adjm)
            v_loss = self.loss_func(valid_y, v_batch_labels)
            sum_loss += v_loss.item()*batch_amount
            sum_amount += batch_amount
            if mode == 'cls':
                p_tag = valid_y.argmax(-1)
                batch_acc = torch.eq(v_batch_labels, p_tag).sum().float().item()
                sum_acc += batch_acc
            else:
                bae = valid_y - v_batch_labels
                bae = bae.abs().sum().float().item()
                ae += bae
        if mode == 'cls':
            self.record_process('[{0} validation] [loss {1}] [acc {2}]\n'.format(tag, round(sum_loss/sum_amount, 4),
            round(sum_acc/sum_amount, 4)))
        else:
            self.record_process('[{0} validation] [loss {1}] [MAE {2}]\n'.format(tag, round(sum_loss/sum_amount, 4),
            round(ae/sum_amount, 4)))
        self.model.train()

    def test(self, tag=''):
        self.model.eval()
        if self.model.prediction_class > 1:
            mode = 'cls'
            sum_acc = 0
        else:
            mode = 'reg'
            ae = 0
        sum_loss = 0
        sum_amount = 0
        for tt_step in range(self.n_tt_step):
            tt_batch_mol, tt_batch_labels = next(self.test_batch_provider)
            batch_amount = tt_batch_labels.shape[0]
            tt_batch_adjm, tt_batch_atoms, tt_batch_ions = tt_batch_mol
            test_y = self.model.forward(tt_batch_atoms, tt_batch_adjm)
            tt_loss = self.loss_func(test_y, tt_batch_labels)
            sum_loss += tt_loss.item()*batch_amount
            sum_amount += batch_amount
            if mode == 'cls':
                p_tag = test_y.argmax(-1)
                batch_acc = torch.eq(tt_batch_labels, p_tag).sum().float().item()
                sum_acc += batch_acc
            else:
                bae = test_y - tt_batch_labels
                bae = bae.abs().sum().float().item()
                ae += bae
        if mode == 'cls':
            self.record_process('[{0} test] [loss {1}] [acc {2}]\n'.format(tag, round(sum_loss/sum_amount, 4),
            round(sum_acc/sum_amount, 4)))
        else:
            self.record_process('[{0} test] [loss {1}] [MAE {2}]\n'.format(tag, round(sum_loss/sum_amount, 4),
            round(ae/sum_amount, 4)))
        self.model.train()
    
    def record_process(self, arg):
        with open(self.log_file, 'a') as f:
            f.write(arg)

    def record_training_info(self):
        with open(self.log_file, 'a') as f:
            f.write('='*50+'\n')
            f.write('starting time : {0}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write('training data file : {0}\n'.format(self.training_data_file))
            f.write('validation data file : {0}\n'.format(self.validation_data_file))
            f.write('test data file : {0}\n'.format(self.test_data_file))
            f.write('shuffle data : {0}\n'.format(self.shuffle_data))
            f.write('learning rate : {0}\n'.format(self.lr))
            f.write('batch size : {0}\n'.format(self.batch_size))
            f.write('warm up epoch : {0}\n'.format(self.warm_up))
            f.write('epoch : {0}\n'.format(self.epoch))
            f.write('device : {0}\n'.format(self.device))
            f.write('task dict size : {0}\n'.format(self.model.dict_size))
            f.write('bond influence : {0}\n'.format(self.model.bond_influence))
            f.write('lower_aromatic : {0}\n'.format(self.lower_aromatic))
            f.write('specify_bond : {0}\n'.format(self.specify_bond))
            f.write('model layers : {0}\n'.format(self.model.layer_num))
            f.write('model dim : {0}\n'.format(self.model.hidden_dim))
            f.write('model heads : {0}\n'.format(self.model.head_num))
            f.write('-'*50+'\n')

    def linear_warmup(self, current_step):
        warmup_lr = self.lr*(current_step+1.0)/self.warmup_step
        return warmup_lr

    def train_step(self, total_step, optimizer, lr_sc=None):
        # training part, the training process is organized by step
        if self.show_process:
            pbar = tqdm(total=1)
            pbar.set_description('Total training step : {0} .                   The current loss -->'.format(total_step))
        for step in range(total_step):
            optimizer.zero_grad()
            batch_mol, batch_labels = next(self.training_batch_provider)
            batch_adjm, batch_atoms, batch_ions = batch_mol
            #self.data_shape_record.append(batch_adjm.shape[-1])
            mol_property = self.model.forward(batch_atoms, batch_adjm)
            #self.memory_record.append(torch.cuda.memory_allocated())
            loss = self.loss_func(mol_property, batch_labels)
            loss.backward()
            loss_item = loss.item()
            optimizer.step()
            if lr_sc:
                lr_sc.step()
            if (step+1) % self.log_record == 0:
                self.record_process('[training] [epoch {0}] [step {1}] [loss {2}]\n'.format(int(step/self.step_per_epoch), step, round(loss_item, 4)))
            if self.valid:
                if (step+1) % self.valid_step == 0:
                    self.validation('step ')
                if (step+1) % self.step_per_epoch == 0:
                    self.validation('epoch ')
            torch.cuda.empty_cache()
            if self.show_process:
                pbar.set_postfix({'loss':round(loss_item, 3)})

    def training_model(self):
        # define loss function
        if self.model.prediction_class == 1:
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        # define the validation step of training process
        if self.valid == 'epoch':
            self.valid_step = self.step_per_epoch
        else:
            self.valid_step = self.valid
        self.record_training_info()
        
        # warm up
        if self.warm_up:
            print('warm up training')
            self.warmup_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(self.warmup_optimizer, lr_lambda=self.linear_warmup)
            self.train_step(self.warmup_step, self.warmup_optimizer, self.warmup_scheduler)
        # define routine training optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print('routine training')
        if self.show_process:
            pbar = tqdm(total=1)
            pbar.set_description('Total training step : {0} .                   The current loss -->'.format(total_step))
        for step in range(self.total_step):
            self.optimizer.zero_grad()
            batch_mol, batch_labels = next(self.training_batch_provider)
            batch_adjm, batch_atoms, batch_ions = batch_mol
            mol_property = self.model.forward(batch_atoms, batch_adjm)
            loss = self.loss_func(mol_property, batch_labels)
            loss.backward()
            loss_item = loss.item()
            self.optimizer.step()
            if (step+1) % self.log_record == 0:
                self.record_process('[training] [epoch {0}] [step {1}] [loss {2}]\n'.format(int(step/self.step_per_epoch), step, round(loss_item, 4)))
            if self.valid:
                if (step+1) % self.valid_step == 0:
                    self.validation('step')
                    self.test('step')
                if (step+1) % self.step_per_epoch == 0:
                    self.validation('epoch')
                    self.test('epoch')
            torch.cuda.empty_cache()
            if self.show_process:
                pbar.set_postfix({'loss':round(loss_item, 3)})

if __name__ == '__main__':
    # model parameter
    hidden_dim = 768
    head_num = 8
    layer_num = 12
    bond_influence = 1
    prediction_class = 2
    # trainer parameter
    epoch = 100
    batch_size = 16
    lr = 1e-4
    # load data
    batch_data = torch.load('hiv/batch_data.pkl')
    batch_label = torch.load('hiv/batch_label.pkl')
    per_step = len(batch_label[0])
    training_step = int(per_step*0.8)
    training_data = [(i,j) for i,j in zip(batch_data[:training_step], batch_label[:training_step])]
    testing_data = [(i,j) for i,j in zip(batch_data[training_step:], batch_label[training_step:])]
    data = [(batch_data[i], batch_label[i]) for i in range(len(batch_label))]
    g = torch.load('hiv/graph_provider.pkl')

    GAT_model = GAT_predictor(hidden_dim, layer_num, head_num, g.dict_size, bond_influence, prediction_class)
    model_trainer = trainer(GAT_model, epoch, lr)
    model_trainer.load_training_data(training_data, testing_data)
    model_trainer.training_model()
