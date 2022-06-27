#!/usr/bin/env python
# coding: utf-8
# %%
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from PIL import Image

import pathlib
import random
import math
# from torch.optim.lr_scheduler import LambdaLR


# %%

def reverse_dic(dic):
    return dict([(v, k) for (k, v) in dic.items()])


def get_label2id(file_name):
    with open(file_name) as file:
        dict = json.load(file)

    return dict


def get_train_set_dic(path,req_labels):
    dict = {}
    list_file = os.listdir(path)

    for item in list_file:
        list_content = req_labels#os.listdir(path + item)
        inside_dict = {}
        for key in list_content:
            inside_dict[key] = os.listdir(path + item + '/' + key)
        dict[item] = inside_dict

    return dict


def get_test_set_images(path):
    # implement a function, return a list of image names

    dir = os.listdir(path)
    # print(dir)
    return dir

def write_result(result, path = 'prediction.json'):

    with open(path, 'w') as f:
        json.dump(result, f)

    print('managed to save result!')
    print('-'*100)




class MyDataSet(Dataset):
    def __init__(self,
                 mode='train',
                 valid_category=None,
                 label2id_path='../input/nico2022/dg_label_id_mapping.json',
                 test_image_path=None,
                 train_image_path='../input/nico2022/track_1/public_dg_0416/train/',
                 transform_type=None,
                 required_labels = None
                 ):
        '''
        :param mode:  train? valid? test?
        :param valid_category: if train or valid, you must pass this parameter
        :param label2id_path:
        :param test_image_path:
        :param train_image_path:  must end by '/'
        '''
        self.mode = mode
        self.transform_type = transform_type
        self.label2id = get_label2id(label2id_path)
        self.id2label = reverse_dic(self.label2id)
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
        self.required_labels = required_labels

        self.transform = transforms.Compose([
            # add transforms here
            transforms.RandomResizedCrop((224,224),scale=(0.75,1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#             transforms.RandAugment(2, 5),
            # transforms.RandomRotation(15),
            # transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # if train or valid, synthesize a dic contain num_images * dic, each subdic contain:
        # path、category_id、context_category

        if mode == 'test':
            self.images = get_test_set_images(test_image_path)

        if mode == 'train':
            self.total_dic = get_train_set_dic(train_image_path,self.required_labels)
            if valid_category is not None:
                del self.total_dic[valid_category]
            self.synthesize_images()

        if mode == 'valid':
            self.total_dic = get_train_set_dic(train_image_path,self.required_labels)
            self.total_dic = dict([(valid_category, self.total_dic[valid_category])])
            self.synthesize_images()

    def synthesize_images(self):
        self.images = {}

        count = 0

        for context_category, context_value in list(self.total_dic.items()):      
            for category_name, image_list in list(context_value.items()):
                for img in image_list:
                    now_dic = {}
                    now_dic['path'] = self.train_image_path + context_category + '/' + category_name + '/' + img
                    now_dic['category_id'] = self.label2id[category_name]
                    now_dic['context_category'] = context_category
                    self.images[len(self.images)] = now_dic

                    # count +=1
                    # if count >= 10:
                    #     return 0

        # self.images = dict(list(self.images.items())[0:10])
        # print(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.mode == 'test':
            img = Image.open(self.test_image_path + self.images[item])
            if self.transform_type == 'test' or self.transform_type is None:
                img = self.test_transform(img)
            else:
                img = self.transform(img)
            return img, self.images[item]

        if self.mode == 'train' or self.mode == 'valid':
            img_dic = self.images[item]
            img = Image.open(img_dic['path'])
            img = self.transform(img)
            y = img_dic['category_id']

            # print(img_dic['path'], self.id2label[y])

            return img, y

    def get_id2label(self):
        return self.id2label



# %%

def get_loader(train_image_path,
               valid_image_path,
               label2id_path,
               batch_size=32,
               valid_category='autumn',
               num_workers=8,
               pin_memory=False):
    '''
    if you are familiar with me, you will know this function aims to get train loader and valid loader
    :return:
    '''
    context_category_list = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    required_classes = ['airplane', 'bird', 'bicycle', 'car', 'cat', 'dog', 'flower', 'horse', 'motorcycle', 'racket', 'ship', 'umbrella', 'truck', 'tiger', 'gun']
    if valid_category == 'rand':
        valid_category = context_category_list[random.randint(0, len(context_category_list) - 1)]

    print(f'we choose {valid_category} as valid, others as train')
    print('-' * 100)

    train_set = MyDataSet(mode='train', train_image_path=train_image_path,
                          label2id_path=label2id_path, valid_category=valid_category,required_labels = required_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    if valid_category is None:
        return train_loader

    valid_set = MyDataSet(mode='valid', valid_category=valid_category,
                          train_image_path=valid_image_path, label2id_path=label2id_path,required_labels = required_classes)

    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    print('managed to get loader!!!!!')
    print('-' * 100)

    return train_loader, valid_loader


def get_test_loader(batch_size=32,
                    test_image_path='./ood_data/track_1/public_dg_0416/public_test_flat/',
                    label2id_path='./ood_data/track_1/dg_label_id_mapping_new.json', transforms=None,
                    num_workers=8):
    '''
    No discriptions
    :return:
    '''
    test_set = MyDataSet(mode='test',
                         test_image_path=test_image_path,
                         label2id_path=label2id_path,
                         transform_type=transforms)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader, test_set.get_id2label()




def hook(module, grad_in, grad_out):
    print('-' * 100)
    print(module)
    print(f'grad in {grad_in}, grad out {grad_out}')


class MixLoader():
    def __init__(self, iteraters, probs=None):
        '''
        :param iteraters: list of dataloader
        '''
        self.loaders = iteraters
        self.iteraters = [iter(loader) for loader in iteraters]
        self.index = [i for i, _ in enumerate(iteraters)]
        lens = [len(i) for i in self.iteraters]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = self.compute_prob(lens)
        self.max_time = sum(lens)
        self.iteration_times = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration_times >= self.max_time:
            self.iteration_times = 0
            raise StopIteration
        else:
            choice = np.random.choice(self.index, p=self.probs)
            try:
                data = next(self.iteraters[choice])
                self.iteration_times += 1
                return data
            except:
                self.iteraters[choice] = iter(self.loaders[choice])
                return self.__next__()

    @staticmethod
    def compute_prob(x):
        total = sum(x)
        result = [i / total for i in x]
        return result

    def __len__(self):
        return self.max_time



# %%
# Model Description class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        from torchvision import models
#         self.model = pyramidnet272(num_classes=60)
        self.model = models.resnet50(num_classes = 15 )
        # print(self.model)
        # self.model = get_pyramidnet(alpha=48, blocks=152, num_classes=60)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.apply(self._init_weights)
        print('We are using model: resnet50 with 15 class output')

    def forward(self, x):
        x = self.model(x)
        return x

    def load_model(self):
        if os.path.exists('model.pth'):
            start_state = torch.load('model.pth', map_location=self.device)

            self.model.load_state_dict(start_state)
            print('using loaded model')
            print('-' * 100)

    def save_model(self,name):
        result = self.model.state_dict()
        torch.save(result, os.path.join(name,'model_wt.pth'))



# %%
# LR scheduler Manual. We are not using it
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    def lr_lambda(now_step):
        if now_step < num_warmup_steps:  # If less than, the learning rate ratio increases monotonically
            return float(now_step) / float(max(1, num_warmup_steps))
        # If it is greater than that, change the ratio to continue to adjust the learning rate
        progress = float(now_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class NoisyStudent():
    def __init__(self,
                 args,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-4,
                 comment= 'name of the process',
                 path_w = 'path_for the weights',
                 path_csv = 'path for the csv',
                 train_image_path='./ood_data/track_1/public_dg_0416/train/',
                 valid_image_path='./ood_data/track_1/public_dg_0416/train/',
                 label2id_path='./ood_data/track_1/dg_label_id_mapping_new.json',
                 test_image_path='./ood_data/track_1/public_dg_0416/public_test_flat/'

                 ):
        self.result = {}
#         from data.data import get_loader, get_test_loader
        self.train_loader,self.valid_loader = get_loader(batch_size=batch_size,
                                       valid_category = 'autumn',
                                       train_image_path=train_image_path,
                                       valid_image_path=valid_image_path,
                                       label2id_path=label2id_path)
        self.test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                                      transforms=None,
                                                      label2id_path=label2id_path,
                                                      test_image_path=test_image_path)
        self.test_loader_student, self.label2id = get_test_loader(batch_size=batch_size,
                                                                  transforms='train',
                                                                  label2id_path=label2id_path,
                                                                  test_image_path=test_image_path)
        # self.train_loader = MixLoader([self.train_loader, self.test_loader_student])
        del self.test_loader_student
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model().to(self.device)

#         if os.path.exists('model.pth'):
#             self.model.load_model()

        self.lr = lr
        self.comment = comment
        self.path_w = path_w
        self.csv_path_dir = path_csv
        self.batch_size = batch_size
        self.args = args
        self.loss_history = {"epoch": [], "lr":[], "train loss": [], "val loss":[]}
        self.accuracy_history = {"epoch":[], "train accuracy":[], "val accuracy":[]}
        
        if self.args.optimizer =='ADAM':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.args.optimizer =='SGD':
            self.optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif self.args.optimizer =='ADAMW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.args.optimizer == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if self.args.scheduler != 'none':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20,40], gamma=0.1)
    def save_result(self, epoch=None):
#         from data.data import write_result
        result = {}
        for name, pre in list(self.result.items()):
            _, y = torch.max(pre, dim=1)
            result[name] = y.item()

        if epoch is not None:
            write_result(result, path='prediction' + str(epoch) + '.json')
        else:
            write_result(result)

        return result

    def predict(self, dataloader):
        with torch.no_grad():
            print('teacher are giving his predictions!')
            self.model.eval()
            for x, names in tqdm(self.test_loader_predict):
                x = x.to(self.device)
                x = self.model(x)
                for i, name in enumerate(list(names)):
                    self.result[name] = x[i, :].unsqueeze(0)  # 1, D

            print('teacher have given his predictions!')
            print('-' * 100)

    def get_label(self, names):
        y = []
        for name in list(names):
            y.append(self.result[name])

        return torch.tensor(y, device=self.device)
    
    
    def train(self,
              total_epoch=3,
              label_smoothing=0.2,
              fp16_training=True,
              warmup_epoch=1,
              warmup_cycle=12000,
              ):
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        prev_loss = 999
        

        
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        loss_history = {"epoch": [], "lr":[], "train loss": [], "val loss":[]}
        accuracy_history = {"epoch":[], "train accuracy":[], "val accuracy":[]}
        
        # early stop parameters
        loss_increase_counter = 0
        early_stop = True  # early stop flag
        early_stop_threshold = 10
        for epoch in range(1, total_epoch + 1):
            train_loss = 0
            train_acc = 0
            step = 0
            self.model.train()
            #self.warm_up(epoch, now_loss = train_loss, prev_loss = prev_loss)

            pbar = tqdm(self.train_loader)
            for x, y in pbar:
                x = x.to(self.device)
                if isinstance(y, tuple):
                    y = self.get_label(y)
                y = y.to(self.device)
                
                with autocast():
                    y_out = self.model(x)  # N, 60
                    _, pre = torch.max(y_out, dim=1)

                    loss = criterion(y_out, y)
                
                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
#                 acc1 = accuracy(y_out,y)

                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()
                
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                # assert False
                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                
                scaler.step(self.optimizer)
                scaler.update()
                
                step += 1
                
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')
            
            if self.args.scheduler != 'none':
                self.scheduler.step()    

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader) 
            print(f'epoch {epoch}, train loader loss = {train_loss}, acc = {train_acc}, lr = {self.optimizer.param_groups[0]["lr"]}')
            with torch.no_grad():
                self.model.eval()
                step = 0
                val_loss = 0
                val_acc = 0
                pbar = tqdm(self.valid_loader)
                for x, y in pbar:
                    x = x.to(self.device)
                    if isinstance(y, tuple):
                        y = self.get_label(y)
                    y = y.to(self.device)

                    with autocast():
                        y_out = self.model(x)  # N, 60
                        _, pre = torch.max(y_out, dim=1)

                        vl_loss = criterion(y_out, y)

                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)
    #                 acc1 = accuracy(y_out,y)
                    val_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    val_loss += vl_loss.item()

                    step += 1
                    # scheduler.step()
                    if step % 10 == 0:
                        pbar.set_postfix_str(f'loss = {val_loss/step} ,acc = {val_acc/step}')

                val_loss /= len(self.valid_loader)
                val_acc /= len(self.valid_loader)                       
                print(f'epoch {epoch},valid loss = {val_loss}, valid acc = {val_acc}') 

           
            self.loss_history["epoch"].append(epoch)
            self.loss_history["lr"].append(self.optimizer.param_groups[0]['lr'])
            self.loss_history["train loss"].append(train_loss)
            self.loss_history["val loss"].append(val_loss)
            self.accuracy_history["epoch"].append(epoch)
            self.accuracy_history["val accuracy"].append(val_acc)   
            self.accuracy_history["train accuracy"].append(train_acc)
            
            
#             # Early stopping
#             if val_loss > prev_loss:
#                 loss_increase_counter += 1
# #                 file_name12 = os.path.join(ckp_path,'model_transf_last_new.pt')
# #                 torch.save(state, file_name12)
#                 if early_stop and (loss_increase_counter > early_stop_threshold):
#                     print("Early Stopping..")
#                     break
#             else:
#                 loss_increase_counter = 0
#                 self.model.save_model(self.path_w)
#                 prev_loss = val_loss
#                 print('saved best checkpoint at epoch {}'.format(epoch))
            self.model.save_model(self.path_w)     
            torch.cuda.empty_cache()
        
        pd.DataFrame.from_dict(data=self.loss_history, orient='columns').to_csv(os.path.join(self.csv_path_dir,'loss.csv'), header=['epoch', 'lr', 'train loss','val loss'])
        pd.DataFrame.from_dict(data=self.accuracy_history, orient='columns').to_csv(os.path.join(self.csv_path_dir,'accuracy.csv'), header=['epoch','train accuracy','val accuracy'])
       

if __name__ == '__main__':
    
    import argparse

    
    paser = argparse.ArgumentParser()
    paser.add_argument('-bt', '--batch_size', default=128)
    paser.add_argument('-opt', '--optimizer', default='ADAM')
    paser.add_argument('-lrn_rt', '--learning_rate', default=5e-4)
    paser.add_argument('-schdlr', '--scheduler', default='none')
    paser.add_argument('-ep', '--epoch', default=50)
    # paser.add_argument('-pt', '--path', default='./logs/new_check')
    
    args = paser.parse_args()
    
    bt_sz = int(args.batch_size) #64
    total_epoch = int(args.epoch)
    l_rate = float(args.learning_rate) #5e-4 #5e-2
    optim_used = str(args.optimizer) #'ADAM' # 'SGD'
    scheduler_used = str(args.scheduler) #'none' #'cyclr'

    comment = f'batch_size_{bt_sz}_lr_{l_rate}_optimizer_{optim_used}' # _scheduler_{scheduler_used}
    print(f'We are running : {comment}')
    print('-' * 100)
    # path 
    if scheduler_used=='none':
        path_req = './logs/new_more'
    else:
        path_req = './logs/new_scheduler'

    filepath_weights = os.path.join(path_req,'weights',comment)
    filepath_csvs = os.path.join(path_req,'csvs',comment)

    pathlib.Path(filepath_weights).mkdir(parents=True, exist_ok=True)
    pathlib.Path(filepath_csvs).mkdir(parents=True, exist_ok=True)

    x = NoisyStudent(args,batch_size=bt_sz, lr=l_rate,comment = comment,path_w = filepath_weights,path_csv = filepath_csvs)
    x.train(total_epoch=total_epoch)
