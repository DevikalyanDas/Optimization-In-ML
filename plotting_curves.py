
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product
import pathlib



# For smoothing the curves
def smooth_columns(data,smooth = 0.6):
    df_l = data.to_frame()
    df_l['smooth'] = df_l.ewm(alpha = smooth).mean()

    return df_l


def plot_graphs_full(data1,logs_path,texti = None):
    
    colors = ['b', 'r', 'm','g' ]
    styles = ['-', '--']
    f, ax1 = plt.subplots(1,1,figsize=(15,15))
#     ax1.set_title(texti+' Plot')
    for k,v in data1.items():

        ep = list(range(len(v)))
        k_sp = '_'.join(k.split('_')[:2])
        flag_sch = k.split('_')[2]
        if flag_sch=='sch':
            if k.split('_')[0]=='ADAM': 
                ax1.plot(ep, v,ls='--',c=colors[0])  #label=str(k_sp),
                
            elif k.split('_')[0]=='SGD': 
                ax1.plot(ep, v,ls='--',c=colors[1])
                
            elif k.split('_')[0]=='RMSPROP': 
                ax1.plot(ep, v,ls='--',c=colors[2])
                
            elif k.split('_')[0]=='ADAMW': 
                ax1.plot(ep, v,ls='--',c=colors[3])
                
        elif flag_sch=='no':
            if k.split('_')[0]=='ADAM': 
                ax1.plot(ep, v,ls='-',c=colors[0])  #label=str(k_sp),
                
            elif k.split('_')[0]=='SGD': 
                ax1.plot(ep, v,ls='-',c=colors[1])
                
            elif k.split('_')[0]=='RMSPROP': 
                ax1.plot(ep, v,ls='-',c=colors[2])
                
            elif k.split('_')[0]=='ADAMW': 
                ax1.plot(ep, v,ls='-',c=colors[3])
                
    
    
    ax1.plot(np.NaN, np.NaN, c=colors[0], label='ADAM')
    ax1.plot(np.NaN, np.NaN, c=colors[1], label='SGD')
    ax1.plot(np.NaN, np.NaN, c=colors[2], label='RMSPROP')
    ax1.plot(np.NaN, np.NaN, c=colors[3], label='ADAMW')
    
    ax2 = ax1.twinx()

    ax2.plot(np.NaN, np.NaN, ls=styles[0],label='No LR Schd ', c='black')  
    ax2.plot(np.NaN, np.NaN, ls=styles[1],label='With LR Schd ', c='black') 
    
    ax1.set_xlabel('Epochs',fontsize=20)
    ax1.set_ylabel('Values',fontsize=20)    

    ax1.xaxis.set_tick_params(labelsize=20)
    ax2.get_yaxis().set_visible(False)

    ax1.legend(loc="center left",bbox_to_anchor=(1, 0.5),fontsize=20)
    ax2.legend(loc="center left",bbox_to_anchor=(1, 0.4),fontsize=20)

    f.tight_layout()
    

    plt.savefig(os.path.join(logs_path, 'Plot {}.png'.format(texti)), dpi=500)
    plt.show()



if __name__ == '__main__':
    
    import argparse

    
    paser = argparse.ArgumentParser()

    paser.add_argument('-lrn_rt', '--learning_rate', default=5e-4)

    # Path to the csvs
    file_loss = 'loss.csv'
    file_acc = 'accuracy.csv'

    from itertools import product

    bt_sz = [64,128]
    total_epoch = 50
    l_rate = float(args.learning_rate)  #1e-2,
    path_select = ['new_more','new_scheduler']
    optim_used = ['ADAM','ADAMW','RMSPROP','SGD']
    # scheduler_used = ['none','cyclr']

    parameters = dict(
        lr = l_rate,
        batch_size = bt_sz,
        optim = optim_used,
        pth = path_select
    )
    param_values = [v for v in parameters.values()]

    train_loss_dict_64 = {}
    valid_loss_dict_64 = {}
    train_acc_dict_64 = {}
    valid_acc_dict_64 = {}

    train_loss_dict_128 = {}
    valid_loss_dict_128 = {}
    train_acc_dict_128 = {}
    valid_acc_dict_128 = {}
    # Training Loop
    for run_id, (lr,batch_size, optim,pth_s) in enumerate(product(*param_values)):
        comment = f'batch_size_{batch_size}_lr_{lr}_optimizer_{optim}'
        id_part = run_id+1
        path = f'./logs/{pth_s}/csvs/{comment}'
        
        plot_paths = f'./logs/plots_new_{l_rate}'
    #     print(comment)
        pathlib.Path(plot_paths).mkdir(parents=True, exist_ok=True)
        
        loss_path = os.path.join(path,file_loss)
        acc_path = os.path.join(path,file_acc)
        
        df_loss = pd.read_csv(loss_path,sep=',',encoding='utf-8')    
        df_acc = pd.read_csv(acc_path,sep=',',encoding='utf-8')

        smooth_train_loss = smooth_columns(df_loss['train loss'])
        smooth_valid_loss = smooth_columns(df_loss['val loss'])
        smooth_train_acc = smooth_columns(df_acc['train accuracy'])
        smooth_valid_acc = smooth_columns(df_acc['val accuracy'])    
        
        if comment.split('_')[2]=='64':
            if pth_s.split('_')[1] == 'scheduler':
                train_loss_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_train_loss['smooth'].tolist()
                valid_loss_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_valid_loss['smooth'].tolist()
                train_acc_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_train_acc['smooth'].tolist()
                valid_acc_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_valid_acc['smooth'].tolist()
            elif pth_s.split('_')[1] == 'more':
                train_loss_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_train_loss['smooth'].tolist()
                valid_loss_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_valid_loss['smooth'].tolist()
                train_acc_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_train_acc['smooth'].tolist()
                valid_acc_dict_64['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_valid_acc['smooth'].tolist()
    #   
        else:
            if pth_s.split('_')[1] == 'scheduler':
                train_loss_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_train_loss['smooth'].tolist()
                valid_loss_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_valid_loss['smooth'].tolist()
                train_acc_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_train_acc['smooth'].tolist()
                valid_acc_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'sch'])] = smooth_valid_acc['smooth'].tolist()
            elif pth_s.split('_')[1] == 'more':
                train_loss_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_train_loss['smooth'].tolist()
                valid_loss_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_valid_loss['smooth'].tolist()
                train_acc_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_train_acc['smooth'].tolist()
                valid_acc_dict_128['_'.join([comment.split('_')[6],comment.split('_')[4],'no'])] = smooth_valid_acc['smooth'].tolist()


    plot_graphs_full(train_loss_dict_64,plot_paths,texti = 'Train loss with bs 64')
    plot_graphs_full(valid_loss_dict_64,plot_paths,texti = 'Val loss with bs 64')
    plot_graphs_full(train_acc_dict_64,plot_paths,texti = 'Train Accuracy with bs 64')
    plot_graphs_full(valid_acc_dict_64,plot_paths,texti = 'Val Accuracy with bs 64')

    plot_graphs_full(train_loss_dict_128,plot_paths,texti = 'Train loss with bs 128')
    plot_graphs_full(valid_loss_dict_128,plot_paths,texti = 'Val loss with bs 128')
    plot_graphs_full(train_acc_dict_128,plot_paths,texti = 'Train Accuracy with bs 128')
    plot_graphs_full(valid_acc_dict_128,plot_paths,texti = 'Val Accuracy with bs 128')



