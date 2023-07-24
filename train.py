import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import optim

from ftcp_pytorch.ftcp import FTCPDataSet
from ftcp_pytorch.vae import VAE, train, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datafile = 'data/example.csv'
params = {  'num_conv_layers' : 3,
            'embedding_dim' : 128,
            'kernel1_size' : 5,
            'kernel2_size' : 3,
            'kernel3_size' : 3,
            'strides1' : 2,
            'strides2' : 2,
            'strides3' : 1,
            'latent_dimensions' : 256,
            'batch_size' : 256,
            'epochs' : 300,
            'dim_pp' : 128,
            'property_predictor':True,
            'learning_rate' : 1e-4,
            'max_elms': 5,
            'max_sites': 40,
            'device': device,
        }


if __name__ == "__main__":
    torch.backends.cudnn.enable = True
    model_path = None

    dataframe = pd.read_csv(datafile)

    dataSet = FTCPDataSet(dataframe, max_elms=params['max_elms'], max_sites=params['max_sites'],predict_property=True, property_name='band_gap')

    print(dataSet.data.shape)

    train_set_size = int(len(dataSet) * 0.8)
    valid_set_size = len(dataSet) - train_set_size
    
    train_set = torch.utils.data.Subset(dataSet, range(train_set_size))
    valid_set = torch.utils.data.Subset(dataSet, range(train_set_size, train_set_size + valid_set_size))

    train_dataloader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(
        valid_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    params['channel_dim'] = dataSet.data.shape[2]
    params['input_dim'] = dataSet.data.shape[1]
    
    current_state = {'name': 'ftcp-vae',
                        'epoch': 0,
                        'model_state_dict': None,
                        'optimizer_state_dict': None,
                        'best_loss': np.inf,
                        'params': params}

    print("Training Model on: " + str(device))
    print("training params: ", params)

    # initialize model
    model = None
    start_epoch = 1
    if model_path:
        loaded_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        for k in current_state.keys():
            try:
                current_state[k] = loaded_checkpoint[k]
            except KeyError:
                current_state[k] = None
        print("checkpoint in ", current_state['epoch'])
        params = current_state['params']
        model = VAE(params).to(device)
        model.load_state_dict(current_state['model_state_dict'])
        start_epoch = current_state['epoch']
    else:
        model = VAE(params).to(device)
    print(model)
    print("start training in epoch ", start_epoch)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr = params['learning_rate'])
    if model_path:
        optimizer.load_state_dict(current_state['optimizer_state_dict'])

    # train model
    epoch = params['epochs']
    train_loss = []
    test_loss = []

    for epoch in range(start_epoch,  epoch+1):
        loss, recon_loss, KLD, MSE = train(model, 
                                              train_dataloader, 
                                              optimizer, 
                                              device, 
                                              epoch
                                              )
        train_loss.append(loss)
        
        test_loss.append(test(model, 
                              test_dataloader,
                                device))    
        
        print("epoch  "+ str(epoch) + "\t" + str(loss) + "\t" + str(recon_loss) + "\t" + str(KLD)+ "\t" + str(MSE))

        ### Update current state and save model
        current_state['epoch'] = epoch
        current_state['model_state_dict'] = model.state_dict()
        current_state['optimizer_state_dict'] = optimizer.state_dict()
        
        if test_loss[-1] < current_state['best_loss']:
            current_state['best_loss'] = test_loss[-1]
            torch.save(current_state, './checkpoints/best.ckpt')

        if epoch % 5 == 0:
            torch.save(current_state, './checkpoints/'+str(epoch)+'.ckpt')
        