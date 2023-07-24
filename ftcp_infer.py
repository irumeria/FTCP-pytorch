import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import joblib

from ftcp_pytorch.ftcp import FTCPDataSet, get_info
from ftcp_pytorch.vae import VAE, train, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "./checkpoints/best.ckpt"

if __name__ == "__main__":
    torch.backends.cudnn.enable = True
    max_sites = 40
    
    dataframe = pd.read_csv('data/example.csv')

    dataSet = FTCPDataSet(dataframe, max_elms=5, max_sites=40,predict_property=True, property_name='band_gap')

    train_set_size = int(len(dataSet) * 0.8)
    valid_set_size = len(dataSet) - train_set_size
    
    train_set = torch.utils.data.Subset(dataSet, range(train_set_size))
    valid_set = torch.utils.data.Subset(dataSet, range(train_set_size, train_set_size + valid_set_size))

    params = None
    current_state = {'name': 'ftcp-vae',
                        'epoch': 0,
                        'model_state_dict': None,
                        'optimizer_state_dict': None,
                        'best_loss': np.inf,
                        'params': params}
    
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
    model.eval()
    print(model)
    print("start training in epoch ", start_epoch)


    ftcp_rebuild = []
    ftcp_predict = []

    for i in range(0, 100):
        (data, prop) = valid_set[i]
        data = torch.tensor([data]).to(device)
        data = data.type(torch.cuda.FloatTensor)
        recon_data, z, mu, logvar, pred_prop = model(data)
        
        ftcp_rebuild.append(recon_data[0].cpu().detach().numpy())
        # la sampling
        z += 0.1 * torch.randn_like(z)
        recon_data = model.decoder(z)
        ftcp_predict.append(recon_data.cpu().detach().numpy())
    ftcp_rebuild = np.array(ftcp_rebuild)

    get_info(ftcp_rebuild, max_elms=params['max_elms'], max_sites=40,
            elm_str=joblib.load('data/element.pkl')
            )
    