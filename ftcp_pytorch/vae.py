import torch
from torch import nn
from torch.nn import functional as F
from torch import nn, optim
import numpy as np

def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1):

    loss_recon = torch.sum(torch.square(x_out - x))
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop, true_prop)
    else:
        MSE = torch.zeros_like(loss_recon)
    if torch.isnan(KLD):
        KLD = torch.zeros_like(loss_recon)
        
    return torch.mean(loss_recon + KLD + MSE), loss_recon, KLD, MSE

class PropertyPredictor(nn.Module):
    def __init__(self, d_latent, d_pp):
        super().__init__()

        self.prediction_layers = nn.Sequential(
            nn.Linear(d_latent, d_pp),
            nn.ReLU(),
            nn.Linear(d_pp, d_pp),
            nn.ReLU(),
            nn.Linear(d_pp, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.prediction_layers(x)
        return x


class VAE(nn.Module):

    def __init__(self, params):

        super(VAE, self).__init__()

        # Load Model Parameters
        self.channel_dim = params['channel_dim']
        self.input_dim = params['input_dim']

        self.num_conv_layers = params['num_conv_layers']
        self.embedding_dim = params['embedding_dim']
        self.kernel1_size = params['kernel1_size']
        self.kernel2_size = params['kernel2_size']
        self.kernel3_size = params['kernel3_size']
        self.strides1 = params['strides1']
        self.strides2 = params['strides2']
        self.strides3 = params['strides3']
        self.latent_dimensions = params['latent_dimensions']
        self.batch_size = params['batch_size']

        self.convl1 = nn.Conv1d(self.input_dim, self.embedding_dim//4,
                                self.kernel1_size)
        self.convl2 = nn.Conv1d(self.embedding_dim//4, self.embedding_dim//2,
                                self.kernel2_size)
        self.convl3 = nn.Conv1d(self.embedding_dim//2, self.embedding_dim,
                                self.kernel3_size)
        # hard codeing for now
        self.map_size = 56
        self.encoder_dense = nn.Linear(self.map_size*self.embedding_dim , 1024)

        self.norm1 = nn.BatchNorm1d(self.embedding_dim//4)
        self.norm2 = nn.BatchNorm1d(self.embedding_dim//2)
        self.norm3 = nn.BatchNorm1d(self.embedding_dim)

        self.normd1 = nn.BatchNorm2d(self.embedding_dim)
        self.normd2 = nn.BatchNorm2d(self.embedding_dim//2)
        self.normd3 = nn.BatchNorm2d(self.embedding_dim//4)

        self.fc_mu = nn.Linear(
            1024, self.latent_dimensions)  
        
        self.fc_logvar = nn.Linear(
            1024, self.latent_dimensions)
        
        self.recover = nn.Linear(self.latent_dimensions, self.embedding_dim*self.map_size)

        self.convd1 = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim//2, 
                                         (1, self.kernel3_size))
        self.convd2 = nn.ConvTranspose2d(self.embedding_dim//2, self.embedding_dim//4,
                                            (1, self.kernel2_size))
        self.convd3 = nn.ConvTranspose2d(self.embedding_dim//4, self.input_dim,
                                         (1, self.kernel1_size))
        
        if params['property_predictor']:
            self.dim_pp = params['dim_pp']
            self.property_predictor = PropertyPredictor(self.latent_dimensions, self.dim_pp)
        else:
            self.property_predictor = None



    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def encoder(self, x):
        x = x.transpose(1, 2)
        x = self.convl1(x)
        x = F.leaky_relu(self.norm1(x), 0.2)
        x = self.convl2(x)
        x = F.leaky_relu(self.norm2(x), 0.2)
        x = self.convl3(x)
        x = F.leaky_relu(self.norm3(x), 0.2)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.encoder_dense(x)
        x = F.sigmoid(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decoder(self, z):
        rz = self.recover(z)
        rz = rz.reshape(-1, self.embedding_dim, 1, self.map_size)
        x = self.normd1(rz)
        x = self.convd1(x)
        x = self.normd2(x)
        x = self.convd2(x)
        x = self.normd3(x)
        x = self.convd3(x)
        x = F.sigmoid(x)
        x_hat = x.squeeze(dim=2)

        return x_hat

    def forward(self, x):
        x = x.transpose(1, 2)

        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z)

        if self.property_predictor:
            prop = self.property_predictor(mu)
        else:
            prop = None
        return x_hat, z, mu, logvar, prop


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    weights = torch.ones(model.channel_dim, dtype=torch.float).to(device)
    recon_losses = []
    KLDs = []
    MSEs = []
    losses = []
    for batch_idx, (data,prop) in enumerate(train_loader):
        data = torch.tensor(data).to(device)
        data = data.type(torch.cuda.FloatTensor)

        recon_data, z, mu, logvar, pred_prop = model(data)
        if pred_prop is not None:
            prop = torch.tensor(prop).to(device)
            prop = prop.type(torch.cuda.FloatTensor)
        loss, recon_loss, KLD, MSE = vae_loss(
            data.squeeze(dim=1),
            recon_data,
            mu, logvar,
            true_prop=prop,  
            pred_prop=pred_prop,
            weights=weights,
            beta=1
        )

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        if batch_idx == 0:
            print("first batch loss ", loss.item())
        recon_losses.append(recon_loss.item())
        KLDs.append(KLD.item())
        MSEs.append(MSE.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.5f}'.format(
        epoch, np.mean(losses)))

    return np.mean(losses), np.mean(recon_losses), np.mean(KLDs), np.mean(MSEs)


def test(model, test_loader, device):
    model.eval()
    test_loss = []
    weights = torch.ones(model.channel_dim, dtype=torch.float).to(device)
    with torch.no_grad():
        for batch_idx, (data,prop) in enumerate(test_loader):
            data = torch.tensor(data).to(device)
            data = data.type(torch.cuda.FloatTensor)

            recon_data, z, mu, logvar, pred_prop = model(data)
            if pred_prop is not None:
                prop = torch.tensor(prop).to(device)
                prop = prop.type(torch.cuda.FloatTensor)
            loss, recon_loss, KLD, MSE = vae_loss(
                data.squeeze(dim=1),
                recon_data,
                mu, logvar,
                true_prop=prop,  
                pred_prop=pred_prop,
                weights=weights,
                beta=1
            )
            test_loss.append(loss.item())

    print('====> Test set loss: {:.5f}'.format(np.mean(test_loss)))

    return np.mean(test_loss)


