from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable, Function
from tqdm import tqdm
from data.build import get_dataloader
from models.model import HyperspecAE
from engine.trainer import SAD, SID
from utils.parse import ArgumentParser
import utils.opts as opts

print(1)
# ------------------ Training -------------------- #
# Load Data
def train(opt):

    train_dataloader, test_set = get_dataloader(BATCH_SIZE=opt.batch_size, DIR=opt.src_dir)

    max_batches = len(train_dataloader)

    # Define Model
    model = HyperspecAE(opt.num_bands, opt.end_members, opt.gaussian_dropout, opt.activation,
                opt.threshold, opt.encoder_type)
    optimizer = optim.Adam(model.parameters(), opt.learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    for epoch in range(opt.epochs):
    
        iterator = iter(train_dataloader)
        loop = tqdm(train_dataloader)
        for batch__ in loop:
            X, _ = next(iterator)
            X = X.view(X.size()[0], -1)
            X = X.cuda()

            enc_out, dec_out = model(X.float())

            if opt.objective=="SAD":
                reconstr_loss = SAD()
            elif opt.objective=='MSE':
                reconstr_loss = nn.MSELoss()
            else:
                reconstr_loss = SID()
            loss = reconstr_loss(dec_out, X.float())

            loss = torch.sum(loss).float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 1 == 0:
                loop.set_postfix(
                    epoch=epoch,
                    training_loss=loss.item()
                )
        if (epoch+1)%10==0:
            print(f'Epoch {epoch + 1:04d} / {opt.epochs:04d}', end='\n=================\n')
            print("Loss: %.4f" %(loss.item()))

        if (opt.save_checkpt!=0 and (epoch+1)%opt.save_checkpt==0):
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, f"{opt.save_dir}/hyperspecae_{epoch+1}.pt")
    
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, f"{opt.save_dir}/hyperspecae_final.pt")
            
    print('Training Finished!')
    
def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.model_opts(parser)
    opts.train_opts(parser)
    
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    train(opt)