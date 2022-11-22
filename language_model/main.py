import argparse
import time
import math
import torch
import torch.nn as nn

import data
import model
import os,sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

os.chdir(sys.path[0])
parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
# you can increase the seqence length to see how well the model works when capturing long-term dependencies
parser.add_argument('--max_sql', type=int, default=35, 
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int,default = 0, help='GPU device id used')
parser.add_argument('--LR', type=int, default= 0.001, help='Learning Rate')
# feel free to add some other arguments
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/wikitext2", batch_size, args.max_sql)

        
# WRITE CODE HERE within two '#' bar                                                           #
# Build model, optimizer and so on                                                             #
################################################################################################
voc_dim = len(data_loader.vocabulary)
useRNN = False
if useRNN:
    rnn = model.RNN(voc_dim,512,512,1)
    #rnn = model.customLSTM(voc_dim,512,512,1)
    rnn = rnn.to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.LR) 
    criterion = nn.CrossEntropyLoss()
else:
    transformer = model.LMTransformer(voc_dim, 512, nhead = 8, d_hid = 512, nlayers = 3, dropout = 0.2).to(device)
    lr = 5.0 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

################################################################################################

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach() 
    else: 
        return tuple(repackage_hidden(v) for v in h)

# WRITE CODE HERE within two '#' bar                                                           #
# Evaluation Function                                                                          #
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       #
################################################################################################
if useRNN:
    def evaluate():
        rnn.eval()
        total_loss = 0.0
        total_correct = 0
        data_loader.set_valid()
        with torch.no_grad():
            for i in range(data_loader.valid.size(0)//args.max_sql):
                origin,target,flag = data_loader.get_batch()
                origin = origin.to(device)
                target = target.to(device)
                predict,hidden = rnn(origin)
                _, predictions = torch.max(predict, 1)
                loss = criterion(predict, target)
                total_loss += loss.item() * origin.size(0)
                total_correct += torch.sum(predictions == target)/len(predictions)
                if i % 1000 == 0:
                    print('validing loss of',i,'000 iter is',loss.item() * origin.size(0))
            epoch_loss = total_loss /data_loader.valid.size(0)
            epoch_acc = total_correct.double()/data_loader.valid.size(0)*args.max_sql
        
        return epoch_loss#, epoch_acc.item()
    ################################################################################################

    # WRITE CODE HERE within two '#' bar                                                           #
    # Training Function                                                                            #     
    # Calculate the average cross-entropy loss between the prediction and the ground truth word    #
    # And then exp(average cross-entropy loss) is perplexity                                       # 
    ################################################################################################
    def train():
        rnn.train()
        total_loss = 0.0
        total_correct = 0
        data_loader.set_train()
        hidden = rnn.init_hidden(args.train_batch_size)
        for i in range(data_loader.train.size(0)//args.max_sql):
            origin, target, flag = data_loader.get_batch()
            origin = origin.to(device)
            target = target.to(device)
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()
            predict, hidden = rnn(origin)
            _, predictions = torch.max(predict, 1)
            loss = criterion(predict, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
            optimizer.step()
            total_loss += loss.item() * origin.size(0)
            total_correct += torch.sum(predictions == target)/len(predictions)
            if i % 1000 == 0:
                print('training loss of',i,'000 iter is',loss.item() * origin.size(0))

        epoch_loss = total_loss /data_loader.train.size(0)
        epoch_acc = total_correct.double()/data_loader.train.size(0)*args.max_sql
        
        return epoch_loss#,epoch_acc.item()
################################################################################################
else:
    def train() :
        transformer.train()  # turn on train mode
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        bptt = args.max_sql
        src_mask = model.generate_square_subsequent_mask(bptt).to(device)

        data_loader.set_train()
        for i in range(data_loader.train.size(0)//args.max_sql):
            origin, target, flag = data_loader.get_batch()
            origin = origin.to(device)
            target = target.to(device)
            seq_len = origin.size(0)
            if seq_len != bptt:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = transformer(origin, src_mask)
            loss = criterion(output.view(-1, voc_dim), target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()* origin.size(0)
        epoch_loss = total_loss /data_loader.train.size(0)
        return epoch_loss

    def evaluate():
        transformer.eval()  # turn on evaluation mode
        bptt = args.max_sql
        total_loss = 0.
        src_mask = model.generate_square_subsequent_mask(bptt).to(device)
        data_loader.set_valid()
        with torch.no_grad():
            for i in range(data_loader.valid.size(0)//args.max_sql):
                origin,target,flag = data_loader.get_batch()
                origin = origin.to(device)
                target = target.to(device)
                seq_len = origin.size(0)
                if seq_len != bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = transformer(origin, src_mask)
                output_flat = output.view(-1, voc_dim)
                total_loss += seq_len * criterion(output_flat, target).item()
        return total_loss / data_loader.valid.size(0)




# WRITE CODE HERE within two '#' bar                                                           #
# Loop over epochs                                                                             #
################################################################################################
loss_curve = []
val_loss_curve = []
for epoch in range(1, args.epochs+1):

    epoch_loss_ = train()
    print("Epoch {0} (total {1} epochs) | AvgLoss {2}".format(epoch, args.epochs, epoch_loss_))
    #print("Epoch {0} (total {1} epochs) | AvgAccu {2}".format(epoch, args.epochs, epoch_acc_))
    loss_curve.append(np.exp(epoch_loss_))

    valid_loss = evaluate()
    print("Valid Epoch {0} (total {1} epochs) | AvgLoss {2}".format(epoch, args.epochs, valid_loss))
    #print("Valid Epoch {0} (total {1} epochs) | AvgAccu {2}".format(epoch, args.epochs, valid_accu))
    val_loss_curve.append(np.exp(valid_loss))

plt.title('Training Curve')
plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.plot(range(len(loss_curve)), loss_curve)
plt.show()
plt.title('Valid Curve')
plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.plot(range(len(val_loss_curve)), val_loss_curve)
plt.show()
################################################################################################