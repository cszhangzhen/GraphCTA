import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from model import *
from utils import *
from datasets import *
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:2', help='specify cuda devices')
parser.add_argument('--source', type=str, default='Citationv1', help='source domain data')
parser.add_argument('--target', type=str, default='DBLPv7', help='target domain data')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--num_layers', type=int, default=2, help='number of gnn layers')
parser.add_argument('--gnn', type=str, default='gcn', help='different types of gnns')
parser.add_argument('--use_bn', type=bool, default=False, help='do not use batchnorm')

args = parser.parse_args()

if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.source)
    source_dataset = CitationDataset(path, args.source)
elif args.source in {'S10', 'M10', 'E10'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Elliptic', args.source)
    source_dataset = EllipticDataset(path, args.source)
elif args.source in {'DE', 'EN', 'FR'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Twitch', args.source)
    source_dataset = TwitchDataset(path, args.source)

if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)
elif args.target in {'S10', 'M10', 'E10'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Elliptic', args.target)
    target_dataset = EllipticDataset(path, args.target)
elif args.target in {'DE', 'EN', 'FR'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Twitch', args.target)
    target_dataset = TwitchDataset(path, args.target)

target_data = target_dataset[0]
data = source_dataset[0]

args.num_classes = len(np.unique(data.y.numpy()))
args.num_features = data.x.size(1)

print(args)

model = Model(args).to(args.device)
data = data.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train_source():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        correct = 0
        output = model(data.x, data.edge_index)
        train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        pred = output[data.train_mask].max(dim=1)[1]
        correct = pred.eq(data.y[data.train_mask]).sum().item()
        train_acc = correct * 1.0 / (data.train_mask).sum().item()

        val_acc, val_loss = compute_test(data.val_mask, model, data)

        print('Epoch: {:04d}'.format(epoch + 1), 'train_loss: {:.6f}'.format(train_loss),
              'train_acc: {:.6f}'.format(train_acc), 'loss_val: {:.6f}'.format(val_loss),
              'acc_val: {:.6f}'.format(val_acc), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(val_loss)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


if __name__ == '__main__':
    if os.path.exists('model.pth'):
        os.remove('model.pth')
    # Model training
    best_model = train_source()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(data.test_mask, model, data)
    print('Source {} test set results, loss = {:.6f}, accuracy = {:.6f}'.format(args.source, test_loss, test_acc))

    target_data = target_data.to(args.device)
    test_acc, test_loss = evaluate(target_data.x, target_data.edge_index, target_data.edge_weight, target_data.y, model)
    print('Target {} test results, loss = {:.6f}, accuracy = {:.6f}'.format(args.target, test_loss, test_acc))

    # Save model for target domain adaptation
    torch.save(model.state_dict(), 'model.pth')
    os.remove('{}.pth'.format(best_model))
