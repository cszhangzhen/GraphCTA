import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from model import *
from utils import *
from layer import *
from datasets import *
import numpy as np
from torch_geometric.transforms import Constant
from torch.nn.parameter import Parameter
from torch_geometric.utils import dropout_adj
from tqdm import tqdm
import random

from torch import Tensor

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
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--tau', type=float, default=0.2, help='tau')
parser.add_argument('--lamb', type=float, default=0.2, help='trade-off parameter lambda')
parser.add_argument('--num_layers', type=int, default=2, help='number of gnn layers')
parser.add_argument('--gnn', type=str, default='gcn', help='different types of gnns')
parser.add_argument('--use_bn', type=bool, default=False, help='do not use batchnorm')
parser.add_argument('--make_undirected', type=bool, default=True, help='directed graph or not')

parser.add_argument('--ratio', type=float, default=0.2, help='structure perturbation budget')
parser.add_argument('--loop_adj', type=int, default=1, help='inner loop for adjacent update')
parser.add_argument('--loop_feat', type=int, default=2, help='inner loop for feature update')
parser.add_argument('--loop_model', type=int, default=3, help='inner loop for model update')
parser.add_argument('--debug', type=int, default=1, help='whether output intermediate results')
parser.add_argument("--K", type=int, default=5, help='number of k-nearest neighbors')

args = parser.parse_args()


if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)
elif args.target in {'S10', 'M10', 'E10'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Elliptic', args.target)
    target_dataset = EllipticDataset(path, args.target)
elif args.target in {'DE', 'EN', 'FR'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Twitch', args.target)    
    target_dataset = TwitchDataset(path, args.target)

data = target_dataset[0]

args.num_classes = len(np.unique(data.y.numpy()))
args.num_features = data.x.size(1)

print(args)

model = Model(args).to(args.device)
data = data.to(args.device)

neighprop = NeighborPropagate()

model.load_state_dict(torch.load('model.pth'))

delta_feat = Parameter(torch.FloatTensor(data.x.size(0), data.x.size(1)).to(args.device))
delta_feat.data.fill_(1e-7)
optimizer_feat = torch.optim.Adam([delta_feat], lr=0.0001, weight_decay=0.0001)

modified_edge_index = data.edge_index.clone()
modified_edge_index = modified_edge_index[:, modified_edge_index[0] < modified_edge_index[1]]
row, col = modified_edge_index[0], modified_edge_index[1]
edge_index_id = (2 * data.x.size(0) - row - 1) * row // 2 + col - row - 1
edge_index_id = edge_index_id.long()
modified_edge_index = linear_to_triu_idx(data.x.size(0), edge_index_id)
perturbed_edge_weight = torch.full_like(edge_index_id, 1e-7, dtype=torch.float32, requires_grad=True).to(args.device)

optimizer_adj = torch.optim.Adam([perturbed_edge_weight], lr=0.0001, weight_decay=0.0001)

n_perturbations = int(args.ratio * data.edge_index.shape[1] // 2)

n = data.x.size(0)

def train_target(target_data, perturbed_edge_weight):
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t = time.time()
    edge_index = target_data.edge_index
    edge_weight = torch.ones(edge_index.shape[1]).to(args.device)
    feat = target_data.x

    mem_fea = torch.rand(target_data.x.size(0), args.nhid).to(args.device)    
    mem_cls = torch.ones(target_data.x.size(0), args.num_classes).to(args.device) / args.num_classes

    for it in tqdm(range(args.epochs//(args.loop_feat+args.loop_adj))):
        for loop_model in range(args.loop_model):
            for k,v in model.named_parameters():
                v.requires_grad = True
            model.train()
            feat = feat.detach()
            edge_weight = edge_weight.detach()
            
            optimizer_model.zero_grad()
            feat_output = model.feat_bottleneck(feat, edge_index, edge_weight)
            cls_output = model.feat_classifier(feat_output)

            onehot = torch.nn.functional.one_hot(cls_output.argmax(1), num_classes=args.num_classes).float()
            proto = (torch.mm(mem_fea.t(), onehot) / (onehot.sum(dim=0) + 1e-8)).t()
        
            prob = neighprop(mem_cls, edge_index)
            weight, pred = torch.max(prob, dim=1)
            cl, weight_ = instance_proto_alignment(feat_output, proto, pred)
            ce = F.cross_entropy(cls_output, pred, reduction='none')
            loss_local = torch.sum(weight_ * ce) / (torch.sum(weight_).item())
            loss = loss_local * (1 - args.lamb) + cl * args.lamb

            loss.backward()
            optimizer_model.step()
            print('Model: ' + str(loss.item()))

            model.eval()
            with torch.no_grad():
                feat_output = model.feat_bottleneck(feat, edge_index, edge_weight)
                cls_output = model.feat_classifier(feat_output)
                softmax_out = F.softmax(cls_output, dim=1)
                outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))
        
            mem_cls = (1.0 - args.momentum) * mem_cls + args.momentum * outputs_target.clone()
            mem_fea = (1.0 - args.momentum) * mem_fea + args.momentum * feat_output.clone()

        for k,v in model.named_parameters():
            v.requires_grad = False
        
        perturbed_edge_weight = perturbed_edge_weight.detach()
        for loop_feat in range(args.loop_feat):
            optimizer_feat.zero_grad()
            delta_feat.requires_grad = True
            loss = test_time_loss(model, target_data.x + delta_feat, edge_index, mem_fea, mem_cls, edge_weight)
            loss.backward()
            optimizer_feat.step()
            print('Feat: ' + str(loss.item()))
        
        new_feat = (data.x + delta_feat).detach()
        for loop_adj in range(args.loop_adj):
            perturbed_edge_weight.requires_grad = True
            edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, args.device, edge_index, edge_weight, args.make_undirected)
            loss = test_time_loss(model, new_feat, edge_index, mem_fea, mem_cls, edge_weight)
            print('Adj: ' + str(loss.item()))

            gradient = grad_with_checkpoint(loss, perturbed_edge_weight)[0]

            with torch.no_grad():
                update_edge_weights(gradient)
                perturbed_edge_weight = project(n_perturbations, perturbed_edge_weight, 1e-7)
        
        if args.loop_adj != 0:
            edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, args.device, edge_index, edge_weight, args.make_undirected)
            edge_weight = edge_weight.detach()
        
        if args.loop_feat != 0:
            feat = (target_data.x + delta_feat).detach()

    edge_index, edge_weight = sample_final_edges(n_perturbations, perturbed_edge_weight, target_data, modified_edge_index, mem_fea, mem_cls)

    test_acc, _ = evaluate(target_data.x + delta_feat, edge_index, edge_weight, target_data.y, model)
    print('acc : ' + str(test_acc))
    print('Optimization Finished!\n')


def instance_proto_alignment(feat, center, pred):
    feat_norm = F.normalize(feat, dim=1)
    center_norm = F.normalize(center, dim=1)
    sim = torch.matmul(feat_norm, center_norm.t())

    num_nodes = feat.size(0)
    weight = sim[range(num_nodes), pred]
    sim = torch.exp(sim / args.tau)
    pos_sim = sim[range(num_nodes), pred]

    sim_feat = torch.matmul(feat_norm, feat_norm.t())
    sim_feat = torch.exp(sim_feat / args.tau)
    ident = sim_feat[range(num_nodes), range(num_nodes)]

    logit = pos_sim / (sim.sum(dim=1) - pos_sim + sim_feat.sum(dim=1) - ident + 1e-8)
    loss = - torch.log(logit + 1e-8).mean()

    return loss, weight


def update_edge_weights(gradient):
    optimizer_adj.zero_grad()
    perturbed_edge_weight.grad = gradient
    optimizer_adj.step()
    perturbed_edge_weight.data[perturbed_edge_weight < 1e-7] = 1e-7


def test_time_loss(model, feat, edge_index, mem_fea, mem_cls, edge_weight=None):
    model.eval()
    feat_output = model.feat_bottleneck(feat, edge_index, edge_weight)
    cls_output = model.feat_classifier(feat_output)
    softmax_out = F.softmax(cls_output, dim=1)
    _, predict = torch.max(softmax_out, 1)
    mean_ent = Entropy(softmax_out)
    est_p = (mean_ent<mean_ent.mean()).sum().item() / mean_ent.size(0)
    value = mean_ent

    predict = predict.cpu().numpy()
    train_idx = np.zeros(predict.shape)

    cls_k = args.num_classes
    for c in range(cls_k):
        c_idx = np.where(predict==c)
        c_idx = c_idx[0]
        c_value = value[c_idx]

        _, idx_ = torch.sort(c_value)
        c_num = len(idx_)
        c_num_s = int(c_num * est_p / 5)

        for ei in range(0, c_num_s):
            ee = c_idx[idx_[ei]]
            train_idx[ee] = 1
                
    train_idx = np.array(train_idx, dtype=bool)
    pred_label = predict[train_idx]
    pseudo_label = torch.from_numpy(pred_label).to(args.device)

    pred_output = cls_output[train_idx]
    loss = F.cross_entropy(pred_output, pseudo_label)

    distance = feat_output @ mem_fea.T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
    idx_near = idx_near[:, 1:]  # batch x K
        
    mem_near = mem_fea[idx_near]  # batch x K x d
    feat_output_un = feat_output.unsqueeze(1).expand(-1, args.K, -1) # batch x K x d
    loss -= torch.mean((feat_output_un * mem_near).sum(-1).sum(1)/args.K) * 0.1

    _, pred_mem = torch.max(mem_cls, dim=1)
    _, pred = torch.max(softmax_out, dim=1)
    idx = pred.unsqueeze(-1) == pred_mem
    neg_num = torch.sum(~idx, dim=1)
    dis = (distance * ~idx).sum(1)/neg_num
    loss += dis.mean() * 0.1

    return loss


@torch.no_grad()
def sample_final_edges(n_perturbations, perturbed_edge_weight, data, modified_edge_index, mem_fea, mem_cls):
    best_loss = float('Inf')
    perturbed_edge_weight = perturbed_edge_weight.detach()
    # TODO: potentially convert to assert
    perturbed_edge_weight[perturbed_edge_weight <= 1e-7] = 0

    # _, feat, labels = self.data.edge_index, self.data.x, self.data.y
    feat = data.x.to(args.device)
    edge_index = data.edge_index.to(args.device)
    edge_weight = torch.ones(edge_index.shape[1]).to(args.device)
    # self.edge_index = data.graph['edge_index'].to(self.device)

    for i in range(20):
        if best_loss == float('Inf'):
            # In first iteration employ top k heuristic instead of sampling
            sampled_edges = torch.zeros_like(perturbed_edge_weight).to(args.device)
            sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
        else:
            sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

        if sampled_edges.sum() > n_perturbations:
            n_samples = sampled_edges.sum()
            if args.debug ==2:
                print(f'{i}-th sampling: too many samples {n_samples}')
            continue
        
        perturbed_edge_weight = sampled_edges

        edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, args.device, edge_index, edge_weight, args.make_undirected)
        with torch.no_grad():
            loss = test_time_loss(model, feat, edge_index, mem_fea, mem_cls, edge_weight)

        # Save best sample
        if best_loss > loss:
            best_loss = loss
            print('best_loss:', best_loss.item())
            best_edges = perturbed_edge_weight.clone().cpu()

    # Recover best sample
    perturbed_edge_weight.data.copy_(best_edges.to(args.device))

    edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, args.device, edge_index, edge_weight, args.make_undirected)
    edge_mask = edge_weight == 1
    make_undirected = args.make_undirected

    allowed_perturbations = 2 * n_perturbations if make_undirected else n_perturbations
    edges_after_attack = edge_mask.sum()
    clean_edges = edge_index.shape[1]
    assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
    
    return edge_index[:, edge_mask], edge_weight[edge_mask]

if __name__ == '__main__':
    train_target(data, perturbed_edge_weight)
