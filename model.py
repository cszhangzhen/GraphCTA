import torch
import torch.nn.functional as F
from layer import *


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        if args.gnn == 'gcn':
            self.gnn = GCN(args)
        elif args.gnn == 'sage':
            self.gnn = SAGE(args)
        elif args.gnn == 'gat':
            self.gnn = GAT(args)
        elif args.gnn == 'gin':
            self.gnn = GIN(args)
        else:
            assert args.gnn in ('gcn', 'sage', 'gat', 'gin'), 'Invalid gnn'
                
        self.reset_parameters()
    
    def reset_parameters(self):
        self.gnn.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x)
        
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        x = self.gnn.feat_bottleneck(x, edge_index, edge_weight)
        return x
    
    def feat_classifier(self, x):
        x = self.gnn.feat_classifier(x)
        
        return x
