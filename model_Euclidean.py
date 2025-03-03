import torch
import torch.nn as nn, torch.nn.functional as F
from dgl.nn.pytorch import TypedLinear
import config
import math
from torch_geometric.nn import LayerNorm
from torch_scatter import scatter
from torch_geometric.utils import softmax
args = config.parse()
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:'+str(eval(args.cuda)+1) if torch.cuda.is_available() else 'cpu')

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class hhgnnConv(nn.Module):
    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=False,device=device):
        super().__init__()
        # self.W = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.type_w=TypedLinear(in_channels,heads * out_channels,num_types=4)
        self.att_v_user=nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_poi = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_class = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_time = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.att_e_friend = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_visit = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_occurrence = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_self = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.edge_num=args.edge_num
        self.layer_norm=args.layer_norm
        self.norm = LayerNorm(heads * out_channels, affine=True, mode='node')
        self.reset_parameters()
        self.edge_type=args.edge_type
        self.node_type=args.node_type
        self.edge_input_length = args.edge_input_length
        self.node_input_length = args.node_input_length

        self.V_raw_index_type= (args.V_raw_index_type).to(device)

        self.V_class=(args.V_class).to(device)
        self.E_class=(args.E_class).to(device)
        self.V_class_index=(args.V_class_index).to(device)
        self.E_class_index=(args.E_class_index).to(device)

        self.V_class_index_0 =(args.V_class_index_0).to(device)
        self.V_class_index_1 =(args.V_class_index_1).to(device)
        self.V_class_index_2 =(args.V_class_index_2).to(device)
        self.V_class_index_3 =(args.V_class_index_3).to(device)
        self.E_class_index_0 =(args.E_class_index_0).to(device)
        self.E_class_index_1 =(args.E_class_index_1).to(device)
        self.E_class_index_2 =(args.E_class_index_2).to(device)
        self.E_class_index_3 =(args.E_class_index_3).to(device)

        self.relu = nn.ReLU()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels, self.out_channels, self.heads)
    def reset_parameters(self):
        glorot(self.att_v_user)
        glorot(self.att_v_poi)
        glorot(self.att_v_class)
        glorot(self.att_v_time)
        glorot(self.att_e_friend)
        glorot(self.att_e_visit)
        glorot(self.att_e_occurrence)
        glorot(self.att_e_self)

    def forward(self, X, vertex, edges):
        '''X represents node features
        # vertex is V  # V is the node index, E is the hyperedge index the node belongs to.
        # Both are (,) and in total, all nodes in the hyperedges sum to:
        # edge is E, which represents the hyperedge index the node belongs to.
        '''

        H, C, N = self.heads, self.out_channels, X.shape[0]
        # X0 = self.W(X) # First apply a linear transformation
        X0 = self.type_w(X, self.V_raw_index_type)  # Apply different transformation matrices for each node in real time.
        # This does not significantly affect memory usage.


        X = X0.view(N, H, C)  # Convert to multi-head representation, X still represents node features.
        Xve = X[vertex]  # Expand indexing based on V
        X = Xve  # Assign a value since it will be used later

        X_e_0 = (torch.index_select(X,0,self.E_class_index_0)*self.att_e_friend).sum(-1)
        X_e_1 = (torch.index_select(X, 0, self.E_class_index_1) * self.att_e_visit).sum(-1)
        X_e_2 = (torch.index_select(X, 0, self.E_class_index_2) * self.att_e_occurrence).sum(-1)
        X_e_3 = (torch.index_select(X, 0, self.E_class_index_3) * self.att_e_self).sum(-1)
        X_e   = torch.cat((X_e_0,X_e_1,X_e_2,X_e_3),0)
        beta_v= torch.gather(X_e,0, self.E_class_index)
        beta =self.leaky_relu(beta_v)
        beta=softmax(beta,edges,num_nodes=self.edge_num)
        beta=beta.unsqueeze(-1)
        Xe=Xve*beta
        # Compute hyperedge embeddings Xe
        Xe = self.relu(scatter(Xe, edges, dim=0, reduce='sum', dim_size=self.edge_num))  # All obtained hyperedge embeddings
        # Xe represents hyperedge embeddings
        # Next, propagate from hyperedges to nodes

        Xe=Xe[edges]
        Xe_2=Xe
        Xe_2_0= (torch.index_select(Xe_2,0,self.V_class_index_0)*self.att_v_user).sum(-1)
        Xe_2_1 = (torch.index_select(Xe_2, 0, self.V_class_index_1) * self.att_v_poi).sum(-1)
        Xe_2_2 = (torch.index_select(Xe_2, 0, self.V_class_index_2) * self.att_v_class).sum(-1)
        Xe_2_3 = (torch.index_select(Xe_2, 0, self.V_class_index_3) * self.att_v_time).sum(-1)
        Xe_2 = torch.cat((Xe_2_0, Xe_2_1, Xe_2_2, Xe_2_3), 0)
        alpha_e=torch.gather(Xe_2,0, self.V_class_index)
        alpha = self.leaky_relu(alpha_e)
        alpha = softmax(alpha, vertex, num_nodes = N)
        alpha = alpha.unsqueeze(-1)
        Xev = Xe * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)  # [N, H, C]
        Xv=Xv.view(N, H * C)
        Xv=self.relu(Xv)

        if self.layer_norm:
            Xv=self.norm(Xv)

        return Xv

class HHGNN_multi(nn.Module):
    def __init__(self, args, nfeat, nhid, out_dim,  nhead, V, E,  node_input_dim,edge_type,node_type):
        super().__init__()
        self.conv_out = hhgnnConv(args, nhid * nhead, out_dim, heads=args.out_nhead,device=device2).to(device2)
        self.conv_in = hhgnnConv(args, nfeat, nhid, heads=nhead, device=device).to(device)
        self.V = V
        self.E = E
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu=nn.ReLU().to(device2)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear( out_dim*args.out_nhead,out_dim,bias=True).to(device2)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim]).to(device)
    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]]=self.relu(self.fc_list_node[i](node_attr[self.node_type[i]].to(device)  ))
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat((X), 0).to(device)
        V, E = (self.V).to(device) , (self.E).to(device)
        X=self.conv_in(X, V, E)
        X = self.conv_out(X.to(device2), V.to(device2), E.to(device2))
        X = self.relu(X)
        X=self.lin_out1(X)
        return  X.to(device)

class HHGNN_Euclidean(nn.Module):
    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type):

        super().__init__()
        self.conv_out = hhgnnConv(args, nhid * nhead, nhid, heads=args.out_nhead,device=device)
        self.conv_in = hhgnnConv(args, nfeat, nhid, heads=nhead, device=device)
        self.V = V.to(device)
        self.E = E.to(device)
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu=nn.ReLU().to(device2)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear( nhid *args.out_nhead,out_dim,bias=True)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim])

        self.user_number=args.user_number

    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]]=self.relu(self.fc_list_node[i](node_attr[self.node_type[i]] ))
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat((X), 0)
        V, E = (self.V) , (self.E)
        X=self.conv_in(X, V, E)
        X = self.conv_out(X, V, E)
        X=self.lin_out1(X)
        c=1 # to unify the output structure
        return  X, c
