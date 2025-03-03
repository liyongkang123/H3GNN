'''
Various models required for the ablation study in this experiment.
'''

from manifold.PoincareManifold import PoincareManifold
import torch.nn.init as init
import torch
import torch.nn as nn, torch.nn.functional as F
from dgl.nn.pytorch import TypedLinear
import config
import math
from torch_scatter import scatter
from torch_geometric.utils import softmax
args = config.parse()

device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')

if args.multi_cuda==0:
    device2=device
else:
    device2 = torch.device('cuda:' + str(eval(args.cuda) + 1) if torch.cuda.is_available() else 'cpu')

EPS= 1e-15
clip_value=0.9899

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

class HypLinear(nn.Module):
    """
    Hyperbolic Poincare linear layer.
    """

    def __init__(self, args, in_features, out_features, c=1, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = PoincareManifold(args)
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.exp_map_zero(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class hhgnnConv_eu_without_hetero(nn.Module): # Transformation based on Euclidean space with the heterogeneous part removed

    def __init__(self, args, in_channels, out_channels,c, device, heads=8,  dropout=0., negative_slope=0.2, skip_sum=False,  ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.device = device
        self.type_w=TypedLinear(in_channels,heads * out_channels,num_types=4).to(self.device)

        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))


        self.att_v_user=nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_poi = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_class = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_time = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.att_ceshi = nn.Parameter(torch.Tensor( heads, out_channels))

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
        self.node_number=args.node_number
        self.edge_to_v =args.edge_to_v
        self.node_to_edge=args.node_to_edge

        self.layer_norm=args.layer_norm
        self.edge_type=args.edge_type
        self.node_type=args.node_type
        self.edge_input_length = args.edge_input_length
        self.node_input_length = args.node_input_length
        self.H_dense=args.H_dense

        self.V_raw_index_type= (args.V_raw_index_type).to(self.device)

        self.V_class=(args.V_class).to(self.device)
        self.E_class=(args.E_class).to(self.device)
        self.V_class_index=(args.V_class_index).to(self.device)
        self.E_class_index=(args.E_class_index).to(self.device)

        self.V_class_index_0 =(args.V_class_index_0).to(self.device)
        self.V_class_index_1 =(args.V_class_index_1).to(self.device)
        self.V_class_index_2 =(args.V_class_index_2).to(self.device)
        self.V_class_index_3 =(args.V_class_index_3).to(self.device)
        self.E_class_index_0 =(args.E_class_index_0).to(self.device)
        self.E_class_index_1 =(args.E_class_index_1).to(self.device)
        self.E_class_index_2 =(args.E_class_index_2).to(self.device)
        self.E_class_index_3 =(args.E_class_index_3).to(self.device)

        self.relu = nn.ReLU()

        self.c_i = c  # Keep the curvature consistent with the previous setting
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)  # Coefficient beta
        self.beta_c = torch.nn.Parameter(torch.tensor(0, dtype=torch.float), requires_grad=True)  # Coefficient c


        self.reset_parameters()

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
        glorot(self.att_e)
        glorot(self.att_v)

    def forward(self, X, vertex, edges):

        H, C, N = self.heads, self.out_channels, X.shape[0]
        X0 = self.W(X)

        X = X0.view(N, H, C)
        Xve = X[vertex]
        X=Xve

        X_e_0 = (torch.index_select(X,0,self.E_class_index_0) * self.att_e).sum(-1)
        X_e_1 = (torch.index_select(X, 0, self.E_class_index_1) * self.att_e).sum(-1)
        X_e_2 = (torch.index_select(X, 0, self.E_class_index_2) * self.att_e).sum(-1)
        X_e_3 = (torch.index_select(X, 0, self.E_class_index_3) * self.att_e).sum(-1)
        X_e   = torch.cat((X_e_0,X_e_1,X_e_2,X_e_3),0)
        beta_v = torch.gather(X_e,0, self.E_class_index)

        beta =self.leaky_relu(beta_v)
        beta =softmax(beta,edges,num_nodes=self.edge_num)
        beta =beta.unsqueeze(-1)
        Xe = Xve*beta
        Xe = self.relu( scatter(Xe,edges,dim=0,reduce='sum',dim_size=self.edge_num))
        Xe=Xe[edges]
        Xe_2=Xe

        Xe_2_0= (torch.index_select(Xe_2,0,self.V_class_index_0)*self.att_v).sum(-1) # Multiply everything by the same att_v, which effectively removes heterogeneous information.
        Xe_2_1 = (torch.index_select(Xe_2, 0, self.V_class_index_1) * self.att_v).sum(-1)
        Xe_2_2 = (torch.index_select(Xe_2, 0, self.V_class_index_2) * self.att_v).sum(-1)
        Xe_2_3 = (torch.index_select(Xe_2, 0, self.V_class_index_3) * self.att_v).sum(-1)
        Xe_2 = torch.cat((Xe_2_0, Xe_2_1, Xe_2_2, Xe_2_3), 0)
        alpha_e=torch.gather(Xe_2,0, self.V_class_index)

        alpha = self.leaky_relu(alpha_e)
        alpha = softmax(alpha, vertex, num_nodes = N)
        alpha = alpha.unsqueeze(-1)
        Xev = Xe * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)  # [N, H, C]
        Xv=Xv.view(N, H * C)
        if self.layer_norm:
            Xv=self.norm(Xv)

        return Xv

class HHGNN_Poincare_without_hetero(nn.Module):
    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type,device_type=device):

        super().__init__()
        self.device = device_type
        self.V = V.to(device)
        self.E = E.to(device)
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu = nn.ReLU() #.to(device2)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear( nhid *args.out_nhead,out_dim,bias=True)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim])

        self.user_number = args.user_number
        self.c = torch.nn.Parameter(torch.tensor(1.,dtype=torch.float), requires_grad=True)

        self.conv_out = hhgnnConv_eu_without_hetero( args, nhid * nhead, nhid, c=self.c, heads=args.out_nhead,device=device,)
        self.conv_in = hhgnnConv_eu_without_hetero( args, nfeat, nhid, c =self.c, heads=nhead, device=device)

        self.manifold = PoincareManifold(args)
        self.r = 2
        self.t = 1
        self.tanh = nn.Tanh()
        self.linear_first = nn.ModuleList( [HypLinear(args, feats_dim, nfeat)   for feats_dim in node_input_dim])


    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            x_tan = self.manifold.proj_tan0( node_attr[self.node_type[i]])
            x_hyp = self.manifold.exp_map_zero(x_tan)
            x_hyp = self.manifold.proj(x_hyp)
            node_feat[self.node_type[i]] = self.linear_first[i](x_hyp)
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat(X, dim=0)
        V, E = (self.V) , (self.E)


        X=self.manifold.log_map_zero(X)
        X=self.manifold.proj_tan0(X)
        X=self.conv_in(X, V, E)
        X=self.manifold.exp_map_zero(X)
        X = self.manifold.proj(X)
        X=self.relu(X)

        X = self.manifold.log_map_zero(X)
        X = self.manifold.proj_tan0(X)
        X = self.conv_out(X, V, E)
        X=self.manifold.exp_map_zero(X)
        X = self.manifold.proj(X)
        X=self.relu(X)

        X = self.manifold.log_map_zero(X)
        X=self.lin_out1(X)
        return  X, self.c


class HHGNN_Poincare_multi_without_hetero(nn.Module):
    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type,device_type=device):

        super().__init__()
        self.device=device_type
        self.V = V.to(self.device)
        self.E = E.to(self.device)
        self.V_2 = V.to(device2)
        self.E_2 = E.to(device2)

        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu = nn.ReLU() #.to(device2)

        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim])

        self.user_number = args.user_number
        self.c = torch.nn.Parameter(torch.tensor(1.,dtype=torch.float), requires_grad=True)

        self.conv_out = hhgnnConv_eu_without_hetero( args, nhid * nhead, nhid, c=self.c, heads=args.out_nhead,device=device2).to(device2)
        self.conv_in = hhgnnConv_eu_without_hetero( args, nfeat, nhid, c =self.c, heads=nhead, device=device).to(device)
        self.lin_out1 = nn.Linear(nhid * args.out_nhead, out_dim, bias=True).to(device2)
        self.manifold_device1 = PoincareManifold(args)
        self.manifold_device2 = PoincareManifold(args)
        self.r = 2
        self.t = 1
        self.tanh = nn.Tanh()
        self.linear_first = nn.ModuleList( [HypLinear(args, feats_dim, nfeat)  for feats_dim in node_input_dim]).to(device)


    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            x_tan = self.manifold_device1.proj_tan0( node_attr[self.node_type[i]])
            x_hyp = self.manifold_device1.exp_map_zero(x_tan)
            x_hyp = self.manifold_device1.proj(x_hyp)
            node_feat[self.node_type[i]] = self.linear_first[i](x_hyp)
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat(X, dim=0)


        X=self.manifold_device1.log_map_zero(X)
        X=self.manifold_device1.proj_tan0(X)
        X=self.conv_in(X, self.V, self.E)
        X=self.manifold_device1.exp_map_zero(X)
        X = self.manifold_device1.proj(X)
        X = self.relu(X).to(device2)

        # X=X.to(device2)
        X = self.manifold_device2.log_map_zero(X)
        X = self.manifold_device2.proj_tan0(X)
        X = self.conv_out(X, self.V_2, self.E_2)
        X=self.manifold_device2.exp_map_zero(X)
        X = self.manifold_device2.proj(X)
        X = self.relu(X)

        X = self.manifold_device2.log_map_zero(X)
        X=self.lin_out1(X)
        return  X, self.c

'''
The above section implements the functionality of removing heterogeneous information.
The following section implements the functionality of removing the hyperbolic space.
'''


class hhgnnConv_eu(nn.Module): # Transformation based on Euclidean space, the same as the normal version with no changes.


    def __init__(self, args, in_channels, out_channels,c, device, heads=8,  dropout=0., negative_slope=0.2, skip_sum=False,  ):
        super().__init__()
        self.device = device
        self.type_w=TypedLinear(in_channels,heads * out_channels,num_types=4).to(self.device)

        self.att_v_user=nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_poi = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_class = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_time = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.att_ceshi = nn.Parameter(torch.Tensor( heads, out_channels))

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
        self.node_number=args.node_number
        self.edge_to_v =args.edge_to_v
        self.node_to_edge=args.node_to_edge

        self.layer_norm=args.layer_norm

        self.edge_type=args.edge_type
        self.node_type=args.node_type
        self.edge_input_length = args.edge_input_length
        self.node_input_length = args.node_input_length
        self.H_dense=args.H_dense

        self.V_raw_index_type= (args.V_raw_index_type).to(self.device)

        self.V_class=(args.V_class).to(self.device)
        self.E_class=(args.E_class).to(self.device)
        self.V_class_index=(args.V_class_index).to(self.device)
        self.E_class_index=(args.E_class_index).to(self.device)

        self.V_class_index_0 =(args.V_class_index_0).to(self.device)
        self.V_class_index_1 =(args.V_class_index_1).to(self.device)
        self.V_class_index_2 =(args.V_class_index_2).to(self.device)
        self.V_class_index_3 =(args.V_class_index_3).to(self.device)
        self.E_class_index_0 =(args.E_class_index_0).to(self.device)
        self.E_class_index_1 =(args.E_class_index_1).to(self.device)
        self.E_class_index_2 =(args.E_class_index_2).to(self.device)
        self.E_class_index_3 =(args.E_class_index_3).to(self.device)

        self.relu = nn.ReLU()

        self.c_i = c  # Keep the curvature consistent with the previous setting
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)  # Coefficient beta
        self.beta_c = torch.nn.Parameter(torch.tensor(0, dtype=torch.float), requires_grad=True)  # Coefficient c


        self.reset_parameters()

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

        H, C, N = self.heads, self.out_channels, X.shape[0]

        X0 = self.type_w( X, self.V_raw_index_type)

        X = X0.view(N, H, C)
        Xve = X[vertex]
        X=Xve

        X_e_0 = (torch.index_select(X,0,self.E_class_index_0) * self.att_e_friend).sum(-1)
        X_e_1 = (torch.index_select(X, 0, self.E_class_index_1) * self.att_e_visit).sum(-1)
        X_e_2 = (torch.index_select(X, 0, self.E_class_index_2) * self.att_e_occurrence).sum(-1)
        X_e_3 = (torch.index_select(X, 0, self.E_class_index_3) * self.att_e_self).sum(-1)
        X_e   = torch.cat((X_e_0,X_e_1,X_e_2,X_e_3),0)
        beta_v = torch.gather(X_e,0, self.E_class_index)
        beta =self.leaky_relu(beta_v)
        beta =softmax(beta,edges,num_nodes=self.edge_num)
        beta =beta.unsqueeze(-1)
        Xe = Xve*beta
        Xe = self.relu( scatter(Xe,edges,dim=0,reduce='sum',dim_size=self.edge_num))

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

        if self.layer_norm:
            Xv=self.norm(Xv)

        return Xv

class HHGNN_Poincare_without_hyperbolic(nn.Module):
    '''
    Removed the hyperbolic space while retaining heterogeneity.
    '''

    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type,device_type=device):

        super().__init__()
        self.device = device_type
        self.V = V.to(device)
        self.E = E.to(device)
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu = nn.ReLU() #.to(device2)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear( nhid *args.out_nhead,out_dim,bias=True)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim])

        self.user_number = args.user_number
        self.c = torch.nn.Parameter(torch.tensor(1.,dtype=torch.float), requires_grad=True)

        self.conv_out = hhgnnConv_eu( args, nhid * nhead, nhid, c=self.c, heads=args.out_nhead,device=device,)
        self.conv_in = hhgnnConv_eu( args, nfeat, nhid, c =self.c, heads=nhead, device=device)

        self.manifold = PoincareManifold(args)
        self.r = 2
        self.t = 1
        self.tanh = nn.Tanh()
        self.linear_first = nn.ModuleList( [HypLinear(args, feats_dim, nfeat)   for feats_dim in node_input_dim])


    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]] = self.fc_list_node[i](node_attr[self.node_type[i]])  #
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat(X, dim=0)
        V, E = (self.V) , (self.E)



        X=self.conv_in(X, V, E)
        X=self.relu(X)

        X = self.conv_out(X, V, E)
        X=self.relu(X)

        X=self.lin_out1(X)
        return  X, self.c


class hhgnnConv_eu_multi(nn.Module):
    def __init__(self, args, in_channels, out_channels,c, device, heads=8,  dropout=0., negative_slope=0.2, skip_sum=False,  ):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.device = device

        self.att_v_user=nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_poi = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_class = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_time = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.att_ceshi = nn.Parameter(torch.Tensor( heads, out_channels))

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
        self.node_number=args.node_number
        self.edge_to_v =args.edge_to_v
        self.node_to_edge=args.node_to_edge

        self.layer_norm=args.layer_norm

        self.edge_type=args.edge_type
        self.node_type=args.node_type
        self.edge_input_length = args.edge_input_length
        self.node_input_length = args.node_input_length
        self.H_dense=args.H_dense

        self.V_raw_index_type= (args.V_raw_index_type).to(self.device)

        self.V_class=(args.V_class).to(self.device)
        self.E_class=(args.E_class).to(self.device)
        self.V_class_index=(args.V_class_index).to(self.device)
        self.E_class_index=(args.E_class_index).to(self.device)

        self.V_class_index_0 =(args.V_class_index_0).to(self.device)
        self.V_class_index_1 =(args.V_class_index_1).to(self.device)
        self.V_class_index_2 =(args.V_class_index_2).to(self.device)
        self.V_class_index_3 =(args.V_class_index_3).to(self.device)
        self.E_class_index_0 =(args.E_class_index_0).to(self.device)
        self.E_class_index_1 =(args.E_class_index_1).to(self.device)
        self.E_class_index_2 =(args.E_class_index_2).to(self.device)
        self.E_class_index_3 =(args.E_class_index_3).to(self.device)

        self.relu = nn.ReLU()

        self.c_i = c  # Keep the curvature consistent with the previous setting
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)  # Coefficient beta
        self.beta_c = torch.nn.Parameter(torch.tensor(0, dtype=torch.float), requires_grad=True)  # Coefficient c


        self.reset_parameters()

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


        H, C, N = self.heads, self.out_channels, X.shape[0]

        X0 = self.W(X)


        X = X0.view(N, H, C)
        Xve = X[vertex]
        X=Xve

        X_e_0 = (torch.index_select(X,0,self.E_class_index_0) * self.att_e_friend).sum(-1)
        X_e_1 = (torch.index_select(X, 0, self.E_class_index_1) * self.att_e_visit).sum(-1)
        X_e_2 = (torch.index_select(X, 0, self.E_class_index_2) * self.att_e_occurrence).sum(-1)
        X_e_3 = (torch.index_select(X, 0, self.E_class_index_3) * self.att_e_self).sum(-1)
        X_e   = torch.cat((X_e_0,X_e_1,X_e_2,X_e_3),0)
        beta_v = torch.gather(X_e,0, self.E_class_index)
        beta =self.leaky_relu(beta_v)
        beta =softmax(beta,edges,num_nodes=self.edge_num)
        beta =beta.unsqueeze(-1)
        Xe = Xve*beta
        Xe = self.relu( scatter(Xe,edges,dim=0,reduce='sum',dim_size=self.edge_num))

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

        if self.layer_norm:
            Xv=self.norm(Xv)

        return Xv

class HHGNN_Poincare_multi_without_hyperbolic(nn.Module):
    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type,device_type=device):

        super().__init__()
        self.device=device_type
        self.V = V.to(self.device)
        self.E = E.to(self.device)
        self.V_2 = V.to(device2)
        self.E_2 = E.to(device2)

        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu = nn.ReLU() #.to(device2)

        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim]).to(device)

        self.user_number = args.user_number
        self.c = torch.nn.Parameter(torch.tensor(1.,dtype=torch.float), requires_grad=True)

        self.conv_out = hhgnnConv_eu_multi( args, nhid * nhead, nhid, c=self.c, heads=args.out_nhead,device=device2).to(device2)
        self.conv_in = hhgnnConv_eu_multi( args, nfeat, nhid, c =self.c, heads=nhead, device=device).to(device)
        self.lin_out1 = nn.Linear(nhid * args.out_nhead, out_dim, bias=True).to(device2)
        self.manifold_device1 = PoincareManifold(args)
        self.manifold_device2 = PoincareManifold(args)
        self.r = 2
        self.t = 1
        self.tanh = nn.Tanh()
        self.linear_first = nn.ModuleList( [HypLinear(args, feats_dim, nfeat)  for feats_dim in node_input_dim]).to(device)


    def forward(self,  node_attr):

        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]] = self.fc_list_node[i](node_attr[self.node_type[i]])  #
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat(X, dim=0)

        X=self.conv_in(X, self.V, self.E)
        X = self.relu(X).to(device2)

        X = self.conv_out(X, self.V_2, self.E_2)
        X = self.relu(X)

        X=self.lin_out1(X)
        return  X, self.c




class HHGNN_Poincare_without_hyperbolic_and_hetero(nn.Module):
    '''
    Removed both the hyperbolic space and heterogeneity.
    '''
    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type,device_type=device):

        super().__init__()
        self.device = device_type
        self.V = V.to(device)
        self.E = E.to(device)
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu = nn.ReLU() #.to(device2)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear( nhid *args.out_nhead,out_dim,bias=True)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim])

        self.user_number = args.user_number
        self.c = torch.nn.Parameter(torch.tensor(1.,dtype=torch.float), requires_grad=True)
        '''
        The main change here is replacing the layer with a non-heterogeneous version.
        '''
        self.conv_out = hhgnnConv_eu_without_hetero( args, nhid * nhead, nhid, c=self.c, heads=args.out_nhead,device=device,)
        self.conv_in = hhgnnConv_eu_without_hetero( args, nfeat, nhid, c =self.c, heads=nhead, device=device)

        self.manifold = PoincareManifold(args)
        self.r = 2
        self.t = 1
        self.tanh = nn.Tanh()
        self.linear_first = nn.ModuleList( [HypLinear(args, feats_dim, nfeat)   for feats_dim in node_input_dim])


    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]] = self.fc_list_node[i](node_attr[self.node_type[i]])  #
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat(X, dim=0)
        V, E = (self.V) , (self.E)


        X=self.conv_in(X, V, E)
        X=self.relu(X)


        X = self.conv_out(X, V, E)
        X=self.relu(X)

        X=self.lin_out1(X)
        return  X, self.c

class HHGNN_Poincare_multi_without_hyperbolic_and_hetero(nn.Module):
    '''
    Dual-GPU model with both heterogeneity and hyperbolic space removed.
    '''

    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type,device_type=device):

        super().__init__()
        self.device=device_type
        self.V = V.to(self.device)
        self.E = E.to(self.device)
        self.V_2 = V.to(device2)
        self.E_2 = E.to(device2)

        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu = nn.ReLU() #.to(device2)

        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim]).to(device)
        self.user_number = args.user_number
        self.c = torch.nn.Parameter(torch.tensor(1.,dtype=torch.float), requires_grad=True)

        self.conv_out = hhgnnConv_eu_without_hetero( args, nhid * nhead, nhid, c=self.c, heads=args.out_nhead,device=device2).to(device2)
        self.conv_in = hhgnnConv_eu_without_hetero( args, nfeat, nhid, c =self.c, heads=nhead, device=device).to(device)
        self.lin_out1 = nn.Linear(nhid * args.out_nhead, out_dim, bias=True).to(device2)
        self.manifold_device1 = PoincareManifold(args)
        self.manifold_device2 = PoincareManifold(args)
        self.r = 2
        self.t = 1
        self.tanh = nn.Tanh()
        self.linear_first = nn.ModuleList( [HypLinear(args, feats_dim, nfeat)  for feats_dim in node_input_dim]).to(device)


    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]] = self.fc_list_node[i](node_attr[self.node_type[i]])  #
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat(X, dim=0)


        X=self.conv_in(X, self.V, self.E)
        X = self.relu(X).to(device2)


        X = self.conv_out(X, self.V_2, self.E_2)
        X = self.relu(X)


        X=self.lin_out1(X)
        return  X, self.c

'''
Through the six models above, we have implemented both single-GPU and multi-GPU versions of:
1. Removing heterogeneity
2. Removing hyperbolic space
3. A combination of both (removing both heterogeneity and hyperbolic space)

6 = 2 * 3 (two computation modes and three configurations).
'''
