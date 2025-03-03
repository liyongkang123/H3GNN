import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import torch_sparse
import pickle
import config
from model_Euclidean import HHGNN_Euclidean
from model_Poincare import HHGNN_Poincare,HHGNN_Poincare_multi
from ablation_study_models import HHGNN_Poincare_without_hetero,HHGNN_Poincare_multi_without_hetero, HHGNN_Poincare_without_hyperbolic,HHGNN_Poincare_multi_without_hyperbolic
from ablation_study_models import HHGNN_Poincare_without_hyperbolic_and_hetero,HHGNN_Poincare_multi_without_hyperbolic_and_hetero

import random

from torch_scatter import scatter

args = config.parse()
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')


# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def accuracy(Z, Y):
    return 100 * Z.argmax(1).eq(Y).float().mean().item()

def fetch_data(args):
    city=args.city
    print('city name: ',city)
    read_friend=open(  city+"/friend_list_index.pkl",'rb' )
    friend_edge=pickle.load(read_friend)
    friend_edge_num=len(friend_edge)
    args.friend_edge_num=friend_edge_num
    print("the number of friendship hyperedge in raw dataset is:", friend_edge_num)
    print("the number of friendship hyperedge used for training is:", round(friend_edge_num *args.split))

    visit_poi=open(  city+"/visit_list_edge_tensor.pkl",'rb' )
    visit_edge=pickle.load(visit_poi)
    visit_edge_num=len(visit_edge)
    args.visit_edge_num=visit_edge_num
    print("the number of check-in hyperedge is:", visit_edge_num)

    tra=open( city+"/trajectory_list_index.pkl",'rb')
    trajectory_edge=pickle.load(tra)
    trajectory_edge_num=len(trajectory_edge)
    args.trajectory_edge_num=trajectory_edge_num
    print("the number of trajectory hyperedge is:", trajectory_edge_num)

    user_number=args.user_number
    poi_number=args.poi_number
    poi_class_number=args.poi_class_number
    time_point_number=args.time_point_number
    user_node_attr = torch.tensor(np.random.randint(0,10,(user_number,args.input_dim)) , dtype=torch.float32)
    poi_node_attr=torch.tensor(np.random.randint(0,10,(poi_number, args.input_dim)) , dtype=torch.float32)

    poi_class_attr=torch.zeros(poi_class_number,poi_class_number)
    index=range(0,poi_class_number,1)
    index=torch.LongTensor( index).view(-1,1)
    poi_class_attr=poi_class_attr.scatter_(dim=1,index=index,value=1)

    time_point_attr=torch.zeros(time_point_number,time_point_number)
    index2=range(0,time_point_number,1)
    index2 = torch.LongTensor(index2).view(-1, 1)
    time_point_attr=time_point_attr.scatter_(dim=1,index=index2,value=1)

    node_attr={}
    node_attr['user']=user_node_attr
    node_attr['poi']=poi_node_attr
    node_attr['poi_class']=poi_class_attr
    node_attr['time_point'] = time_point_attr


    train_rate=args.split
    friend_edge_train_len  =round(friend_edge_num *train_rate)
    all_index = list(np.arange(0, friend_edge_num, 1))
    train_edge_index = sorted(random.sample(all_index, friend_edge_train_len))
    test_edge_index_true = sorted(list(set(all_index).difference(set(train_edge_index))))

    friend_edge_train={}
    for i in range(len(train_edge_index)):
        friend_edge_train[i]=friend_edge[train_edge_index[i]]


    friend_edge_test=[]
    for i in range(len(test_edge_index_true)):
        friend_edge_test.append(list(friend_edge[ test_edge_index_true[i]]))

    for i in range(len(test_edge_index_true)):
        friend_edge_test.append([list(friend_edge[ test_edge_index_true[i]])[0], random.randint(0, user_number-1) ])

    test_label=[]
    for i in range(2* len(test_edge_index_true)):
        if i <len(test_edge_index_true):
            test_label.append(1)
        else:
            test_label.append(0)
    test_label=np.array(test_label)
    friend_edge_test=(torch.tensor(friend_edge_test,dtype=torch.long)).t().contiguous()
    K=args.negative_K
    friend_edge_train_all=[]
    for i in range(len(friend_edge_train)):
        friend_edge_train_all.append(list(friend_edge_train[i]))

    for i in range (len(friend_edge_train)):
        for j in range(K):
            friend_edge_train_all.append( [ list(friend_edge_train[i])[0],  random.randint(0, user_number-1)])

    friend_edge_train_all =(torch.tensor(np.array(friend_edge_train_all),dtype=torch.long)).t().contiguous()

    friend_edge_train_all_label=[]
    for i in range(len(friend_edge_train)):
        friend_edge_train_all_label.append(1)
    for i in range (len(friend_edge_train)):
        for j in range(K):
            friend_edge_train_all_label.append(0)
    friend_edge_train_all_label=torch.tensor(np.array(friend_edge_train_all_label),dtype=torch.long)

    G={}
    G['friend']=friend_edge_train
    G['check_in'] = visit_edge
    G['trajectory'] = trajectory_edge

    print("There are", user_number, "user nodes in this hypergraph.")
    print("There are", poi_number, "POI nodes in this hypergraph.")
    print("There are", poi_class_number, "POI category nodes in this hypergraph.")
    print("There are", time_point_number, "time-day nodes in this hypergraph.")


    return   G, node_attr,friend_edge_train, friend_edge_test, test_label,friend_edge_train_all, friend_edge_train_all_label,len(friend_edge_train)

def initialise( G,node_attr , args, node_type,edge_type, unseen=None):

    G2={}
    z=0
    for i in edge_type:
        for j in range(len(G[i])):
            G2[z]=G[i][j]
            z+=1
    print("There are", len(G2), "original hyperedges in this hypergraph.")

    G=G2.copy()
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)

    node_number= args.user_number+ args.poi_number+ args.poi_class_number+ args.time_point_number
    if args.add_self_loop:
        Vs = set(range(node_number))
        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and list(nodes)[0] in Vs:
                Vs.remove(list(nodes)[0])
        for v in Vs:
            G[f'self-loop-{v}'] = [v]

    print("There are", len(G), "hyperedges in this hypergraph.")
    print("Among them,", len(G) - len(G2), "are added self-loop hyperedges.")


    args.self_loop_edge_number=len(G)-len(G2)
    edge_type.append('self-loop')
    N, M = node_number, len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()
    H_dense=H.todense()
    H_dense=H_dense.transpose()
    H_dense=torch.tensor(H_dense,dtype=torch.long)
    H_dense=H_dense.to_sparse_coo()
    args.H_dense=H_dense.to(device)
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()
    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col

    degE = scatter(degV[V], E, dim=0, reduce='sum')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge


    args.edge_num=max(E)+1
    args.node_number=node_number
    args.degV = degV.to(device)
    args.degE = degE.to(device)
    args.degE2 = degE2.pow(-1.).to(device)

    nhid = args.nhid
    nhead = args.nhead
    edge_input_length=[round(args.friend_edge_num*args.split),args.visit_edge_num,args.trajectory_edge_num,args.self_loop_edge_number]
    node_input_dim=[]
    for i in node_type:
        node_input_dim.append(node_attr[i].shape[1])
    args.node_input_dim=node_input_dim
    node_input_length = [args.user_number,args.poi_number,args.poi_class_number,args.time_point_number]
    V_raw_index_type=[0 for i in range(args.user_number)]+ [1 for i in range(args.poi_number)]+[2 for i in range(args.poi_class_number)]+[3 for i in range(args.time_point_number)]
    args.V_raw_index_type=torch.tensor(V_raw_index_type,dtype=torch.long)
    args.edge_type=edge_type
    args.node_type=node_type

    a=0
    edge_input_length_raw = []
    for i in range(len(edge_input_length)):
        edge_input_length_raw.append(a+edge_input_length[i])
        a=a + edge_input_length[i]

    b=0
    node_input_length_raw=[]
    for i in range(len(node_input_length)):
        node_input_length_raw.append(b+node_input_length[i])
        b= b+ node_input_length[i]

    V_class=[]
    V_class_index_0,V_class_index_1,V_class_index_2,V_class_index_3=[],[],[],[]
    for i in range(V.shape[0]):
        if V[i] <node_input_length_raw[0]:
            V_class.append(0) #user
            V_class_index_0.append(i)
        elif node_input_length_raw[0]<= V[i] <node_input_length_raw[1]:
            V_class.append(1)#POI
            V_class_index_1.append(i)
        elif node_input_length_raw[1]<= V[i] <node_input_length_raw[2]:
            V_class.append(2)#POItype
            V_class_index_2.append(i)
        elif node_input_length_raw[2]<= V[i] <node_input_length_raw[3]:
            V_class.append(3)#timepoint
            V_class_index_3.append(i)

    E_class=[]
    E_class_index_0,E_class_index_1,E_class_index_2,E_class_index_3=[],[],[],[]
    # E_class_index=[]
    for i in range(E.shape[0]):
        if E[i]<edge_input_length_raw[0]:
            E_class.append(0) #friend
            E_class_index_0.append(i)
        elif edge_input_length_raw[0]<=E[i]<edge_input_length_raw[1]:
            E_class.append(1) #check-in
            E_class_index_1.append(i)
        elif edge_input_length_raw[1]<=E[i]<edge_input_length_raw[2]:
            E_class.append(2)#Trajectory
            E_class_index_2.append(i)
        elif edge_input_length_raw[2]<=E[i]<edge_input_length_raw[3]:
            E_class.append(3) #self-loop
            E_class_index_3.append(i)

    edge_to_v = {}  # Records the node indices within each hyperedge
    print(E.shape[0])
    for i in range(E.shape[0]):
        # print(edge_to_v.keys())
        if E[i].item() in edge_to_v.keys():
            edge_to_v[E[i].item()].append(V[i].item())
        else:
            edge_to_v[E[i].item()] = [V[i].item()]

    # Convert to tensor
    for i in range(args.edge_num):
        edge_to_v[i] = torch.tensor(edge_to_v[i], dtype=torch.long).to(device)
    args.edge_to_v = edge_to_v

    node_to_edge = {}  # Records the number of hyperedges each node belongs to
    for i in range(V.shape[0]):
        if V[i].item() in node_to_edge.keys():
            node_to_edge[V[i].item()].append(E[i].item())
        else:
            node_to_edge[V[i].item()] = [E[i].item()]

    # Convert to tensor
    for i in range(node_number):
        node_to_edge[i] = torch.tensor(node_to_edge[i], dtype=torch.long).to(device)
    args.node_to_edge = node_to_edge

    args.V_class = torch.tensor(V_class, dtype=torch.long)
    args.E_class = torch.tensor(E_class, dtype=torch.long)

    # V_class_index_0 records the sequence index of each V node type corresponding to E
    args.V_class_index_0 = torch.tensor(V_class_index_0, dtype=torch.long)
    args.V_class_index_1 = torch.tensor(V_class_index_1, dtype=torch.long)
    args.V_class_index_2 = torch.tensor(V_class_index_2, dtype=torch.long)
    args.V_class_index_3 = torch.tensor(V_class_index_3, dtype=torch.long)

    # E_class_index_0 records the sequence index of each E type corresponding to V for message passing
    args.E_class_index_0 = torch.tensor(E_class_index_0, dtype=torch.long)
    args.E_class_index_1 = torch.tensor(E_class_index_1, dtype=torch.long)
    args.E_class_index_2 = torch.tensor(E_class_index_2, dtype=torch.long)
    args.E_class_index_3 = torch.tensor(E_class_index_3, dtype=torch.long)

    E_class_index=torch.unsqueeze( torch.cat((args.E_class_index_0,args.E_class_index_1,args.E_class_index_2,args.E_class_index_3) ,0),1)
    args.E_class_index  =E_class_index.repeat(1, nhead)
    V_class_index=torch.unsqueeze(torch.cat((args.V_class_index_0,args.V_class_index_1,args.V_class_index_2,args.V_class_index_3) ,0) ,1)
    args.V_class_index= V_class_index.repeat(1, nhead)

    args.V=V
    args.E=E
    args.edge_input_length=edge_input_length_raw
    args.node_input_length=node_input_length_raw


    args.dataset_dict={'hypergraph':G,'n':N,'features':torch.randn(N,args.input_dim)}

    if args.ablation_study == 0:  # 0 indicates no ablation study, proceed with normal training
        if args.manifold_name == 'euclidean':
            model = HHGNN_Euclidean(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
            model.to(device)
        elif args.manifold_name == 'poincare':
            if args.multi_cuda == 0:
                model = HHGNN_Poincare(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                model.to(device)
            elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                model = HHGNN_Poincare_multi(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
            # optimiser = RiemannianAMSGrad(args, model.parameters(), lr=args.lr)

    elif args.ablation_study == 1:  # 1 indicates ablation study training
        # Check the ablation state. By default, the ablation study is only conducted in the Poincaré disk model.
        if args.ablation_state == 0:  # 0 removes heterogeneity
            if args.multi_cuda == 0:
                model = HHGNN_Poincare_without_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                model.to(device)
            elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                model = HHGNN_Poincare_multi_without_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
        elif args.ablation_state == 1:  # 1 removes the hyperbolic space
            if args.multi_cuda == 0:
                model = HHGNN_Poincare_without_hyperbolic(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                model.to(device)
            elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                model = HHGNN_Poincare_multi_without_hyperbolic(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
        elif args.ablation_state == 10:  # 10 removes both hyperbolic space and heterogeneity
            if args.multi_cuda == 0:
                model = HHGNN_Poincare_without_hyperbolic_and_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                model.to(device)
            elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                model = HHGNN_Poincare_multi_without_hyperbolic_and_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)


    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    return model, optimiser, G


def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """
    d = np.array(M.sum(1))
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}
    return DI.dot(M)
