#coding=utf-8
import pickle
import wandb

from utils import *

args = config.parse()
wandb.init(
    # set the wandb project where this run will be logged
    project="hypergraph",
    # track hyperparameters and run metadata
    config= vars(args),
)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
if args.multi_cuda==0:
    device2=device
else:
    device2 = torch.device('cuda:' + str(eval(args.cuda) + 1) if torch.cuda.is_available() else 'cpu')

add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'
model_name = args.model_name

now_time=time.strftime("%m%d%H%M", time.localtime()) # Month, day, hour, and minute

# Define node and edge types
node_type=['user','poi','poi_class','time_point']
edge_type=['friend','check_in','trajectory']

file='output/'+now_time+'_'+args.city+'_'+args.manifold_name+'_'+'input_dim_'+str(args.input_dim)+'_'+'hidden_dim_'+str(args.nhid)+'_'+\
     'out_dim_'+str(args.out_dim)+'_multi_head_'+str(args.out_nhead)+'+lambda_1_'+str(args.lam_1)+'+lambda_2_'+str(args.lam_2)+'+lambda_3_'+str(args.lam_3)+\
     '_'+'negative_K_'+str(args.negative_K)+'_multi_cuda_'+str(args.multi_cuda)+'_ablation_study_'+str(args.ablation_study)+'_ablation_state_'+str(args.ablation_state) +'.txt'

embedding_file='output_embedding/'+file[7:-4]+'.pkl'
print(file)

G, node_attr, friend_edge_train, friend_edge_test, test_label,friend_edge_train_all, friend_edge_train_all_label,k =fetch_data(args)
model, optimizer,G = initialise(G, node_attr , args,node_type,edge_type)
friend_edge_train_all_label=(torch.tensor( friend_edge_train_all_label, dtype=torch.float32)).to(device2)

friend_edge_train_list = []
for i in range(len(friend_edge_train)):
    f = list(friend_edge_train[i])
    friend_edge_train_list.append((f[0], f[1]))
    friend_edge_train_list.append((f[1], f[0]))

friend_edge_train_list = np.array(friend_edge_train_list)
friend_edge_train_list = torch.tensor(friend_edge_train_list, dtype=torch.long).t().contiguous()
g = dgl.graph((friend_edge_train_list[0], friend_edge_train_list[1]))
g=g.to(device2)

tanh=torch.nn.Tanh()

for i in node_type:
    node_attr[i]= torch.tensor(node_attr[i],).to(device)

best_test_auc, test_auc, Z = 0, 0, None
tic_epoch = time.time()
for epoch in range(args.epochs):

    optimizer.zero_grad()
    model.train()
    Z, c = model(node_attr)
    predic_label = F.cosine_similarity( Z[friend_edge_train_all[0] ], Z[friend_edge_train_all[1] ])

    loss_cross = F.binary_cross_entropy_with_logits( predic_label, friend_edge_train_all_label)
    loss_margin = margin_loss( predic_label[: k], predic_label[k: ]  )
    con_loss = contrastive_loss(Z,g)
    loss= con_loss* args.lam_1 + loss_cross* args.lam_2 + loss_margin * args.lam_3

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        # End time
        end_time = time.time()
        # Calculate execution time
        execution_time = end_time - tic_epoch
        print(f"Execution time for 100 epochs: {execution_time} seconds")
        tic_epoch = time.time()


    if epoch > 500:
        if epoch%100 == 0:
            auc, ap, top_k = test( Z, test_label, g, friend_edge_test)

            print("Epoch:",epoch, "AUC:",auc)
            if auc>best_test_auc:
                best_test_auc = auc
                print("best_test_auc:",best_test_auc)
                f=open(embedding_file,'wb')
                pickle.dump(Z[:args.user_number], f)
            print("best_test_auc:",best_test_auc)
            need_write = "epoch" + str(epoch) + " best_auc: " + str(best_test_auc) +"_auc: " + str(auc) + " _ap: " + str(ap)
            top = 'top_1+' + str(top_k[0]) + ' top_5+' + str(top_k[1]) + ' top_10+' + str(top_k[2]) + ' top_15+' + str(
                top_k[3]) + ' top_20+' + str(top_k[4])
            with open(file, 'a+') as f:
                f.write(need_write + '\n')  ## Add newline for better readability
                f.write(top + '\n')  # # Add newline for better readability