import torch
import numpy as np
import os
import argparse
import copy 
import json
from task_manager import semi_task_manager
from models.gnns import *
from models.model import *
from models.warmup import LinearWarmupCosineAnnealingLR
from models.evaluate import LogReg, LogReg_fit
from models.argument import GraphAugmentor
from utils import *
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data


"""
# Args Settings  
"""

# The hyperparameter settings on different datasets are slightly different and need to be adjusted by yourself!

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CoraFull-CL')  # 'CoraFull-CL' 'Arxiv-CL'  'Reddit-CL'  'Products-CL'
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--in_dim', type=int, default=64)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--fast_num_epochs', type=int, default=300)
parser.add_argument('--slow_num_epochs', type=int, default=300)
parser.add_argument('--warmup_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--inference_batch_size', type=int, default=300)
parser.add_argument('--n_cls_per_task', type=int, default=2)
parser.add_argument('--n_cls', type=int, default=70)
parser.add_argument('--init_num_clu', type=int, default=10)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--variance_threshold', type=float, default=0.1)
parser.add_argument('--compactness_threshold_mode', type=str, default='avg') # 28, avg
parser.add_argument('--memory_retention_threshold', type=float, default=0.8)
args = parser.parse_args()

if args.dataset == 'Arxiv-CL' or args.dataset == 'Reddit-CL':
    args.n_cls = 40
elif args.dataset == 'Products-CL':
    args.n_cls = 46
if args.device == None:
    args.device = 'cpu'
print(f"Args:\n{args}")

cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)] # this line will remove the final task if only one class included
task_seq = cls
n_tasks = len(task_seq)
task_manager = semi_task_manager()

acc_matrix = np.zeros([n_tasks, n_tasks])
outs_dir = f"./result/taskIL/{args.dataset}/"
os.makedirs(outs_dir, exist_ok=True)
model_dir = outs_dir + 'models/'
os.makedirs(model_dir, exist_ok=True)


"""
Model Settings
"""
if args.dataset == "CoraFull-CL":
    slow_gnn = BatchedGAT(
        in_dim=8710,
        out_dim=args.emb_dim,
    )
if args.dataset == "Arxiv-CL":
    slow_gnn = BatchedThreeLayerGCN(
        in_dim=128,
        out_dim=args.emb_dim,
    )
if args.dataset == "Reddit-CL":
    slow_gnn = BatchedGAT(
        in_dim=602,
        out_dim=args.emb_dim,
    )
if args.dataset == "Products-CL":
    slow_gnn = BatchedGAT(
        in_dim=100,
        out_dim=args.emb_dim,
    )

augmentor = GraphAugmentor(
    p_x_1=0.1,
    p_e_1=0.2,
    )
neighbor_loader_sizes = [10, ]*slow_gnn.num_layers
model = TRACE(
    slow_encoder=slow_gnn,
    augmentor=augmentor,
    train_loader_sizes= neighbor_loader_sizes,
    inference_batch_size=args.inference_batch_size,
    device=args.device,
    n_tasks=n_tasks
)

slow_optimizer = torch.optim.AdamW(
    params=model.slow_encoder.parameters(),
    lr=args.lr,
    weight_decay=1e-5,
)
slow_scheduler = LinearWarmupCosineAnnealingLR(
    optimizer=slow_optimizer,
    warmup_epochs=args.warmup_epochs,
    max_epochs=args.slow_num_epochs,
)


"""
Task CL
"""
n_cls_so_far = 0
best_models = []
model_copy = None
novelty_list = [0.0] * n_tasks

for task, task_cls in enumerate(task_seq):
    print(f"task-{task}, task_cls-{task_cls}")
    n_cls_so_far += len(task_cls)
    task_manager.add_task(task, n_cls_so_far)

    # data : no_inter_tsk_edge
    adj, features, labels, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = load_data(args.dataset, task_cls)
    data, masks = data_to_pyg(adj, features, labels, ids_per_cls_all, train_ids, valid_ids_, test_ids_)

    train_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=neighbor_loader_sizes,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(os.environ.get("NUM_WORKERS", 0)))

    if task>0:
        model.load_state_dict(torch.load(f=os.path.join(model_dir, f"best_model_{args.dataset}_{task-1}.pkl")))
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(args.device)

        for name, param in model_copy.named_parameters():
            param.requires_grad = False
        
        best_models.append(model_copy)
    
    if task >= 2:
        model.memory_retention_processing(task, novelty_list)
        
    model = model.to(args.device)
    # fast learning system
    loss_min = 1e9
    total_loss = 0

    model.init_fast_encoder(args)
    fast_optimizer = torch.optim.AdamW(
        params=model.fast_encoder.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    fast_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=fast_optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.fast_num_epochs,
    )
    for epoch in range(args.fast_num_epochs):
        for _, n_id, adjs in train_loader:
            fast_optimizer.zero_grad()
            loss_fast = model.fast_learning(data = data, n_id = n_id, adjs = adjs)
            total_loss += loss_fast.item()
            loss_fast.backward()
            fast_optimizer.step()

        fast_scheduler.step()
        torch.cuda.empty_cache()    

        avg_loss = total_loss / len(train_loader)
        if avg_loss <= loss_min:
            loss_min = avg_loss
            torch.save(model.state_dict(), f=os.path.join(model_dir, f"best_model_{args.dataset}_{task}.pkl"))

    model.load_state_dict(torch.load(f=os.path.join(model_dir, f"best_model_{args.dataset}_{task}.pkl")))
    model.node_proxies_processing(data, task, args)      

    # slow learning system
    loss_min = 1e9
    total_loss = 0

    if task > 0:
        for epoch in range(args.slow_num_epochs):
            for _, n_id, adjs in train_loader:
                slow_optimizer.zero_grad()
                loss_slow = model.slow_learning(data = data, n_id = n_id, adjs = adjs, t = task, cp_model = model_copy, args = args, best_models = best_models)
                total_loss += loss_slow.item()
                loss_slow.backward()
                slow_optimizer.step()
            
            slow_scheduler.step()
            torch.cuda.empty_cache()

            avg_loss = total_loss / len(train_loader)
            if avg_loss <= loss_min:
                loss_min = avg_loss
                torch.save(model.state_dict(), f=os.path.join(model_dir, f"best_model_{args.dataset}_{task}.pkl"))

    # Evaluate
    model.load_state_dict(torch.load(f=os.path.join(model_dir, f"best_model_{args.dataset}_{task}.pkl")))
    embs = model.predict(data = model.rep_memory_graph[task]['g']).to(args.device)
    embs_target = embs[model.rep_memory_graph[task]['idx']]
    novelty_list[task] = torch.mean(cos_sim(embs_target, embs_target)).item()

    acc_mean = []
    for t in range(task + 1):
        adj, features, labels, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = load_data(args.dataset, task_seq[t])
        data, masks = data_to_pyg(adj, features, labels, ids_per_cls_all, train_ids, valid_ids_, test_ids_)
        
        embs = model.predict(data = data).to(args.device) 
        test_ids = []
        test_ids.extend(valid_ids_)
        test_ids.extend(test_ids_)
        train_embs = embs[train_ids]
        test_embs = embs[test_ids]
        label_offset1 = task_manager.get_label_offset(t - 1)[1]
        labels = labels - label_offset1
        train_lbls =  torch.LongTensor(labels[train_ids]).to(args.device)
        test_lbls = torch.LongTensor(labels[test_ids]).to(args.device)

        # LogisticRegression classifier
        logreg = LogReg_fit(embs.shape[1], 2, args.weight_decay, args.device)
        epoch_flag, acc = logreg.fit(train_embs, train_lbls, test_embs, test_lbls)

        acc_matrix[task][t] = round(acc * 100, 2)
        acc_mean.append(acc)
        print(f"T{t:02d} {acc * 100:.2f}|", end="")
        
    acc_mean = round(np.mean(acc_mean) * 100, 2)
    print('')
    print(f"T{t:02d} {acc_mean:.2f}")
    print('=================='*5)


"""
# Save Acc matrix、AP、AF
"""
print("Finish " * 5)
print('AP: ', acc_mean)
backward = []
for t in range(n_tasks - 1):
    b = acc_matrix[n_tasks - 1][t] - acc_matrix[t][t]
    backward.append(round(b, 2))
fgt_mean = round(np.mean(backward), 2)
print('AF: ', fgt_mean)
np.save(outs_dir+'acc_matrix_{}.npy'.format(args.dataset), acc_matrix)