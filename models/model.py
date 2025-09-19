import copy
import torch
import os
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from task_node_proxy import progressive_clustering
from models.gnns import BatchedGAT, BatchedThreeLayerGCN, BatchedTwoLayerGCN


def cos_sim(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)

    return torch.mm(normalized_tensor_1, torch.t(normalized_tensor_2))

"""
TRACE
"""
class TRACE(nn.Module):
    def __init__(self, slow_encoder, augmentor, train_loader_sizes, inference_batch_size, device, n_tasks):
        super(TRACE, self).__init__()

        self._device = device
        self.fast_encoder = None
        self.slow_encoder = slow_encoder.to(self._device)
        self._augmentor = augmentor
        self._train_loader_sizes = train_loader_sizes
        self._inference_batch_size = inference_batch_size
        
        self.current_task = -1
        self.rep_memory_node = []
        self.memory_retention_list = [0.0] * n_tasks
        self.rep_memory_graph = []

        self.mse_loss = nn.MSELoss()
    
    def memory_retention_processing(self, t, novelty_list):
        for tidx in range(t-1):
            embs = self.predict(data = self.rep_memory_graph[tidx]['g']).to(self._device)
            embs_target = embs[self.rep_memory_graph[tidx]['idx']]
            self.memory_retention_list[tidx] = round(torch.mean(cos_sim(embs_target, embs_target)).item()/novelty_list[tidx], 2)
            #offset = torch.abs(cos_sim(embs_target, embs_target) - novelty_list[tidx])
            #self.memory_retention_list[tidx] = round(torch.mean(torch.div(offset, torch.abs(novelty_list[tidx]))).item(),2)  
            #you could also use different measures of deviation, such as relative changes

    def init_fast_encoder(self, args):
        if args.dataset == "CoraFull-CL":
            self.fast_encoder = BatchedGAT(
                in_dim=8710,
                out_dim=args.emb_dim,
            ).to(args.device)
        if args.dataset == "Arxiv-CL":
            self.fast_encoder = BatchedThreeLayerGCN(
                in_dim=128,
                out_dim=args.emb_dim,
            ).to(args.device)
        if args.dataset == "Reddit-CL":
            self.fast_encoder = BatchedGAT(
                in_dim=602,
                out_dim=args.emb_dim,
            ).to(args.device)
        if args.dataset == "Products-CL":
            self.fast_encoder = BatchedGAT(
                in_dim=100,
                out_dim=args.emb_dim,
            ).to(args.device)
                
    def fast_learning(self, data, n_id, adjs):
        self.fast_encoder.train()

        adjs = [adj.to(self._device) for adj in adjs]
        (x_a, eis_a), (x_b, eis_b) = self._augmentor.augment_batch(x=data.x[n_id].to(self._device), adjs=adjs)
        sizes = [adj.size[1] for adj in adjs]
        z_a = self.fast_encoder(x=x_a, edge_indexes=eis_a, sizes=sizes)
        z_b = self.fast_encoder(x=x_b, edge_indexes=eis_b, sizes=sizes)
        loss_fast = self.barlow_twins_loss(z_a=z_a, z_b=z_b)

        return loss_fast

    def node_proxies_processing(self, data, t, args):
        self.fast_encoder.eval()
        with torch.no_grad():
            embs = self.fast_encoder.inference(
                x_all=data.x,
                edge_index_all=data.edge_index,
                inference_batch_size=self._inference_batch_size,
                device=self._device,
            )

        cur_rep_node = []
        node_proxies = progressive_clustering(cur_rep_node, embs, np.arange(embs.shape[0]), args.init_num_clu, args, random_state=1)
        
        if t != self.current_task:
            self.current_task = t
            self.rep_memory_node.append(node_proxies)

            # cur sample subgraph
            subg_node, subg_edge, target_idx, _ = k_hop_subgraph(node_proxies, self.fast_encoder.num_layers, data.edge_index, relabel_nodes=True)
            self.rep_memory_graph.append({'g': Data(x=data.x[subg_node], edge_index=subg_edge), 'idx':target_idx})

    def slow_learning(self, data, n_id, adjs, t, cp_model, args, best_models): 
        self.slow_encoder.train()

        adjs = [adj.to(self._device) for adj in adjs]
        (x_a, eis_a), (x_b, eis_b) = self._augmentor.augment_batch(x=data.x[n_id].to(self._device), adjs=adjs)
        sizes = [adj.size[1] for adj in adjs]
        z_a = self.slow_encoder(x=x_a, edge_indexes=eis_a, sizes=sizes)
        z_b = self.slow_encoder(x=x_b, edge_indexes=eis_b, sizes=sizes)
        loss_slow = self.barlow_twins_loss(z_a=z_a, z_b=z_b)

        sample_cur_g = NeighborSampler(
            self.rep_memory_graph[t]['g'].edge_index,
            node_idx=self.rep_memory_graph[t]['idx'],
            sizes=self._train_loader_sizes,
            batch_size=len(self.rep_memory_graph[t]['idx']),
            shuffle=False,
            num_workers=int(os.environ.get("NUM_WORKERS", 0)))

        sample_last_g = NeighborSampler(
            self.rep_memory_graph[t-1]['g'].edge_index,
            node_idx=self.rep_memory_graph[t-1]['idx'],
            sizes=self._train_loader_sizes,
            batch_size=len(self.rep_memory_graph[t-1]['idx']),
            shuffle=False,
            num_workers=int(os.environ.get("NUM_WORKERS", 0)))

        for _, sample_n_id, sample_adjs in sample_cur_g:
            sample_adjs = [adj.to(self._device) for adj in sample_adjs]
            (cur_x_a, cur_eis_a), (cur_x_b, cur_eis_b) = self._augmentor.augment_batch(x=self.rep_memory_graph[t]['g'].x[sample_n_id].to(self._device), adjs=sample_adjs)
            cur_sizes = [adj.size[1] for adj in sample_adjs]
        for _, sample_n_id, sample_adjs in sample_last_g:
            sample_adjs = [adj.to(self._device) for adj in sample_adjs]
            (last_x_a, last_eis_a), (last_x_b, last_eis_b) = self._augmentor.augment_batch(x=self.rep_memory_graph[t-1]['g'].x[sample_n_id].to(self._device), adjs=sample_adjs)
            last_sizes = [adj.size[1] for adj in sample_adjs]

        cur_rep_a = cp_model.slow_encoder(x=cur_x_a, edge_indexes=cur_eis_a, sizes=cur_sizes)
        cur_rep_b = cp_model.slow_encoder(x=cur_x_b, edge_indexes=cur_eis_b, sizes=cur_sizes)
        last_rep_a = cp_model.slow_encoder(x=last_x_a, edge_indexes=last_eis_a, sizes=last_sizes)
        last_rep_b = cp_model.slow_encoder(x=last_x_b, edge_indexes=last_eis_b, sizes=last_sizes)

        sim_old_a = cos_sim(cur_rep_a, last_rep_a)
        sim_old_b = cos_sim(cur_rep_b, last_rep_b)

        cor_old_a = cos_sim(last_rep_a,last_rep_a)
        cor_old_b = cos_sim(last_rep_b,last_rep_b)

        enc_cur_rep_a  = self.slow_encoder(x=cur_x_a, edge_indexes=cur_eis_a, sizes=cur_sizes)
        enc_cur_rep_b = self.slow_encoder(x=cur_x_b, edge_indexes=cur_eis_b, sizes=cur_sizes)
        enc_last_rep_a = self.slow_encoder(x=last_x_a, edge_indexes=last_eis_a, sizes=last_sizes)
        enc_last_rep_b = self.slow_encoder(x=last_x_b, edge_indexes=last_eis_b, sizes=last_sizes)

        enc_sim_old_a = cos_sim(enc_cur_rep_a, enc_last_rep_a)
        enc_sim_old_b = cos_sim(enc_cur_rep_b, enc_last_rep_b)

        enc_cor_old_a = cos_sim(enc_last_rep_a, enc_last_rep_a)
        enc_cor_old_b = cos_sim(enc_last_rep_b, enc_last_rep_b)

        loss_cos1 = (self.mse_loss(enc_sim_old_a,sim_old_a)+self.mse_loss(enc_sim_old_b,sim_old_b))/2.0
        loss_cor1 = (self.mse_loss(enc_cor_old_a,cor_old_a)+self.mse_loss(enc_cor_old_b,cor_old_b))/2.0
        loss_slow += args.beta*(loss_cos1+loss_cor1)


        for pretask in range(t-1):  # spaced replay tasks that were done a long time ago
            if self.memory_retention_list[pretask] < args.memory_retention_threshold or self.memory_retention_list[pretask] > (2.0-args.memory_retention_threshold):
            #if self.memory_retention_list[pretask] > args.memory_retention_threshold:  # set thresholds based on different measurement methods, such as this one for relative changes

                cp_model.load_state_dict(best_models[pretask])
                pre_model = copy.deepcopy(cp_model)
                for name, param in pre_model.named_parameters():
                    param.requires_grad = False

                sample_pre_g = NeighborSampler(        
                    self.rep_memory_graph[pretask]['g'].edge_index,
                    node_idx=self.rep_memory_graph[pretask]['idx'],
                    sizes=self._train_loader_sizes,
                    batch_size=len(self.rep_memory_graph[pretask]['idx']),
                    shuffle=False,
                    num_workers=int(os.environ.get("NUM_WORKERS", 0)))
                
                for _, sample_n_id, sample_adjs in sample_pre_g:
                    sample_adjs = [adj.to(self._device) for adj in sample_adjs]
                    (pre_x_a, pre_eis_a), (pre_x_b, pre_eis_b) = self._augmentor.augment_batch(x=self.rep_memory_graph[pretask]['g'].x[sample_n_id].to(self._device), adjs=sample_adjs)
                    pre_sizes = [adj.size[1] for adj in sample_adjs]
                
                cur_rep_a = pre_model.slow_encoder(x=cur_x_a, edge_indexes=cur_eis_a, sizes=cur_sizes)
                cur_rep_b = pre_model.slow_encoder(x=cur_x_b, edge_indexes=cur_eis_b, sizes=cur_sizes)
                pre_rep_a = pre_model.slow_encoder(x=pre_x_a, edge_indexes=pre_eis_a, sizes=pre_sizes)
                pre_rep_b = pre_model.slow_encoder(x=pre_x_b, edge_indexes=pre_eis_b, sizes=pre_sizes)
                
                sim_old_a = cos_sim(cur_rep_a, pre_rep_a)
                sim_old_b = cos_sim(cur_rep_b, pre_rep_b)
                cor_old_a = cos_sim(pre_rep_a,pre_rep_a)
                cor_old_b = cos_sim(pre_rep_b,pre_rep_b)
                
                enc_pre_rep_a = self.slow_encoder(x=pre_x_a, edge_indexes=pre_eis_a, sizes=pre_sizes)
                enc_pre_rep_b = self.slow_encoder(x=pre_x_b, edge_indexes=pre_eis_b, sizes=pre_sizes)
                
                enc_sim_old_a = cos_sim(enc_cur_rep_a, enc_pre_rep_a)
                enc_sim_old_b = cos_sim(enc_cur_rep_b, enc_pre_rep_b)
                enc_cor_old_a = cos_sim(enc_pre_rep_a, enc_pre_rep_a)
                enc_cor_old_b = cos_sim(enc_pre_rep_b, enc_pre_rep_b)


                loss_cos2 = (self.mse_loss(enc_sim_old_a,sim_old_a)+self.mse_loss(enc_sim_old_b,sim_old_b))/2.0
                loss_cor2 = (self.mse_loss(enc_cor_old_a,cor_old_a)+self.mse_loss(enc_cor_old_b,cor_old_b))/2.0
                
                loss_slow += (args.beta*min(1.0,self.memory_retention_list[pretask])) * (loss_cos2+loss_cor2)

        return loss_slow
    
    def predict(self, data):
        self.slow_encoder.eval()
        with torch.no_grad():
            z = self.slow_encoder.inference(
                x_all=data.x,
                edge_index_all=data.edge_index,
                inference_batch_size=self._inference_batch_size,
                device=self._device,
            )
        return z

    def barlow_twins_loss(self, z_a, z_b):
        EPS = 1e-15
        batch_size = z_a.size(0)
        feature_dim = z_a.size(1)
        _lambda = 1 / feature_dim

        # Apply batch normalization
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

        # Cross-correlation matrix
        c = (z_a_norm.T @ z_b_norm) / batch_size

        # Loss function
        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        loss = (
            (1 - c.diagonal()).pow(2).sum()
            + _lambda * c[off_diagonal_mask].pow(2).sum()
        )

        return loss