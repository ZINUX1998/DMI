from utils_new import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'

split_RS_train_test_matrix()

train_data_loader, train_csr, test_csr = get_RS_train_test_edges(20)

NUMBER_U = train_csr.shape[0]
NUMBER_V = train_csr.shape[1]


class get_node_emb(nn.Module):
    def __init__(self, NUM_U, NUM_V, 
                 U_CLUSTER_ARC, V_CLUSTER_ARC, IN_DIM = 64):
        super(get_node_emb, self).__init__()
        
        get_ini_emb = get_ini_emds(IN_DIM, IN_DIM, NUM_V, NUM_U)
        V_embedding, U_embedding = get_ini_emb()
        self.V_emb = Parameter(V_embedding)
        self.U_emb = Parameter(U_embedding)
        
        self.U_CLUSTER_NETWORK = MLP(U_CLUSTER_ARC)
        self.V_CLUSTER_NETWORK = MLP(V_CLUSTER_ARC)
        
        self.STRUCTURE_CLUSTERS = torch.nn.Embedding(
            num_embeddings = U_CLUSTER_ARC[0], embedding_dim = IN_DIM)
        nn.init.normal_(self.STRUCTURE_CLUSTERS.weight, std = 1)

        
    def forward(self, idx_U, idx_V, set_alpha):
        
        lhs = self.U_emb[idx_U]     # (batch_size, out_features)
        rhs = self.V_emb[idx_V]     # (batch_size, 1 + neg_edges, out_features)
        
        U_DIS = F.cosine_similarity(lhs.unsqueeze(1), self.STRUCTURE_CLUSTERS.weight.unsqueeze(0), dim=2)
        U_numerator = self.U_CLUSTER_NETWORK(U_DIS)       #soft assignment
        U_soft_assignments = (U_numerator.t() / torch.sum(U_numerator, 1)).t()       #soft assignment
        
        rhs_flat = rhs.view(-1, rhs.shape[-1])  # shape: (15, 8) - 将5*3个二维张量视为15个向量
        CLUSTERS_flat = self.STRUCTURE_CLUSTERS.weight.unsqueeze(0).expand(rhs_flat.shape[0], -1, -1).contiguous()  # shape: (15, 4, 8)
        cos_sim = F.cosine_similarity(rhs_flat.unsqueeze(1), CLUSTERS_flat, dim=-1)  # shape: (15, 4)
        V_DIS = cos_sim.view(rhs.shape[0], rhs.shape[1], self.STRUCTURE_CLUSTERS.weight.shape[0])  # shape: (5, 3, 4)
        V_assignments = self.V_CLUSTER_NETWORK(V_DIS) # (batch_size, 1 + neg_edges, cluster_numbers)
        
        PROB_CLUSTER = torch.sum(U_soft_assignments.unsqueeze(1) * V_assignments, dim=2)
        
        distance_U_V = F.cosine_similarity(lhs.unsqueeze(1), rhs, dim=2) + 1
        PROB_U_V = distance_U_V / 2.0  # (batch_size, 1 + neg_edges)
        
        LINK_PROB = set_alpha*PROB_CLUSTER + (1-set_alpha)*PROB_U_V

        return LINK_PROB.view(-1)

class update_nodes_embedding(nn.Module):
    def __init__(self, NODES_U_NUMBER, NODES_V_NUMBER,
                 U_CLUSTERS_DIM, V_CLUSTERS_DIM):
        super(update_nodes_embedding, self).__init__()
        self.prediction_module = get_node_emb(NODES_U_NUMBER,
                                              NODES_V_NUMBER,
                                              U_CLUSTERS_DIM, 
                                              V_CLUSTERS_DIM).cuda()
        self.optimizer = torch.optim.Adam(params=self.prediction_module.parameters(), lr=5e-3)
        self.loss_function = torch.nn.MSELoss(reduction='sum')

    def pairwise_distances(self, clusters):
        norm_squared = torch.sum(clusters**2, dim=1, keepdim=True)  # 每行的范数平方
        distances = torch.sqrt(torch.clamp(norm_squared - 2 * torch.matmul(clusters, clusters.transpose(0, 1)) + norm_squared.transpose(0, 1), min=1e-12))
        return distances

    def forward(self, DATA_TRAIN, TRAIN_CSR, TEST_CSR, TEST_BETCH, 
                TOP_K, MAX_K, SET_BETA, print_max_K = 0):
        
        best_ndcg = 0.
        best_metrics = {}

        target_metric = f'ndcg@{MAX_K}'
        
        self.prediction_module.train()
        
        for epoch in range(100):
            total_loss = 0
            
#             PRED_U_EMDS = self.prediction_module.U_emb.detach()
#             self.prediction_module.STRUCTURE_CLUSTERS = PRED_U_EMDS[self.structure_centers] + torch.sparse.mm(CLUSTER_GRAPH, PRED_U_EMDS)

            ##################### 批训练 #################
            for batch_id, batch in enumerate(DATA_TRAIN):
                idx_U, pos_idx_V, neg_idx_V = batch
                V_idx = torch.cat((pos_idx_V.unsqueeze(dim=1), neg_idx_V), dim=1).to(device)
    
                pos_lables = torch.ones_like(pos_idx_V).to(device)
                neg_lables = torch.zeros_like(neg_idx_V).to(device)
                true_labels = torch.cat((pos_lables.unsqueeze(dim=1), neg_lables), dim=1).to(device)
                true_labels = true_labels.float().to(device)

                nodes_similar = self.prediction_module(idx_U, V_idx, 0.7).to(device)
                
                pair_loss = self.loss_function(nodes_similar, true_labels.view(-1))
                
                cluster_distances = self.pairwise_distances(self.prediction_module.STRUCTURE_CLUSTERS.weight)
                cluster_loss = torch.sum(cluster_distances)
                
                loss = pair_loss - cluster_loss*SET_BETA
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
#                 if train_step%100 == 0:
#                     print('iter loss: ', loss.item())
            
            print(f'Epoch: {epoch:02d}, Loss: {total_loss:.4f}')
            
            metrics = batch_evaluation(self.prediction_module, 
                                       TEST_CSR, TRAIN_CSR, 
                                       epoch, TEST_BETCH, TOP_K, MAX_K, 0.7)
            
            if metrics[target_metric] >= best_ndcg:
                best_metrics = metrics.copy()
                best_ndcg = metrics[target_metric]

            print('** epoch', epoch, 'total_loss: ', total_loss, '**')
            print('Epoch', epoch, '|', end='\t')
            print_metrics(metrics, TOP_K, MAX_K, print_max_K)
            print('** best performance: epoch', best_metrics['epoch'], '**')
            print('Epoch', best_metrics['epoch'], '|', end='\t')
            print_metrics(best_metrics, TOP_K, MAX_K, print_max_K)

        return best_metrics

set_random_seed(42)

################ 超参数 #####################
topk = [3, 5, 10]
test_batch = 1000
# cluster_number = len(cluster_centers)
cluster_number = 32
max_K = max(topk)
U_hidden_dim, U_output_dim, V_hidden_dim, V_output_dim = 12, 32, 12, 32
U_network_arc = [cluster_number, U_hidden_dim, U_output_dim]
V_network_arc = [cluster_number, V_hidden_dim, V_output_dim]

ALL_BEST_RESULT = []

if __name__ == '__main__':
    for setting_beta in [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3]:
    #for setting_alpha in [0.5]:
        print('############################# beta : ', setting_beta)
        UPDATE_NODES_MODULE = update_nodes_embedding(NUMBER_U, NUMBER_V, U_network_arc, V_network_arc)
        best_result = UPDATE_NODES_MODULE(train_data_loader, train_csr, test_csr, test_batch, topk, max_K, setting_beta)
        ALL_BEST_RESULT.append(best_result)

    print('######################################### best result ###########################')
    print(ALL_BEST_RESULT)
