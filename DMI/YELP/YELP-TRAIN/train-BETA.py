from utils_new import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'

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

        
    def forward(self, idx_U, idx_V):
        
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
        
        LINK_PROB = 0.5*PROB_CLUSTER + 0.5*PROB_U_V

        return LINK_PROB.view(-1)

class update_nodes_embedding(nn.Module):
    def __init__(self, NODES_U_NUMBER, NODES_V_NUMBER,
                 U_CLUSTERS_DIM, V_CLUSTERS_DIM):
        super(update_nodes_embedding, self).__init__()
        self.prediction_module = get_node_emb(NODES_U_NUMBER,
                                              NODES_V_NUMBER,
                                              U_CLUSTERS_DIM, 
                                              V_CLUSTERS_DIM).cuda()
        self.optimizer = torch.optim.Adam(params=self.prediction_module.parameters(), 
                                          lr=5e-3)
        self.loss_function = torch.nn.MSELoss(reduction='sum')

    def pairwise_distances(self, clusters):
        norm_squared = torch.sum(clusters**2, dim=1, keepdim=True)  # 每行的范数平方
        distances = torch.sqrt(torch.clamp(norm_squared - 2 * torch.matmul(clusters, clusters.transpose(0, 1)) + norm_squared.transpose(0, 1), min=1e-12))
        return distances

    def forward(self, DATA_TRAIN, DATA_TEST, SET_BETA):        
        best_auc = 0
        best_ap = 0

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

                nodes_similar = self.prediction_module(idx_U, V_idx).to(device)
                 
                pair_loss = self.loss_function(nodes_similar, true_labels.view(-1))
                
                cluster_distances = self.pairwise_distances(self.prediction_module.STRUCTURE_CLUSTERS.weight)
                cluster_loss = torch.sum(cluster_distances)
                
                loss = pair_loss - cluster_loss*SET_BETA
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f'Epoch: {epoch:02d}, Loss: {total_loss:.4f}')
            
            FINAL_U_EMDS = self.prediction_module.U_emb.detach()
            FINAL_V_EMDS = self.prediction_module.V_emb.detach()

            all_predict_edges = []
            all_true_edges = []
            for _, (test_idx_U, test_idx_V, test_edge_label) in enumerate(DATA_TEST):
        
                test_lhs = FINAL_U_EMDS[test_idx_U]
                test_rhs = FINAL_V_EMDS[test_idx_V]

                predict_test_edges = predict_edges(test_lhs, test_rhs, self.prediction_module).cpu()

                predict_label = np.array(predict_test_edges)
                true_label = np.array(test_edge_label)
                all_predict_edges = np.concatenate((all_predict_edges, predict_label), axis=0)
                all_true_edges = np.concatenate((all_true_edges, true_label),axis=0)

            predict_ap, predict_auc = computer_prediction(all_true_edges, all_predict_edges)
            print('epoch: ', epoch, 'predict_auc_roc = ', predict_auc, 'predict_auc_pr = ', predict_ap)
	    
            if predict_ap > best_ap:
                best_ap = predict_ap
            if predict_auc > best_auc:
                best_auc = predict_auc

        return best_auc, predict_ap

set_random_seed(42)
NUMBER_U, NUMBER_V, train_data_loader, test_data_loader = get_LP_train_test_data(4)

################ 超参数 #####################
cluster_number = 32
U_hidden_dim, U_output_dim, V_hidden_dim, V_output_dim = 12, 32, 12, 32
U_network_arc = [cluster_number, U_hidden_dim, U_output_dim]
V_network_arc = [cluster_number, V_hidden_dim, V_output_dim]

ALL_BEST_AP = []
ALL_BEST_AUC = []


if __name__ == '__main__':
    for setting_beta in [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3]:
        print('############################# BETA: ', setting_beta)

        UPDATE_NODES_MODULE = update_nodes_embedding(NUMBER_U, NUMBER_V, U_network_arc, V_network_arc)
    
        best_result_auc, best_result_ap = UPDATE_NODES_MODULE(train_data_loader, test_data_loader, setting_beta)

        ALL_BEST_AUC.append(best_result_auc)
        ALL_BEST_AP.append(best_result_ap)
    
    print('################################## BEST AUC RESULTS ##################################')
    print(ALL_BEST_AUC)
    print('################################## BEST AP RESULTS  ##################################')
    print(ALL_BEST_AP)
