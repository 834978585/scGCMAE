import torch
import torch.nn as nn
from torch.nn.functional import mse_loss as mse
from GAE import GCN
import torch.nn.functional as F




class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        num_genes,
        gene_matrix,
        hidden_size=128,
        dropout=0,
        masked_data_weight=.75,
        mask_loss_weight=0.7,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight

        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_genes, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

        self.mask_predictor = nn.Linear(hidden_size, num_genes)
        self.decoder = nn.Linear(
            in_features=hidden_size+num_genes, out_features=num_genes)

        self.GAE = GCN(in_dim=num_genes, hidden1=4000, hidden2=1000, hidden3=hidden_size, z_emb_size=hidden_size,
                       dropout_rate=0.1)
        self.fc4 = nn.Linear(hidden_size, 512)
        self.fc5 = nn.Linear(512, num_genes)
        self.gene_matrix = nn.Parameter(gene_matrix.float())

    def contrastive_loss(self, output1, output2, label, margin=1.0):
        distance = nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(distance, 2) + (label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
        return loss.mean()

    def sim(self,z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())  # 返回相似度矩阵

    def semi_loss(self,z1: torch.Tensor, z2: torch.Tensor):
        tau = 10
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))  # 计算两个相似度矩阵之间的损失

    def forward_mask(self, x, y, adj):
        score = self.gene_matrix
        x = x * score
        y = y * score
        latent1 = self.encoder(x)

        g1 = self.GAE.gcn_encoder(x, adj)
        g2 = self.GAE.gcn_encoder(y, adj)

        l1 = self.semi_loss(g1, g2).mean()
        l2 = self.semi_loss(g2, g1).mean()
        ret_knn = (l1 + l2) * 0.5
        contrastive_loss = ret_knn

        latent = latent1 * 1 + (g1+g2) * 0.0001

        reconstruction = self.fc5(F.relu(self.fc4(latent)))


        adj_hat = self.GAE.gcn_decoder(g1)
        loss_gae = F.mse_loss(adj_hat, adj)


        return latent, reconstruction, contrastive_loss, loss_gae, score

    def loss_mask(self, x, y, adj):
        latent, reconstruction, contrastive_loss, loss_gae, score = self.forward_mask(x, y, adj)

        reconstruction_loss = mse(reconstruction, y, reduction='none')
        reconstruction_loss = reconstruction_loss.mean()

        return latent, reconstruction_loss, contrastive_loss, loss_gae, score

    def feature(self, x):
        latent = self.encoder(x)
        return latent

    def forward(self, x):
        latent = self.encoder(x)
        predicted_mask = self.mask_predictor(latent)
        reconstruction = self.decoder(
            torch.cat([latent, predicted_mask], dim=1))
        return latent, predicted_mask, reconstruction


