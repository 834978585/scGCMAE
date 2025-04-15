import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import random
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from datasets import Loader, apply_noise
from model import AutoEncoder
from evaluate import evaluate
from util import AverageMeter
from thop import profile
from thop import clever_format
from sklearn.neighbors import kneighbors_graph
from scipy import sparse as sp
from time import time as get_time


def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def inference(net, data_loader_test):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    net.eval()
    feature_vector = []
    labels_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader_test):
            feature_vector.extend(net.feature(x.to(device)).detach().cpu().numpy())
            labels_vector.extend(y.numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    dis = []
    resolutions = sorted(list(np.arange(0.01, 2.5, increment)), reverse=True)
    i = 0
    res_new = []
    for res in resolutions:
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(
            adata.obs['leiden']).leiden.unique())
        dis.append(abs(count_unique_leiden-fixed_clus_count))
        res_new.append(res)
        if count_unique_leiden == fixed_clus_count:
            break
    reso = resolutions[np.argmin(dis)]

    return reso

def buildGraphNN(X, neighborK):
    A = kneighbors_graph(X, neighborK, mode='connectivity', metric='cosine', include_self=True)
    return A


def kldloss(p, q):
    c1 = -torch.sum(p * torch.log(q), dim=-1)
    c2 = -torch.sum(p * torch.log(p), dim=-1)
    return torch.mean(c1 - c2)

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def cal_latent(z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / 1
        num = torch.pow(1.0 + num, -(1 + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p


def train(args):
    data_load = Loader(args, dataset_name=args["dataset"], drop_last=True)
    data_loader = data_load.train_loader
    data_loader_test = data_load.test_loader
    x_shape = args["data_dim"]

    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    gene = data_load.gene[:x_shape]
    # scoreall= pd.DataFrame(columns=['Epoch'] + [f'Gene_{i+1}' for i in range(1000)])
    scoreall = pd.DataFrame(columns=['Epoch'] + list(gene))
    selected_genes = pd.read_csv("gene.csv").values

    uppercase_gene = [s.upper() for s in gene]
    result_matrix = []

    # 遍历字符串列表
    for value in uppercase_gene:
        # 检查是否与目标字符串列表中的任一字符串完全匹配
        if value in selected_genes:
            result_matrix.append(1.1)  # 相符为 2
        else:
            result_matrix.append(1)  # 不相符为 0

    count = {}
    for number in result_matrix:
        if number in count:
            count[number] += 1  # 如果数字已存在，计数加 1
        else:
            count[number] = 1  # 如果数字不存在，初始化计数为 1


    # Hyper-params
    init_lr = args["learning_rate"]
    max_epochs = args["epochs"]
    mask_probas = [0.3]*x_shape

    # setup model
    model = AutoEncoder(
        num_genes=x_shape,
        hidden_size=128,
        masked_data_weight=0.75,
        mask_loss_weight=0.7,
        gene_matrix=torch.tensor(result_matrix),
    ).to(device)
    model_checkpoint = 'model_checkpoint.pth'


    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    max_value = -1

    # train model
    for epoch in range(max_epochs):
        model.train()
        meter = AverageMeter()
        loss_ae_list = []
        contrastive_loss_list = []
        loss_gae_list = []
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            x_corrputed, mask = apply_noise(x, mask_probas)
            KNN_adj = buildGraphNN(x, 30)
            adj1 = sp.csr_matrix(KNN_adj)
            adj1 = torch.FloatTensor(adj1.todense()).to(device)


            optimizer.zero_grad()
            x_corrputed_latent, reconstruction_loss, contrastive_loss, loss_gae, score = model.loss_mask(x_corrputed, x, adj1)
            loss_ae = reconstruction_loss * 1 + contrastive_loss * 0.01 + loss_gae * 0.1
            loss_ae.backward()
            optimizer.step()
            meter.update(loss_ae.detach().cpu().numpy())
            loss_ae_list.append(loss_ae.item())
            contrastive_loss_list.append(contrastive_loss.item())
            loss_gae_list.append(loss_gae.item())

    
        if epoch %  1 ==0 or epoch == max_epochs - 1:
            # Generator in eval mode
            latent, true_label = inference(model, data_loader_test)
            clustering_model = KMeans(n_clusters=args["n_classes"], n_init='auto',init='k-means++')
            clustering_model.fit(latent)
            pred_label = clustering_model.labels_

            nmi, ari, acc = evaluate(true_label, pred_label)
            ss = silhouette_score(latent, pred_label)
            db = davies_bouldin_score(latent, pred_label)

            res = {}
            res["nmi"] = nmi
            res["ari"] = ari
            res["acc"] = acc
            res["sil"] = ss
            res["db"] = db
            res["dataset"] = args["dataset"]
            res["epoch"] = epoch
            results.append(res)
            scoreall.loc[epoch] = [epoch + 1] + score.detach().cpu().numpy().tolist()

            if nmi + ari >= max_value:
              max_value = nmi + ari
              best_ari, best_nmi = ari, nmi
              best_acc = acc
              best_epoch = epoch
              best_ss = ss
              best_db = db

            print("epoch:" + str(epoch))
            print('loss_ae:%.4f' % np.mean(loss_ae_list))

            np.save(args["save_path"]+"/embedding_"+str(epoch)+".npy", 
                    latent)
            pd.DataFrame({"True": true_label, 
                        "Pred": pred_label}).to_csv(args["save_path"]+"/types_"+str(epoch)+".txt")
    print("best_ari: %f best_nmi: %f best_acc: %f best_epoch: %.0f best_ss: %f best_db: %f" % (best_ari, best_nmi, best_acc, best_epoch, best_ss, best_db))
    best = pd.DataFrame(
                {'dataset': args["dataset"], 'best_ari': best_ari, 'best_nmi': best_nmi,
                 'best_acc': best_acc, 'best_epoch': best_epoch, 'best_ss': best_ss, 'best_db': best_db,}, index=[0])

    torch.save({
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict()
    }, model_checkpoint
    )

    return results, best, scoreall


if __name__ == "__main__":
    time_start = get_time()
    for i in range(1):
        seed = random.randint(1,100)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        args = {}
        args["num_workers"] = 4
        args["paths"] = {"data": "D:/单细胞/医院数据/聚类结果/AttentionAE-sc-main/Data/AnnData/",
                        "results": "./res/"}
        args['batch_size'] = 128
        args["data_dim"] = 1000
        args['n_classes'] = 4
        args['epochs'] = 100
        args["dataset"] = "10X_PBMC"
        args["learning_rate"] = 1e-3
        args["latent_dim"] = 32

        print(args)

        path = args["paths"]["data"]
        '''''
        files = ["Pollen", "Quake_Smart-seq2_Lung", "Limb_Muscle", "Quake_10x_Limb_Muscle"
                 "worm_neuron_cell", "Melanoma_5K", "Young", "Guo", "Baron", 'Muraro'
                 "Wang", "Quake_10x_Spleen", "Shekhar", "Macosko", "Klein",'Enge'
                 "Tosches", "Bach", "hrvatin"]
'''
        files = ["Quake_Smart-seq2_Lung"]

        results = pd.DataFrame()
        bestall = pd.DataFrame()
        scoreall = pd.DataFrame()
        save_dir = make_dir(args["paths"]["results"], "a_summary")
        for dataset in files:
            print(f">> {dataset}")
            args["dataset"] = dataset
            args["save_path"] = make_dir("./data/sc_data/scMAE/"+str(i), dataset)

            res, best, score = train(args)
            results = results._append(res)
            bestall = bestall._append(best)
            scoreall = scoreall._append(score)
            results.to_csv(args["paths"]["results"] +
                        "/res_all_data_test"+str(i)+".csv", header=True)
            bestall.to_csv(args["paths"]["results"] +
                           "/best_all_data_test" + str(i) + ".csv", header=True)
            scoreall.to_csv(args["paths"]["results"] +
                            "/score_all_data_test" + str(i) + ".csv", header=True, index=False)
        time = get_time() - time_start
        print("Running Time:" + str(time))