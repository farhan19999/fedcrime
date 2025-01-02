import sys
import os
import shutil
import numpy as np
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import argparse as Ap
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics,parameters_to_ndarrays, ndarrays_to_parameters
from collections import OrderedDict
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import math

seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)

parser = Ap.ArgumentParser()

parser.add_argument("--cluster_head", type=int, default = 50, help="Number of cluster head")
parser.add_argument("--target_region", type=int, default = 7, help="Region for prediction")
parser.add_argument("--target_cat", type=int, default = 1, help="Crime category for prediction")
parser.add_argument("--perc_clients", type=int, default = 100, help="Percentage of cluster heads perticipating")
parser.add_argument("--server_perc", type=int, default = 50, help="Percentage of data in Server")
parser.add_argument("--epoch",type = int, default = 1, help = "Epoch in client training")
parser.add_argument("--round", type = int, default = 20, help = "Total Round of Communication between server and client")
parser.add_argument("--mu", type = float, default = 0, help = "Proximal Term for Fed Prox")
parser.add_argument("--aggr_mthd", type = int, default = 0, help = "Aggregation Method : (0)-sample ratio, (1)-crime ratio, (3)-crime ratio with exponential smoothing")
parser.add_argument("--smoothing", type= int, default=0, help="Use smoothing to fill unknown values : (0): no (1): yes")
parser.add_argument("--beta1", type= int, default=80, help="filtering threshold before merging")
parser.add_argument("--beta2", type= int, default=70, help="filtering threshold after merging")
# -------------------------------------- FED PROX Experiments -----------------------------------
args = parser.parse_args()

cluster_head = args.cluster_head
test_region = args.target_region
sample_clients = int(cluster_head)
perc_of_cluster = int(args.perc_clients)
smoothing = int(args.smoothing)
beta1 = int(args.beta1)/100
beta2 = int(args.beta2)/100

print(f'cluster head : ${cluster_head}')
print(f'percent of cluster : ${perc_of_cluster}')
participant_clients = (sample_clients*perc_of_cluster)//100
# participant_clients = 25

BATCH_SIZE = 32
if((1530//participant_clients)<32):
    BATCH_SIZE = 1530//participant_clients

top = True


hidden_state_of_lstm = 40
learning_rate = 0.001
server_percentage = args.server_perc/100 # percent of data in server
isolated_users_percentage = 0.1  # Change this according to your requirement
overlapping_percentage = 0  # Change this according to your requirement
number_of_rounds = int(args.round)
EPOCHS = int(args.epoch)
Fed_Prox_Proximal_MU = float(args.mu)
aggregation_method=0

save_file_path = "k_means_clustered_" + "hs_" + str(hidden_state_of_lstm) + "_lr_" + str(int(learning_rate * 10000)) + "_Sp_" + str(int(server_percentage * 100))+"_Iu_" + str(int(isolated_users_percentage * 100)) + "_Op_" + str(int(overlapping_percentage * 100)) + "_Cl_" +str(sample_clients)




'''
    total_partition: total number of partitions to be created.
      If server_partition is True, then total_partition is the number of clients plus one. If server_partition is False, then total_partition is the number of clients.
    server_sample_size: number of samples in server
'''
def create_server_data_partition(total_partition, server_sample_size, server_partition=True):
    if(server_partition):
        total_partition+=1
    partition = np.arange(server_sample_size)
    # now shuffle the partition
    # np.random.seed(96)
    np.random.shuffle(partition)
    partition_size = server_sample_size // total_partition
    with open('server_partition.txt', 'w') as f:
        for i in range(total_partition):
            f.write(' '.join(map(str, partition[i * partition_size:(i + 1) * partition_size])) + '\n')



create_server_data_partition(participant_clients, 1530, server_partition=False) # plus one is for server


if(os.path.isdir(f'./random/Region_{test_region}_{save_file_path}')):
    shutil.rmtree(f'./random/Region_{test_region}_{save_file_path}')


os.makedirs(f'./random/Region_{test_region}_{save_file_path}')





# seed_value = 42
# np.random.seed(seed_value)

root_folder="."
# root_folder = "/content/drive/MyDrive/Undergrad_thesis_final/Entire_fed_crime_folder"

DATA_READ_FILE_PATH = f'{root_folder}/toy_data'

TOY_EXP_RESULT_PATH = './external/r_'+str(test_region)+"_"+save_file_path
DATA_WRITE_FILE_PATH = f'./temp_{test_region}_{save_file_path}'
if(os.path.isdir(DATA_WRITE_FILE_PATH)):
  shutil.rmtree(DATA_WRITE_FILE_PATH)
if(os.path.isdir(TOY_EXP_RESULT_PATH)):
  shutil.rmtree(TOY_EXP_RESULT_PATH)


os.makedirs(DATA_WRITE_FILE_PATH)
os.makedirs(TOY_EXP_RESULT_PATH)


np.set_printoptions(threshold=np.inf)



div = 6 ##############################################################################################


def filtering(userid,input_data,server_data_to_be_added, tw = 120):
    forecast = 1  # Num of ts to forecast in the future
    L = input_data.shape[0]
    mask = np.zeros(L-tw-forecast+1)
    for i in range(L - tw - forecast+1):
        train_seq = input_data[i:i + tw, :]
        server_seq = server_data_to_be_added[i:i + tw, :]
        #count number of zeros in the train_seq
        count_zeros = np.count_nonzero(train_seq == 0)
        if(count_zeros > (tw*beta1)):
            continue
        # add the server data to the input data
        train_seq += server_seq
        count_zeros = np.count_nonzero(train_seq == 0)
        if(count_zeros > (tw*beta2)):
            continue
        mask[i] = 1
    np.savetxt(f"{DATA_WRITE_FILE_PATH}/external/{userid}/mask.txt", mask, fmt='%d')
    # print(f"mask saved for user {userid}")
    

#! here server_data_to_be_added is the data that is to be added to the client data
#! remove_extra_samples is a boolean variable that is used to remove the extra data from the inout sequences
def create_inout_sequences_for_client(input_data,server_data_to_be_added,list_of_sample_indices,userid,remove_extra_samples = True, tw=120):

    forecast = 1  # Num of ts to forecast in the future
    # recent_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int32))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int32))
    mask = np.loadtxt(f"{DATA_WRITE_FILE_PATH}/external/{userid}/mask.txt", dtype=np.int32)
    
    

    L = input_data.shape[0]
    for i in range(L - tw - forecast+1):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    #filter the input data and server data from the mask
    in_seq1 = in_seq1[mask == 1]
    out_seq1 = out_seq1[mask == 1]

    #! now create the inout sequence for server data again in the same process so that we can extract the specified index of data from the list provided in the parameters from the server data
    server_in_seq = torch.from_numpy(np.ones((8000, tw), dtype=np.int32))
    server_out_seq = torch.from_numpy(np.ones((8000, forecast), dtype=np.int32))
    LL = server_data_to_be_added.shape[0]
    for i in range(LL - tw - forecast+1):
        server_train_seq = server_data_to_be_added[i:i + tw, :]
        server_in_seq[i] = server_train_seq.view(server_train_seq.shape[0] * server_train_seq.shape[1])
        server_train_label = server_data_to_be_added[i + tw:i + tw + forecast, :]
        server_out_seq[i] = server_train_label.view(server_train_label.shape[0] * server_train_label.shape[1])
    server_in_seq = server_in_seq[:i + 1, :]
    server_out_seq = server_out_seq[:i + 1, :]

    #filter the input data and server data from the mask
    server_in_seq = server_in_seq[mask == 1]
    server_out_seq = server_out_seq[mask == 1]

    #TODO: now we have to process in sequence 1 so that it has overlapping data of both server and client and remove the extra samples if the boolen is True
    #! now we have to add the server_in_sequece to the in_seq1 and server_out_sequence to the out_seq1 in only the specified indices
    # for i in list_of_sample_indices:
    #     in_seq1[i] += server_in_seq[i]
    #     out_seq1[i] += server_out_seq[i]
    # if(remove_extra_samples):
    #     #TODO: remove the extra samples from the in_seq1 and out_seq1. In the list of sample indices are the indices for whom we have added the server and input data. Now remove other indices from the in_seq1 and out_seq1
    #     in_seq1 = in_seq1[list_of_sample_indices]
    #     out_seq1 = out_seq1[list_of_sample_indices]
    in_seq1 = in_seq1 + server_in_seq
    out_seq1 = out_seq1 + server_out_seq

    # fill the zero values of the in_seq1 with average of the previous 6 values and next 6 values but not the filled values
    if(smoothing):
        temp_in_seq1 = torch.zeros_like(in_seq1)
        for i in range(in_seq1.shape[0]):
            for j in range(in_seq1.shape[1]):
                temp_in_seq1[i][j] = in_seq1[i][j]
                if(in_seq1[i][j] == 0):
                    sum = 0
                    count = 0
                    for k in range(-6, 7):
                        if(j+k >= 0 and j+k < in_seq1.shape[1] and in_seq1[i][j+k] != 0):
                            sum += in_seq1[i][j+k]
                            count += 1
                    if(count != 0):
                        temp_in_seq1[i][j] = sum//count
        in_seq1 = temp_in_seq1

    # daily_temporal_data_generation
    batch_size = in_seq1.shape[0]
    time_step_daily = int(tw / div)
    in_seq2 = torch.from_numpy(np.ones((batch_size, time_step_daily), dtype=np.int32))
    out_seq2 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % div == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

    # weekly_temporal_data_generation
    time_step_weekly = int(tw / (div * 7)) + 1
    in_seq3 = torch.from_numpy(np.ones((batch_size, time_step_weekly), dtype=np.int32))
    out_seq3 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % (div * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
    return in_seq1, out_seq1, in_seq2, in_seq3

def create_inout_sequences_for_server(input_data,list_of_sample_indices,keep_only_list_indices = False, tw=120):

    forecast = 1  # Num of ts to forecast in the future

    # recent_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int32))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int32))
    L = input_data.shape[0]
    for i in range(L - tw - forecast+1):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    if(keep_only_list_indices):
        in_seq1 = in_seq1[list_of_sample_indices]
        out_seq1 = out_seq1[list_of_sample_indices]


    # daily_temporal_data_generation
    batch_size = in_seq1.shape[0]
    time_step_daily = int(tw / div)
    in_seq2 = torch.from_numpy(np.ones((batch_size, time_step_daily), dtype=np.int32))
    out_seq2 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % div == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

    # weekly_temporal_data_generation
    time_step_weekly = int(tw / (div * 7)) + 1
    in_seq3 = torch.from_numpy(np.ones((batch_size, time_step_weekly), dtype=np.int32))
    out_seq3 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % (div * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
    return in_seq1, out_seq1, in_seq2, in_seq3

def create_inout_sequences(input_data, tw=120):
    forecast = 1  # Num of ts to forecast in the future

    # recent_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int32))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int32))
    L = input_data.shape[0]
    for i in range(L - tw - forecast+1):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]


    # daily_temporal_data_generation
    batch_size = in_seq1.shape[0]
    time_step_daily = int(tw / div)
    in_seq2 = torch.from_numpy(np.ones((batch_size, time_step_daily), dtype=np.int32))
    out_seq2 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % div == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

    # weekly_temporal_data_generation
    time_step_weekly = int(tw / (div * 7)) + 1
    in_seq3 = torch.from_numpy(np.ones((batch_size, time_step_weekly), dtype=np.int32))
    out_seq3 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % (div * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
    return in_seq1, out_seq1, in_seq2, in_seq3


def load_data_GAT(bs, userid):
    # build features
    idx_features_labels = np.genfromtxt(f"{DATA_WRITE_FILE_PATH}/external/{userid}/gat_feat.txt", dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)
    # build features_ext
    idx_features_labels_ext = np.genfromtxt(f"{DATA_WRITE_FILE_PATH}/external/{userid}/gat_feat_ext.txt",
                                            dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    features_ext = sp.csr_matrix(idx_features_labels_ext[:, 1:], dtype=np.float32)  # (Nodes, features)

    # build features
    idx_crime_side_features_labels = np.genfromtxt(f"{DATA_WRITE_FILE_PATH}/external/{userid}/gat_crime_side.txt",
                                                   dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    crime_side_features = sp.csr_matrix(idx_crime_side_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)

    # build graph
    num_reg = int(idx_features_labels.shape[0] / bs)
    idx = np.array(idx_features_labels[:num_reg, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{DATA_WRITE_FILE_PATH}/external/{userid}/tem_gat_adj.txt", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    if(len(edges.shape) == 1):
      edges = edges.reshape(1, 2)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(num_reg, num_reg),
                        dtype=np.float32)  # replaced 5
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    features_ext = torch.FloatTensor(np.array(features_ext.todense()))

    crime_side_features = torch.FloatTensor(np.array(crime_side_features.todense()))

    return adj, features, features_ext, crime_side_features


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data_regions(bs, target_crime_cat, target_region, target_city, userid,partition_list,remove_extra_samples=True,tw=120 ):
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    com = gen_neighbor_index_zero(target_region, target_city)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    aggregated_data_scaler_2 = MinMaxScaler(feature_range=(-1,1))
    device = torch.device('cpu')
    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    for i in com:
        loaded_train_data = ''
        if(userid == f'server_{save_file_path}'):
          loaded_train_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + f"/{userid}/r_" + str(i) + ".txt", dtype=np.float32)).T
          loaded_aggregated_train_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH + "/"+target_city+'/aggregated/r_'+str(i)+'.txt',dtype=np.float32)).T
          loaded_aggregated_train_data = loaded_aggregated_train_data[:,target_crime_cat:target_crime_cat+1]

        else:
          loaded_train_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + f"/user_{save_file_path}/{userid}/r_" + str(i) + ".txt", dtype=np.float32)).T
        loaded_train_data = loaded_train_data[:, target_crime_cat:target_crime_cat + 1]

        if(userid == f'server_{save_file_path}'):
            train_x, train_y, train_x_daily, train_x_weekly = create_inout_sequences_for_server(loaded_train_data, partition_list,keep_only_list_indices=False, tw=tw)
            train_x_for_aggregated_data_2, train_y_aggregated_data_2, train_x_daily_aggregated_data_2, train_x_weekly_aggregated_data_2 = create_inout_sequences_for_server(loaded_aggregated_train_data, partition_list,keep_only_list_indices=False) # here loaded_aggregated_train_data is actually loaded_aggregated_data


        else:
            loaded_server_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH+"/" + target_city + f"/server_{save_file_path}/r_" + str(i) + ".txt", dtype=int)).T
            loaded_server_data = loaded_server_data[:, target_crime_cat:target_crime_cat+1]
            train_x, train_y, train_x_daily, train_x_weekly = create_inout_sequences_for_client(loaded_train_data, loaded_server_data, partition_list,remove_extra_samples=remove_extra_samples, userid=userid,tw=tw)

        loaded_test_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + "/test/r_" + str(i) + ".txt", dtype=int)).T
        loaded_test_data = loaded_test_data[:, target_crime_cat:target_crime_cat+1]

        # train_x, train_y, train_x_daily, train_x_weekly = create_inout_sequences(loaded_train_data)
        test_x, test_y, test_x_daily, test_x_weekly = create_inout_sequences(loaded_test_data)

        if(userid == f'server_{save_file_path}'):
            train_x_aggregated_scaled_2=torch.from_numpy(aggregated_data_scaler_2.fit_transform(train_x_for_aggregated_data_2))
            test_x = torch.from_numpy(aggregated_data_scaler_2.transform(test_x))
            train_y_aggregated_scaled_2 = torch.from_numpy(aggregated_data_scaler_2.fit_transform(train_y_aggregated_data_2))
            test_y = torch.from_numpy(aggregated_data_scaler_2.transform(test_y))

        train_x = torch.from_numpy(scaler.fit_transform(train_x))
        # test_x = torch.from_numpy(scaler.transform(test_x))

        train_y = torch.from_numpy(scaler.fit_transform(train_y))
        # test_y = torch.from_numpy(scaler.transform(test_y))

        # Divide your data into train set & test set

        train_extra = train_x.shape[0] % bs
        train_x = train_x[:train_x.shape[0]-train_extra, :]
        train_y = train_y[:train_y.shape[0]-train_extra, :]

        test_extra = test_x.shape[0] % bs
        test_x = test_x[:test_x.shape[0] - test_extra, :]
        test_y = test_y[:test_y.shape[0] - test_extra, :]

        train_x = train_x.view(int(train_x.shape[0] / bs), bs, tw)
        test_x = test_x.view(int(test_x.shape[0] / bs), bs, tw)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x.to(device))
        add_test.append(test_x.to(device))

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    return batch_add_train, batch_add_test


def load_data_regions_external(bs, nxfeatures, target_region, target_city, tw=120):
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    com = gen_neighbor_index_one_with_target(target_region, target_city)
    poi_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + "/poi.txt", dtype=np.int32)) # (num_of_regions, poi_categories) -> (77, 10)
    device = torch.device('cpu')
    if(torch.cuda.is_available()):
            device = torch.device('cuda')

    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + "/act_ext/taxi_" + str(i) + ".txt", dtype=np.int32)).T
        # loaded_data -> (num_of_days*time_frame_per_day, 2) -> (259 * 2, 2 ) # here 2 in dim=1 means in and out flow

        loaded_data1 = loaded_data[:, 0:1]
        loaded_data2 = loaded_data[:, 1:2]
        x_in, y_in, z_in, m_in = create_inout_sequences(loaded_data1)
        x_out, y_out, z_out, m_out = create_inout_sequences(loaded_data2)

        x_in = x_in.unsqueeze(2).double()
        x_out = x_out.unsqueeze(2).double()
        poi = poi_data[i - 1].double()
        poi = poi.repeat(x_in.shape[0], tw, 1)

        x = torch.cat([x_in, x_out, poi], dim=2)

        # Divide into train_test data
        # train_x_size = int(x.shape[0] * .7)
        train_x_size = 1530
        test_x_size = 420
        # train_x = x[: train_x_size, :, :]  # (bs, tw) = (1386, 120)
        # test_x = x[train_x_size+tw+1:, :, :]  # (bs, tw) = (683, 120)
        # test_x = test_x[:test_x.shape[0] - 11, :, :]

        test_x = x[: test_x_size, :, :]
        train_x = x[test_x_size+tw:, :, :]
        # print('train_x shape in load regionals ',train_x.shape)
        # print('train_x shape in load regionals ',test_x.shape)

        ######################### BY PRANGON #########################
        # train_extra = train_x_size % bs  # (ns_tr % bs) = 30
        train_extra = train_x.shape[0] % bs  # (ns_tr % bs) = 30
        test_extra = test_x.shape[0] % bs  # (ns_te % bs) = 21

        train_x = train_x[:train_x.shape[0]-train_extra, :]  # (ns_tr, ts) = (1356, ts)                                    ### (1356, 1)
        test_x = test_x[:test_x.shape[0]-test_extra, :]  # (ns_te, ts) = (672, ts)                                                   ### (672, 1)
        ######################### BY PRANGON #########################

        train_x = train_x.view(int(train_x.shape[0] / bs), bs, tw, nxfeatures)
        test_x = test_x.view(int(test_x.shape[0] / bs), bs, tw, nxfeatures)

        train_x = train_x.transpose(2, 1)  # (num_regions, tw, bs, nxfeatures)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x.to(device))
        add_test.append(test_x.to(device))

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    # batch_add_train -> (num_of_batches, num_of_regions, time_width=120, batch_size=42, num_of_feature=10+2)
    return batch_add_train, batch_add_test


def load_data_sides_crime(bs, target_crime_cat, target_region, target_city, userid, partition_list, remove_extra_samples=True, tw=120):
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions

    com = gen_neighbor_index_zero_with_target(target_region, target_city)
    side = gen_com_side_adj_matrix(com, target_city)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    aggregated_data_scaler_3 = MinMaxScaler(feature_range=(-1,1))
    device = torch.device('cpu')
    if(torch.cuda.is_available()):
        device = torch.device('cuda')

    for i in range(len(com)):
        loaded_train_data = ''
        if(userid == f'server_{save_file_path}'):
          loaded_train_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + f'/{userid}/s_' + str(side[i]) + ".txt", dtype=np.int32)).T
          loaded_aggregated_data_3 = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH + "/"+target_city+'/aggregated/s_'+str(side[i])+'.txt',dtype=np.int32)).T
          loaded_aggregated_data_3 = loaded_aggregated_data_3[:, target_crime_cat:target_crime_cat + 1]
        else:
          loaded_train_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + f'/user_{save_file_path}/{userid}/s_' + str(side[i]) + ".txt", dtype=np.int32)).T
        loaded_train_data = loaded_train_data[:, target_crime_cat:target_crime_cat + 1]
        tensor_ones = torch.from_numpy(np.ones((loaded_train_data.size(0), loaded_train_data.size(1)), dtype=np.int32))

        loaded_train_data = torch.where(loaded_train_data > 1, tensor_ones, loaded_train_data)
        if(userid == f'server_{save_file_path}'):
            tensor_ones_2 = torch.from_numpy(np.ones((loaded_aggregated_data_3.size(0), loaded_aggregated_data_3.size(1)), dtype=np.int32))
            loaded_aggregated_data_3 = torch.where(loaded_aggregated_data_3 > 1, tensor_ones_2, loaded_aggregated_data_3)

        if(userid == f'server_{save_file_path}'):
            train_x, train_y, train_x_daily, train_x_weekly = create_inout_sequences_for_server(loaded_train_data, partition_list,keep_only_list_indices=False, tw=tw)
            train_x_for_aggregated_data_3, train_y_aggregated_data_3, train_x_daily_aggregated_data_3, train_x_weekly_aggregated_data_3 = create_inout_sequences_for_server(loaded_aggregated_data_3, partition_list,keep_only_list_indices=False,tw=tw) # here loaded_aggregated_data is actually loaded_aggregated_data


        else:
            loaded_server_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH+"/" + target_city + f"/server_{save_file_path}/s_" + str(side[i]) + ".txt", dtype=int)).T
            loaded_server_data = loaded_server_data[:, target_crime_cat:target_crime_cat+1]
            train_x, train_y, train_x_daily, train_x_weekly = create_inout_sequences_for_client(loaded_train_data, loaded_server_data, partition_list, remove_extra_samples=remove_extra_samples, userid = userid, tw=tw)
        tensor_ones = torch.from_numpy(np.ones((train_x.size(0), train_x.size(1)), dtype=np.int32))
        train_x = torch.where(train_x > 1, tensor_ones, train_x)
        tensor_ones = torch.from_numpy(np.ones((train_y.size(0), train_y.size(1)), dtype=np.int32))
        train_y = torch.where(train_y > 1, tensor_ones, train_y)


        # modified by changra farhan for Round 0 same initial test metrics problem solving
        if(userid == f'server_{save_file_path}'):
            tensor_ones_3 = torch.from_numpy(np.ones((train_x_for_aggregated_data_3.size(0), train_x_for_aggregated_data_3.size(1)), dtype=np.int32))
            train_x_for_aggregated_data_3 = torch.where(train_x_for_aggregated_data_3 > 1, tensor_ones_3, train_x_for_aggregated_data_3)
            tensor_ones_3 = torch.from_numpy(np.ones((train_y_aggregated_data_3.size(0), train_y_aggregated_data_3.size(1)), dtype=np.int32))
            train_y_aggregated_data_3 = torch.where(train_y_aggregated_data_3 > 1, tensor_ones_3, train_y_aggregated_data_3)

        # train_x, train_y, _,__ = create_inout_sequences(loaded_train_data)

        loaded_test_data = torch.from_numpy(np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + "/test/s_" + str(side[i]) + ".txt", dtype=np.int32)).T
        loaded_test_data = loaded_test_data[:, target_crime_cat:target_crime_cat + 1]
        tensor_ones = torch.from_numpy(np.ones((loaded_test_data.size(0), loaded_test_data.size(1)), dtype=np.int32))
        loaded_test_data = torch.where(loaded_test_data > 1, tensor_ones, loaded_test_data)
        test_x, test_y,_,__ = create_inout_sequences(loaded_test_data)

        if(userid == f'server_{save_file_path}'):
            train_x_aggregated_result = torch.from_numpy(aggregated_data_scaler_3.fit_transform(train_x_for_aggregated_data_3))
            test_x = torch.from_numpy(aggregated_data_scaler_3.transform(test_x))

            train_y_aggregated_result = torch.from_numpy(aggregated_data_scaler_3.fit_transform(train_y_aggregated_data_3))
            test_y = torch.from_numpy(aggregated_data_scaler_3.transform(test_y))

        train_x = torch.from_numpy(scaler.fit_transform(train_x))
        # test_x = torch.from_numpy(scaler.transform(test_x))

        train_y = torch.from_numpy(scaler.fit_transform(train_y))
        # test_y = torch.from_numpy(scaler.transform(test_y))

        # Divide your data into train set & test set

        train_extra = train_x.shape[0] % bs
        train_x = train_x[:train_x.shape[0]-train_extra, :]
        train_y = train_y[:train_y.shape[0]-train_extra, :]

        test_extra = test_x.shape[0] % bs
        test_x = test_x[:test_x.shape[0] - test_extra, :]
        test_y = test_y[:test_y.shape[0] - test_extra, :]

        train_x = train_x.view(int(train_x.shape[0] / bs), bs, tw)
        test_x = test_x.view(int(test_x.shape[0] / bs), bs, tw)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x.to(device))
        add_test.append(test_x.to(device))

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    return batch_add_train, batch_add_test


def gen_com_adj_matrix(target_region):
    adj_matrix = np.zeros((77, 77), dtype=np.int32)
    edges_unordered = np.genfromtxt(f"{DATA_READ_FILE_PATH}/chicago/com_adjacency.txt", dtype=np.int32)
    for i in range(edges_unordered.shape[0]):
        src = edges_unordered[i][0] - 1
        dst = edges_unordered[i][1] - 1
        adj_matrix[src][dst] = 1
        adj_matrix[src][dst] = 1
    np.savetxt(f"{DATA_READ_FILE_PATH}/chicago/com_adj_matrix.txt", adj_matrix, fmt="%d")
    return


def gen_com_side_adj_matrix(regions, target_city):
    idx = np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + "/side_com_adj.txt", dtype=np.int32)
    idx_map = {j: i for i, j in iter(idx)}
    side = [idx_map.get(x + 1) % 101 for x in regions]  # As it starts with 0
    return side


def gen_neighbor_index_zero(target_region, target_city):
    adj_matrix = np.loadtxt(f"{DATA_READ_FILE_PATH}/" + target_city + "/com_adj_matrix.txt")
    adj_matrix = adj_matrix[target_region]
    neighbors = []
    for i in range(adj_matrix.shape[0]):
        if adj_matrix[i] == 1:
            neighbors.append(i)
    return neighbors


def gen_neighbor_index_zero_with_target(target_region, target_city):
    neighbors = gen_neighbor_index_zero(target_region, target_city)
    neighbors.append(target_region)
    return neighbors


def gen_neighbor_index_one_with_target(target_region, target_city):
    neighbors = gen_neighbor_index_zero(target_region, target_city)
    neighbors.append(target_region)
    neighbors = [x + 1 for x in neighbors]
    return neighbors


def gen_gat_adj_file(target_city, target_region, userid):
    neighbors = gen_neighbor_index_zero(target_region, target_city)
    adj_target = torch.zeros(len(neighbors), 2)
    for i in range(len(neighbors)):
        adj_target[i][0] = target_region
        adj_target[i][1] = neighbors[i]
    np.savetxt(f"{DATA_WRITE_FILE_PATH}/external/{userid}/tem_gat_adj.txt", adj_target, fmt="%d")
    return




class Sparse_attention(nn.Module):
    def __init__(self, top_k=5):
        super(Sparse_attention, self).__init__()
        self.top_k = top_k

    def forward(self, attn_s):
        eps = 10e-8
        batch_size = attn_s.size()[0]
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            return attn_s
        else:
            delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps

        attn_w = attn_s - delta.reshape((batch_size, 1)).repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min=0)
        attn_w_sum = torch.sum(attn_w, dim=1)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.reshape((batch_size, 1)).repeat(1, time_step)
        return attn_w_normalize


# Built on original code base of (Check layers_torch.py):
# https://github.com/nke001/sparse_attentive_backtracking_release
class self_LSTM_sparse_attn_predict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 truncate_length=100, predict_m=10, block_attn_grad_past=False, attn_every_k=1, top_k=5):
        super(self_LSTM_sparse_attn_predict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.tanh = torch.nn.Tanh()

        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.w_t.data, gain=1.414)

        self.sparse_attn = Sparse_attention(top_k=self.top_k)
        self.predict_m = nn.Linear(hidden_size, 2)  # hidden_size
        self.device = None
        if(torch.cuda.is_available()):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x):
        # print('x shape in lstm ',x.shape)
        batch_size = x.size(0)
        time_size = x.size(1)
        input_size = self.input_size
        hidden_size = self.hidden_size

        h_t = Variable(torch.zeros(batch_size, hidden_size))  # h_t = (batch_size, hidden_size)
        c_t = Variable(torch.zeros(batch_size, hidden_size))  # c_t = (batch_size, hidden_size)
        predict_h = Variable(torch.zeros(batch_size, hidden_size))  # predict_h = (batch_size, hidden_size)

        h_old = h_t.view(batch_size, 1, hidden_size).to(self.device)  # h_old = (batch_size, 1, hidden_size) --> Memory

        outputs = []
        attn_all = []
        attn_w_viz = []
        predicted_all = []
        outputs_new = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)

            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = h_t.detach(), c_t.detach()

            # Feed LSTM Cell
            # print('batch size ',batch_size)
            # print('input size ',input_size)
            # print('input_t size',input_t.shape)
            input_t = input_t.contiguous().view(batch_size, input_size)  # input_t = (batch_size, input_size)
            h_t, c_t = self.lstm1(input_t, (h_t.to(self.device), c_t.to(self.device)))  # h_t/ c_t = (batch_size, hidden dimension)
            h_t_naive_lstm = h_t
            predict_h = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            predicted_all.append(h_t)  # changed predict_h

            # Broadcast and concatenate current hidden state against old states
            h_repeated = h_t.unsqueeze(1).repeat(1, remember_size,
                                                 1)  # h_repeated = (batch_size, remember_size = memory, hidden_size)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            mlp_h_attn = self.tanh(mlp_h_attn)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if False:  # PyTorch 0.2.0
                attn_w = torch.matmul(mlp_h_attn, self.w_t)
            else:  # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size * remember_size,
                                             2 * hidden_size)  # mlp_h_attn = (batch_size * remember_size, 2* hidden_size)
                attn_w = torch.mm(mlp_h_attn, self.w_t)  # attn_w = (batch_size * remember_size, 1)
                attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w = attn_w.view(batch_size, remember_size)  # attn_w = (batch_size, remember_size)
            attn_w = self.sparse_attn(attn_w)  # attn_w = (batch_size, remember_size)
            attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)

            # if i >= 100:
            # print(attn_w.mean(dim=0).view(remember_size))
            attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))  # you should return it
            out_attn_w = attn_w
            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w = attn_w.repeat(1, 1, hidden_size)  # attn_w = (batch_size, remember_size, hidden_size)
            h_old_w = attn_w * h_old  # attn_w = (batch_size, remember_size, hidden_size)
            attn_c = torch.sum(h_old_w, 1).squeeze(1)  # att_c = (batch_size, hidden_size)

            # Feed attn_c to hidden state h_t
            h_t = h_t + attn_c  # h_t = (batch_size, hidden_size)

            #
            # At regular intervals, remember a hidden state, store it in memory
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            predict_real_h_t = self.predict_m(
                h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            outputs_new += [predict_real_h_t]

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        return attn_c, out_attn_w


# Built on original code base of (Check layers.py):
# https://github.com/Diego999/pyGAT
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_cfeatures, in_xfeatures, out_features, att_dim, bs, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_cfeatures = in_cfeatures
        self.in_xfeatures = in_xfeatures
        self.out_features = out_features
        self.att_dim = att_dim
        self.bs = bs
        self.emb_dim = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_cfeatures, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.Wf = nn.Parameter(torch.zeros(size=(in_cfeatures, out_features)))
        # nn.init.xavier_uniform_(self.Wf.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # self.WS = nn.Parameter(torch.zeros(size=(in_cfeatures, out_features)))
        # nn.init.xavier_uniform_(self.WS.data, gain=1.414)

        # self.aS = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        # nn.init.xavier_uniform_(self.aS.data, gain=1.414)

        # self.WQ = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        # nn.init.xavier_uniform_(self.WQ.data, gain=1.414)

        # self.WK = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        # nn.init.xavier_uniform_(self.WK.data, gain=1.414)

        # self.WV = nn.Parameter(torch.zeros(size=(1, out_features)))
        # nn.init.xavier_uniform_(self.WV.data, gain=1.414)

        # self.WF = nn.Linear(self.in_xfeatures, out_features, bias=False)  # For other one

    def forward(self, input, adj, ext_input, side_input):
        input = input.view(self.bs, -1, 1)
        ext_input = ext_input.view(self.bs, -1, self.in_xfeatures) #(bs, num_of_region, num_of_feature)
        side_input = side_input.view(self.bs, -1, 1)
        adj = adj.repeat(self.bs, 1, 1)

        """
            Step 1: Generate c_{i,t}^k
        """

        # Find the attention vectors for region_wise crime similarity
        h = torch.matmul(input, self.W)  # h = [h_1, h_2, h_3, ... , h_N] * W
        N = h.size()[1]  # N = Number of Nodes (regions)
        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2).view(h.shape[0],
                N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)  # shape = (bs, N, N)
        attention = torch.where(adj > 0, e, zero_vec)  # shape = (bs, N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (bs, N, N)

        # Find the attention vectors for side_wise crime similarity
        # h_side = torch.matmul(side_input, self.WS)  # h = [h_1, h_2, h_3, ... , h_N] * W
        # a_input_side = torch.cat([h_side.repeat(1, 1, N).view(self.bs, N * N, -1), h_side.repeat(1, N, 1)], dim=2).view(
        #             self.bs, N, -1, 2 * self.out_features)
        # e_side = self.leakyrelu(torch.matmul(a_input_side, self.aS).squeeze(3))
        # attention_side = torch.where(adj > 0, e_side, zero_vec)  # shape = (bs, N, N)
        # attention_side = F.dropout(attention_side, self.dropout, training=self.training)  # shape = (bs, N, N)

        # Find the crime representation of a region: c_{i,t}^k
        attention = attention 
        attention = torch.where(attention > 0, attention, zero_vec)  # shape = (bs, N, N)
        attention = F.softmax(attention, dim=2)  # shape = (bs, N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (bs, N, N)
        h_prime = torch.matmul(attention, h)  # shape = (bs, N, F')

        """
            Step 2: Generate e_{i,t}^k
        """
        # Generate Query Vector
        # q = torch.cat([input.repeat(1, 1, N).view(input.shape[0], N * N, -1), input.repeat(1, N, 1)], dim=2).view(
        #     input.shape[0], N, N, -1)
        # q = torch.matmul(q, self.WQ)  # (bs, N, N, dq) = (bs, N, N, 2) * (2, dq)
        # q = q / (self.att_dim ** 0.5)
        # q = q.unsqueeze(3)  # (bs, N, N, 1, dq)

        # # Generate Key Vector
        # ext_input = ext_input.unsqueeze(3) #(bs, num_of_region, num_of_feature = 12, 1)

        # k = torch.cat([ext_input.repeat(1, 1, N, 1).view(ext_input.shape[0], N * N, self.in_xfeatures, -1),
        #                ext_input.repeat(1, N, 1, 1).view(ext_input.shape[0], N * N, self.in_xfeatures, -1)], dim=3)

        # k = k.view(ext_input.shape[0], N, N, self.in_xfeatures, 2)
        # k = torch.matmul(k, self.WK)  # (bs, N, N, in_xfeatures, dk) = (bs, N, N, in_xfeatures, 2)* (2, dk)
        # k = torch.transpose(k, 4, 3)  # (bs, N, N, dk, in_xfeatures)

        # # Generate Value Vector
        # v = torch.matmul(ext_input, self.WV)  # (bs, N, N, in_xfeatures, dv)

        # # Generate dot product attention
        # dot_attention = torch.matmul(q, k).squeeze(3)  # (bs, N, N, in_xfeatures)
        # zero_vec = -9e15 * torch.ones_like(dot_attention)
        # dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (bs, N, N, in_xfeatures)
        # dot_attention = F.softmax(dot_attention, dim=3)  # shape = (bs, N, N, in_xfeatures)

        # # Generate the external feature representation of the regions: e_{i,t}^k
        # crime_attention = attention.unsqueeze(3).repeat(1, 1, 1, self.in_xfeatures)
        # final_attention = dot_attention * crime_attention
        # ext_rep = torch.matmul(final_attention, v)  # (bs, N, N, dv)
        # ext_rep = ext_rep.sum(dim=2)  # (bs, N, N, dv)

        if self.concat:
            # return F.elu(h_prime), F.elu(ext_rep)
            return F.elu(h_prime)
        else:
            return h_prime
            # return h_prime, ext_rep

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_cfeatures) + ' -> ' + str(self.out_features) + ')'



class AIST(nn.Module):
    def __init__(self, ncfeat, nxfeat, gout, gatt, rhid, ratt, rlayer, bs, rts, city, tr, tc, userid, fgat = True):
        super(AIST, self).__init__()
        self.ncfeat = ncfeat
        self.nxfeat = nxfeat
        self.gout = gout
        self.gatt = gatt
        self.rhid = rhid
        self.ratt = ratt
        self.rlayer = rlayer
        self.bs = bs
        self.rts = rts
        self.tr = tr
        self.city = city
        self.tc = tc
        self.userid= userid
        self.fgat = fgat

        self.smod = Spatial_Module(self.ncfeat, self.nxfeat, self.gout, self.gatt, 0.5, 0.6, self.rts, self.bs,
                        self.tr, self.tc, self.city, self.userid, self.fgat)
        if(fgat):
            sab1_input = 2 * self.gout
        else:
            sab1_input = 1 * self.gout
        self.sab1 = self_LSTM_sparse_attn_predict(sab1_input, self.rhid, self.rlayer, 1,
                    truncate_length=5, top_k=4, attn_every_k=5, predict_m=10)
        self.sab2 = self_LSTM_sparse_attn_predict(1, self.rhid, self.rlayer, 1,
                    truncate_length=5, top_k=4, attn_every_k=5, predict_m=10)
        self.sab3 = self_LSTM_sparse_attn_predict(1, self.rhid, self.rlayer, 1,
                    truncate_length=1, top_k=3, attn_every_k=1, predict_m=10)

        self.fc1 = nn.Linear(self.rhid, 1)

        self.wv = nn.Linear(self.rhid, self.ratt)  # (S, E) x (E, 1) = (S, 1)
        self.wu = nn.Parameter(torch.zeros(size=(self.bs, self.ratt)))  # attention of the trends
        nn.init.xavier_uniform_(self.wu.data, gain=1.414)
        self.dropout_layer = nn.Dropout(p=0.2)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    
    def forward(self, x_crime, x_crime_daily, x_crime_weekly, x_regions, x_ext, s_crime):
        x_crime = self.smod(x_crime, x_regions, x_ext, s_crime)
        x_con, x_con_attn = self.sab1(x_crime)  # (bs, rts)
        x_con = self.dropout_layer(x_con)
        x_con = x_con.unsqueeze(1)

        x_daily, x_daily_attn = self.sab2(x_crime_daily)  # x_daily = (bs, dts=20)
        x_daily = self.dropout_layer(x_daily)
        x_daily = x_daily.unsqueeze(1)

        x_weekly, x_weekly_attn = self.sab3(x_crime_weekly)  # x_weekly = (bs, wts=3):
        x_weekly = self.dropout_layer(x_weekly)
        x_weekly = x_weekly.unsqueeze(1)

        x = torch.cat((x_con, x_daily, x_weekly), 1)

        um = torch.tanh(self.wv(x))  # (bs, 3, ratt)
        um = um.transpose(2, 1)  # (bs, ratt, 3)
        wu = self.wu.unsqueeze(1)
        alpha_m = torch.bmm(wu, um)  # (bs, 1, 3)
        alpha_m = alpha_m.squeeze(1)  # (bs, 3)
        alpha_m = torch.softmax(alpha_m, dim=1)
        attn_trend = alpha_m.detach()
        alpha_m = alpha_m.unsqueeze(1)

        x = torch.bmm(alpha_m, x)
        x = x.squeeze(1)
        x = torch.tanh(self.fc1(x))

        return x, attn_trend


class Spatial_Module(nn.Module):
    def __init__(self, ncfeat, nxfeat, nofeat, gatt, dropout, alpha, ts, bs, tr, tc, city, userid, fgat=True):
        super(Spatial_Module, self).__init__()
        self.ncfeat = ncfeat
        self.nxfeat = nxfeat
        self.nofeat = nofeat
        self.att = gatt
        self.bs = bs
        self.ts = ts
        self.tr = tr
        self.tc = tc
        self.city = city
        self.userid = userid
        self.fgat = fgat
        self.gat = [GraphAttentionLayer(self.ncfeat, self.nxfeat, self.nofeat, self.att, self.bs, dropout=dropout,
                    alpha=alpha) for _ in range(self.ts)]
        for i, g in enumerate(self.gat):
            self.add_module('gat{}'.format(i), g)

        self.device = None
        if(torch.cuda.is_available()):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x_crime, x_regions, x_ext, s_crime):
        T = x_crime.shape[1]
        tem_x_regions = x_regions.copy()
        reg = gen_neighbor_index_zero_with_target(self.tr, self.city)
        label = torch.tensor(reg).to(self.device)  #! here 1st mistake got for not passing the reg in cuda, so passed the reg in cuda later
        label = label.repeat(T * self.bs, 1)  # (T*bs, N)
        label = label.view(label.shape[0] * label.shape[1], 1).long()  # (T * bs * N, 1)
        x_crime = x_crime.transpose(1, 0)  # (T, bs)
        tem_x_regions.append(x_crime)

        N = len(tem_x_regions)  # Num of actual nodes
        feat = torch.stack(tem_x_regions, 2)  # (T, bs, N)
        feat = feat.view(feat.shape[0] * feat.shape[1] * feat.shape[2], 1).long()  # (T*bs*N, 1)
        feat = torch.cat([label, feat], dim=1)  # (T*bs*N, 2) --> (Node Label, features)
        feat = feat.view(T, self.bs * N, 2)

        # x_ext :(num_of_regions, time_step_width = 120, batch_size = 42, num_of_feature = 12)
        feat_ext = torch.stack(x_ext, 2) # will stack #num_of_regions tensor of shape(time_step_width, batch_size, num_of_features) in dim=2(new) -> (time_step_width, batch_size, num_of_regions, num_of_features)
        feat_ext = feat_ext.view(feat_ext.shape[0] * feat_ext.shape[1] * feat_ext.shape[2], -1).long()  # (T*bs*N, nxfeat)
        feat_ext = torch.cat([label, feat_ext], dim=1)  # (T*bs*N, nxfeature+1)
        feat_ext = feat_ext.view(T, self.bs * N, self.nxfeat + 1) #(T, bs*N, nxfeature+1)

        crime_side = torch.stack(s_crime, 2)
        crime_side = crime_side.view(crime_side.shape[0] * crime_side.shape[1] * crime_side.shape[2], -1).long()  # (T*bs*N, 1)
        crime_side = torch.cat([label, crime_side], dim=1)  # (T*bs*N, 2)
        crime_side = crime_side.view(T, self.bs * N, 2)  # (T, bs*N, 2)

        spatial_output = []
        j = 0
        for i in range(T-self.ts, T):
            np.savetxt(f"{DATA_WRITE_FILE_PATH}/external/{self.userid}/gat_feat.txt", feat[i].cpu(), fmt='%d')
            np.savetxt(f"{DATA_WRITE_FILE_PATH}/external/{self.userid}/gat_feat_ext.txt", feat_ext[i].cpu(), fmt='%d')
            np.savetxt(f"{DATA_WRITE_FILE_PATH}/external/{self.userid}/gat_crime_side.txt", crime_side[i].cpu(), fmt='%d')
            adj, features, features_ext, crime_side_features = load_data_GAT(self.bs, self.userid)

            # out, ext = self.gat[j](features.to(self.device), adj.to(self.device), features_ext.to(self.device), crime_side_features.to(self.device))  # (N, F')(N, N, dv)               out = out[:, -1, :]
            out = self.gat[j](features.to(self.device), adj.to(self.device), features_ext.to(self.device), crime_side_features.to(self.device))  # (N, F')(N, N, dv)               out = out[:, -1, :]
            # print('out shape in Spatial Module forward ',out.shape)
            out = out[:, -1, :]
            # ext = ext[:, -1, :]
            # if(self.fgat):
            #   out = torch.stack((out, ext), dim=2)
            spatial_output.append(out)
            j = j + 1
            # print(f'spatial output shape after {i}th iteration {spatial_output.shape}')

        spatial_output = torch.stack(spatial_output, 1)
        # print(" spatial_output shape : ", spatial_output.shape )
        return spatial_output



class AistClient(fl.client.NumPyClient):
    def __init__(self,target_city,target_region,target_cat,time_step,recent_time_step,batch_size,gat_out,gat_att,ncfeature,nxfeature,slstm_nhid,slstm_nlayer,slstm_att,userid,partition_id,data_folder='data',round = -1, fgat = True):
        self.target_city = target_city
        self.target_region = target_region
        self.target_cat = target_cat
        self.time_step = time_step
        self.recent_time_step = recent_time_step
        self.batch_size = batch_size
        self.gat_out = gat_out
        self.gat_att = gat_att
        self.ncfeature = ncfeature
        self.nxfeature = nxfeature
        self.slstm_nhid = slstm_nhid
        self.slstm_nlayer = slstm_nlayer
        self.slstm_att = slstm_att
        self.criterion = nn.L1Loss()
        self.userid = userid
        self.crime_count_per_client = 0

        #adding round variable by Farhan
        self.round = round

        #here in place of data in data_folder, pass the mouted folder path for the dataset

        gen_gat_adj_file(self.target_city, self.target_region, self.userid)   # generate the adj_matrix file for GAT layers
        loaded_data = None
        if(self.userid == f"server_{save_file_path}"):
          loaded_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH+"/" + target_city + f"/{self.userid}/r_" + str(target_region) + ".txt", dtype=int)).T
        #   modified by changra farhan
          loaded_aggregated_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH + "/"+target_city+'/aggregated/r_'+str(target_region)+'.txt',dtype=int)).T
          loaded_aggregated_data = loaded_aggregated_data[:,target_cat:target_cat+1]
        else:
          loaded_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH+"/" + target_city + f"/user_{save_file_path}/{self.userid}/r_" + str(target_region) + ".txt", dtype=int)).T
        loaded_data = loaded_data[:, target_cat:target_cat+1]
        loaded_test_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH + "/" + target_city + "/test/r_" + str(target_region) + ".txt", dtype=int)).T
        loaded_test_data = loaded_test_data[:, target_cat:target_cat+1]

        train_x = None
        train_y = None
        train_x_daily = None
        train_x_weekly = None
        test_x = None
        test_y = None
        test_x_daily = None
        test_x_weekly = None
        partition_list = np.loadtxt('server_partition.txt', dtype=int)
        partition_list = partition_list[partition_id]
        if(self.userid ==f"server_{save_file_path}"):
          train_x, train_y, train_x_daily, train_x_weekly = create_inout_sequences_for_server(loaded_data, partition_list,keep_only_list_indices=False) # here loaded_data is actually loaded_train_data
          train_x_for_aggregated_data, train_y_aggregated_data, train_x_daily_aggregated_data, train_x_weekly_aggregated_data = create_inout_sequences_for_server(loaded_aggregated_data, partition_list,keep_only_list_indices=False) # here loaded_aggregated_data is actually loaded_aggregated_data

        else:
          loaded_server_data = torch.from_numpy(np.loadtxt(DATA_READ_FILE_PATH+"/" + target_city + f"/server_{save_file_path}/r_" + str(target_region) + ".txt", dtype=int)).T
          loaded_server_data = loaded_server_data[:, target_cat:target_cat+1]
          filtering(self.userid, loaded_data, loaded_server_data)
          train_x, train_y, train_x_daily, train_x_weekly = create_inout_sequences_for_client(loaded_data, loaded_server_data, partition_list, userid=self.userid)

        test_x, test_y, test_x_daily, test_x_weekly = create_inout_sequences(loaded_test_data)

        # ---------------------- Getting crime count per client from train_x by Farhan --------------
        # --------------------------- Smoothing the weight of the client by Farhan and Sukarna Sir ------------------
        total_crimes_of_client = torch.sum(train_x)
        total_crimes_of_client_count = total_crimes_of_client.item()
        self.crime_count_per_client = total_crimes_of_client_count

        #print the shape of the train_x
        # print('train_x shape for inout sequence ',train_x.shape)
        # print('test_x shape for inout sequence ',test_x.shape)


        # print("train_x: ", train_x.shape)
        # print("test_x: ", test_x.shape)
        # if(self.userid == f"server_{save_file_path}"):
            # test_x_debug_file = open(f'./Lunar_Lander/test_x_debug_{server_percentage}.txt','w')
            # print(test_x,file=test_x_debug_file)
        # ---------------- New Test Scale created by Changra Farhan For Test Shape Solving Problem---------------
        #  The problem was: For different server percentage 0.30,0.50 etc, the test data was scaled from different min max value, thats why
        #  though seed was used, In the server round 0, the result was different where server was never trained, It was the difference of test data ---------------
        aggregated_data_scaler = MinMaxScaler(feature_range=(-1,1))
        scale = MinMaxScaler(feature_range=(-1, 1))
        if(self.userid ==f"server_{save_file_path}"):
            train_x_aggregated_scaled=torch.from_numpy(aggregated_data_scaler.fit_transform(train_x_for_aggregated_data))
            test_x = torch.from_numpy(aggregated_data_scaler.transform(test_x))
            train_x_daily_aggregated_scaled = torch.from_numpy(aggregated_data_scaler.fit_transform(train_x_daily_aggregated_data))
            test_x_daily = torch.from_numpy(aggregated_data_scaler.transform(test_x_daily))
            train_x_weekly_aggregated_scaled = torch.from_numpy(aggregated_data_scaler.fit_transform(train_x_weekly_aggregated_data))
            test_x_weekly  = torch.from_numpy(aggregated_data_scaler.transform(test_x_weekly))
            train_y_aggregated_scaled = torch.from_numpy(aggregated_data_scaler.fit_transform(train_y_aggregated_data))
            test_y = torch.from_numpy(aggregated_data_scaler.transform(test_y))

        # this line is not useful now
        train_x = torch.from_numpy(scale.fit_transform(train_x))
        # test_x = torch.from_numpy(scale.transform(test_x))
        # joblib.dump(scale,f'./saved_models/r_{target_region}/r_{target_region}_c_{target_cat}_x.save')

        train_x_daily = torch.from_numpy(scale.fit_transform(train_x_daily))
        # joblib.dump(scale,f'./saved_models/r_{target_region}/r_{target_region}_c_{target_cat}_x_daily.save')

        train_x_weekly = torch.from_numpy(scale.fit_transform(train_x_weekly))
        # joblib.dump(scale,f'./saved_models/r_{target_region}/r_{target_region}_c_{target_cat}_x_weekly.save')

        train_y = torch.from_numpy(scale.fit_transform(train_y))
        # joblib.dump(scale,f'./saved_models/r_{target_region}/r_{target_region}_c_{target_cat}_y.save')
        self.scale = scale
        self.aggregated_data_scaler = aggregated_data_scaler

        # Divide your data into train set & test set


        train_extra = train_x.shape[0] % batch_size
        train_x = train_x[:train_x.shape[0]-train_extra, :]
        train_x_daily = train_x_daily[:train_x_daily.shape[0]-train_extra, :]
        train_x_weekly = train_x_weekly[:train_x_weekly.shape[0]-train_extra, :]
        train_y = train_y[:train_y.shape[0]-train_extra, :]

        self.train_x_size = train_x.shape[0]

        test_extra = test_x.shape[0] % batch_size
        test_x = test_x[:test_x.shape[0] - test_extra, :]  # 11 is subtracted to make ns_te compatible with bs
        test_x_daily = test_x_daily[:test_x_daily.shape[0] - test_extra, :]
        test_x_weekly = test_x_weekly[:test_x_weekly.shape[0] - test_extra, :]
        test_y = test_y[:test_y.shape[0] - test_extra, :]

        self.test_x_size = test_x.shape[0]


        # print('test_x shape ',test_x.shape)
        # Divide it into batches
        self.train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)  # (nb=num of batches, bs, rts)
        self.train_x_daily = train_x_daily.view(int(train_x_daily.shape[0] / batch_size), batch_size, train_x_daily.shape[1])  # (nb, bs, dts)
        self.train_x_weekly = train_x_weekly.view(int(train_x_weekly.shape[0] / batch_size), batch_size, train_x_weekly.shape[1])  # (nb, bs, rws)
        self.train_y = train_y.view(int(train_y.shape[0] / batch_size), batch_size, 1)

        self.test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)
        self.test_x_daily = test_x_daily.view(int(test_x_daily.shape[0] / batch_size), batch_size, test_x_daily.shape[1])
        self.test_x_weekly = test_x_weekly.view(int(test_x_weekly.shape[0] / batch_size), batch_size, test_x_weekly.shape[1])
        self.test_y = test_y.view(int(test_y.shape[0] / batch_size), batch_size, 1)


        # load data for external_features and side_features
        self.train_feat, self.test_feat = load_data_regions(batch_size, target_cat, target_region, target_city, userid,partition_list)

        self.train_feat_ext, self.test_feat_ext = load_data_regions_external(batch_size, nxfeature, target_region, target_city)
        # print(len(self.train_feat_ext), len(self.train_feat_ext[0]), self.train_feat_ext[0][0].shape)

        self.train_crime_side, self.test_crime_side = load_data_sides_crime(batch_size, target_cat, target_region, target_city, userid,partition_list)

        # Model and optimizer
        self.aist_model = AIST(ncfeature, nxfeature, gat_out, gat_att, slstm_nhid, slstm_att, slstm_nlayer, batch_size,
                    recent_time_step, target_city, target_region, target_cat, userid = self.userid, fgat = fgat)
        n = sum(p.numel() for p in self.aist_model.parameters() if p.requires_grad)

        if(torch.cuda.is_available()):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.train_x = self.train_x.to(device)
        self.train_x_daily = self.train_x_daily.to(device)
        self.train_x_weekly = self.train_x_weekly.to(device)
        self.train_y = self.train_y.to(device)

        self.test_x = self.test_x.to(device)
        self.test_x_daily = self.test_x_daily.to(device)
        self.test_x_weekly = self.test_x_weekly.to(device)
        self.test_y = self.test_y.to(device)

        self.aist_model = self.aist_model.to(device)

    def get_model(self):
      return self.aist_model.state_dict()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.aist_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.aist_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.aist_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # ------- changing the train function for FED PROX 
        self.train(config)
        if(args.aggr_mthd == 0):
            count = self.train_x_size
        else:
            count = self.crime_count_per_client
        # return self.get_parameters(config={}), self.crime_count_per_client, {}
        return self.get_parameters(config={}), count, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = self.compute_test()
        return float(loss), self.test_x_size, {"mae": metrics["mae"], "mse": metrics["mse"]}

    def train(self,config=None):
        # if('server' not in self.userid):
            # print(f'Weight for {self.userid} : {self.get_parameters(config={})[0][0][0]}')
            # print(f'-------------- server trained yeah ----------------')
        lr = learning_rate
        weight_decay = 5e-4
        optimizer = optim.Adam(self.aist_model.parameters(), lr=lr)
        # criterion = nn.MSELoss()


        # epochs = 1
        # epochs = 5
        epochs = EPOCHS
        best = epochs + 1
        best_epoch = 0
        t_total = time.time()
        loss_values = []
        mae_train_loss_values = [] #TODO; newly added for scaled up training errors
        bad_counter = 0
        patience = 100
        best_model = self.aist_model.state_dict()
        train_batch = self.train_x.shape[0]

        #writing epoch and training loss in a csv file
        server_training_loss_storing_csv_file = f'./external/r_{test_region}_{save_file_path}/server_training_loss_vs_rounds.csv'

        #--------------- get the global parameters for fed prox --------------------- 
        # global_params = [val.detach().clone() for val in self.aist_model.parameters()]
        # # -------------- get the proximal mu from the config -------------------------
        # _proximal_mu_ = config['proximal_mu']

        with open(server_training_loss_storing_csv_file, mode='a', newline='') as csv_file:
          for epoch in range(epochs):
              i = 0
              #TODO: here setting the mae_train = 0
              mae_train = 0
              loss_values_batch = []
              for i in range(train_batch):
                  t = time.time()

                  x_crime = Variable(self.train_x[i]).float()
                  x_crime_daily = Variable(self.train_x_daily[i]).float()
                  x_crime_weekly = Variable(self.train_x_weekly[i]).float()
                  y = Variable(self.train_y[i]).float()

                  self.aist_model.train()
                  optimizer.zero_grad()
                  output, attn = self.aist_model(x_crime, x_crime_daily, x_crime_weekly, self.train_feat[i], self.train_feat_ext[i], self.train_crime_side[i])
                  y = y.view(-1, 1)
                  
                #  -------------------------- updated for Fed Prox ----------------
                #   proximal_term = 0.0
                #   local_model_params = self.aist_model.parameters()
                #   for local_weights, global_weights in zip(local_model_params, global_params):
                #         proximal_term += torch.square((local_weights - global_weights).norm(2))
                #   loss_train = self.criterion(output, y) + (_proximal_mu_ / 2) * proximal_term
                  loss_train = self.criterion(output, y)
                #   ------------------- update done ------------------------
                  loss_train.backward()
                  optimizer.step()
                  mae_train = MAE(self.scale.inverse_transform(y.cpu().detach().numpy()), self.scale.inverse_transform(output.cpu().detach().numpy()))

                  # now store the training loss for batch_epoch as it was done in the aggregated AIST model
                #   print('Epoch: {:04d}'.format(epoch*train_batch + i + 1),
                #       'loss_train: {:.4f}'.format(loss_train.data.item()),
                #       'time: {:.4f}s'.format(time.time() - t))

                  # print('loss train ',loss_train.data.item())
                  loss_values.append(loss_train.data.item())
                  mae_train_loss_values.append(mae_train)
                #   torch.save(self.aist_model.state_dict(), f'{DATA_WRITE_FILE_PATH}/external/{self.userid}/{epoch*train_batch + i + 1}.pkl')
                  if loss_values[-1] < best:
                      best = loss_values[-1]
                      best_epoch = epoch*train_batch + i + 1
                      bad_counter = 0
                    
                  else:
                      bad_counter += 1

                  if bad_counter == patience:
                      break

                #   files = glob.glob(f'{DATA_WRITE_FILE_PATH}/external/{self.userid}/*.pkl')
                #   for file in files:
                #       epoch_nb = int(file.split('/')[-1].split('.')[0])
                #       if epoch_nb < best_epoch:
                #           os.remove(file)

            #   files = glob.glob(f'{DATA_WRITE_FILE_PATH}/external/{self.userid}/*.pkl')
            #   for file in files:
            #       epoch_nb = int(file.split('/')[-1].split('.')[0])
            #       if epoch_nb > best_epoch:
            #           os.remove(file)

            #   if epoch*train_batch + i + 1 >= 800:
            #       break

          if(self.userid == f"server_{save_file_path}"):
            average_training_loss_for_this_round = sum(mae_train_loss_values)/len(mae_train_loss_values)
            print('average training loss for round ',self.round, ' is ',average_training_loss_for_this_round)
            if(self.round==1):
              csv_file.write('round,loss\n')
            # csv_file.write(f'{self.round},{best}\n')
            csv_file.write(f'{self.round},{average_training_loss_for_this_round}\n')
        

    def compute_train(self):
        loss = 0
        mae = 0
        mse = 0
        train_batch = self.train_x.shape[0]
        for i in range(train_batch):
            self.aist_model.eval()

            x_crime_train = Variable(self.train_x[i]).float()
            x_crime_daily_train = Variable(self.train_x_daily[i]).float()
            x_crime_weekly_train = Variable(self.train_x_weekly[i]).float()
            y_train = Variable(self.train_y[i]).float()


            output_train, list_att = self.aist_model(x_crime_train, x_crime_daily_train, x_crime_weekly_train, self.train_feat[i], self.train_feat_ext[i], self.train_crime_side[i])
            y_train = y_train.view(-1, 1)
            y_train = torch.from_numpy(self.scale.inverse_transform(y_train.cpu()))
            output_train = torch.from_numpy(self.scale.inverse_transform(output_train.cpu().detach()))

            # self.stat_y.append(y_test.detach().numpy())
            # self.stat_y_prime.append(output_test.numpy())

            loss_train = self.criterion(output_train, y_train)
            mae_train = MAE(y_train.detach().numpy(), output_train.numpy())
            mse_train = MSE(y_train.detach().numpy(), output_train.numpy())

            # for j in range(42):
            #     print(y_test[j, :].data.item(), output_test[j, :].data.item())

            loss += loss_train.data.item()
            mae += mae_train
            mse += mse_train

            # print("Test set results:",
            #     "loss= {:.4f}".format(loss_test.data.item()))
            # print("mae= {:.4f}".format(mae_test))
            # print("mse= {:.4f}".format(mse_test))

        #print(self.target_region, " ", self.target_cat, " ", loss/i)
        #print(self.target_region, " ", self.target_cat, " ", loss/i, file=f)
        return loss/train_batch, {"mae": mae/train_batch, "mse": mse/train_batch}

    def compute_test(self):
        #f = open('result/aist.txt','a')
        stat_y = []
        stat_y_prime = []
        loss = 0
        mae = 0
        mse = 0
        test_batch = self.test_x.shape[0]
        result_file = None
        if(self.userid == f"server_{save_file_path}"):
            result_file = open(f'./external/r_{test_region}_{save_file_path}/round_{self.round}_y_pred.csv','w')
            result_file.write('y_true,y_pred\n')

        for i in range(test_batch):
            self.aist_model.eval()

            x_crime_test = Variable(self.test_x[i]).float()
            x_crime_daily_test = Variable(self.test_x_daily[i]).float()
            x_crime_weekly_test = Variable(self.test_x_weekly[i]).float()
            y_test = Variable(self.test_y[i]).float()


            output_test, list_att = self.aist_model(x_crime_test, x_crime_daily_test, x_crime_weekly_test, self.test_feat[i], self.test_feat_ext[i], self.test_crime_side[i])
            y_test = y_test.view(-1, 1)
            # here changing by changra farhan for test scaling problem solving
            # y_test = torch.from_numpy(self.scale.inverse_transform(y_test.cpu()))
            # output_test = torch.from_numpy(self.scale.inverse_transform(output_test.cpu().detach()))

            y_test = torch.from_numpy(self.aggregated_data_scaler.inverse_transform(y_test.cpu()))
            output_test = torch.from_numpy(self.aggregated_data_scaler.inverse_transform(output_test.cpu().detach()))

            stat_y.append(y_test.detach().numpy())
            stat_y_prime.append(output_test.numpy())

            loss_test = self.criterion(output_test, y_test)
            mae_test = MAE(y_test.detach().numpy(), output_test.numpy())
            mse_test = MSE(y_test.detach().numpy(), output_test.numpy())
            if(self.userid == f"server_{save_file_path}"):
                for j in range(self.batch_size):
                    result_file.write(f'{y_test[j, :].data.item()},{output_test[j, :].data.item()}\n')

            loss += loss_test.data.item()
            mae += mae_test
            mse += mse_test

            # print("Test set results:",
            #     "loss= {:.4f}".format(loss_test.data.item()))
            # print("mae= {:.4f}".format(mae_test))
            # print("mse= {:.4f}".format(mse_test))

        #print(self.target_region, " ", self.target_cat, " ", loss/i)
        #print(self.target_region, " ", self.target_cat, " ", loss/i, file=f)
        return loss/test_batch, {"mae": mae/test_batch, "mse": mse/test_batch}


from typing import Callable, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from functools import reduce
from logging import WARNING
from flwr.common.logger import log



class FedCustom(fl.server.strategy.FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        # proximal_mu : float = 1.0
        # initial_parameters : Parameters = None,
        # eta : float = 1e-1,
        # eta_l: float = 1e-1,
        # beta_1 : float = 0.0,
        # beta_2 : float = 0.0,
        # tau: float = 1e-9
    ) -> None:
        super().__init__(fraction_fit = fraction_fit, fraction_evaluate= fraction_evaluate,
                        min_fit_clients = min_fit_clients, min_evaluate_clients = min_evaluate_clients,
                        min_available_clients = min_available_clients)
        # super().__init__(fraction_fit = fraction_fit, fraction_evaluate= fraction_evaluate,
        #         min_fit_clients = min_fit_clients, min_evaluate_clients = min_evaluate_clients,
        #         min_available_clients = min_available_clients,initial_parameters = initial_parameters,eta=eta)                
        # super().__init__(fraction_fit = fraction_fit, fraction_evaluate= fraction_evaluate,
        #                 min_fit_clients = min_fit_clients, min_evaluate_clients = min_evaluate_clients,
        #                 min_available_clients = min_available_clients, proximal_mu=proximal_mu)
        # super().__init__(fraction_fit = fraction_fit, fraction_evaluate= fraction_evaluate,
        #                 min_fit_clients = min_fit_clients, min_evaluate_clients = min_evaluate_clients,
        #                 min_available_clients = min_available_clients,eta = eta, eta_l = eta_l,beta_1 = beta_1, beta_2 = beta_2, tau = tau)
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        if(os.path.isdir(f'{DATA_WRITE_FILE_PATH}/external/server_{save_file_path}')):
          shutil.rmtree(f'{DATA_WRITE_FILE_PATH}/external/server_{save_file_path}')
        os.makedirs(f'{DATA_WRITE_FILE_PATH}/external/server_{save_file_path}')

        self.server_model = AistClient(target_city='chicago',
                                       target_region=test_region, target_cat=args.target_cat, batch_size=BATCH_SIZE,
                                       time_step=120, recent_time_step=20, ncfeature=1, nxfeature=12,
                                       gat_out=8, gat_att=40, slstm_nhid=hidden_state_of_lstm, slstm_nlayer=1,
                                       slstm_att=30,data_folder=f'{DATA_READ_FILE_PATH}/chicago2/', round = 0, userid=f'server_{save_file_path}',partition_id=participant_clients-1, fgat=False)

        # print('server model : ',(self.server_model.aist_model.state_dict()))
        # self.farhan_model = self.server_model
        # for param_tensor in self.farhan_model.aist_model.state_dict():
            # print(param_tensor, "\t", self.farhan_model.aist_model.state_dict()[param_tensor].size())
        # count_parameters(self.server_model)

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # trained_param, _, __ = self.server_model.fit(config={},parameters=self.server_model.get_parameters(config={}))
        # ndarrays = trained_param
        ndarrays = self.server_model.get_parameters(config={})
        return fl.common.ndarrays_to_parameters(ndarrays)


    def farhan_aggregate_for_fedsum(self,results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute Normal Fed Sum"""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            #TODO: converted for FEDSUM
            [layer * 1 for layer in weights] for weights, num_examples in results
            # [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            #TODO: converted for FEDSUM
            reduce(np.add, layer_updates) / 1
            # reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def farhan_aggregate_for_weight_smoothing(self,results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute Normal Fed Sum"""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])
        exp_total = sum([math.exp(1.2*(num_examples/num_examples_total)) for _, num_examples in results])
        # weight_total = sum([(num_examples/num_examples_total) for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        # --------- calculate the crime proportion -------------
        weighted_weights = [
            #  converted for FEDSUM
            [layer * ((math.exp(1.2*(num_examples/num_examples_total)))/exp_total) for layer in weights] for weights, num_examples in results
            # [layer * (num_examples/num_examples_total) for layer in weights] for weights, num_examples in results
            # [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            # converted for FEDSUM
            reduce(np.add, layer_updates) / 1
            # reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def farhan_aggregate_fit_for_fedsum(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self.farhan_aggregate_for_fedsum(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided, warning by Farhan")

        return parameters_aggregated, metrics_aggregated

    def farhan_aggregate_fit_for_weight_smoothing(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        if(args.aggr_mthd == 0 or args.aggr_mthd == 1):
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        elif(args.aggr_mthd == 2):
            parameters_aggregated = ndarrays_to_parameters(self.farhan_aggregate_for_weight_smoothing(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided, warning by Farhan")

        return parameters_aggregated, metrics_aggregated

    # # ---------------------- using our modified configure_fit function for dynamic proximal_mu, the default proximal_mu is used in the fed_prox original code base -------------
    # def configure_fit(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     """Configure the next round of training.

    #     Sends the proximal factor mu to the clients
    #     """
    #     # Get the standard client/config pairs from the FedAvg super-class
    #     client_config_pairs = super().configure_fit(
    #         server_round, parameters, client_manager
    #     )

    #     # Return client/config pairs with the proximal factor mu added
    #     # ------------------ changing the proximal_mu at each round using an exponential function -------------------
    #     self.proximal_mu = Fed_Prox_Proximal_MU*(1-math.exp(-1*0.5*server_round))
    #     return [
    #         (
    #             client,
    #             FitIns(
    #                 fit_ins.parameters,
    #                 {**fit_ins.config, "proximal_mu": self.proximal_mu},
    #             ),
    #         )
    #         for client, fit_ins in client_config_pairs
    #     ]
    # def configure_fit(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     """Configure the next round of training."""

    #     # Sample clients
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )

    #     # Create custom configs
    #     n_clients = len(clients)
    #     half_clients = n_clients // 2
    #     standard_config = {"lr": 0.001}
    #     higher_lr_config = {"lr": 0.003}
    #     fit_configurations = []
    #     for idx, client in enumerate(clients):
    #         if idx < half_clients:
    #             fit_configurations.append((client, FitIns(parameters, standard_config)))
    #         else:
    #             fit_configurations.append(
    #                 (client, FitIns(parameters, higher_lr_config))
    #             )
    #     return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     for _, fit_res in results
        # ]
        # parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        # client_parameters, metrics_aggregated = super().aggregate_fit(server_round=server_round, results=results, failures=failures)
        '''
            Uncomment the following line to use fedsum
        '''
        # client_parameters, metrics_aggregated = self.farhan_aggregate_fit_for_fedsum(server_round=server_round, results=results, failures=failures)
        client_parameters, metrics_aggregated = self.farhan_aggregate_fit_for_weight_smoothing(server_round=server_round, results=results, failures=failures)
        client_parameters = parameters_to_ndarrays(client_parameters)
        # self.server_model.set_parameters(client_parameters)
        self.server_model.round = server_round
        # trained_param, _, __ = self.server_model.fit(config={},parameters=client_parameters)
        trained_param = client_parameters
        torch.save(self.server_model.get_model(),
                  f'./external/r_{test_region}_{save_file_path}/{server_round}.pkl')
        parameters_aggregated = ndarrays_to_parameters(trained_param)
        return parameters_aggregated, metrics_aggregated

    # def configure_evaluate(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, EvaluateIns]]:
    #     """Configure the next round of evaluation."""
    #     if self.fraction_evaluate == 0.0:
    #         return []
    #     config = {}
    #     evaluate_ins = EvaluateIns(parameters, config)

    #     # Sample clients
    #     sample_size, min_num_clients = self.num_evaluation_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )

    #     # Return client/config pairs
    #     return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""


        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        parameters = parameters_to_ndarrays(parameters)
        loss, __,  metrics = self.server_model.evaluate(parameters = parameters, config={})
        train_loss, train_metrics = self.server_model.compute_train()
        if(server_round == 0):
          with open(f'./external/r_{test_region}_{save_file_path}/Evaluation_Loss_MAE_MSE_FED.csv', 'w') as file:
            file.write(f'round,mae,mse,train_mae,train_mse\n')
        with open(f'./external/r_{test_region}_{save_file_path}/Evaluation_Loss_MAE_MSE_FED.csv', 'a') as file:
          file.write(f'{server_round},{metrics["mae"]},{metrics["mse"]},{train_metrics["mae"]},{train_metrics["mse"]}\n')
        print(f'After round {server_round}: \n\
              train_metrics : {train_metrics}\n\
              test_metrics : {metrics}')

        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients




crimes_with_uid_csv = pd.read_csv(f'{root_folder}/crimes_with_uid_{save_file_path}.csv')
crimes_of_current_region = crimes_with_uid_csv[crimes_with_uid_csv['neighbourhood_id']== (test_region+1)]
uid_row_count = crimes_of_current_region['uid'].value_counts().reset_index()
uid_row_count.columns = ['uid', 'crime_count']

# Sort the DataFrame by 'uid'
uid_row_count = uid_row_count.sort_values(by='crime_count',ascending=False)
# uid_row_count.to_csv(f'region_{test_region}_crime_counts.csv', index=False)
# print('uid row count dataframe ',uid_row_count)


total_users_of_this_region = uid_row_count['uid'].unique().tolist()
# Get the top 1946 uid values
top_uid_list = None
if top == True:
    # top_uid_list = uid_row_count.head(sample_clients)['uid'].tolist()
    top_uid_list = uid_row_count.head(participant_clients)['uid'].tolist()
else:
    top_uid_list = total_users_of_this_region
# top_uid_list = uid_row_count.head(760)['uid'].tolist()
# uid_list_neighbourhood_test_region = crimes_with_uid_csv.loc[crimes_with_uid_csv['neighbourhood_id'] == test_region, 'uid'].unique().tolist()
# print('len of uid list ',len(uid_list_neighbourhood_test_region))
print('len of selected uids ',len(top_uid_list))
# print('top uid list is ',top_uid_list)
print('total users length for this region ',test_region,' is ',len(total_users_of_this_region))


def create_strategy():
  # self.server_model = AistClient(target_city='chicago',
  #                                      target_region=test_region, target_cat=1, batch_size=BATCH_SIZE,
  #                                      time_step=120, recent_time_step=20, ncfeature=1, nxfeature=12,
  #                                      gat_out=8, gat_att=40, slstm_nhid=hidden_state_of_lstm, slstm_nlayer=1,
  #                                      slstm_att=30,data_folder=f'{DATA_READ_FILE_PATH}/chicago2/', round = 0, userid=f'server_{save_file_path}',partition_id=participant_clients-1, fgat=False)
#   new_farhan_model = AIST(ncfeat=1, nxfeat=12, gout=8, gatt=40, rhid=hidden_state_of_lstm, ratt=30, rlayer=1, bs=BATCH_SIZE,
                    # rts=20, city='chicago', tr=test_region, tc=1, userid =f'server_{save_file_path}', fgat = False)
#   initial_params = new_farhan_model.get_parameters()
  fraction_level = None
  if(top == True):
    fraction_level = 1
  else:
    fraction_level = sample_clients/len(total_users_of_this_region)
  strategy = FedCustom(
      fraction_fit= fraction_level,  # Sample 100% of available clients for training
      fraction_evaluate=0.0,  # Sample 100% of available clients for evaluation
      min_fit_clients=1,  # Never sample less than 2 clients for training
      min_evaluate_clients=1,  # Never sample less than 2 clients for evaluation
      min_available_clients=1,  # Wait until all 2 clients are available
    #   initial_parameters = ndarrays_to_parameters(initial_params),
    #   proximal_mu= Fed_Prox_Proximal_MU
    #   eta = 0.001,
      # eta_l = 0.1,
      # beta_1 = 0.1,
      # beta_2 = 0.1,
      # tau = 1e-7
  )
  return strategy


def client_fn(cid: str):
  """Returns a FlowerClient containing the cid-th data partition"""
  userid = str(top_uid_list[int(cid)])
#   print(f'client mapping :{cid}->{userid}')

  if(os.path.isdir(f'{DATA_WRITE_FILE_PATH}/external/{userid}')):
    shutil.rmtree(f'{DATA_WRITE_FILE_PATH}/external/{userid}')
  os.makedirs(f'{DATA_WRITE_FILE_PATH}/external/{userid}')
  # here cid should be passed in the data_folder path to map each client number with the corresponding client data folder
  # here target region 28 got 2775, 27 got 1293, crime category 1  is theft
  #target region is now 22
  return AistClient(target_city='chicago', target_region=test_region, target_cat=args.target_cat, batch_size=BATCH_SIZE, time_step=120, recent_time_step=20, ncfeature=1, nxfeature=12, gat_out=8, gat_att=40, slstm_nhid=hidden_state_of_lstm, slstm_nlayer=1, slstm_att=30,data_folder='data', userid=userid, partition_id=int(cid), fgat= False).to_client()


# if(participant_clients % 5 == 0 and (participant_clients//5)%2==1):
#     num_gpus_fraction = 2/(participant_clients+1)
# else:
#     num_gpus_fraction = 2/participant_clients
num_gpus_fraction = 0.1

def run_simulation(NUM_CLIENTS,strategy):
    client_resources = {"num_cpus": 1, "num_gpus": num_gpus_fraction}

    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a callback to construct a client
        num_clients=NUM_CLIENTS,  # total number of clients in the experiment
        config=fl.server.ServerConfig(num_rounds=number_of_rounds),  # let's run for 15 round only
        strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
        client_resources=client_resources,
        # ray_init_args = {'num_cpus': 32, 'num_gpus': 2}
        # ray_init_args = {'num_cpus': 128, 'num_gpus': 1}
    )
    return history




def plot_histories():
    test_data = pd.read_csv(f'./external/r_{test_region}_{save_file_path}/Evaluation_Loss_MAE_MSE_FED.csv')
    train_data = pd.read_csv(f'./external/r_{test_region}_{save_file_path}/server_training_loss_vs_rounds.csv')
    round = train_data['round']
    train_loss = train_data['loss']
    plt.plot(round, train_loss, label='MAE')
    plt.yscale('log')  # Set y-axis to log scale
    plt.title(f'Train_loss(MAE) vs Round (Region {test_region})')
    plt.xlabel('Round')
    plt.ylabel('MAE (log scale)')
    plt.legend()
    # Save the figure with learning rate and network count in the filename
    plt.savefig(f'./random/Region_{test_region}_{save_file_path}/train.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Clear the plot
    plt.clf()

    # Remove rows where 'round' is 0
    test_data = test_data[test_data['round'] != 0]

    round = test_data['round']
    mae_test_loss = test_data['mae']
    mse_test_loss = test_data['mse']
    plt.plot(round, mae_test_loss,label='MAE')
    plt.plot(round, mse_test_loss,label='MSE')
    plt.title(f'Test_loss(MAE/MSE) vs Round (Region {test_region})')
    plt.xlabel('Round')
    plt.ylabel('MAE/MSE')
    plt.legend()
    # Save the figure with learning rate and network count in the filename
    plt.savefig(f'./random/Region_{test_region}_{save_file_path}/test.png', dpi=300, bbox_inches='tight')
    plt.show()
    # Clear the plot
    plt.clf()

def store_best_result(history):
    mae = history.metrics_centralized['mae']
    mse = history.metrics_centralized['mse']
    mae = np.array([m[1] for m in mae])
    mse = np.array([m[1] for m in mse])
    min_ind = np.argmin(mae)

    with open(f'./external/r_{test_region}_{save_file_path}/best_result.csv', 'w') as f:
        f.write(f'best_round,mae,mse\n')
        f.write(f'{min_ind},{mae[min_ind]},{mse[min_ind]}\n')




num_of_client = None
if top == True:
    num_of_client = [participant_clients]
    # num_of_client = [sample_clients]
else:
    num_of_client = [len(total_users_of_this_region)]

histories = []

for n in num_of_client:
    strategy = create_strategy()
    history = run_simulation(n,strategy=strategy)
    histories.append(history)
    # Append each history to the file
    # file.write(f'n={n}\n')
    # file.write(str(history) + '\n')
# print(history)

# plot_histories(num_of_client, histories, 1)


# plot_histories()


print(histories)
store_best_result(histories[0])
print("BATCH SIZE : ", BATCH_SIZE)
if(os.path.isdir(DATA_WRITE_FILE_PATH)):
  shutil.rmtree(DATA_WRITE_FILE_PATH)




