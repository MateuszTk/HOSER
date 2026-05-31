import gc
import os
import argparse
import json
import math
from contextlib import nullcontext
from datetime import datetime
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import LineString
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from utils import set_seed, create_nested_namespace, get_angle, resolve_device
from models.hoser import HOSER
from dataset import Dataset


class MyCollateFn:
    def __init__(self, timestamp_label_log1p_mean, timestamp_label_log1p_std):
        self.timestamp_label_log1p_mean = timestamp_label_log1p_mean
        self.timestamp_label_log1p_std = timestamp_label_log1p_std
        self.batch_counter = 0

    def __call__(self, items):
        batch_trace_road_id = []
        batch_temporal_info = []
        batch_trace_distance_mat = []
        batch_trace_time_interval_mat = []
        batch_trace_len = []
        batch_destination_road_id = []
        batch_candidate_road_id = []
        batch_metric_dis = []
        batch_metric_angle = []
        batch_candidate_len = []
        batch_road_label = []
        batch_timestamp_label = []

        for trace_road_id, temporal_info, trace_distance_mat, trace_time_interval_mat, trace_len, destination_road_id, candidate_road_id, metric_dis, metric_angle, candidate_len, road_label, timestamp_label in items:
            batch_trace_road_id.append(np.array(trace_road_id, copy=True))
            batch_temporal_info.append(np.array(temporal_info, copy=True))
            batch_trace_distance_mat.append(np.array(trace_distance_mat, copy=True))
            batch_trace_time_interval_mat.append(np.array(trace_time_interval_mat, copy=True))
            batch_trace_len.append(trace_len)
            batch_destination_road_id.append(destination_road_id)
            batch_candidate_road_id.append([np.array(candidate, copy=True) for candidate in candidate_road_id])
            batch_metric_dis.append([np.array(distance, copy=True) for distance in metric_dis])
            batch_metric_angle.append([np.array(angle, copy=True) for angle in metric_angle])
            batch_candidate_len.append(np.array(candidate_len, copy=True))
            batch_road_label.append(np.array(road_label, copy=True))
            batch_timestamp_label.append(np.array(timestamp_label, copy=True))

        max_trace_len = max(batch_trace_len)
        min_trace_len = min(batch_trace_len)
        mean_trace_len = sum(batch_trace_len) / len(batch_trace_len)
        max_candidate_len = max([max(x) for x in batch_candidate_len])

        # Statististics
        total_trace_cells = len(batch_trace_len) * max_trace_len
        used_trace_cells = sum(batch_trace_len)
        trace_utilization = used_trace_cells / total_trace_cells

        total_mat_cells = len(batch_trace_len) * max_trace_len * max_trace_len
        used_mat_cells = sum(t * t for t in batch_trace_len)
        mat_utilization = used_mat_cells / total_mat_cells

        all_candidate_lens = [c for cl in batch_candidate_len for c in cl]
        total_candidate_cells = len(batch_trace_len) * max_trace_len * max_candidate_len
        used_candidate_cells = sum(int(c) for c in all_candidate_lens)
        candidate_utilization = used_candidate_cells / total_candidate_cells if total_candidate_cells > 0 else 0

        self.batch_counter += 1
        # print(f"  [Batch {self.batch_counter}] samples={len(batch_trace_len)} | "
        #       f"trace_len min={min_trace_len} mean={mean_trace_len:.1f} max={max_trace_len} | "
        #       f"max_candidates={max_candidate_len} | "
        #       f"utilization: seq={trace_utilization:.1%} mat={mat_utilization:.1%} cand={candidate_utilization:.1%} | "
        #       f"est_size: mat={total_mat_cells*4*2/1024/1024:.1f}MB cand={total_candidate_cells*4*3/1024/1024:.1f}MB")

        for i in range(len(batch_trace_road_id)):
            trace_pad_len = max_trace_len - batch_trace_len[i]

            batch_trace_road_id[i] = np.pad(batch_trace_road_id[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_temporal_info[i] = np.pad(batch_temporal_info[i], (0, trace_pad_len), 'constant', constant_values=0.0)
            batch_trace_distance_mat[i] = np.pad(batch_trace_distance_mat[i], ((0, trace_pad_len), (0, trace_pad_len)), 'constant', constant_values=0.0)
            batch_trace_time_interval_mat[i] = np.pad(batch_trace_time_interval_mat[i], ((0, trace_pad_len), (0, trace_pad_len)), 'constant', constant_values=0.0)
            
            for j in range(len(batch_candidate_road_id[i])):
                candidate_pad_len = max_candidate_len - batch_candidate_len[i][j]

                batch_candidate_road_id[i][j] = np.pad(batch_candidate_road_id[i][j], (0, candidate_pad_len), 'constant', constant_values=0)
                batch_metric_dis[i][j] = np.pad(batch_metric_dis[i][j], (0, candidate_pad_len), 'constant', constant_values=0.0)
                batch_metric_angle[i][j] = np.pad(batch_metric_angle[i][j], (0, candidate_pad_len), 'constant', constant_values=0.0)

            batch_candidate_road_id[i] = np.concatenate((np.stack(batch_candidate_road_id[i]), np.zeros((trace_pad_len, max_candidate_len), dtype=np.int64)), axis=0)
            batch_metric_dis[i] = np.concatenate((np.stack(batch_metric_dis[i]), np.zeros((trace_pad_len, max_candidate_len), dtype=np.float32)), axis=0)
            batch_metric_angle[i] = np.concatenate((np.stack(batch_metric_angle[i]), np.zeros((trace_pad_len, max_candidate_len), dtype=np.float32)), axis=0)

            batch_candidate_len[i] = np.pad(batch_candidate_len[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_road_label[i] = np.pad(batch_road_label[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_timestamp_label[i] = np.pad(batch_timestamp_label[i], (0, trace_pad_len), 'constant', constant_values=0.0)

        batch_timestamp_label = (np.log1p(batch_timestamp_label) - self.timestamp_label_log1p_mean) / self.timestamp_label_log1p_std

        batch_trace_road_id = torch.from_numpy(np.array(batch_trace_road_id))
        batch_temporal_info = torch.from_numpy(np.array(batch_temporal_info))
        batch_trace_distance_mat = torch.from_numpy(np.array(batch_trace_distance_mat))
        batch_trace_time_interval_mat = torch.from_numpy(np.array(batch_trace_time_interval_mat))
        batch_trace_len = torch.from_numpy(np.array(batch_trace_len))
        batch_destination_road_id = torch.from_numpy(np.array(batch_destination_road_id))
        batch_candidate_road_id = torch.from_numpy(np.array(batch_candidate_road_id))
        batch_metric_dis = torch.from_numpy(np.array(batch_metric_dis))
        batch_metric_angle = torch.from_numpy(np.array(batch_metric_angle))
        batch_candidate_len = torch.from_numpy(np.array(batch_candidate_len))
        batch_road_label = torch.from_numpy(np.array(batch_road_label))
        batch_timestamp_label = torch.from_numpy(np.array(batch_timestamp_label))

        return batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle, batch_candidate_len, batch_road_label, batch_timestamp_label


class GroupBatchSampler:
    def __init__(self, trace_lengths, batch_size, shuffle=True, jitter=4):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.jitter = jitter
        self.trace_lengths = np.array(trace_lengths)
        self.num_samples = len(trace_lengths)

    def __iter__(self):
        if self.shuffle:
            jittered = self.trace_lengths + np.random.randint(-self.jitter, self.jitter + 1, size=self.num_samples)
            order = np.argsort(jittered)
        else:
            order = np.argsort(self.trace_lengths)

        batches = [order[i:i + self.batch_size].tolist() for i in range(0, self.num_samples, self.batch_size)]

        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None) # path to checkpoint file
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device, args.cuda)
    use_amp = device.type == 'cuda'
    amp_context = (lambda: torch.amp.autocast(device_type=device.type)) if use_amp else nullcontext

    # Prepare model config and related features

    geo_file = f'./data/{args.dataset}/roadmap.geo'
    rel_file = f'./data/{args.dataset}/roadmap.rel'
    train_traj_file = f'./data/{args.dataset}/train.csv'
    val_traj_file = f'./data/{args.dataset}/val.csv'
    test_traj_file = f'./data/{args.dataset}/test.csv'
    road_network_partition_file = f'./data/{args.dataset}/road_network_partition'
    zone_trans_mat_file = f'./data/{args.dataset}/zone_trans_mat.npy'

    save_dir = f'./save/{args.dataset}/seed{args.seed}'
    tensorboard_log_dir = f'./tensorboard_log/{args.dataset}/seed{args.seed}'
    loguru_log_dir = f'./log/{args.dataset}/seed{args.seed}'

    with open(f'./config/{args.dataset}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = create_nested_namespace(config)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_log_dir)
    os.makedirs(loguru_log_dir, exist_ok=True)
    logger.add(os.path.join(loguru_log_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'), level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {message}")
    logger.info(f'using device {device}')

    geo = pd.read_csv(geo_file)
    rel = pd.read_csv(rel_file)
    num_roads = len(geo)

    road_attr_len = geo['length'].to_numpy().astype(np.float32)
    road_attr_len = np.log1p(road_attr_len)
    road_attr_len = (road_attr_len - np.mean(road_attr_len)) / np.std(road_attr_len)

    road_attr_type = geo['highway'].values.tolist()
    if args.dataset in ['Beijing', 'San_Francisco']:
        for i in range(len(road_attr_type)):
            if road_attr_type[i].startswith('[') and road_attr_type[i].endswith(']'):
                info = eval(road_attr_type[i])
                road_attr_type[i] = info[0] if info[0] != 'unclassified' else info[1]
    le = LabelEncoder()
    road_attr_type = le.fit_transform(road_attr_type)

    road_attr_lon = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.x for _, row in geo.iterrows()]).astype(np.float32)
    road_attr_lon = (road_attr_lon - np.mean(road_attr_lon)) / np.std(road_attr_lon)
    road_attr_lat = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.y for _, row in geo.iterrows()]).astype(np.float32)
    road_attr_lat = (road_attr_lat - np.mean(road_attr_lat)) / np.std(road_attr_lat)

    adj_row = []
    adj_col = []
    adj_angle = []
    adj_reachability = []

    reachable_road_id_dict = dict()
    for i in range(num_roads):
        reachable_road_id_dict[i] = []
    for _, row in rel.iterrows():
        origin_id = row['origin_id']
        destination_id = row['destination_id']
        reachable_road_id_dict[origin_id].append(destination_id)

    coord2road_id = dict()
    for road_id, row in geo.iterrows():
        coord = json.loads(row['coordinates'], parse_float=str)
        start_coord = tuple(coord[0])
        end_coord = tuple(coord[-1])
        if start_coord not in coord2road_id:
            coord2road_id[start_coord] = [road_id]
        else:
            coord2road_id[start_coord].append(road_id)
        if end_coord not in coord2road_id:
            coord2road_id[end_coord] = [road_id]
        else:
            coord2road_id[end_coord].append(road_id)

    road_adj = np.zeros((num_roads, num_roads), dtype=bool)
    for k, v in coord2road_id.items():
        for road_id1 in v:
            for road_id2 in v:
                if road_id1 != road_id2:
                    road_adj[road_id1, road_id2] = True

    for road_id in range(num_roads):
        adj_road_id_list = np.where(road_adj[road_id])[0]
        for adj_road_id in adj_road_id_list:
            adj_row.append(road_id)
            adj_col.append(adj_road_id)

            road_id_coord = eval(geo.loc[road_id, 'coordinates'])
            adj_road_id_coord = eval(geo.loc[adj_road_id, 'coordinates'])
            
            road_id_angle = get_angle(road_id_coord[0][1], road_id_coord[0][0], road_id_coord[-1][1], road_id_coord[-1][0])
            adj_road_id_angle = get_angle(adj_road_id_coord[0][1], adj_road_id_coord[0][0], adj_road_id_coord[-1][1], adj_road_id_coord[-1][0])
            angle = abs(road_id_angle - adj_road_id_angle)
            if angle > math.pi:
                angle = math.pi * 2 - angle
            angle /= math.pi
            adj_angle.append(angle)

            if adj_road_id in reachable_road_id_dict[road_id]:
                adj_reachability.append(1.0)
            else:
                adj_reachability.append(0.0)

    road_edge_index = np.stack([
        np.array(adj_row).astype(np.int64),
        np.array(adj_col).astype(np.int64),
    ], axis=0)
    intersection_attr = np.stack([
        np.array(adj_angle).astype(np.float32),
        np.array(adj_reachability).astype(np.float32),
    ], axis=1)

    del adj_row, adj_col, adj_angle, adj_reachability
    del road_adj, coord2road_id, reachable_road_id_dict

    zone_trans_mat = np.load(zone_trans_mat_file)
    zone_edge_index = np.stack(zone_trans_mat.nonzero())

    zone_trans_mat = zone_trans_mat.astype(np.float32)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.sum(zone_trans_mat, axis=1), 1.0))
    zone_trans_mat_norm = zone_trans_mat * D_inv_sqrt[:, np.newaxis] * D_inv_sqrt[np.newaxis, :]
    zone_edge_weight = zone_trans_mat_norm[zone_edge_index[0], zone_edge_index[1]]

    del zone_trans_mat, zone_trans_mat_norm, D_inv_sqrt

    config.road_network_encoder_config.road_id_num_embeddings = num_roads
    config.road_network_encoder_config.type_num_embeddings = len(np.unique(road_attr_type))
    config.road_network_encoder_feature.road_attr.len = road_attr_len
    config.road_network_encoder_feature.road_attr.type = road_attr_type
    config.road_network_encoder_feature.road_attr.lon = road_attr_lon
    config.road_network_encoder_feature.road_attr.lat = road_attr_lat
    config.road_network_encoder_feature.road_edge_index = road_edge_index
    config.road_network_encoder_feature.intersection_attr = intersection_attr
    config.road_network_encoder_feature.zone_edge_index = zone_edge_index
    config.road_network_encoder_feature.zone_edge_weight = zone_edge_weight

    del road_attr_len, road_attr_type, road_attr_lon, road_attr_lat
    del road_edge_index, intersection_attr, zone_edge_index, zone_edge_weight
    del geo, rel, le

    road2zone = []
    with open(road_network_partition_file, 'r') as file:
        for line in file:
            road2zone.append(int(line.strip()))
    road2zone = np.array(road2zone)

    # Prepare dataset and dataloader

    train_dataset = Dataset(geo_file, rel_file, train_traj_file)
    val_dataset = Dataset(geo_file, rel_file, val_traj_file)

    # histogram
    trace_lengths = np.array(train_dataset.trace_len)
    logger.info(f'Trace length statistics: min={trace_lengths.min()}, max={trace_lengths.max()}, '
                f'mean={trace_lengths.mean():.1f}, median={np.median(trace_lengths):.1f}, '
                f'std={trace_lengths.std():.1f}')
    hist_bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]
    hist_counts, _ = np.histogram(trace_lengths, bins=hist_bins)
    logger.info('Trace length histogram:')
    for i in range(len(hist_counts)):
        bar = '#' * int(hist_counts[i] / max(hist_counts) * 40)
        logger.info(f'  [{hist_bins[i]:>4d}, {hist_bins[i+1]:>4d}) : {hist_counts[i]:>6d} {bar}')
    percentiles = [50, 75, 90, 95, 99]
    pct_values = np.percentile(trace_lengths, percentiles)
    logger.info(f'Percentiles: ' + ', '.join(f'p{p}={int(v)}' for p, v in zip(percentiles, pct_values)))
    del trace_lengths

    timestamp_label_array = []
    for item in train_dataset:
        timestamp_label_array.extend(item[11])
    timestamp_label_array = np.array(timestamp_label_array)
    timestamp_label_array_log1p_mean = np.log1p(timestamp_label_array).mean()
    timestamp_label_array_log1p_std = np.log1p(timestamp_label_array).std()

    del timestamp_label_array
    gc.collect()


    logger.info(f'timestamp_label_array_log1p_mean {timestamp_label_array_log1p_mean:.3f}')
    logger.info(f'timestamp_label_array_log1p_std {timestamp_label_array_log1p_std:.3f}')

    train_batch_sampler = GroupBatchSampler(
        train_dataset.trace_len, config.optimizer_config.batch_size, shuffle=True)
    val_batch_sampler = GroupBatchSampler(
        val_dataset.trace_len, config.optimizer_config.batch_size, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=MyCollateFn(timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std),
        num_workers=0,
        pin_memory=use_amp,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=MyCollateFn(timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std),
        num_workers=0,
        pin_memory=use_amp,
    )

    # Start training

    model = HOSER(
        config.road_network_encoder_config,
        config.road_network_encoder_feature,
        config.trajectory_encoder_config,
        config.navigator_config,
        road2zone,
    ).to(device)

    logger.info(f'config.road_network_encoder_config {config.road_network_encoder_config}')
    logger.info(f'config.road_network_encoder_feature {config.road_network_encoder_feature}')
    logger.info(f'config.trajectory_encoder_config {config.trajectory_encoder_config}')
    logger.info(f'config.navigator_config {config.navigator_config}')
    logger.info(f'road2zone {road2zone}')

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer_config.learning_rate, weight_decay=config.optimizer_config.weight_decay)

    metrics_list = []
    start_epoch = 0

    if args.resume:
        logger.info(f'resuming from checkpoint {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        metrics_list = checkpoint['metrics_list']
        logger.info(f'resumed after epoch {start_epoch}')
        del checkpoint

    total_iters = config.optimizer_config.max_epoch * len(train_dataloader)
    warmup_iters = config.optimizer_config.max_epoch * len(train_dataloader) * config.optimizer_config.warmup_ratio
    iter_num = start_epoch * len(train_dataloader)
    def get_lr(it):
        if it < warmup_iters:
            return config.optimizer_config.learning_rate * it / warmup_iters
        assert it <= total_iters
        decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return coeff * config.optimizer_config.learning_rate

    for epoch_id in range(start_epoch, config.optimizer_config.max_epoch):
        model.train()
        for batch_id, (batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle, batch_candidate_len, batch_road_label, batch_timestamp_label) in enumerate(tqdm(train_dataloader, desc=f'[training] epoch{epoch_id+1}')):
            with amp_context():
                model.setup_road_network_features()

            batch_trace_road_id = batch_trace_road_id.to(device)
            batch_temporal_info = batch_temporal_info.to(device)
            batch_trace_distance_mat = batch_trace_distance_mat.to(device)
            batch_trace_time_interval_mat = batch_trace_time_interval_mat.to(device)
            batch_trace_len = batch_trace_len.to(device)
            batch_destination_road_id = batch_destination_road_id.to(device)
            batch_candidate_road_id = batch_candidate_road_id.to(device)
            batch_metric_dis = batch_metric_dis.to(device)
            batch_metric_angle = batch_metric_angle.to(device)
            batch_candidate_len = batch_candidate_len.to(device)
            batch_road_label = batch_road_label.to(device)
            batch_timestamp_label = batch_timestamp_label.to(device)

            iter_num += 1
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()

            with amp_context():
                logits, time_pred = model(batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle)

                logits_mask = torch.arange(logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < batch_trace_len.unsqueeze(1)
                selected_logits = logits[logits_mask]
                selected_candidate_len = batch_candidate_len[logits_mask]
                selected_road_label = batch_road_label[logits_mask]

                candidate_mask = torch.arange(selected_logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < selected_candidate_len.unsqueeze(1)
                masked_selected_logits = selected_logits.masked_fill(~candidate_mask, float('-inf'))

                loss_next_step = F.cross_entropy(masked_selected_logits, selected_road_label)

                selected_time_pred = time_pred[logits_mask][torch.arange(time_pred[logits_mask].size(0)), selected_road_label]
                selected_time_pred = selected_time_pred * timestamp_label_array_log1p_std + timestamp_label_array_log1p_mean
                selected_timestamp_label = batch_timestamp_label[logits_mask]
                selected_timestamp_label = selected_timestamp_label * timestamp_label_array_log1p_std + timestamp_label_array_log1p_mean

                loss_time_pred = torch.mean(torch.abs(selected_time_pred - selected_timestamp_label) / torch.clamp(selected_timestamp_label, min=1.0))

                loss = loss_next_step + loss_time_pred

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.optimizer_config.max_norm)
            scaler.step(optimizer)
            scaler.update()

            step = len(train_dataloader) * epoch_id + batch_id
            writer.add_scalar('loss_next_step', loss_next_step.item(), step)
            writer.add_scalar('loss_time_pred', loss_time_pred.item(), step)
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)

        logger.info(f'[training] epoch{epoch_id+1}, loss_next_step {loss_next_step.item():.3f}, loss_time_pred {loss_time_pred.item():.3f}')
        writer.flush()

        model.eval()
        with torch.no_grad():
            with amp_context():
                model.setup_road_network_features()

        val_next_step_correct_cnt, val_next_step_total_cnt = 0, 0
        val_time_pred_mape_sum, val_time_pred_total_cnt = 0, 0

        for batch_id, (batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle, batch_candidate_len, batch_road_label, batch_timestamp_label) in enumerate(tqdm(val_dataloader, desc=f'[validating] epoch{epoch_id+1}')):
            batch_trace_road_id = batch_trace_road_id.to(device)
            batch_temporal_info = batch_temporal_info.to(device)
            batch_trace_distance_mat = batch_trace_distance_mat.to(device)
            batch_trace_time_interval_mat = batch_trace_time_interval_mat.to(device)
            batch_trace_len = batch_trace_len.to(device)
            batch_destination_road_id = batch_destination_road_id.to(device)
            batch_candidate_road_id = batch_candidate_road_id.to(device)
            batch_metric_dis = batch_metric_dis.to(device)
            batch_metric_angle = batch_metric_angle.to(device)
            batch_candidate_len = batch_candidate_len.to(device)
            batch_road_label = batch_road_label.to(device)
            batch_timestamp_label = batch_timestamp_label.to(device)

            with torch.no_grad():
                with amp_context():
                    logits, time_pred = model(batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle)

                    logits_mask = torch.arange(logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < batch_trace_len.unsqueeze(1)
                    selected_logits = logits[logits_mask]
                    selected_candidate_len = batch_candidate_len[logits_mask]
                    selected_road_label = batch_road_label[logits_mask]

                    candidate_mask = torch.arange(selected_logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < selected_candidate_len.unsqueeze(1)
                    masked_selected_logits = selected_logits.masked_fill(~candidate_mask, float('-inf'))

                    val_next_step_correct_cnt += torch.sum(torch.argmax(masked_selected_logits, dim=1) == selected_road_label).item()
                    val_next_step_total_cnt += torch.sum(batch_trace_len).item()

                    selected_time_pred = time_pred[logits_mask][torch.arange(time_pred[logits_mask].size(0)), selected_road_label]
                    selected_time_pred = selected_time_pred * timestamp_label_array_log1p_std + timestamp_label_array_log1p_mean
                    selected_timestamp_label = batch_timestamp_label[logits_mask]
                    selected_timestamp_label = selected_timestamp_label * timestamp_label_array_log1p_std + timestamp_label_array_log1p_mean

                    val_time_pred_mape_sum += torch.sum(torch.abs(selected_time_pred - selected_timestamp_label) / torch.clamp(selected_timestamp_label, min=1.0))
                    val_time_pred_total_cnt += torch.sum(batch_trace_len).item()

        writer.add_scalar('val_next_step_acc', val_next_step_correct_cnt/val_next_step_total_cnt, len(train_dataloader) * epoch_id + batch_id)
        writer.add_scalar('val_time_pred_mape', val_time_pred_mape_sum/val_time_pred_total_cnt, len(train_dataloader) * epoch_id + batch_id)
        logger.info(f'[validating] epoch{epoch_id+1}, val_next_step_acc {val_next_step_correct_cnt/val_next_step_total_cnt:.3f}, val_time_pred_mape {val_time_pred_mape_sum/val_time_pred_total_cnt:.3f}')
        metrics_list.append(val_next_step_correct_cnt/val_next_step_total_cnt)

        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch_id+1}.pth'))
        torch.save({
            'epoch': epoch_id + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'metrics_list': metrics_list,
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch_id+1}.pth'))

    best_epoch = np.argmax(metrics_list)
    logger.info(f'loading epoch_{best_epoch+1}.pth')

    del train_dataset, train_dataloader, val_dataset, val_dataloader
    del scaler, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    best_model_state_dict = torch.load(os.path.join(save_dir, f'epoch_{best_epoch+1}.pth'), map_location=device)
    model.load_state_dict(best_model_state_dict)

    model.eval()
    with torch.no_grad():
        with amp_context():
            model.setup_road_network_features()

    test_next_step_correct_cnt, test_next_step_total_cnt = 0, 0
    test_time_pred_mape_sum, test_time_pred_total_cnt = 0, 0

    test_dataset = Dataset(geo_file, rel_file, test_traj_file)
    test_batch_sampler = GroupBatchSampler(
        test_dataset.trace_len, config.optimizer_config.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=MyCollateFn(timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std),
        num_workers=0,
        pin_memory=use_amp,
    )

    for batch_id, (batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle, batch_candidate_len, batch_road_label, batch_timestamp_label) in enumerate(tqdm(test_dataloader, desc=f'[testing]')):
        batch_trace_road_id = batch_trace_road_id.to(device)
        batch_temporal_info = batch_temporal_info.to(device)
        batch_trace_distance_mat = batch_trace_distance_mat.to(device)
        batch_trace_time_interval_mat = batch_trace_time_interval_mat.to(device)
        batch_trace_len = batch_trace_len.to(device)
        batch_destination_road_id = batch_destination_road_id.to(device)
        batch_candidate_road_id = batch_candidate_road_id.to(device)
        batch_metric_dis = batch_metric_dis.to(device)
        batch_metric_angle = batch_metric_angle.to(device)
        batch_candidate_len = batch_candidate_len.to(device)
        batch_road_label = batch_road_label.to(device)
        batch_timestamp_label = batch_timestamp_label.to(device)

        with torch.no_grad():
            with amp_context():
                logits, time_pred = model(batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle)

                logits_mask = torch.arange(logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < batch_trace_len.unsqueeze(1)
                selected_logits = logits[logits_mask]
                selected_candidate_len = batch_candidate_len[logits_mask]
                selected_road_label = batch_road_label[logits_mask]

                candidate_mask = torch.arange(selected_logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < selected_candidate_len.unsqueeze(1)
                masked_selected_logits = selected_logits.masked_fill(~candidate_mask, float('-inf'))

                test_next_step_correct_cnt += torch.sum(torch.argmax(masked_selected_logits, dim=1) == selected_road_label).item()
                test_next_step_total_cnt += torch.sum(batch_trace_len).item()

                selected_time_pred = time_pred[logits_mask][torch.arange(time_pred[logits_mask].size(0)), selected_road_label]
                selected_time_pred = selected_time_pred * timestamp_label_array_log1p_std + timestamp_label_array_log1p_mean
                selected_timestamp_label = batch_timestamp_label[logits_mask]
                selected_timestamp_label = selected_timestamp_label * timestamp_label_array_log1p_std + timestamp_label_array_log1p_mean

                test_time_pred_mape_sum += torch.sum(torch.abs(selected_time_pred - selected_timestamp_label) / torch.clamp(selected_timestamp_label, min=1.0))
                test_time_pred_total_cnt += torch.sum(batch_trace_len).item()

    logger.info(f'[testing] test_next_step_acc {test_next_step_correct_cnt/test_next_step_total_cnt:.3f}, test_time_pred_mape {test_time_pred_mape_sum/test_time_pred_total_cnt:.3f}')

    torch.save(model.state_dict(), os.path.join(save_dir, f'best.pth'))
