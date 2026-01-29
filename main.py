import argparse
from data.dataset import DatasetManager
from relbench.tasks import get_task_names
import torch
from data.utils import get_stype_proposal, make_pkey_fkey_graph, data_loader, get_configs
from torch_frame.config import TextEmbedderConfig
from encoders import GloveTextEmbedding
import warnings
from train import Experiment
from torch_geometric.seed import seed_everything
import numpy as np
import os
import json
import time
warnings.filterwarnings("ignore")


def main(args, Data, graph_path):
    # download dataset
    database = Data.download_dataset(args.dataset)
    db = database.get_db()
    print(f'******************** DATASET {args.dataset} *********************')
    print(db.table_dict.keys())
    print('*********************************************************')

    # download task
    # task_names = get_task_names(args.dataset)
    task = Data.download_task(args.dataset, args.task) # The task is the training table.
    task_des = Data.get_task_description(args.dataset, args.task)
    if args.model == 'RelGNN':
        model_config, loader_config = get_configs(args.dataset, args.task)
        args.subgraph_type = loader_config['subgraph_type']
        args.num_neighbors = loader_config['num_neighbors']

    training_table_train = task.get_table("train")
    training_table_val = task.get_table("val")
    training_table_test = task.get_table("test", mask_input_cols=False)
    print(f'******************** TASK *********************')
    # print(training_table_train)
    print(args.task,":",task_des)
    print('Task type:', task.task_type)
    print('*********************************************************')

    # gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, "is OK!")
    
    all_results = []
    all_times = []
    for seed in [0,1]:
        seed_everything(seed)
        # get types of each table each column
        col_to_stype_dict = get_stype_proposal(db)

        # creat the heterogenous graph
        text_embedder_cfg = TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        )
        data, col_stats_dict = make_pkey_fkey_graph(
            db,
            col_to_stype_dict=col_to_stype_dict,  # speficied column types
            text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder, Convert some text columns into dense vectors (using the text embedding model you specified)
            cache_dir=graph_path,  # store materialized graph for convenience
        )

        entity_table, loader_dict = data_loader(args, training_table_train, training_table_val, training_table_test, data, task)
        
        exp = Experiment(args, data, col_stats_dict, entity_table, loader_dict, task, device)
        exp.fit()
        
        test_metric, avg_inference_time = exp.test_with_time()
        all_results.append(test_metric)
        all_times.append(avg_inference_time)
        
        
        print(f"\n=== Inference Time Comparison (Seed {seed}) ===")
        print(f"Model: {args.model}")
        if args.model == 'HeteroGraphSAGE':
            print(f"Type Gate: {args.lambda_g > 0}")
            print(f"Col Gate: {args.lambda_g > 0}")
            print(f"VIB: {args.beta_vib > 0}")
            print(f"Struct: {args.lambda_h > 0}")
        print(f"Average inference time: {avg_inference_time:.2f} s")
    
    print('ave_metric: {:.4f}'.format(np.mean(all_results)), '+/- {:.4f}'.format(np.std(all_results)))

    results_dict = {}
    results_dict['test_metric_mean'] = float(np.mean(all_results))
    results_dict['test_metric_std'] = float(np.std(all_results))
    results_dict['test_time'] = float(np.mean(all_times))

    outfile_name = f"{args.dataset}_{args.task}_{args.struct_strategies}" +\
                f"_results.txt"
    Hyperparameters = f"lr{args.lr}_weight_decay{args.wd}_channels{args.channels}" +\
                    f"_lambda_w{args.lambda_w}_lambda_g{args.lambda_g}_augmentation{args.lambda_h}_vib{args.beta_vib}" +\
                    f"_knn_k{args.knn_k}_recent_k{args.recent_k}"
                    
                       
    print("train and val outfile_name", outfile_name)

    with open(os.path.join('', outfile_name), 'a') as outfile:
        outfile.write(Hyperparameters)
        outfile.write('\n')
        outfile.write(json.dumps(results_dict))
        outfile.write('\n')



        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for RDL tasks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default='rel-f1', help="The RDL database.")
    parser.add_argument("--task", type=str, default='driver-top3', help="The task of the database.")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--temporal_strategy", type=str, default="uniform")

    parser.add_argument("--epochs", type=int, default=30, help="The epochs of training.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=5e-4)

    parser.add_argument("--model", type=str, default='HeteroGraphSAGE', help="The backbone model.") 
    parser.add_argument("--subgraph_type", type=str, default='directional')

    parser.add_argument("--struct_strategies", type=str, default="knn") # 2hop_agg, 2hop_shot, recent_k

    parser.add_argument("--lambda_g", type=float, default=1e-4)
    parser.add_argument("--lambda_w", type=float, default=1e-4)
    parser.add_argument("--lambda_h", type=float, default=1e-4)
    parser.add_argument("--beta_vib", type=float, default=1e-4)

    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--hc_tau_start", type=float, default=2.0)
    parser.add_argument("--hc_tau_end",   type=float, default=0.3)
    parser.add_argument("--hc_init_logit", type=float, default=2.0)

    parser.add_argument("--knn_k", type=int, default=10)   
    parser.add_argument("--recent_k", type=int, default=3) 
    
    args = parser.parse_args()
    Data = DatasetManager()

    graph_path = ''

    main(args, Data, graph_path)

