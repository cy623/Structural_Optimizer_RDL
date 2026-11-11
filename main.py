"""Single-task entry point. HPO never reads or scores the test target."""
import argparse
import json
import os
from pathlib import Path
import sys

from common import ROOT, task_id

DEFAULTS = dict(seed=42, epochs=50, batch_size=256, num_layers=2,
                num_neighbors=64, temporal_strategy='uniform', aggr='sum',
                max_steps_per_epoch=0, warmup_epochs=5,
                hc_tau_start=2., hc_tau_end=.3, hc_init_logit=2.,
                channels=128, lr=.001, wd=.005, lambda_w=.001,
                lambda_g=.01, lambda_h=.01, beta_vib=.001,
                knn_k=10, recent_k=5, strategies='all',
                gate_threshold=.5, sim_threshold=.8)
TEXT_REVISION = 'e5e8fec6971be8960cfaa853a77a6ddc62a265d7'


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'train', 'predict'], default='train')
    parser.add_argument('--dataset', default='rel-f1')
    parser.add_argument('--task', default='driver-top3')
    parser.add_argument('--gpu', default='0', help='Physical CUDA index; or cpu for a smoke check')
    parser.add_argument('--data_root', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--cache_root', type=Path, default=ROOT / 'data' / 'graph')
    parser.add_argument('--output_root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--prepare_device', default='cpu')
    parser.add_argument('--text_revision', default=TEXT_REVISION)
    parser.add_argument('--config', type=Path, help='JSON containing only the model/training configuration')
    parser.add_argument('--trial_id', default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_neighbors', type=int, default=64)
    parser.add_argument('--temporal_strategy', choices=['uniform', 'last'], default='uniform')
    parser.add_argument('--aggr', choices=['sum'], default='sum')
    parser.add_argument('--max_steps_per_epoch', type=int, default=0, help='0 means full training epoch')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--hc_tau_start', type=float, default=2.)
    parser.add_argument('--hc_tau_end', type=float, default=.3)
    parser.add_argument('--hc_init_logit', type=float, default=2.)
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--wd', type=float, default=.005)
    parser.add_argument('--lambda_w', type=float, default=.001)
    parser.add_argument('--lambda_g', type=float, default=.01)
    parser.add_argument('--lambda_h', type=float, default=.01)
    parser.add_argument('--beta_vib', type=float, default=.001)
    parser.add_argument('--knn_k', type=int, default=10)
    parser.add_argument('--recent_k', type=int, default=5)
    parser.add_argument('--strategies', choices=['all'], default='all')
    parser.add_argument('--gate_threshold', type=float, default=.5)
    parser.add_argument('--sim_threshold', type=float, default=.8)
    return parser


def parse(argv=None):
    p = parser()
    initial = p.parse_args(argv)
    if initial.config:
        values = json.loads(initial.config.read_text())
        if set(values) - set(DEFAULTS):
            p.error('Unknown configuration keys: ' + str(set(values) - set(DEFAULTS)))
        p.set_defaults(**values)
    args = p.parse_args(argv)
    task_id(args.dataset, args.task)
    config = {k: getattr(args, k) for k in DEFAULTS}
    for key in ['epochs', 'batch_size', 'num_layers', 'num_neighbors', 'channels', 'knn_k', 'recent_k']:
        if config[key] <= 0:
            p.error(key + ' must be positive')
    if args.max_steps_per_epoch < 0 or args.warmup_epochs < 0:
        p.error('Step limit and warmup must be nonnegative')
    return args, config


def main():
    args, config = parse()
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, dataset=args.dataset, task=args.task,
                              config=config,
                              selection='validation AUROC only', test_labels_loaded=False), indent=2))
        return
    env = dict(PYTHONHASHSEED=str(args.seed), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='' if args.gpu == 'cpu' else args.gpu)
    if any(os.environ.get(k) != v for k, v in env.items()):
        os.execve(sys.executable, [sys.executable, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                  dict(os.environ, **env))
    from common import configure
    configure(args.seed)
    if args.mode == 'prepare':
        from data_pipeline import prepare_graph
        print(prepare_graph(args))
    elif args.mode == 'train':
        from train import train
        train(args, config)
    else:
        from train import predict
        predict(args)


if __name__ == '__main__':
    main()
