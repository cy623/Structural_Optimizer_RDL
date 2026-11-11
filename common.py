import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
OFFICIAL_COMMIT = '25cca45dd7a7cd2e17e116bd44a7951ebbdc06d0'
DATA_REPO = 'stanford-star/relbench-v1'
DATA_REVISION = '21d7284df2a9621c71f21bc4a1a30b7e8effb8dd'
TEXT_MODEL = 'sentence-transformers/average_word_embeddings_glove.6B.300d'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    paths = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'vendor').rglob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): sha256(p) for p in paths},
                                     sort_keys=True).encode()).hexdigest()


def task_id(dataset, task):
    matches = [t for t in task_manifest()['tasks'] if (t['dataset'], t['task']) == (dataset, task)]
    if not matches:
        raise ValueError(f'Not one of the 12 classification leaderboard tasks: {dataset}/{task}')
    return dataset + '__' + task


def task_manifest():
    return json.loads((ROOT / 'tasks.json').read_text())


@contextlib.contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def configure(seed=42):
    import numpy as np
    import torch
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        raise RuntimeError('Set PYTHONHASHSEED before launching Python (main.py does this).')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reseed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
