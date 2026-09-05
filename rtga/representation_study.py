"""Development ablation of interference between motion and binary predictions."""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .models import DynamicsEnsemble, EnsembleConfig
from .envs import PuckConfig
from .planning import PlannerConfig
from .online_experiment import frozen_goal_evaluation
from .event_learning_study import OfflineReplay, evaluate, write_json


class SplitNetwork(nn.Module):
    """Identical initial outputs, then independent parameters for output types."""

    def __init__(self, original, binary_dims):
        super().__init__()
        self.physical = original
        self.binary = copy.deepcopy(original)
        self.binary_dims = tuple(binary_dims)

    def forward(self, inputs):
        physical = self.physical(inputs)
        binary = self.binary(inputs)
        # Only the Bernoulli logits use the separate network. Its unused
        # continuous and variance outputs receive no loss/gradient.
        result = physical.clone()
        result[..., list(self.binary_dims)] = binary[..., list(self.binary_dims)]
        return result


class SplitDynamicsEnsemble(DynamicsEnsemble):
    def __init__(self, config, device='cpu'):
        super().__init__(config,device)
        self.network = SplitNetwork(self.network,config.binary_dims)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=config.learning_rate)

    def _clip_member_gradients(self, max_norm):
        # Separate clipping too: a physical gradient must not rescale the
        # binary branch, or vice versa.
        with torch.no_grad():
            for branch in [self.network.physical,self.network.binary]:
                params = [p for p in branch.parameters() if p.grad is not None]
                norm2 = sum(p.grad.reshape(self.config.members,-1).square().sum(1) for p in params)
                factors = (max_norm/(norm2.sqrt()+1e-8)).clamp(max=1)
                for param in params:
                    param.grad.mul_(factors.reshape(-1,*([1]*(param.ndim-1))))


def run(output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('Use a new output directory to preserve this experiment')
    source = output/'source'
    source.mkdir()
    names = ['__init__.py','representation_study.py','models.py','agent.py','envs.py',
             'planning.py','event_learning_study.py','online_experiment.py','experiments.py','viewer.py']
    hashes = {}
    for name in names:
        original = Path(__file__).parent/name
        hashes[name] = hashlib.sha256(original.read_bytes()).hexdigest()
        shutil.copy2(original,source/name)
    previous = json.loads(Path('results/007-reset-collection/results.json').read_text())
    if not previous['completed']:
        raise ValueError('Study007 must be complete')
    protocol = {
        'experiment':'009-output-representation','stage':'development',
        'question':'Can separate output-type networks reduce shared-trunk interference?',
        'datasets':'Study007 unchanged seed0/1/2 continuing and500step-reset random datasets,10k actual rows each.',
        'sampler':'Unchanged generic high_change: half uniform, half highest1% normalized observed changes.',
        'arms':['shared10','shared100','split10'],'updates':3000,
        'budget':'Equal optimizer updates and sample presentations, unequal parameter counts and wall time.',
        'split':'Clone identical initial shared network; independently train and clip physical versus binary outputs.',
        'control':'shared100 raises binary BCE weight10→100 without adding parameters.',
        'evaluation':'Same006 spatial/velocity/action and motion probes. All arms receive three100step frozen goals.',
        'selection':'Design prompted by007 failures on these data/probes. Development, not fresh confirmation.',
        'baseline':'Replay shared10 rather than substitute a previous artifact; check agreement with007 predictions.',
        'source_sha256':hashes,
    }
    write_json(output/'protocol.json',protocol)
    with np.load('results/007-reset-collection/grid-probes.npz') as z:
        grid = {key:z[key].copy() for key in z.files}
    with np.load('results/007-reset-collection/motion-probes.npz') as z:
        motion = {key:z[key].copy() for key in z.files}
    legacy = previous['legacy_probe']
    base = EnsembleConfig(**previous['model_config'])
    results = {'protocol':protocol,'records':[],'dataset_sha256':{},'completed':False}
    for seed in [0,1,2]:
        for interval in ['none','500']:
            label = f'seed{seed}-reset{interval}'
            path = Path('results/007-reset-collection')/f'{label}-actual.npz'
            results['dataset_sha256'][label] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:
                data = tuple(z[key].copy() for key in ['states','actions','next_states'])
            openings = np.flatnonzero((data[0][:,4]<.5)&(data[2][:,4]>=.5))
            for arm in protocol['arms']:
                config = replace(base,seed=seed,binary_loss_weight=100 if arm=='shared100' else 10)
                model = (SplitDynamicsEnsemble if arm=='split10' else DynamicsEnsemble)(config)
                replay = OfflineReplay(data,np.array(base.observation_scale),'high_change',seed+101,
                                       int(openings[0]) if len(openings) else -1)
                start = perf_counter()
                model.train_steps(replay,3000,128)
                fit_seconds = perf_counter()-start
                key = f'{label}-{arm}'
                metrics = evaluate(model,legacy,grid,motion,output/f'{key}-predictions.npz')
                metrics['legacy_probe_probability_members'] = metrics.pop('event_probability_members')
                metrics['legacy_probe_probability_mean'] = metrics.pop('event_probability_mean')
                baseline_max_difference = None
                if arm=='shared10':
                    with np.load(Path('results/007-reset-collection')/f'{label}-high_change-predictions.npz') as a, np.load(output/f'{key}-predictions.npz') as b:
                        baseline_max_difference = max(float(np.max(abs(a[name]-b[name]))) for name in a.files)
                    if baseline_max_difference != 0:
                        raise AssertionError('Shared10 failed to reproduce study007 predictions')
                goals = frozen_goal_evaluation(model,PuckConfig(variant='mechanism'),
                            PlannerConfig(population=48,horizon=12,generations=3),seed)
                record = {'dataset':label,'seed':seed,'arm':arm,'opening_count':len(openings),
                          'model_config':asdict(config),'fit_seconds':fit_seconds,
                          'parameter_count':sum(p.numel() for p in model.network.parameters()),
                          'optimizer_updates':model.version,'training_member_samples':3000*128*3,
                          'shared10_prediction_max_difference_from007':baseline_max_difference,
                          'frozen_goals':goals,**metrics}
                results['records'].append(record)
                model.save(output/f'{key}.pt')
                write_json(output/'results.json',results)
                print(json.dumps({'dataset':label,'arm':arm,'recall':metrics['grid_all']['recall_at_half'],
                                  'outside_fp':metrics['grid_outside_interact']['false_positive_rate_at_half'],
                                  'goals':[g['success'] for g in goals]}),flush=True)
    results['source_unchanged'] = all(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==value for name,value in hashes.items())
    if not results['source_unchanged']:
        raise AssertionError('Numerical source changed during run')
    results['completed'] = True
    write_json(output/'results.json',results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/009-output-representation')
    args=parser.parse_args()
    run(args.output)


if __name__=='__main__':
    main()
