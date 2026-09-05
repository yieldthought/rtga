"""Does repeated environmental opportunity improve rare-mechanism learning?

Offline diagnostic with identical random actions across fixed reset schedules.
No observation, goal, or model prediction influences collection or resets.
"""

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import sys

import numpy as np
import torch

from .envs import PuckConfig, PuckLab


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def collect(seed, steps=10_000, reset_every=None):
    if steps < 1 or (reset_every is not None and reset_every < 1):
        raise ValueError('Collection and episode lengths must be positive')
    env = PuckLab(PuckConfig(variant='mechanism'), seed=seed+3001)
    rng = np.random.default_rng(seed+201)
    states, actions, targets, episodes = [], [], [], []
    observation = env.observe()
    for step in range(steps):
        if reset_every is not None and step and step % reset_every == 0:
            observation = env.reset()
        action = int(rng.integers(env.n_actions))
        target = env.step(action)
        states.append(observation.copy())
        actions.append(action)
        targets.append(target.copy())
        episodes.append(0 if reset_every is None else step // reset_every)
        observation = target
    return (np.asarray(states,dtype=np.float32), np.asarray(actions),
            np.asarray(targets,dtype=np.float32)), np.asarray(episodes)


def snapshot(output):
    source = output/'source'
    source.mkdir()
    names = ['__init__.py','agent.py','models.py','planning.py','envs.py',
             'event_learning_study.py','reset_collection_study.py','online_experiment.py',
             'experiments.py','viewer.py']
    for name in names:
        shutil.copy2(Path(__file__).parent/name,source/name)
    package = '_rtga_reset_snapshot'
    spec = importlib.util.spec_from_file_location(package,source/'__init__.py',
                                                submodule_search_locations=[str(source.resolve())])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    modules = {name: importlib.import_module(f'{package}.{name}')
               for name in ['agent','models','planning','envs','event_learning_study','online_experiment']}
    return modules, {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}


def run(output, steps=10_000, fit_updates=3000):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('Preserve completed or partial runs; use a new output directory')
    protocol = {
        'experiment':'007-reset-collection','seeds':[0,1,2], 'steps':steps,
        'reset_every':[None,500,1000], 'samplers':['uniform','high_change'],
        'fit_updates':fit_updates,'command':sys.orig_argv,
        'scope':'Offline data-opportunity diagnostic, not curiosity or online sample-efficiency evidence.',
        'collection':'Same uniform random action sequence per seed across schedules; reset fixed by time only.',
        'reset':'Return to the same initial observed state; record no reset transition. No random-generator reseeding.',
        'fitting':'Fresh K=3, hidden64 ensemble per dataset/sampler; one model seed paired to each collection seed.',
        'sampling':'Unchanged study006 uniform or half uniform plus half largest1% normalized transition changes.',
        'evaluation':'Study006 fixed spatial/action/velocity grid and physical probes; all models receive three100step frozen goal trials.',
        'decision':'Pilot only. Report every condition and seed; no tuning or selection from outcomes.',
    }
    write(output/'protocol.json',protocol)
    runtime, hashes = snapshot(output)
    study = runtime['event_learning_study']
    # Fixed legacy event probe; this is not called an experienced event for
    # datasets from other seeds. Its velocity is one of the grid conditions.
    legacy = json.loads(Path('results/006-event-learning/results.json').read_text())['event']
    grid, motion = study.make_probes(runtime,legacy)
    np.savez_compressed(output/'grid-probes.npz',**grid)
    np.savez_compressed(output/'motion-probes.npz',**motion)
    env = runtime['envs'].PuckLab(runtime['envs'].PuckConfig(variant='mechanism'))
    scale = np.maximum(np.maximum(abs(env.observation_low),abs(env.observation_high)),1e-3)
    base = runtime['models'].EnsembleConfig(6,6,binary_dims=(4,),observation_scale=tuple(scale),
                                           observation_low=tuple(env.observation_low),
                                           observation_high=tuple(env.observation_high))
    results = {'protocol':protocol,'source_sha256':hashes,'model_config':asdict(base),
               'legacy_probe':legacy,'datasets':[],'fits':[],'completed':False,
               'probe_generation_interactions':len(grid['states'])+len(motion['states'])}
    for seed in protocol['seeds']:
        reference_actions = None
        for interval in protocol['reset_every']:
            label = f'seed{seed}-reset{interval or "none"}'
            begin = perf_counter()
            data, episodes = collect(seed,steps,interval)
            if reference_actions is None:
                reference_actions = data[1].copy()
            assert np.array_equal(reference_actions,data[1])
            np.savez_compressed(output/f'{label}-actual.npz',states=data[0],actions=data[1],
                                next_states=data[2],episodes=episodes)
            openings = np.flatnonzero((data[0][:,4] < .5) & (data[2][:,4] >= .5))
            cells = np.minimum((np.concatenate([data[0][:,:2],data[2][:,:2]])*16).astype(int),15)
            dataset_record = {'label':label,'seed':seed,'reset_every':interval,
                              'steps':steps,'episodes':int(episodes.max()+1),
                              'opening_count':len(openings),'opening_indices':openings.tolist(),
                              'opening_states':data[0][openings].tolist(),
                              'coverage':len(np.unique(cells,axis=0))/256,
                              'collection_seconds':perf_counter()-begin}
            results['datasets'].append(dataset_record)
            for sampler in protocol['samplers']:
                model = runtime['models'].DynamicsEnsemble(replace(base,seed=seed))
                replay = study.OfflineReplay(data,scale,sampler,seed+101,
                                            int(openings[0]) if len(openings) else -1)
                begin = perf_counter()
                model.train_steps(replay,fit_updates,128)
                fit_seconds = perf_counter()-begin
                key = f'{label}-{sampler}'
                measured = study.evaluate(model,legacy,grid,motion,output/f'{key}-predictions.npz')
                measured['legacy_probe_probability_members'] = measured.pop('event_probability_members')
                measured['legacy_probe_probability_mean'] = measured.pop('event_probability_mean')
                goals = runtime['online_experiment'].frozen_goal_evaluation(
                    model,env.config,runtime['planning'].PlannerConfig(population=48,horizon=12,generations=3),seed)
                record = {'dataset':label,'seed':seed,'sampler':sampler,'fit_updates':fit_updates,
                          'fit_seconds':fit_seconds,'opening_count':len(openings),
                          'opening_events_in_high_change_tail':int(np.isin(openings,replay.tail).sum()),
                          'first_opening_draws_per_member':replay.exposure.tolist(),
                          'frozen_goals':goals,**measured}
                results['fits'].append(record)
                model.save(output/f'{key}.pt')
                write(output/'results.json',results)
                print(json.dumps({'dataset':label,'sampler':sampler,'openings':len(openings),
                                  'recall':measured['grid_all']['recall_at_half'],
                                  'outside_fp':measured['grid_outside_interact']['false_positive_rate_at_half'],
                                  'goal_success':[g['success'] for g in goals]}),flush=True)
    results['completed'] = True
    results['collection_interactions'] = steps*len(results['datasets'])
    results['goal_evaluation_interactions'] = sum(g['steps'] for f in results['fits'] for g in f['frozen_goals'])
    write(output/'results.json',results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/007-reset-collection')
    parser.add_argument('--steps',type=int,default=10_000)
    parser.add_argument('--fit-updates',type=int,default=3000)
    args=parser.parse_args()
    run(args.output,args.steps,args.fit_updates)


if __name__=='__main__':
    main()
