"""Fixed paired-seed baseline for planned, reactive, and random exploration."""

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import shutil

from .agent import AgentConfig
from .experiments import provenance, write_json
from .online_experiment import run_online


def run_suite(output, seed_start=1000, seeds=5, steps=2500):
    output = Path(output)
    if (output / 'protocol.json').exists():
        raise FileExistsError('Choose a new output directory to preserve this study')
    if seeds < 1 or steps < 1:
        raise ValueError('Seeds and steps must be positive')
    config = AgentConfig()
    source_dir = output / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for name in ['agent.py','models.py','planning.py','envs.py','online_experiment.py',
                 'online_suite.py','experiments.py','viewer.py','__init__.py']:
        path = Path(__file__).parent / name
        shutil.copy2(path, source_dir / name)
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    modes = ['curiosity', 'reactive', 'random']
    protocol = {
        'experiment': '005-online-baseline', 'provenance': provenance(),
        'source_sha256': source_hashes,
        'seeds': list(range(seed_start, seed_start + seeds)),
        'variants': ['mechanism', 'noise'], 'modes': modes, 'steps': steps,
        'agent_config_template': asdict(config),
        'protocol': {
            'experience': 'Same 256-step random warmup; 2500 total actual transitions per run by default.',
            'training': 'Same scheduled fitting and replay capacity, including random control.',
            'compute': 'Equal interactions and optimizer updates, unequal planning compute; all counted separately.',
            'primary': ['final spatial coverage', 'door opening incidence', 'frozen right-room goal success'],
            'secondary': ['three frozen goals', 'prediction errors', 'noisy-source visits', 'runtime'],
            'analysis': 'Report every seed, paired bootstrap intervals for coverage; small-sample event counts are descriptive.',
            'selection': 'Five new seeds selected before results; no tuning within this suite.',
            'evaluation': 'Probe and goal simulator transitions never enter training.',
            'noise': 'Stochastic sensor source added to the same closed-door layout; no environment rewards reach the agent.',
            'execution': 'Sequential, rotating mode order by seed; other research processes may share CPU.',
        },
    }
    write_json(output / 'protocol.json', protocol)
    runs = []
    for variant in protocol['variants']:
        for index, seed in enumerate(protocol['seeds']):
            for mode in modes[index % 3:] + modes[:index % 3]:
                destination = output / f'{variant}-{mode}-{seed}'
                record = run_online(destination, seed, mode, variant, steps, config)
                runs.append({'path': str(destination.relative_to(output)),
                             'seed': seed, 'variant': variant, 'mode': mode,
                             'final_metrics': record['final_metrics'],
                             'door_open_step': record['door_open_step'],
                             'frozen_goals': record['frozen_goals']})
                write_json(output / 'index.json', {'completed': False, 'runs': runs})
    write_json(output / 'index.json', {'completed': True, 'runs': runs})
    print(json.dumps({'completed': True, 'runs': len(runs), 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=2500)
    args = parser.parse_args()
    run_suite(args.output, args.seed_start, args.seeds, args.steps)


if __name__ == '__main__':
    main()
