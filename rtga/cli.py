"""Command line entry point for reproducible RTGA experiments."""

import argparse

from .planning import PlannerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    oracle = sub.add_parser('oracle', help='Compare search methods using known dynamics')
    oracle.add_argument('--output', default='runs/oracle')
    oracle.add_argument('--seeds', type=int, default=10)
    oracle.add_argument('--seed-start', type=int, default=0)
    oracle.add_argument('--steps', type=int, default=100)
    oracle.add_argument('--population', type=int, default=96)
    oracle.add_argument('--horizon', type=int, default=32)
    oracle.add_argument('--generations', type=int, default=6)
    oracle.add_argument('--methods', nargs='+', default=['persistent', 'fresh', 'random', 'cem'])
    oracle.add_argument('--variants', nargs='+', default=['open', 'rooms'])
    oracle.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'oracle':
        from .experiments import run_oracle_suite
        config = PlannerConfig(population=args.population, horizon=args.horizon,
                               generations=args.generations, initial_repeat=args.repeat)
        run_oracle_suite(args.output, range(args.seed_start, args.seed_start + args.seeds),
                         args.methods, args.variants, args.steps, config)


if __name__ == '__main__':
    main()
