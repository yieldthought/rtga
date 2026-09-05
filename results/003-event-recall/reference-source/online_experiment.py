"""Online-control and curiosity experiments with evaluator-only probes."""

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from .agent import AgentConfig, RTGAAgent, LearnedEvaluator
from .envs import PuckLab, PuckConfig
from .experiments import provenance, write_json, environment_description, trace_frame
from .planning import EvolutionPlanner


def probe_transitions(config, seed):
    """State-controlled evaluator transitions, never given to agent training.

    This probe distribution deliberately covers both rooms, including states
    the agent may not have reached. It is diagnostic, not exploration credit.
    """
    env=PuckLab(config,seed=seed+9001)
    rng=np.random.default_rng(seed+9002)
    states,actions,targets=[],[],[]
    for x in [.1,.3,.7,.9]:
        for y in [.2,.5,.8]:
            for action in range(env.n_actions):
                snap=env.snapshot()
                snap['state']=np.array([x,y,*rng.uniform(-.2,.2,size=2),
                                       float(config.variant in {'open','rooms'}),0])
                env.restore(snap)
                states.append(env.observe())
                actions.append(action)
                targets.append(env.step(action))
    return np.array(states),np.array(actions),np.array(targets)


def frozen_goal_evaluation(model, env_config, planner_config, seed, steps=100):
    """Three withheld goals; frozen models, no evaluation data retained."""
    records=[]
    for goal_index,goal in enumerate([(.8,.5),(.25,.8),(.25,.2)]):
        env=PuckLab(env_config,seed=seed+30_000+goal_index)
        planner=EvolutionPlanner(env.n_actions,replace(planner_config,seed=seed+40_000+goal_index))
        obs=env.observe()
        reached=None
        begin=perf_counter()
        transitions=0
        for t in range(steps):
            decision=planner.plan(LearnedEvaluator(model,obs,mode='goal',goal=goal))
            obs=env.step(decision.action)
            transitions+=decision.model_transitions
            if reached is None and np.linalg.norm(obs[:2]-goal)<=.06:
                reached=t+1
        records.append({'goal':goal,'success':reached is not None,'first_success_step':reached,
                        'final_distance':float(np.linalg.norm(obs[:2]-goal)),
                        'steps':steps,'model_transitions':transitions,
                        'wall_seconds':perf_counter()-begin})
    return records


def run_online(output, seed=0, mode='curiosity', variant='open', steps=1500,
               agent_config=None, checkpoint_every=250, evaluate_goals=True):
    torch.set_num_threads(1)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config=replace(agent_config or AgentConfig(),seed=seed,mode=mode)
    config.planner=replace(config.planner,seed=seed+7101)
    env=PuckLab(PuckConfig(variant=variant),seed=seed+3001)
    goal=np.array([.8,.7]) if mode=='goal' else None
    agent=RTGAAgent(env.observation_low,env.observation_high,env.n_actions,
                    config,binary_dims=(4,),goal=goal)
    probes=probe_transitions(env.config,seed)
    data={'experiment':'online_learning','provenance':provenance(),'seed':seed,'mode':mode,
          'variant':variant,'steps_requested':steps,'config':asdict(config),
          'probe_description':'72 state-controlled real simulator transitions, evaluator-only',
          'probe_interactions':len(probes[0]),'checkpoints':[],'frames_completed':0}
    trace={'schema_version':1,'metadata':{'title':'RTGA online learning','seed':seed,'method':mode,
           'config':asdict(config),'path_semantics':'mean of deterministic ensemble carrier trajectories'},
           'environment':environment_description(env),'frames':[]}
    if goal is not None:
        trace['environment']['goal']=goal.tolist()
    obs=env.observe()
    wall_start=perf_counter()
    agent_seconds=evaluation_seconds=0.
    action_latencies=[]
    model_transitions=0
    distance_curve=[]
    first_goal=door_open_at=None
    training_updates=0
    for step in range(steps):
        action=agent.act(obs,is_first=step==0)
        info=agent.last_info
        action_latencies.append(info['action_latency_ms'])
        agent_seconds+=info['action_latency_ms']/1000
        training_updates+=info['training_steps']
        next_obs=env.step(action)
        metrics=env.metrics()
        if door_open_at is None and next_obs[4]>=.5:
            door_open_at=step+1
        if goal is not None:
            dist=float(np.linalg.norm(next_obs[:2]-goal))
            distance_curve.append(dist)
            if first_goal is None and dist<=.06:
                first_goal=step+1
        decision=agent.last_decision
        if decision is not None:
            model_transitions+=decision.model_transitions
        elif mode=='reactive' and step>=config.warmup:
            model_transitions+=config.members*env.n_actions
        # Keep a bounded, regularly sampled trace, with full learning curves in JSON.
        if step%max(1,steps//600)==0:
            if decision is not None:
                frame=trace_frame(step,obs,action,decision,next_obs,metrics)
                frame['paths']=frame['paths'][:4]
                frame['plans']=frame['plans'][:16]
            else:
                frame={'step':step,'state':obs.tolist(),'action':action,'objective':0.,
                       'plans':[],'paths':[],'selected_path':[],'latency_ms':info['action_latency_ms'],
                       'metrics':metrics,'prediction_error':None,'disagreement':None}
            frame['prediction_error']=info['prediction_error_before_update']
            frame['disagreement']=info['disagreement_before_update']
            frame['latency_ms']=info['action_latency_ms']
            frame['model_version']=info['model_version']
            trace['frames'].append(frame)
        obs=next_obs
        if (step+1)%checkpoint_every==0 or step+1==steps:
            eval_start=perf_counter()
            measured=agent.model.evaluate(*probes)
            evaluation_seconds+=perf_counter()-eval_start
            checkpoint={'step':step+1,**metrics,'model_version':agent.model.version,
                        'probe':measured,'agent_seconds':agent_seconds,
                        'model_transitions':model_transitions,
                        'recent_latency_ms':float(np.mean(action_latencies[-checkpoint_every:]))}
            if goal is not None:
                checkpoint['distance']=distance_curve[-1]
                checkpoint['first_goal_step']=first_goal
            data['checkpoints'].append(checkpoint)
            data['frames_completed']=step+1
            write_json(output/'results.json',data)
            print(json.dumps({'mode':mode,'variant':variant,'seed':seed,**checkpoint}),flush=True)
    agent.observe_final(obs)
    replay_states,replay_actions,replay_next=agent.replay.transitions()
    np.savez_compressed(output/'actual_transitions.npz',states=replay_states,
                        actions=replay_actions,next_states=replay_next)
    data.update({'completed':True,'agent_seconds':agent_seconds,'evaluation_seconds':evaluation_seconds,
                 'collection_wall_seconds':perf_counter()-wall_start,'model_transitions':model_transitions,
                 'training_updates':training_updates,'door_open_step':door_open_at,
                 'first_goal_step':first_goal,'distance_curve':distance_curve,
                 'action_latency_p50_ms':float(np.percentile(action_latencies,50)),
                 'action_latency_p95_ms':float(np.percentile(action_latencies,95)),
                 'action_latency_p99_ms':float(np.percentile(action_latencies,99)),
                 'final_metrics':env.metrics()})
    data['actual_transitions_file']='actual_transitions.npz'
    if evaluate_goals:
        data['frozen_goals']=frozen_goal_evaluation(agent.model,env.config,config.planner,seed)
    agent.model.save(output/'model.pt')
    trace['metadata']['metrics']=data['final_metrics']
    write_json(output/'results.json',data)
    write_json(output/'trace.json',trace)
    from .viewer import write_viewer
    write_viewer(trace,output/'viewer.html')
    return data


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['curiosity','goal','random','reactive'],default='curiosity')
    parser.add_argument('--variant',choices=['open','rooms','mechanism','noise'],default='open')
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--warmup',type=int,default=256)
    parser.add_argument('--initial-fit',type=int,default=200)
    parser.add_argument('--train-steps',type=int,default=8)
    parser.add_argument('--train-every',type=int,default=8)
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--population',type=int,default=48)
    parser.add_argument('--horizon',type=int,default=12)
    parser.add_argument('--generations',type=int,default=3)
    parser.add_argument('--no-goal-evaluation',action='store_true')
    args=parser.parse_args()
    from .planning import PlannerConfig
    config=AgentConfig(warmup=args.warmup,initial_fit=args.initial_fit,
                       train_steps=args.train_steps,train_every=args.train_every,
                       warmup_repeat=args.repeat,
                       planner=PlannerConfig(population=args.population,horizon=args.horizon,
                                              generations=args.generations))
    run_online(args.output,args.seed,args.mode,args.variant,args.steps,config,
               evaluate_goals=not args.no_goal_evaluation)


if __name__=='__main__':
    main()
