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


def noise_probe_transitions(config, seed, repeats=32):
    """Independent actual outcomes at one identical noisy-source state/action.

    Restoring the physical input preserves the advancing evaluator RNG. These
    repeated observations estimate outcome variation, not exploration credit.
    """
    if repeats < 1:
        raise ValueError('noise probe repeats must be positive')
    if config.variant != 'noise':
        return None
    env = PuckLab(config, seed=seed + 19_001)
    states, actions, targets = [], [], []
    for _ in range(repeats):
        snapshot = env.snapshot()
        snapshot['state'] = np.array([*config.noise_source, 0, 0, 0, 0], dtype=float)
        env.restore(snapshot)
        states.append(env.observe())
        actions.append(5)  # Environment evaluator's explicit interact action.
        targets.append(env.step(5))
    return np.array(states), np.array(actions), np.array(targets)


def evaluate_noise_probe(model, probes):
    """Describe sensor error and predicted uncertainty with one inference pass.

    A small repeated sample is a diagnostic, not proof of calibration. The
    Gaussian mixture density is scored as returned by the learned model.
    """
    states, actions, targets = probes
    means, variances = model.predict(states, actions)
    sensor_means = means[:, :, 5]
    sensor_variances = np.maximum(variances[:, :, 5], 1e-12)
    outcomes = targets[:, 5]
    ensemble_mean = sensor_means.mean(axis=0)
    epistemic = sensor_means.var(axis=0)
    aleatoric = sensor_variances.mean(axis=0)
    log_density = -.5 * (np.log(2 * np.pi * sensor_variances)
                         + (outcomes[None] - sensor_means) ** 2 / sensor_variances)
    mixture_log_density = np.logaddexp.reduce(log_density, axis=0) - np.log(model.config.members)
    return {
        'sample_count': len(outcomes),
        'sensor_mse': float(np.mean((ensemble_mean - outcomes) ** 2)),
        'sensor_mixture_nll': float(-mixture_log_density.mean()),
        'observed_sensor_mean': float(outcomes.mean()),
        'observed_sensor_variance': float(outcomes.var(ddof=1)) if len(outcomes) > 1 else None,
        'predicted_sensor_mean': float(ensemble_mean.mean()),
        'predicted_sensor_aleatoric_variance': float(aleatoric.mean()),
        'predicted_sensor_epistemic_variance': float(epistemic.mean()),
        'predicted_sensor_total_variance': float((epistemic + aleatoric).mean()),
        'interpretation': 'Repeated held-out source outcomes; descriptive mean/variance/density diagnostics, not a calibration claim.',
    }


def frozen_goal_evaluation(model, env_config, planner_config, seed, steps=100):
    """Three withheld goals; frozen models, no evaluation data retained."""
    if steps < 1:
        raise ValueError('goal evaluation steps must be positive')
    records=[]
    for goal_index,goal in enumerate([(.8,.5),(.25,.8),(.25,.2)]):
        begin=perf_counter()
        env=PuckLab(env_config,seed=seed+30_000+goal_index)
        planner=EvolutionPlanner(env.n_actions,replace(planner_config,seed=seed+40_000+goal_index))
        obs=env.observe()
        reached=None
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
                        'environment_interactions':steps,
                        'planning_model_transitions':transitions,
                        'wall_seconds':perf_counter()-begin})
    return records


def run_online(output, seed=0, mode='curiosity', variant='open', steps=1500,
               agent_config=None, checkpoint_every=250, evaluate_goals=True,
               goal_evaluation_steps=100, noise_probe_repeats=32):
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not isinstance(checkpoint_every, (int, np.integer)) or checkpoint_every < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    if goal_evaluation_steps < 1 or noise_probe_repeats < 1:
        raise ValueError('goal evaluation steps and noise probe repeats must be positive')
    run_start=perf_counter()
    torch.set_num_threads(1)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config=replace(agent_config or AgentConfig(),seed=seed,mode=mode)
    config.planner=replace(config.planner,seed=seed+7101)
    env=PuckLab(PuckConfig(variant=variant),seed=seed+3001)
    goal=np.array([.8,.7]) if mode=='goal' else None
    agent=RTGAAgent(env.observation_low,env.observation_high,env.n_actions,
                    config,binary_dims=(4,),goal=goal)
    probe_start=perf_counter()
    probes=probe_transitions(env.config,seed)
    noise_probes=noise_probe_transitions(env.config,seed,noise_probe_repeats)
    probe_generation_seconds=perf_counter()-probe_start
    noise_probe_count=0 if noise_probes is None else len(noise_probes[0])
    data={'experiment':'online_learning','provenance':provenance(),'seed':seed,'mode':mode,
          'variant':variant,'steps_requested':steps,'config':asdict(config),
          'probe_description':'72 state-controlled real simulator transitions, evaluator-only',
          'probe_interactions':len(probes[0]),'noise_probe_interactions':noise_probe_count,
          'probe_generation_seconds':probe_generation_seconds,
          'noise_probe_description':('Repeated actual interact transitions at the noise source, with identical '
                                     'physical inputs and advancing evaluator RNG; never used for training.'
                                     if noise_probes is not None else None),
          'accounting_notes':{
              'model_transitions':'Legacy alias of planning_model_transitions; includes reactive action scoring only, not surprise/probe/goal evaluation.',
              'total_model_inference_transitions':'Sum of individual member predictions for planning, surprise, checkpoint probes and frozen goals; excludes fitting forwards/backwards.',
              'training_member_samples':'Optimizer updates × batch size × members; fitting workload, not novel environment experience.',
              'checkpoint_timing':'Recorded after the environment step but before ingesting its observation: step N normally has N−1 consumed transitions.',
              'model_last_fit_consumed_transitions':'Cumulative actual transitions available at the most recent update; bootstrap fitting need not sample every retained transition.',
              'final_transition':'observe_final records the last transition without fitting; original agent update semantics are preserved.',
              'collection_wall_seconds':'Collection loop, checkpoint evaluation, and final replay serialization; excludes initial probe generation and frozen-goal evaluation.',
              'evaluation_seconds':'Probe generation + standard/noise checkpoint evaluation + frozen-goal evaluation, including evaluation environment setup.',
          },
          'checkpoints':[],'frames_completed':0}
    trace={'schema_version':1,'metadata':{'title':'RTGA online learning','seed':seed,'method':mode,
           'config':asdict(config),'path_semantics':'mean of deterministic ensemble carrier trajectories'},
           'environment':environment_description(env),'frames':[]}
    if goal is not None:
        trace['environment']['goal']=goal.tolist()
    obs=env.observe()
    wall_start=perf_counter()
    agent_seconds=training_seconds=0.
    checkpoint_evaluation_seconds=noise_probe_evaluation_seconds=0.
    evaluation_seconds=probe_generation_seconds
    action_latencies=[]
    planning_model_transitions=surprise_model_transitions=0
    probe_model_transitions=noise_probe_model_transitions=0
    distance_curve=[]
    first_goal=door_open_at=None
    training_updates=0
    model_last_fit_consumed_transitions=None
    for step in range(steps):
        action=agent.act(obs,is_first=step==0)
        info=agent.last_info
        action_latencies.append(info['action_latency_ms'])
        agent_seconds+=info['action_latency_ms']/1000
        training_updates+=info['training_steps']
        training_seconds+=info['train_ms']/1000
        if info['training_steps']:
            model_last_fit_consumed_transitions=agent.transitions
        if info['prediction_error_before_update'] is not None:
            surprise_model_transitions+=config.members
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
            planning_model_transitions+=decision.model_transitions
        elif mode=='reactive' and step>=config.warmup:
            planning_model_transitions+=config.members*env.n_actions
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
            checkpoint_evaluation_seconds+=perf_counter()-eval_start
            probe_model_transitions+=config.members*len(probes[0])
            noise_measured=None
            if noise_probes is not None:
                noise_eval_start=perf_counter()
                noise_measured=evaluate_noise_probe(agent.model,noise_probes)
                noise_probe_evaluation_seconds+=perf_counter()-noise_eval_start
                noise_probe_model_transitions+=config.members*noise_probe_count
            evaluation_seconds=(probe_generation_seconds+checkpoint_evaluation_seconds
                                +noise_probe_evaluation_seconds)
            checkpoint={'step':step+1,**metrics,'model_version':agent.model.version,
                        'probe':measured,'agent_seconds':agent_seconds,
                        'consumed_transitions':agent.transitions,
                        'replay_size':len(agent.replay),
                        'pending_transition_count':step+1-agent.transitions,
                        'model_last_fit_consumed_transitions':model_last_fit_consumed_transitions,
                        'model_transitions':planning_model_transitions,
                        'planning_model_transitions':planning_model_transitions,
                        'surprise_model_transitions':surprise_model_transitions,
                        'probe_model_transitions':probe_model_transitions,
                        'noise_probe_model_transitions':noise_probe_model_transitions,
                        'total_model_inference_transitions':(planning_model_transitions+surprise_model_transitions
                                                            +probe_model_transitions+noise_probe_model_transitions),
                        'training_updates':training_updates,
                        'training_member_samples':training_updates*config.batch_size*config.members,
                        'evaluation_seconds':evaluation_seconds,
                        'recent_latency_ms':float(np.mean(action_latencies[-checkpoint_every:]))}
            if noise_measured is not None:
                checkpoint['noise_probe']=noise_measured
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
    probe_arrays={'states':probes[0],'actions':probes[1],'next_states':probes[2]}
    if noise_probes is not None:
        probe_arrays.update(noise_states=noise_probes[0],noise_actions=noise_probes[1],
                            noise_next_states=noise_probes[2])
    np.savez_compressed(output/'evaluation_probes.npz',**probe_arrays)
    data.update({'completed':True,'agent_seconds':agent_seconds,'evaluation_seconds':evaluation_seconds,
                 'collection_wall_seconds':perf_counter()-wall_start,
                 'model_transitions':planning_model_transitions,
                 'planning_model_transitions':planning_model_transitions,
                 'surprise_model_transitions':surprise_model_transitions,
                 'probe_model_transitions':probe_model_transitions,
                 'noise_probe_model_transitions':noise_probe_model_transitions,
                 'consumed_transitions':agent.transitions,'replay_size':len(agent.replay),
                 'pending_transition_count':steps-agent.transitions,
                 'model_last_fit_consumed_transitions':model_last_fit_consumed_transitions,
                 'final_model_version':agent.model.version,
                 'final_transition_recorded_without_training':True,
                 'training_seconds':training_seconds,
                 'training_member_samples':training_updates*config.batch_size*config.members,
                 'checkpoint_evaluation_seconds':checkpoint_evaluation_seconds,
                 'noise_probe_evaluation_seconds':noise_probe_evaluation_seconds,
                 'training_updates':training_updates,'door_open_step':door_open_at,
                 'first_goal_step':first_goal,'distance_curve':distance_curve,
                 'action_latency_p50_ms':float(np.percentile(action_latencies,50)),
                 'action_latency_p95_ms':float(np.percentile(action_latencies,95)),
                 'action_latency_p99_ms':float(np.percentile(action_latencies,99)),
                 'final_metrics':env.metrics()})
    data['actual_transitions_file']='actual_transitions.npz'
    data['evaluation_probes_file']='evaluation_probes.npz'
    frozen_goal_seconds=0.
    if evaluate_goals:
        goal_eval_start=perf_counter()
        data['frozen_goals']=frozen_goal_evaluation(agent.model,env.config,config.planner,seed,
                                                 steps=goal_evaluation_steps)
        frozen_goal_seconds=perf_counter()-goal_eval_start
    goal_records=data.get('frozen_goals',[])
    frozen_goal_model_transitions=sum(record['planning_model_transitions'] for record in goal_records)
    frozen_goal_interactions=sum(record['environment_interactions'] for record in goal_records)
    data.update({
        'frozen_goal_evaluation_seconds':frozen_goal_seconds,
        'evaluation_seconds':evaluation_seconds+frozen_goal_seconds,
        'frozen_goal_model_transitions':frozen_goal_model_transitions,
        'frozen_goal_environment_interactions':frozen_goal_interactions,
        'collection_environment_interactions':steps,
        'evaluation_environment_interactions':len(probes[0])+noise_probe_count+frozen_goal_interactions,
        'total_environment_interactions':steps+len(probes[0])+noise_probe_count+frozen_goal_interactions,
        'total_model_inference_transitions':(planning_model_transitions+surprise_model_transitions
                                            +probe_model_transitions+noise_probe_model_transitions
                                            +frozen_goal_model_transitions),
    })
    agent.model.save(output/'model.pt')
    agent.save(output/'agent.pt')
    data['agent_checkpoint_file']='agent.pt'
    trace['metadata']['metrics']=data['final_metrics']
    write_json(output/'results.json',data)
    write_json(output/'trace.json',trace)
    from .viewer import write_viewer
    write_viewer(trace,output/'viewer.html')
    data['run_wall_seconds']=perf_counter()-run_start
    write_json(output/'results.json',data)
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
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--goal-evaluation-steps',type=int,default=100)
    parser.add_argument('--noise-probe-repeats',type=int,default=32)
    args=parser.parse_args()
    from .planning import PlannerConfig
    config=AgentConfig(warmup=args.warmup,initial_fit=args.initial_fit,
                       train_steps=args.train_steps,train_every=args.train_every,
                       warmup_repeat=args.repeat,
                       planner=PlannerConfig(population=args.population,horizon=args.horizon,
                                              generations=args.generations))
    run_online(args.output,args.seed,args.mode,args.variant,args.steps,config,
               checkpoint_every=args.checkpoint_every,evaluate_goals=not args.no_goal_evaluation,
               goal_evaluation_steps=args.goal_evaluation_steps,noise_probe_repeats=args.noise_probe_repeats)


if __name__=='__main__':
    main()
