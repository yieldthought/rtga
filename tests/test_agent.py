import inspect
import numpy as np
import torch
import pytest

from rtga.agent import RTGAAgent, AgentConfig, LearnedEvaluator
from rtga.planning import PlannerConfig


def small_agent(mode='curiosity'):
    torch.set_num_threads(1)
    return RTGAAgent([0,0], [1,1], 3, AgentConfig(warmup=8, initial_fit=2,
        train_every=2, train_steps=1, hidden=16, mode=mode,
        planner=PlannerConfig(population=8, horizon=3, generations=2)))


def test_agent_learns_actual_transitions_and_plans_without_env():
    agent=small_agent()
    for t in range(12):
        action=agent.act(np.array([t/20, .3]), is_first=t==0)
        assert 0 <= action < 3
    assert len(agent.replay)==11
    assert agent.model.version==3
    assert agent.last_decision.model_transitions==3*3*8*3*2
    assert not hasattr(agent,'env')
    assert 'envs' not in inspect.getsource(__import__('rtga.agent',fromlist=['']))


def test_final_then_reset_does_not_train_reset_transition():
    agent=small_agent('random')
    agent.act([.1,.1],is_first=True)
    agent.observe_final([.2,.2])
    agent.act([.9,.9],is_first=True)
    assert len(agent.replay)==1
    states,actions,next_states=agent.replay.transitions()
    np.testing.assert_allclose(next_states, [[.2,.2]])


def test_frozen_snapshot_and_same_input_disagreement():
    agent=small_agent()
    evaluator=LearnedEvaluator(agent.model,[.3,.3])
    plans=np.zeros((5,4),dtype=int)
    before=evaluator(plans)
    assert before.paths.shape==(5,4,2)
    assert before.model_transitions==3*3*5*4
    # Identical parameters imply zero local disagreement, even after rollout.
    with torch.no_grad():
        for param in agent.model.network.parameters():
            param[1:].copy_(param[:1].expand_as(param[1:]))
    after=evaluator(plans)
    assert np.max(np.abs(after.scores)) < 1e-12
    agent.model.version+=1
    with pytest.raises(RuntimeError):
        evaluator(plans)


def test_unbounded_gymnasium_observations_keep_finite_model_scale():
    import gymnasium as gym
    from rtga.adapters import ObservationActionAdapter
    adapter=ObservationActionAdapter(gym.make('CartPole-v1'))
    agent=RTGAAgent(adapter.observation_low,adapter.observation_high,adapter.n_actions,
                    AgentConfig(mode='random',warmup=4,initial_fit=2,hidden=16))
    obs=adapter.reset(seed=7)
    for t in range(8):
        transition=adapter.step(agent.act(obs,is_first=t==0))
        obs=transition.observation
        if transition.terminated or transition.truncated:
            break
    assert np.isfinite(agent.model.config.observation_scale).all()
    assert agent.model.version > 0
    adapter.close()


@pytest.mark.parametrize('checkpoint_at', [3, 15, 27])
def test_agent_resume_reproduces_actions_models_and_wrapped_replay(tmp_path, checkpoint_at):
    config=AgentConfig(warmup=8, initial_fit=2, train_every=2, train_steps=1,
                      hidden=16, replay_capacity=16, warmup_repeat=3,
                      planner=PlannerConfig(population=8,horizon=3,generations=2))
    agent=RTGAAgent([0,0],[1,1],3,config)
    # A deterministic actual transition function; each resumed action affects
    # future input, so a diverging population/RNG cannot hide behind fixed data.
    obs=np.array([.2,.3],dtype=np.float32)
    def step(state, action):
        return (state + np.array([action*.007,.011],dtype=np.float32)) % 1
    for t in range(checkpoint_at):
        obs=step(obs,agent.act(obs,is_first=t==0))
    agent.save(tmp_path/'agent.pt')
    resumed=RTGAAgent.load(tmp_path/'agent.pt')
    for _ in range(12):
        action=agent.act(obs)
        assert resumed.act(obs)==action
        for original, restored in zip(agent.model.network.parameters(),resumed.model.network.parameters()):
            torch.testing.assert_close(original,restored,rtol=0,atol=0)
        np.testing.assert_array_equal(agent.planner.population,resumed.planner.population)
        obs=step(obs,action)
    agent.observe_final(obs)
    resumed.observe_final(obs)
    assert agent.transitions==resumed.transitions
    for original, restored in zip(agent.replay.transitions(),resumed.replay.transitions()):
        np.testing.assert_array_equal(original,restored)
