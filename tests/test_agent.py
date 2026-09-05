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
