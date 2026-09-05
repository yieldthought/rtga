import numpy as np
import torch
from rtga.models import DynamicsEnsemble, EnsembleConfig, ReplayBuffer
from rtga.representation_study import SplitDynamicsEnsemble


def test_split_starts_identically_and_preserves_other_output_gradients():
    config=EnsembleConfig(3,2,hidden=8,binary_dims=(2,))
    base=DynamicsEnsemble(config)
    split=SplitDynamicsEnsemble(config)
    states=np.array([[.1,.2,0],[.2,.3,1]],dtype=np.float32)
    actions=np.array([0,1])
    for a,b in zip(base.predict(states,actions),split.predict(states,actions)):
        np.testing.assert_array_equal(a,b)
    inputs=torch.cat((torch.tensor(states).expand(3,-1,-1),
                      torch.nn.functional.one_hot(torch.tensor(actions),2).float().expand(3,-1,-1)),dim=-1)
    output=split.network(inputs)
    output[...,2].sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in split.network.binary.parameters())
    assert all(p.grad is None or not torch.count_nonzero(p.grad) for p in split.network.physical.parameters())


def test_split_checkpoint_reproduces_predictions(tmp_path):
    config=EnsembleConfig(3,2,hidden=8,binary_dims=(2,))
    model=SplitDynamicsEnsemble(config)
    replay=ReplayBuffer(10,3)
    for n in range(10):
        replay.append([.1,.2,0],n%2,[.2,.3,n%2])
    model.train_steps(replay,3,8)
    model.save(tmp_path/'split.pt')
    restored=SplitDynamicsEnsemble(config)
    restored.load_state_dict(torch.load(tmp_path/'split.pt',weights_only=True))
    for a,b in zip(model.predict([[.1,.2,0]],[1]),restored.predict([[.1,.2,0]],[1])):
        np.testing.assert_array_equal(a,b)
