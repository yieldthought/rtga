import numpy as np
import pytest
import torch

from rtga.models import DynamicsEnsemble, EnsembleConfig, ReplayBuffer


def _transitions(count, seed):
    rng = np.random.default_rng(seed)
    states = rng.uniform(-1, 1, (count, 3)).astype(np.float32)
    states[:, 2] = rng.integers(2, size=count)
    actions = rng.integers(3, size=count)
    next_states = states.copy()
    next_states[:, 0] += 0.15 * states[:, 1]
    next_states[:, 1] = 0.8 * states[:, 1] + 0.12 * (actions - 1)
    next_states[:, 2] = np.where(actions == 2, 1.0, states[:, 2])
    return states, actions, next_states


def _replay(count=512, seed=12):
    replay = ReplayBuffer(count, 3, seed=seed)
    for state, action, next_state in zip(*_transitions(count, seed)):
        replay.append(state, int(action), next_state)
    return replay


def test_predict_shapes_probabilities_and_independent_initialization():
    states, actions, _ = _transitions(17, 1)
    model = DynamicsEnsemble(EnsembleConfig(3, 3, binary_dims=(2,), seed=42))
    means, variances = model.predict_tensor(torch.from_numpy(states), actions)
    assert means.shape == variances.shape == (3, 17, 3)
    assert not means.requires_grad and not variances.requires_grad
    assert torch.isfinite(means).all() and torch.isfinite(variances).all()
    assert torch.all(variances > 0)
    assert torch.all((means[..., 2] >= 0) & (means[..., 2] <= 1))
    torch.testing.assert_close(variances[..., 2], means[..., 2] * (1 - means[..., 2]))
    assert not torch.equal(means[0], means[1])
    again, _ = model.predict_tensor(states, actions)
    torch.testing.assert_close(means, again, rtol=0, atol=0)
    numpy_means, numpy_variances = model.predict(states, actions)
    np.testing.assert_array_equal(numpy_means, means.numpy())
    np.testing.assert_array_equal(numpy_variances, variances.numpy())


def test_training_improves_real_heldout_predictions_without_fitting_on_evaluation():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = DynamicsEnsemble(EnsembleConfig(3, 3, hidden=32, binary_dims=(2,), seed=3))
        replay = _replay()
        heldout = _transitions(256, 77)
        before = model.evaluate(*heldout)
        result = model.train_steps(replay, 250, batch_size=64)
        version = model.version
        after = model.evaluate(*heldout)
        assert after["normalized_mse"] < before["normalized_mse"] * 0.12
        assert after["binary_brier"] < before["binary_brier"] * 0.12
        assert after["nll"] < before["nll"]
        assert version == model.version == 250
        assert result["steps"] == 250 and result["samples"] == 250 * 64 * 3
        assert np.isfinite(list(after.values())).all()
        assert len(replay) == 512
    finally:
        torch.set_num_threads(previous_threads)


def test_replay_bootstraps_ring_order_and_rng_roundtrip():
    replay = ReplayBuffer(20, 1, seed=9)
    for index in range(25):
        replay.append([index], index % 2, [index + 1])
    assert len(replay) == 20
    states, _, _ = replay.transitions()
    np.testing.assert_array_equal(states[:, 0], np.arange(5, 25))
    samples = replay.sample(40, members=3)[0]
    assert samples.shape == (3, 40, 1)
    assert not np.array_equal(samples[0], samples[1])
    samples[:] = -100
    assert (replay.transitions()[0] >= 5).all()
    restored = ReplayBuffer(20, 1)
    restored.load_state_dict(replay.state_dict())
    for original, copied in zip(replay.sample(12, 3), restored.sample(12, 3)):
        np.testing.assert_array_equal(original, copied)


def test_checkpoint_preserves_predictions_optimizer_and_version(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = EnsembleConfig(3, 3, hidden=16, binary_dims=(2,), seed=5)
        model = DynamicsEnsemble(config)
        replay = _replay(80)
        model.train_steps(replay, 5, 16)
        checkpoint = tmp_path / "dynamics.pt"
        model.save(checkpoint)
        restored = DynamicsEnsemble.load(checkpoint)
        states, actions, _ = _transitions(11, 7)
        for expected, actual in zip(model.predict(states, actions), restored.predict(states, actions)):
            np.testing.assert_array_equal(expected, actual)
        assert restored.version == model.version == 5
        restored_replay = ReplayBuffer(80, 3)
        restored_replay.load_state_dict(replay.state_dict())
        model.train_steps(replay, 2, 16)
        restored.train_steps(restored_replay, 2, 16)
        for expected, actual in zip(model.predict(states, actions), restored.predict(states, actions)):
            np.testing.assert_array_equal(expected, actual)
    finally:
        torch.set_num_threads(previous_threads)


def test_normalization_bounds_single_member_and_input_validation():
    model = DynamicsEnsemble(EnsembleConfig(
        2, 2, members=1, observation_scale=(2.0, 0.1),
        observation_low=(-1.0, -0.5), observation_high=(1.0, 0.5),
    ))
    states = np.array([[2.0, -0.8]], dtype=np.float32)
    means, variances = model.predict(states, np.array([1]))
    assert np.all(means <= [1.0, 0.5]) and np.all(means >= [-1.0, -0.5])
    assert (variances > 0).all()
    metrics = model.evaluate(states, [1], [[1.0, -0.5]])
    assert metrics["epistemic"] == 0.0
    with pytest.raises(ValueError, match="states must"):
        model.predict_tensor([1.0, 2.0], [0])
    with pytest.raises(ValueError, match="strictly positive"):
        EnsembleConfig(2, 2, observation_scale=(1.0, 0.0))
    with pytest.raises(ValueError, match="empty"):
        ReplayBuffer(10, 2).sample(3)


def test_model_initialization_does_not_mutate_global_torch_rng():
    before = torch.random.get_rng_state().clone()
    DynamicsEnsemble(EnsembleConfig(2, 3, seed=15))
    assert torch.equal(before, torch.random.get_rng_state())


def test_stochastic_variance_is_distinct_from_member_disagreement():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        replay = ReplayBuffer(1200, 1, seed=8)
        rng = np.random.default_rng(15)
        for outcome in rng.normal(0, 0.2, size=1200):
            replay.append([0.0], 0, [outcome])
        model = DynamicsEnsemble(EnsembleConfig(1, 1, hidden=16, seed=7))
        model.train_steps(replay, 200, 128)
        means, variances = model.predict_tensor([[0.0]], [0])
        assert 0.02 < float(variances.mean()) < 0.08
        assert float(means.var(dim=0, unbiased=False).mean()) < 0.005
        # Identical member parameters must produce zero epistemic disagreement,
        # while their predicted stochastic variance remains positive.
        with torch.no_grad():
            for parameter in model.network.parameters():
                parameter[1:].copy_(parameter[0:1].expand_as(parameter[1:]))
        means, variances = model.predict_tensor([[0.0]], [0])
        assert float(means.var(dim=0, unbiased=False).sum()) == 0.0
        assert float(variances.min()) > 0.02
    finally:
        torch.set_num_threads(previous_threads)
