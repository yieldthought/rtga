import numpy as np
from rtga.reset_collection_study import collect


def test_fixed_resets_change_states_without_creating_training_transitions():
    continuing,_=collect(0,steps=520)
    episodic,episode=collect(0,steps=520,reset_every=500)
    np.testing.assert_array_equal(continuing[1],episodic[1])
    np.testing.assert_array_equal(continuing[0][:500],episodic[0][:500])
    assert len(episodic[0])==520
    assert episode[499]==0 and episode[500]==1
    np.testing.assert_allclose(episodic[0][500],[.2,.5,0,0,0,0])
    np.testing.assert_array_equal(episodic[2][:499],episodic[0][1:500])
    np.testing.assert_array_equal(episodic[2][500:-1],episodic[0][501:])
    # Latched state can reset between episodes, but never closes in replay.
    assert not np.any((episodic[0][:,4]==1)&(episodic[2][:,4]==0))
