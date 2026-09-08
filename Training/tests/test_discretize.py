from ichigo_train.discretize import PrefixSchedule


def test_schedule_boundaries():
    s = PrefixSchedule(1000, 4, 0.2)
    assert (s.state_at(0).tau, s.state_at(0).frozen_prefix, s.state_at(0).heads_only, s.state_at(0).stage) == (1.0, 0, False, 1) and s.state_at(599).stage == 1
    st = s.state_at(600)
    assert st.stage == 2 and st.frozen_prefix == 0 and abs(st.tau - 1.0) < 1e-9
    assert s.freeze_steps() == [675, 750, 825, 900]
    assert s.state_at(674).frozen_prefix == 0 and s.state_at(675).frozen_prefix == 1
    assert s.state_at(899).frozen_prefix == 3 and abs(s.state_at(899).tau - (1 - 0.8 * 299 / 300)) < 1e-9
    st = s.state_at(900)
    assert st.frozen_prefix == 4 and st.heads_only and st.stage == 3
    assert s.state_at(999).heads_only


def test_tiny_step_counts_do_not_crash():
    s = PrefixSchedule(10, 4)
    seen = {s.state_at(i).frozen_prefix for i in range(10)}
    assert 4 in seen and s.state_at(9).heads_only
