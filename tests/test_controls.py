"""Tests for the control experiments. A bug there wouldn't crash, it would just
give a wrong number, so we pin the properties the claims rely on."""

import numpy as np
import torch
from torch_geometric.data import Data

from factnet.graph.feature_alignment import dead_dimensions, quantile_align, stack
from factnet.graph.root_ablation import _mask_root, _remove_root
from factnet.graph.trivial_baselines import represent
from factnet.nlp.significance import paired_bootstrap


def _cascade() -> Data:
    x = torch.tensor([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 0, 0, 1], [1, 2, 3, 2]])
    return Data(x=x, edge_index=edge_index, y=torch.tensor([1]))


def test_masking_the_root_touches_nothing_else():
    """The ablation must remove the article text and leave the topology."""
    out = _mask_root(_cascade())
    assert out.x[0].abs().sum() == 0
    assert torch.equal(out.x[1:], _cascade().x[1:])
    assert torch.equal(out.edge_index, _cascade().edge_index)


def test_removing_the_root_renumbers_without_dangling_edges():
    """Every surviving edge must point at a node that still exists."""
    out = _remove_root(_cascade())
    assert out.num_nodes == 3
    assert torch.equal(out.x, _cascade().x[1:])
    assert out.edge_index.numel() > 0, "the 1->2 edge should survive"
    assert int(out.edge_index.max()) < out.num_nodes
    assert int(out.edge_index.min()) >= 0


def test_removing_the_root_from_a_star_leaves_no_edges():
    """A cascade where everything hangs off the root has nothing left to read."""
    x = torch.tensor([[9.0, 9.0], [1.0, 1.0], [2.0, 2.0]])
    star = Data(x=x, edge_index=torch.tensor([[0, 0], [1, 2]]), y=torch.tensor([0]))
    out = _remove_root(star)
    assert out.num_nodes == 2
    assert out.edge_index.shape[1] == 0


def test_trivial_baselines_never_look_at_an_edge():
    """The point of the baseline is that rewiring the cascade cannot change it."""
    graph = _cascade()
    rewired = Data(x=graph.x, y=graph.y,
                   edge_index=torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]]))
    for kind in ("root", "users", "mean-pool", "size"):
        assert np.allclose(represent(graph, kind), represent(rewired, kind)), kind


def test_root_baseline_reads_the_root_and_users_baseline_does_not():
    graph = _cascade()
    assert np.allclose(represent(graph, "root"), [9.0, 9.0])
    assert np.allclose(represent(graph, "users"), [3.0, 4.0])


def test_quantile_alignment_preserves_the_order_of_accounts():
    """Alignment may move the scale, never the ranking, or it changes the data."""
    rng = np.random.default_rng(0)
    target = rng.random((200, 3)) * 0.01
    source = rng.random((500, 3)) * 100
    aligned = quantile_align(target, source)
    for column in range(3):
        before = target[:, column].argsort()
        after = aligned[:, column].argsort()
        assert np.array_equal(before, after), column
    assert aligned.max() > target.max() * 10, "the scale should have moved"


def test_dead_dimensions_finds_counters_the_collector_never_fills():
    matrix = np.array([[0.0, 1.0, 5.0], [0.0, 2.0, 5.0], [0.0, 3.0, 5.0]])
    assert dead_dimensions(matrix) == [0, 2]


def test_constant_columns_survive_alignment_untouched():
    """A counter with no equivalent on the target platform has no ranks to map."""
    target = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    source = np.array([[7.0, 1.0], [8.0, 2.0], [9.0, 3.0]])
    assert np.allclose(quantile_align(target, source)[:, 0], 0.0)


def test_stack_concatenates_every_node_of_every_cascade():
    assert stack([_cascade(), _cascade()]).shape == (8, 2)


def test_identical_predictions_produce_no_gap():
    """The paired bootstrap must cancel the sample, not manufacture a difference."""
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 300)
    pred = rng.integers(0, 2, 300)
    gaps = paired_bootstrap(truth, pred, pred.copy(), rounds=200)
    assert np.allclose(gaps, 0.0)


def test_a_real_gap_is_detected_and_signed():
    """A model that is genuinely better must land on the positive side."""
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, 400)
    good = truth.copy()
    good[rng.choice(400, 40, replace=False)] ^= 1        # 90 percent correct
    poor = truth.copy()
    poor[rng.choice(400, 160, replace=False)] ^= 1       # 60 percent correct
    gaps = paired_bootstrap(truth, good, poor, rounds=300)
    assert gaps.mean() > 0.15
    assert (gaps > 0).mean() > 0.99
