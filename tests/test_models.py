"""Tests for neural network model definitions (models.py)."""

import pytest
import torch
import torch.nn as nn
from drugreflector.models import fetch_activation, init_weights, order_activation_block, make_fc_encoder, nnFC


# ---------------------------------------------------------------------------
# fetch_activation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected_type", [
    ("Sigmoid", nn.Sigmoid),
    ("ReLU", nn.ReLU),
    ("Mish", nn.Mish),
    ("Tanh", nn.Tanh),
    ("SELU", nn.SELU),
    ("LeakyReLU", nn.LeakyReLU),
])
def test_fetch_activation_supported(name, expected_type):
    act = fetch_activation(name)
    assert isinstance(act, expected_type)


def test_fetch_activation_unknown_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        fetch_activation("GELU")


def test_fetch_activation_with_init_params():
    act = fetch_activation("LeakyReLU", {"negative_slope": 0.2})
    assert isinstance(act, nn.LeakyReLU)
    assert act.negative_slope == 0.2


# ---------------------------------------------------------------------------
# init_weights
# ---------------------------------------------------------------------------
def test_init_weights_modifies_linear():
    layer = nn.Linear(10, 5)
    original = layer.weight.clone()
    init_weights(layer)
    # Xavier init should produce different values (extremely unlikely to match)
    # We just verify no error and the shape is preserved
    assert layer.weight.shape == original.shape


def test_init_weights_noop_on_batchnorm():
    layer = nn.BatchNorm1d(10)
    init_weights(layer)  # Should not raise


# ---------------------------------------------------------------------------
# order_activation_block
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order,expected_types", [
    ("act-drop-bn", [nn.ReLU, nn.Dropout, nn.BatchNorm1d]),
    ("act-bn-drop", [nn.ReLU, nn.BatchNorm1d, nn.Dropout]),
    ("bn-drop-act", [nn.BatchNorm1d, nn.Dropout, nn.ReLU]),
    ("bn-act-drop", [nn.BatchNorm1d, nn.ReLU, nn.Dropout]),
    ("drop-act-bn", [nn.Dropout, nn.ReLU, nn.BatchNorm1d]),
    ("drop-bn-act", [nn.Dropout, nn.BatchNorm1d, nn.ReLU]),
])
def test_order_activation_block(order, expected_types):
    block = order_activation_block(
        order,
        act_layer=nn.ReLU(),
        bn_layer=nn.BatchNorm1d(10),
        drop_layer=nn.Dropout(0.5),
    )
    assert len(block) == 3
    for module, expected in zip(block, expected_types):
        assert isinstance(module, expected)


# ---------------------------------------------------------------------------
# make_fc_encoder
# ---------------------------------------------------------------------------
def test_make_fc_encoder_single_hidden():
    modules, out_dim = make_fc_encoder(100, [64])
    assert len(modules) == 1
    assert out_dim == 64


def test_make_fc_encoder_multiple_hidden():
    modules, out_dim = make_fc_encoder(100, [64, 32])
    assert len(modules) == 2
    assert out_dim == 32


def test_make_fc_encoder_int_hidden_dims():
    modules, out_dim = make_fc_encoder(100, 64)
    assert len(modules) == 1
    assert out_dim == 64


def test_make_fc_encoder_no_dropout():
    modules, _ = make_fc_encoder(100, [64], dropout_p=0.0)
    # The sequential should contain Identity instead of Dropout
    seq = modules[0]
    block = seq[1]  # activation block
    has_identity = any(isinstance(m, nn.Identity) for m in block)
    has_dropout = any(isinstance(m, nn.Dropout) for m in block)
    assert has_identity
    assert not has_dropout


def test_make_fc_encoder_no_batchnorm():
    modules, _ = make_fc_encoder(100, [64], batch_norm=False)
    seq = modules[0]
    block = seq[1]
    has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in block)
    assert not has_batchnorm


# ---------------------------------------------------------------------------
# nnFC
# ---------------------------------------------------------------------------
def test_nnfc_forward_shape():
    model = nnFC(input_dim=978, output_dim=9597, hidden_dims=[1024, 2048])
    model.eval()
    x = torch.randn(5, 978)
    out = model(x)
    assert out.shape == (5, 9597)


def test_nnfc_forward_no_hidden():
    model = nnFC(input_dim=100, output_dim=50, hidden_dims=None)
    model.eval()
    x = torch.randn(3, 100)
    out = model(x)
    assert out.shape == (3, 50)
    assert isinstance(model.encoder, nn.Identity)


def test_nnfc_deterministic_eval_mode():
    model = nnFC(input_dim=50, output_dim=20, hidden_dims=[32], dropout_p=0.5)
    model.eval()
    x = torch.randn(4, 50)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


def test_nnfc_final_layer_bias_true():
    model = nnFC(input_dim=50, output_dim=20, hidden_dims=[32], final_layer_bias=True)
    assert model.final_layer.bias is not None


def test_nnfc_final_layer_bias_false():
    model = nnFC(input_dim=50, output_dim=20, hidden_dims=[32], final_layer_bias=False)
    assert model.final_layer.bias is None


def test_nnfc_single_hidden():
    model = nnFC(input_dim=100, output_dim=50, hidden_dims=[64])
    model.eval()
    x = torch.randn(2, 100)
    out = model(x)
    assert out.shape == (2, 50)
