import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.model import _six_channel_conv, build_model, count_parameters  # noqa: E402


@pytest.mark.parametrize("arch", ["siamese", "stack"])
def test_output_is_four_scores_per_building(arch):
    model = build_model(arch, pretrained=False).eval()
    pre, post = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(pre, post)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["siamese", "stack"])
def test_gradients_reach_the_first_layer(arch):
    model = build_model(arch, pretrained=False)
    out = model(torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64))
    out.sum().backward()
    assert model.backbone.conv1.weight.grad.abs().sum() > 0


def test_models_use_both_images():
    for arch in ("siamese", "stack"):
        model = build_model(arch, pretrained=False).eval()
        pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            base = model(pre, post)
            assert not torch.allclose(base, model(pre, torch.randn(1, 3, 64, 64)))
            assert not torch.allclose(base, model(torch.randn(1, 3, 64, 64), post))


def test_six_channel_conv_averages_the_two_images():
    conv = torch.nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False)
    new = _six_channel_conv(conv)
    assert new.weight.shape == (8, 6, 7, 7)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        # the same image in both halves must give the original layer's output
        assert torch.allclose(new(torch.cat([x, x], dim=1)), conv(x), atol=1e-5)


def test_siamese_shares_one_backbone():
    siamese, stack = build_model("siamese", False), build_model("stack", False)
    # one ResNet-18 (~11.2M weights) either way; the siamese head adds ~0.4M
    assert 11.0e6 < count_parameters(stack) < 11.4e6
    assert count_parameters(siamese) - count_parameters(stack) < 0.5e6


def test_unknown_arch():
    with pytest.raises(ValueError):
        build_model("transformer")
