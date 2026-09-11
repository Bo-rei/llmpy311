import torch
from scripts.experiments.search_historical_known_training import BASE, configure


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.encoder = torch.nn.Module()
        self.encoder.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(6)])
        self.projection = torch.nn.Linear(2,2)


def test_requested_layers_and_projection_warmup():
    model = TinyModel()
    recipe = {**BASE, 'layers': 4}
    configure(model, recipe, True)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.projection.parameters())
    configure(model, recipe, False)
    assert [all(p.requires_grad for p in block.parameters()) for block in model.encoder.encoder.layer] == [False,False,True,True,True,True]


def test_disabled_projection_trains_backbone_even_in_first_epoch():
    model = TinyModel()
    configure(model, {**BASE, 'projection_enabled': False}, True)
    assert not any(p.requires_grad for p in model.projection.parameters())
    assert all(p.requires_grad for p in model.encoder.encoder.layer[-1].parameters())
