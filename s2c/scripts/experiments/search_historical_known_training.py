"""Bounded MiniLM recipe screening on Known validation, without test access."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from protocol_v2.experiments.racal_v1.representation import build_racal_model, encode_rows, set_seed
from protocol_v2.experiments.racal_v1.runner import _class_centers, _center_losses
from tools.eval.run_historical_trainable_parameter_full_pipeline import DATA_ROOT, MODEL_ROOT, write_csv

OUT = ROOT / 'results/analysis/historical_known_training'
ART = ROOT.parent / 'artifacts/s2c/runs/historical_known_training'
BASE = dict(layers=2, projection=256, projection_enabled=True, temperature=.07,
            intra=.1, inter=.1, margin=.2, lr=2e-5, projection_lr=2e-4, epochs=9)
RECIPES = {
    'last2': BASE,
    'last4': {**BASE, 'layers': 4},
    'all6': {**BASE, 'layers': 6},
    'last1': {**BASE, 'layers': 1},
    'no_projection': {**BASE, 'projection_enabled': False},
    'projection128': {**BASE, 'projection': 128},
    'temperature10': {**BASE, 'temperature': .1},
    'compact_margin': {**BASE, 'intra': .5, 'inter': .5, 'margin': .4},
    'classification_only': {**BASE, 'intra': 0., 'inter': 0.},
    'lr_low': {**BASE, 'lr': 1e-5, 'projection_lr': 1e-4},
    'lr_high': {**BASE, 'lr': 4e-5, 'projection_lr': 4e-4},
}
CELLS = [('clinc150', .75), ('stackoverflow', .5), ('clinc150', .25),
         ('clinc150', .5), ('stackoverflow', .25), ('stackoverflow', .75),
         ('banking77_oos', .25), ('banking77_oos', .5), ('banking77_oos', .75)]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def configure(model, recipe, warmup):
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.projection.parameters():
        p.requires_grad_(recipe['projection_enabled'])
    # A disabled zero-initialized residual projection remains exactly identity.
    if not warmup or not recipe['projection_enabled']:
        for layer in model.encoder.encoder.layer[-recipe['layers']:]:
            for p in layer.parameters():
                p.requires_grad_(True)
    groups = []
    for prefix, lr in [('encoder.', recipe['lr']), ('projection.', recipe['projection_lr'])]:
        params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:
            groups.append(dict(params=params, lr=lr))
    return torch.optim.AdamW(groups)


def train_cell(dataset, kir, seed, name, device):
    recipe = RECIPES[name]
    tag = f'{dataset}_kir{round(kir*100):02d}_seed{seed}_{name}'
    folder = ART / tag
    record_path = OUT / f'{tag}.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['recipe'] == recipe and Path(record['checkpoint']).is_file()
        return record
    folder.mkdir(parents=True, exist_ok=False)
    data = DATA_ROOT / dataset / f'kir{round(kir*100):02d}_seed{seed}' / 'gate'
    train = json.loads((data / 'train.json').read_text())
    known = {r['intent'] for r in train}
    validation = [r for r in json.loads((data / 'val.json').read_text()) if r['intent'] in known]
    assert train and validation and all(int(r['label']) == 0 for r in train + validation)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / 'all-MiniLM-L6-v2', local_files_only=True)
    mode = 'last2_minilm_plus_projection'
    model = build_racal_model(MODEL_ROOT / 'all-MiniLM-L6-v2', mode, recipe['projection']).to(device)
    names = sorted(known)
    mapping = {name: i for i, name in enumerate(names)}
    target = torch.tensor([mapping[r['intent']] for r in train], device=device)
    gold = np.array([mapping[r['intent']] for r in validation])
    best = -1.
    history = []
    for epoch in range(1, recipe['epochs'] + 1):
        if epoch in (1, 2):
            optimizer = configure(model, recipe, epoch == 1)
        values = encode_rows(model, tokenizer, train, device, 128, 256)
        centers = torch.as_tensor(_class_centers(values, train)[0], device=device)
        model.train()
        losses = []
        order = np.random.default_rng(seed + epoch * 7919).permutation(len(train))
        for start in range(0, len(order), 64):
            indices = order[start:start+64]
            tokens = tokenizer([train[int(i)]['text'] for i in indices], padding=True, truncation=True,
                               max_length=256, return_tensors='pt').to(device)
            loss, _ = _center_losses(model(tokens), target[torch.as_tensor(indices, device=device)], centers,
                recipe['temperature'], recipe['intra'], recipe['inter'], 1., recipe['margin'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        values = encode_rows(model, tokenizer, train, device, 128, 256)
        centers_np = _class_centers(values, train)[0]
        val = encode_rows(model, tokenizer, validation, device, 128, 256)
        predicted = np.argmax(val @ centers_np.T, axis=1)
        score = float(f1_score(gold, predicted, average='macro'))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), known_validation_f1=score))
        if score > best + 1e-12:
            best, best_epoch = score, epoch
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            mode=mode, epoch=epoch, recipe=recipe), folder / 'checkpoint.pt')
        write_csv(OUT / f'{tag}_history.csv', history)
        print(f'TRAIN {tag} epoch={epoch} Known-F1={score:.6f}', flush=True)
    record = dict(dataset=dataset, kir=kir, seed=seed, name=name, recipe=recipe,
                  epoch=best_epoch, known_validation_f1=best, checkpoint=str(folder / 'checkpoint.pt'),
                  device=str(device), test_read=False, real_oos_used=False, pseudo_oos_used=False)
    dump(record_path, record)
    del model, optimizer
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--priority-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    cells = CELLS[:2] if args.priority_only else CELLS
    manifest = dict(status='running', protocol='historical_v19_paper_main', evidence='H1',
                    recipes=RECIPES, cells=cells, screening_seed=42, expansion_seeds=[13,87],
                    top_recipes=2, selection='Known validation macro F1; stable recipe ties',
                    real_oos_used=False, pseudo_oos_used=False, test_read=False, device='cuda')
    dump(OUT / 'MANIFEST.json', manifest)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.zeros(1, device=device)
    selected = []
    for dataset, kir in cells:
        records = [train_cell(dataset, kir, 42, name, device) for name in RECIPES]
        winners = sorted(records, key=lambda r: -r['known_validation_f1'])[:2]
        selected.extend(winners)
        dump(OUT / 'screening_selection.json', selected)
        for winner in winners:
            for seed in (13,87):
                train_cell(dataset, kir, seed, winner['name'], device)
    manifest.update(status='validation_complete')
    dump(OUT / 'MANIFEST.json', manifest)


if __name__ == '__main__':
    main()
