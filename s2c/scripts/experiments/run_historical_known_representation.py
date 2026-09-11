"""CLINC longer training with Known classification checkpoint selection."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.run_historical_trainable_checkpoint_selection import set_phase, optimizer_for
from protocol_v2.experiments.racal_v1.representation import build_racal_model, encode_rows, set_seed
from protocol_v2.experiments.racal_v1.runner import _class_centers, _center_losses
from tools.eval.run_historical_trainable_parameter_full_pipeline import DATA_ROOT, MODEL_ROOT, _fit_detector, _metrics_from_output, write_csv
from tools.eval.run_historical_trainable_extended_boundary_search import _distance_matrix, _score_from_distances
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder

OUT = ROOT / 'results/analysis/historical_known_representation'
ART = ROOT.parent / 'artifacts/s2c/runs/historical_known_representation'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    ART.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.zeros(1, device=device)
    mode = 'last2_minilm_plus_projection'
    manifest = dict(status='running', protocol='historical_v19_paper_main', evidence='H1 Gate-only',
                    dataset='clinc150', kirs=[.5, .75], seeds=[13,42,87], device='cuda',
                    epochs=9, warmup_epochs=1, backbone_lr=2e-5, projection_lr=2e-4,
                    temperature=.07, intra_weight=.1, inter_weight=.1, margin=.2,
                    selection='Known validation nearest cosine centroid macro F1; earliest tie',
                    validation_oos_used=False, test_used_for_selection=False,
                    test_access='after_all_six_checkpoint_locks',
                    boundary='K1 diagonal Mahalanobis lambda1 threshold1 nearest_sphere',
                    base_commit='d0b10e267a3ca41c2130b804e26900c1e5c53077')
    dump(OUT / 'MANIFEST.json', manifest)
    locks = []
    for kir in manifest['kirs']:
        for seed in manifest['seeds']:
            tag = f'kir{round(kir*100):02d}_seed{seed}'
            folder = ART / tag
            folder.mkdir()
            data = DATA_ROOT / 'clinc150' / tag / 'gate'
            train = json.loads((data/'train.json').read_text())
            val = [r for r in json.loads((data/'val.json').read_text()) if int(r['label']) == 0]
            assert all(int(r['label']) == 0 for r in train)
            set_seed(seed)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT/'all-MiniLM-L6-v2', local_files_only=True)
            model = build_racal_model(MODEL_ROOT/'all-MiniLM-L6-v2', mode, 256).to(device)
            names = sorted({r['intent'] for r in train})
            mapping = {name:i for i,name in enumerate(names)}
            targets = torch.tensor([mapping[r['intent']] for r in train], device=device)
            gold = np.array([mapping[r['intent']] for r in val])
            best, history = -1., []
            for epoch in range(1,10):
                if epoch in (1,2):
                    set_phase(model, epoch == 1)
                    optimizer = optimizer_for(model, 2e-5)
                values = encode_rows(model, tokenizer, train, device, 128, 256)
                centers = torch.as_tensor(_class_centers(values, train)[0], device=device)
                model.train()
                losses = []
                order = np.random.default_rng(seed+epoch*7919).permutation(len(train))
                for start in range(0,len(order),64):
                    ids = order[start:start+64]
                    tokens = tokenizer([train[int(i)]['text'] for i in ids], padding=True, truncation=True,
                                       max_length=256, return_tensors='pt').to(device)
                    loss,_ = _center_losses(model(tokens),targets[torch.as_tensor(ids,device=device)],centers,.07,.1,.1,1.,.2)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                values = encode_rows(model,tokenizer,train,device,128,256)
                centers_np = _class_centers(values,train)[0]
                validation = encode_rows(model,tokenizer,val,device,128,256)
                score = float(f1_score(gold,np.argmax(validation@centers_np.T,axis=1),average='macro'))
                history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_known_f1=score))
                if score > best+1e-12:
                    best, best_epoch = score, epoch
                    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},mode=mode,epoch=epoch),folder/'checkpoint.pt')
                write_csv(OUT/f'{tag}_history.csv',history)
                print(f'TRAIN {tag} epoch={epoch} known_f1={score:.5f} best={best:.5f}',flush=True)
            lock = dict(kir=kir,seed=seed,checkpoint=str(folder/'checkpoint.pt'),epoch=best_epoch,validation_known_f1=best)
            locks.append(lock)
            dump(OUT/'selection_lock.json',locks)
            del model, optimizer
            torch.cuda.empty_cache()
    results=[]
    for lock in locks:
        tag=f"kir{round(lock['kir']*100):02d}_seed{lock['seed']}"
        data=DATA_ROOT/'clinc150'/tag/'gate'
        train=json.loads((data/'train.json').read_text())
        test=json.loads((data/'test.json').read_text())
        encoder=_RacalGateEncoder(MODEL_ROOT/'all-MiniLM-L6-v2',Path(lock['checkpoint']),device)
        detector=_fit_detector(encoder.encode([r['text'] for r in train]),train,1,1.,'nearest_sphere')
        output=_score_from_distances(_distance_matrix(detector,encoder.encode([r['text'] for r in test])),detector,'nearest_sphere')
        result={**lock,**_metrics_from_output(detector,output,test,1.)}
        results.append(result)
        print(f'TEST {tag} OOS={result["f1_u"]:.5f}',flush=True)
        del encoder
        torch.cuda.empty_cache()
    write_csv(OUT/'per_seed.csv',results)
    manifest.update(status='complete',completed_training_units=6,completed_test_units=6,full_pipeline_verified=False)
    dump(OUT/'MANIFEST.json',manifest)


if __name__=='__main__':
    main()
