"""Known-only selective classification geometry search. Never opens test data."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.eval.run_historical_trainable_parameter_full_pipeline import DATA_ROOT, MODEL_ROOT, DEFAULT_H1_ROOT, write_csv
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector

OUT = ROOT / 'results/analysis/historical_known_geometry'


def _distance_matrix(detector, embeddings, chunk_size=4096):
    """Compute the same distances with BLAS-backed quadratic forms."""
    values = np.asarray(embeddings, dtype=np.float64)
    if detector.l2_normalize:
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    centers = np.asarray([sphere.center for sphere in detector.spheres], dtype=np.float64)
    if detector.distance_metric == 'euclidean':
        weighted_centers = centers
        inverse = None
    else:
        inverse = np.asarray([sphere.inv_diag_cov for sphere in detector.spheres], dtype=np.float64)
        weighted_centers = centers * inverse
    center_term = np.sum(centers * weighted_centers, axis=1)
    outputs = []
    for start in range(0, len(values), chunk_size):
        batch = values[start:start + chunk_size]
        if inverse is None:
            batch_term = np.sum(batch * batch, axis=1)
            cross = batch @ centers.T
        else:
            batch_term = (batch * batch) @ inverse.T
            cross = batch @ weighted_centers.T
        if inverse is None:
            squared = batch_term[:, None] + center_term[None, :] - 2.0 * cross
        else:
            squared = batch_term + center_term[None, :] - 2.0 * cross
        outputs.append(np.sqrt(np.maximum(squared, 0.0)).astype(np.float64))
    return np.concatenate(outputs, axis=0)


def search(train, validation, train_intents, truth):
    candidates = []
    for k in (1, 2, 3, 4, 5):
        for metric in ('euclidean', 'mahalanobis_diag'):
            detector = MultiSphereOOSDetector(center_mode='class_centroid_mixture', subcenters_per_intent=k,
                radius_method='mean_std', radius_lambda=1., distance_metric=metric,
                covariance_eps=1e-6, l2_normalize=True, random_state=42, acceptance_mode='nearest_sphere')
            detector.fit(train, train_intents)
            td, vd = _distance_matrix(detector, train), _distance_matrix(detector, validation)
            owners = np.array([detector.cluster_to_intent[i] for i in range(len(detector.spheres))])
            own_assignment = np.argmin(np.where(train_intents[:, None] == owners[None, :], td, np.inf), axis=1)
            boundaries = []
            for lam in (.0, .25, .5, .75, 1., 1.5, 2., 3.):
                detector.radius_lambda = lam
                detector._compute_radii()
                boundaries.append((f'mean_std_{lam}', np.array([s.radius for s in detector.spheres])))
            for q in (.5, .7, .8, .9, .95, .99):
                center_radius = np.array([np.quantile(td[own_assignment == i, i], q)
                    if np.any(own_assignment == i) else detector.spheres[i].radius for i in range(len(owners))])
                boundaries.append((f'center_quantile_{q}', center_radius))
                intent_radius = np.array([np.quantile(td[train_intents == name].min(axis=1), q) for name in owners])
                boundaries.append((f'intent_quantile_{q}', intent_radius))
            # The second distance is to another intent, never another center of the same intent.
            nearest = vd.argmin(axis=1)
            d1 = vd[np.arange(len(vd)), nearest]
            d2 = np.min(np.where(owners[None, :] != owners[nearest, None], vd, np.inf), axis=1)
            ambiguity = d1 / np.maximum(d2, 1e-12)
            for boundary, radius in boundaries:
                normalized = vd / np.maximum(radius, 1e-12)
                for rule in ('nearest_sphere', 'normalized_union'):
                    ids = nearest if rule == 'nearest_sphere' else normalized.argmin(axis=1)
                    base = normalized[np.arange(len(vd)), ids]
                    correct = owners[ids] == truth
                    for weight in (0., .25, .5, 1., 2.):
                        scores = base + weight * ambiguity
                        for threshold in np.arange(.4, 2.401, .05):
                            accepted = scores <= threshold
                            utility = float(np.mean(accepted * np.where(correct, 1., -4.)))
                            candidates.append(dict(k=k, distance=metric, boundary=boundary, rule=rule,
                                fusion_weight=weight, threshold=round(float(threshold), 6),
                                utility=utility, known_coverage=float(accepted.mean()),
                                correct_accept_rate=float(np.mean(accepted & correct))))
    # Stable loop order resolves ties without consulting test metrics.
    return max(candidates, key=lambda r: r['utility']), candidates


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', protocol='historical_v19_paper_main', device='cuda',
        selection='known_correct_accept_minus_4_wrong_accept', real_oos_used=False,
        pseudo_oos_used=False, test_read=False, source_checkpoint='known_only_validation_for_checkpoint',
        planned_units=27, base_commit='d0b10e267a3ca41c2130b804e26900c1e5c53077')
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.zeros(1, device=device)
    locks=[]
    cells=[('clinc150',.75),('stackoverflow',.5),('clinc150',.25),('clinc150',.5),
           ('stackoverflow',.25),('stackoverflow',.75),('banking77_oos',.25),('banking77_oos',.5),('banking77_oos',.75)]
    for dataset,kir in cells:
        for seed in (13,42,87):
            tag=f'kir{round(kir*100):02d}_seed{seed}'
            data=DATA_ROOT/dataset/tag/'gate'
            train=json.loads((data/'train.json').read_text())
            known={r['intent'] for r in train}
            assert all(int(r['label'])==0 for r in train)
            val=[r for r in json.loads((data/'val.json').read_text()) if r['intent'] in known]
            assert all(int(r['label'])==0 for r in val)
            checkpoint=DEFAULT_H1_ROOT/dataset/tag/'trainable_k1/checkpoint.pt'
            assert json.loads(checkpoint.with_name('run_manifest.json').read_text())['selection']=='known_only_validation_for_checkpoint'
            encoder=_RacalGateEncoder(MODEL_ROOT/'all-MiniLM-L6-v2',checkpoint,device)
            arrays=[encoder.encode([r['text'] for r in rows]) for rows in (train,val)]
            del encoder
            torch.cuda.empty_cache()
            winner, candidates=search(*arrays,np.array([r['intent'] for r in train]),np.array([r['intent'] for r in val]))
            write_csv(OUT/f'{dataset}_{tag}_candidates.csv',candidates)
            locks.append(dict(dataset=dataset,kir=kir,seed=seed,checkpoint=str(checkpoint),**winner))
            (OUT/'selection_lock.json').write_text(json.dumps(locks,indent=2))
            print(f'LOCK {dataset}/{tag}: {winner}',flush=True)
    manifest.update(status='validation_complete',completed_units=len(locks))
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
