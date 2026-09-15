"""Check all selections, then evaluate locked methods and build the report."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.run_kir_sensitivity_known_only import ART,KIRS,SEEDS,dump


def main():
    selection_root=ART/'coverage_repair'
    selected=[]
    missing=[]
    for dataset in ('banking77','clinc150','stackoverflow'):
        for kir in KIRS:
            for seed in SEEDS:
                tag=f'kir{round(kir*100):02d}_seed{seed}'
                for path in (selection_root/'locks'/f'{dataset}_{tag}.json',ART/'components'/dataset/tag/'selection_complete.json'):
                    if not path.exists():missing.append(str(path))
                for method in ('DOC','KNNCL','ADB','DA-ADB'):
                    candidates=[]
                    for manifest in (ART/'baselines'/dataset/tag).glob(f'{method}*/run_manifest.json'):
                        record=json.loads(manifest.read_text())
                        if record.get('status')=='selection_complete' and (manifest.parent/'selected_method.pt').exists():
                            candidates.append((manifest.parent,record))
                    if len(candidates)!=1:missing.append(f'{dataset}/{tag}/{method}: selected attempts={len(candidates)}')
                    else:selected.extend(candidates)
    if missing:
        print(f'Not ready: {len(missing)} missing/ambiguous selections; no test evaluated',flush=True)
        return 2
    barrier=selection_root/'all_selections_locked.json'
    dump(barrier,dict(status='all_selections_locked',units=495))
    subprocess.run([sys.executable,str(ROOT/'scripts/experiments/evaluate_kir_ours.py')],check=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    for folder,manifest in selected:
        interpreter=manifest['command'][0]
        subprocess.run([interpreter,str(ROOT/'tools/compat/textoir/evaluate_selected_method.py'),
                        '--run-dir',str(folder),'--selection-barrier',str(barrier)],env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/analysis/summarize_kir_sensitivity_known_only.py'),
                    '--selection-root',str(selection_root),
                    '--output-dir',str(ROOT/'results/analysis/kir_sensitivity_known_only/coverage_repair')],check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
