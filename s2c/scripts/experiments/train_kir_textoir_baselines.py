"""Run native TextOIR selection, preserving selected state and deferring test."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.experiments.run_kir_sensitivity_known_only import ART, KIRS, SEEDS


def run_unit(dataset, kir, seed, method, python_executable):
    runtime={'banking77':'banking','clinc150':'oos','stackoverflow':'stackoverflow'}[dataset]
    tag=f'kir{round(kir*100):02d}_seed{seed}'
    data=ART/'data'/dataset/tag
    out=ART/'baselines'/dataset/tag/method
    attempts=list(out.parent.glob(f'{method}*/run_manifest.json'))
    if any(json.loads(p.read_text()).get('status')=='selection_complete' for p in attempts):
        return f'SKIP {dataset}/{tag}/{method}'
    if out.exists():
        index=1
        while out.with_name(f'{method}_attempt{index}').exists():index+=1
        out=out.with_name(f'{method}_attempt{index}')
    command=[sys.executable,str(ROOT/'tools/compat/textoir/run_external_textoir.py'),
        '--dataset',runtime,'--method',method,'--known-cls-ratio',str(kir),'--seed',str(seed),
        '--data-root',str(data/'textoir'),'--known-labels-file',str(data/'known_labels.json'),
        '--run-dir',str(out),'--bert-model',str(ART/'bert_legacy_format'),
        '--python-executable',python_executable,
        '--defer-test']
    try:
        subprocess.run(command,check=True,cwd=ROOT)
    except subprocess.CalledProcessError as error:
        return f'FAIL {dataset}/{tag}/{method} exit={error.returncode}'
    return f'OK {dataset}/{tag}/{method}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['banking77','clinc150','stackoverflow'])
    parser.add_argument('--methods',nargs='+',default=['DOC','KNNCL','ADB','DA-ADB'])
    parser.add_argument('--kirs', nargs='+', type=float, choices=KIRS, default=list(KIRS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--python-executable', default='/home/bo/anaconda3/envs/textoir-py39/bin/python')
    parser.add_argument('--max-workers',type=int,default=1,
                        help='Concurrent TextOIR runs on the same CUDA device')
    args=parser.parse_args()
    # Fail before creating a matrix of failed attempts if CUDA is inaccessible.
    subprocess.run([args.python_executable, '-c',
        'import torch; x=torch.zeros(1, device="cuda:0"); print("CUDA preflight:", x.device, flush=True)'],
        check=True, timeout=120)
    tasks=[(dataset,kir,seed,method)
           for dataset in args.datasets for kir in args.kirs for seed in args.seeds for method in args.methods]
    if args.max_workers == 1:
        failed = False
        for task in tasks:
            result = run_unit(*task,args.python_executable)
            print(result,flush=True)
            failed |= result.startswith('FAIL ')
        return int(failed)
    failed = False
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures=[executor.submit(run_unit,*task,args.python_executable) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(result,flush=True)
            failed |= result.startswith('FAIL ')
    return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
