"""Train matched downstream components on the shared Known-only views."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.run_kir_sensitivity_known_only import ART, KIRS, SEEDS, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['banking77','clinc150','stackoverflow'])
    parser.add_argument('--data-root', type=Path, default=ART / 'data')
    parser.add_argument('--components-root', type=Path, default=ART / 'components')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    components_root = args.components_root.resolve()
    env = dict(os.environ, CONDA_DEFAULT_ENV='bo', PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT/'src'))))
    for dataset in args.datasets:
        for kir in KIRS:
            for seed in SEEDS:
                tag = f'kir{round(100*kir):02d}_seed{seed}'
                data = data_root / dataset / tag
                out = components_root / dataset / tag
                out.mkdir(parents=True, exist_ok=True)
                domains = sorted(p.name for p in (data/'experts').iterdir() if p.is_dir())
                experts = out/'experts'
                attempt=0
                while any((experts/d).exists() and not (experts/d/'best_model.pt').exists() for d in domains):
                    attempt+=1
                    experts=out/f'experts_attempt{attempt}'
                commands = []
                router_dir = out/'router'
                router = router_dir/'best_model.pt'
                if len(domains) == 1:
                    router = out/'router/constant_router.json'
                    dump(router, dict(router_mode='constant', domain=domains[0], domain_label=0))
                elif not router.exists():
                    router_dir = out/'router'
                    attempt = 0
                    while router_dir.exists():
                        attempt += 1
                        router_dir = out/f'router_attempt{attempt}'
                    router = router_dir/'best_model.pt'
                    commands.append([sys.executable, str(ROOT/'tools/train/train_router_v19.py'),
                        '--data_dir', str(data/'router'), '--output_dir', str(router_dir),
                        '--epochs','10','--batch_size','32','--patience','5','--num_workers','0','--seed',str(seed)])
                for domain in domains:
                    if not (experts/domain/'best_model.pt').exists():
                        commands.append([sys.executable, str(ROOT/'tools/train/train_expert_v19.py'),
                            '--domain',domain,'--data_dir',str(data/'experts'),'--output_dir',str(experts),
                            '--epochs','15','--batch_size','32','--patience','5','--num_workers','0','--seed',str(seed)])
                with (out/'training.log').open('a') as log:
                    for command in commands:
                        command.append('--defer_test')
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                dump(out/'selection_complete.json',dict(dataset=dataset,kir=kir,seed=seed,
                     router=str(router),experts=str(experts),domains=domains,test_read=False))
                print(f'COMPONENTS {dataset}/{tag}',flush=True)


if __name__ == '__main__':
    main()
