"""Restore a locally produced TextOIR selection and evaluate the shared test.

Run with the original runtime interpreter. Only load this campaign's trusted
local selected_method.pt files (they contain Python method objects).
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--selection-barrier',type=Path,
                        help='Optional campaign barrier; a deferred selection manifest is sufficient for one run')
    args=parser.parse_args()
    run=args.run_dir.resolve()
    manifest=json.loads((run/'run_manifest.json').read_text())
    if args.selection_barrier is not None:
        barrier=json.loads(args.selection_barrier.read_text())
        assert barrier['status']=='all_selections_locked'
    else:
        assert manifest.get('status') == 'selection_complete'
        assert manifest.get('test_deferred') is True
    result=run/'final_predictions.npz'
    if result.exists():return
    runtime=run/'runtime_overlay/open_intent_detection'
    sys.path.insert(0,str(runtime))
    import torch
    import numpy as np
    from dataloaders.bert_loader import get_examples,get_loader
    saved=torch.load(run/'selected_method.pt',weights_only=False)
    method,config=saved['method'],saved['args']
    data=saved.get('data',getattr(method,'data',None))
    if data is None:raise RuntimeError('Selected state lacks original data attributes')
    examples=get_examples(config,data.dataloader.base_attrs,'test')
    loader=get_loader(examples,config,data.label_list,'test',sampler_mode='sequential')
    method.test_dataloader=loader
    data.dataloader.test_loader=loader
    data.dataloader.test_examples=examples
    if hasattr(method,'data'):method.data=data
    outputs=method.test(config,data)
    np.savez_compressed(result,y_true=np.asarray(outputs['y_true']),y_pred=np.asarray(outputs['y_pred']),
                        known_labels=np.asarray(data.known_label_list),oos_id=data.unseen_label_id)


if __name__=='__main__':main()
