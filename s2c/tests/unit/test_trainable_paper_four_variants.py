import pytest
from tools.eval.run_trainable_paper_four_variants import metrics, compose, select


def test_known_f1_includes_oos_false_positives():
    rows=[dict(label=0,intent='a',domain='d'),dict(label=1,intent='unknown',domain='unknown')]
    pred=compose([dict(intent='a',domain='d')]*2,[.1,.8],1.)
    m=metrics(rows,pred)
    assert m['known_f1']==pytest.approx(200/3)
    assert m['oos_f1']==0
    assert m['overall_accuracy']==50


def test_confidence_threshold_keeps_rejection_active():
    rows=[dict(label=0,intent='a',domain='d'),dict(label=1,intent='unknown',domain='unknown')]
    downstream=[dict(intent='a',domain='d')]*2
    t,sweep=select(rows,downstream,[.1,.9])
    assert .1<=t<.9
    assert max(r['f1_all'] for r in sweep)==100
    assert [p['is_oos'] for p in compose(downstream,[.1,.9],t)]==[False,True]
