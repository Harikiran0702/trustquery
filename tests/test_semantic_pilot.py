import copy
import pytest
from trustquery.evaluation.semantic_pilot import verify_components

class Fake:
    def __init__(self, scores):
        self.scores=scores
        self.calls=[]
    def logits(self, premise, hypothesis):
        self.calls.append((premise,hypothesis))
        return self.scores[premise] if isinstance(self.scores,dict) else self.scores

def claim(kind='policy_requirement',text='Reviews are required.',id='c1'):
    return dict(id=id,type=kind,text=text)
def passage(text='Reviews are required.',id='p1'):
    return dict(chunk_id=id,source='policy.pdf',page=1,text=text,
                evidence_type='synthetic_policy_requirement',operational_proof=False)

@pytest.mark.parametrize('scores,expected', [([0,8,0],'text_supported'),([8,0,0],'contradicted'),([0,0,8],'insufficient_evidence'),([0,0,0],'insufficient_evidence'),([0,2,0],'insufficient_evidence')])
def test_mapping_threshold(scores,expected):
    backend=Fake(scores)
    assert verify_components([claim()],[passage()],backend)['status']==expected
    assert backend.calls==[('Reviews are required.','Reviews are required.')]

@pytest.mark.parametrize('kind', ['operational_fact','unknown'])
def test_type_gate_precedes_nli(kind):
    b=Fake([0,9,0])
    out=verify_components([claim(kind)],[passage()],b)
    assert not b.calls and out['status']!='text_supported'
    assert out['answers']==out['citations']==[] and out['operational_proof'] is False

def test_conflict_order_and_provenance():
    ps=[passage('Required.','yes'),passage('Forbidden.','no')]
    b=Fake({'Required.':[0,9,0],'Forbidden.':[9,0,0]})
    for rows in (ps,list(reversed(ps))):
        out=verify_components([claim()],rows,b)
        assert out['status']=='conflicted'
        assert {p['passage_id'] for p in out['components'][0]['pairs']}=={'yes','no'}

def test_partial_mixed_and_empty():
    ps=[passage()]
    assert verify_components([claim(),claim(text='Daily review.',id='c2')],ps)['status']=='partial'
    assert verify_components([claim(),claim('operational_fact',id='c2')],ps)['status']=='implementation_evidence_required'
    assert verify_components([],ps)['status']=='insufficient_evidence'
    assert verify_components([claim()],[])['status']=='insufficient_evidence'

def test_ids_mutation_lexical_and_tbd():
    cs=[claim()]
    ps=[passage('TBD. Reviews may occur.')]
    before=copy.deepcopy((cs,ps))
    assert verify_components(cs,ps)['status']=='insufficient_evidence'
    assert (cs,ps)==before
    with pytest.raises(ValueError):
        verify_components(cs+cs,ps)
    with pytest.raises(ValueError):
        verify_components(cs,ps+ps)
    with pytest.raises(ValueError):
        verify_components([claim('bad')],ps)

@pytest.mark.parametrize('field,value',[('source',''),('page',0),('evidence_type','operational'),('operational_proof',True)])
def test_invalid_provenance(field,value):
    p=passage()
    p[field]=value
    b=Fake([0,9,0])
    assert verify_components([claim()],[p],b)['status']=='insufficient_evidence'
    assert not b.calls

def test_failure_abstains():
    class Broken:
        def logits(self,*args): raise RuntimeError('offline failure')
    out=verify_components([claim()],[passage()],Broken())
    assert out['status']=='insufficient_evidence'
    assert out['components'][0]['pairs'][0]['error']=='offline failure'

def test_source_status_never_approval():
    out=verify_components([claim('source_status','It has not been approved.')],[passage('It has not been approved.')])
    assert out['status']=='text_supported' and out['source_approval']=='not_established'
