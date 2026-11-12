import pytest
from trustquery.batch_assessment import assess
from trustquery.evaluation.batch_retrieval_eval import evaluate


def passage(text, id='p1'):
    return dict(text=text, chunk_id=id, evidence_type='synthetic_policy_requirement', operational_proof=False)


@pytest.mark.parametrize('question,text,label', [
    ('Does policy require MFA?', 'Policy must require MFA.', 'policy_supported'),
    ('Does policy require MFA and encryption?', 'Policy must require MFA.', 'partially_supported'),
    ('Does policy require MFA?', 'Policy must require backups.', 'insufficient_evidence'),
    ('Does policy require MFA?', 'MFA is a glossary term.', 'insufficient_evidence'),
    ('Does policy require MFA?', 'Policy must not require MFA.', 'insufficient_evidence'),
    ('Does policy require MFA?', 'Policy must require MFA. TBD.', 'insufficient_evidence'),
    ('Is MFA actually deployed?', 'Policy must require MFA.', 'implementation_evidence_required'),
    ('Does policy require MFA and provide recent logs?', 'Policy must require MFA.', 'implementation_evidence_required'),
    ('Do you encrypt data?', 'We must encrypt data.', 'implementation_evidence_required'),
    ('Does policy require MFA?', 'MFA must be used unless exempt.', 'insufficient_evidence'),
])
def test_triage(question,text,label):
    result=assess(question,[passage(text)])
    assert result['classification']==label
    assert result['operational_proof'] is False
    assert result['uncertainty'] and result['reasons']
    assert result['passages'][0]['passage_id']=='p1'


def test_empty_both_and_untrusted_type():
    assert assess('Does policy require MFA?',[])['classification']=='insufficient_evidence'
    r=assess('Does policy require MFA and actual logs?',[])
    assert r['both_policy_and_implementation_required']
    assert not r['evidence_passage_ids']
    p=passage('Policy must require MFA.')
    p['evidence_type']='unknown'
    assert assess('Does policy require MFA?',[p])['classification']=='insufficient_evidence'


def test_no_score_dependency_and_order():
    ps=[passage('Policy must require MFA.','z'),passage('Policy must require encryption.','a')]
    a=assess('Does policy require MFA and encryption?',ps)
    ps[0]['scores']={'reranker_score':999999}
    assert assess('Does policy require MFA and encryption?',ps)==a
    assert a['evidence_passage_ids']==['z','a']


def test_eval_denominators():
    rows=[dict(id='q',retrieved_passages=[dict(source='p.pdf',page=2)],evidence_assessment={'classification':'policy_supported'})]
    key=dict(records=[dict(id='q',evidence=[dict(source='p.pdf',page=2)]),dict(id='other')])
    result=evaluate(rows,key)
    assert not result['complete']
    assert result['source_page_hit_at_k']['1']==dict(hits=1,denominator=1,rate=1)
    with pytest.raises(ValueError):
        evaluate(rows+rows,key)


def test_conflicting_passages_prevent_full_support():
    r=assess('Does policy require MFA?', [passage('Policy must require MFA.'), passage('Policy must not require MFA.','p2')])
    assert r['classification']=='partially_supported'
    assert len(r['reasons'])==2


def test_eval_rank_empty_errors_and_no_locator_denominator():
    rows=[dict(id='q',retrieved_passages=[dict(source='other.pdf',page=2),dict(source='p.pdf',page=2)],evidence_assessment={'classification':'partially_supported'}),
          dict(id='r',retrieved_passages=[],retrieval_error={'message':'offline'},evidence_assessment={'classification':'insufficient_evidence'})]
    key=dict(records=[dict(id='q',evidence=[dict(source='p.pdf',page=2)]),dict(id='r')])
    result=evaluate(rows,key)
    assert result['complete']
    assert result['locator_eligible_questions']==1
    assert result['source_page_hit_at_k']['1']['hits']==0
    assert result['source_page_hit_at_k']['3']['hits']==1
    assert result['empty_passages']==result['retrieval_errors']==['r']


def test_assessment_does_not_mutate_passages():
    import copy
    ps=[passage('Policy must require MFA.')]
    before=copy.deepcopy(ps)
    assess('Does policy require MFA?',ps)
    assert ps==before
