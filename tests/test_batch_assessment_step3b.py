"""Precision boundaries for deterministic triage, not semantic gold labels."""
import copy
import pytest
from trustquery.batch_assessment import assess


def p(text, id='p'):
    return dict(text=text, chunk_id=id, evidence_type='synthetic_policy_requirement', operational_proof=False)


@pytest.mark.parametrize('question', [
    'Where does your policy require production keys to be kept?',
    'What does your plan require responders to record when collecting evidence?',
    'How much restore testing does your policy require, and what should the test record contain?',
    'What does your process require before a production change?',
])
def test_policy_domain_nouns_are_not_operating_claims(question):
    r = assess(question, [p('Platform Engineering must store production keys in KMS.')])
    assert r['policy_requested'] and not r['implementation_required']


@pytest.mark.parametrize('question', [
    'Does policy require MFA, and is it enforced today?',
    'Under your policy, have you tested recovery?',
    'Does your policy require records, and can you provide recent records?',
    'Does policy require MFA and can you show the current inventory?',
    'Has your information security policy been approved by management?',
    'Does policy require MFA and is MFA enabled?',
    'How quickly would you remove access, and can you share a recent example?',
])
def test_mixed_and_operational_proof_stays_unmet(question):
    r = assess(question, [p('Policy must require MFA.')])
    assert r['both_policy_and_implementation_required']
    assert r['classification'] == 'implementation_evidence_required'
    assert not r['operational_proof']


@pytest.mark.parametrize('text,label', [
    ('Policy must require MFA. Backups are not deployed.', 'policy_supported'),
    ('Policy must require MFA unless exempt.', 'insufficient_evidence'),
    ('Policy must require MFA. MFA is not required for contractors.', 'partially_supported'),
    ('Policy must require MFA. MFA is optional.', 'partially_supported'),
    ('Policy must require MFA. MFA applies only to staff.', 'partially_supported'),
    ('Policy must require MFA. Unknown.', 'insufficient_evidence'),
    ('Policy must require MFA for staff.', 'insufficient_evidence'),
    ('Policy must require MFA for customers.', 'insufficient_evidence'),
    ('MFA must require policy.', 'insufficient_evidence'),
    ('Policy must require multi-factor authentication.', 'insufficient_evidence'),
    ('Policy must not require MFA.', 'insufficient_evidence'),
    ('Policy must require backups.', 'insufficient_evidence'),
])
def test_polarity_scope_order_and_unproven_synonyms(text, label):
    assert assess('Does policy require MFA?', [p(text)])['classification'] == label


def test_all_terms_across_sentences_never_establish_entailment():
    r = assess('Does policy require customer MFA?', [p('Policy must require staff MFA. Customer is a glossary term.')])
    assert r['classification'] == 'insufficient_evidence'


def test_bounded_partial_and_broad_claim_abstention():
    assert assess('Does policy require MFA and encryption?', [p('Policy must require MFA.')])['classification'] == 'partially_supported'
    r = assess('Does your policy protect every customer system?', [p('Policy must protect staff systems. Customer systems may be excluded.')])
    assert r['classification'] == 'insufficient_evidence'
    assert r['policy_coverage'] == 'partial_lexical'


def test_caution_and_usable_sentences_retain_provenance():
    r = assess('Does policy require MFA?', [p('Policy must require MFA. MFA is not required for contractors.')])
    assert r['passages'][0]['usable_sentences'] == ['Policy must require MFA.']
    assert r['passages'][0]['blocked_sentences'][0]['text'].startswith('MFA is not')


def test_order_scores_no_mutation_and_empty():
    ps = [p('Policy must require MFA.', 'one'), p('Backups are not deployed.', 'two')]
    old = copy.deepcopy(ps)
    a = assess('Does policy require MFA?', ps)
    b = assess('Does policy require MFA?', ps[::-1])
    assert a['classification'] == b['classification']
    assert set(a['evidence_passage_ids']) == set(b['evidence_passage_ids'])
    assert old == ps
    assert assess('Does policy require MFA?', [])['classification'] == 'insufficient_evidence'
    assert assess('Do you have customer SAML?', [])['classification'] == 'implementation_evidence_required'


def test_source_declaration_is_not_real_world_approval_or_negative_inference():
    r = assess('Has your policy been approved?', [p('This policy has not been approved.')])
    assert r['source_status_observations'][0]['status'] == 'source_declares_not_approved'
    assert r['classification'] == 'implementation_evidence_required'
    assert not assess('Has your policy been approved?', [p('Record CTO approval before publishing.')])['source_status_observations']


def test_retrieval_still_produces_blank_answers_and_citations():
    from trustquery.batch_retrieval import retrieve_drafts
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import json
    with TemporaryDirectory() as directory:
        path = Path(directory)/'questions.json'
        path.write_text(json.dumps([dict(id='q', question='Does policy require MFA?', response_type='yes_no')]))
        chunk = dict(text='Policy must require MFA.', metadata=dict(source='access_control_policy.pdf', page=1, chunk_index=0))
        rows = retrieve_drafts(path, chunks=[chunk], collection=None, reranker_model=None, retrieve=lambda **kw: [chunk])
        assert rows[0]['answer'] == '' and rows[0]['citations'] == []
        assert rows[0]['evidence_assessment']['classification'] == 'policy_supported'
