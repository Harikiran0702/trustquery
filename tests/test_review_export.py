import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import pytest
from trustquery.review_export import export_bytes, main, worksheet, snapshot
from trustquery.review_store import ReviewStore, ReviewConflict, load_batch, policy_extract


def record():
    return dict(id='q1',question='Does policy require MFA?',answer='Batch draft',outcome='abstained',
                provenance={'source':'input.json','input_sha256':'abc','corpus_sha256':'def','input_record':1},
                evidence_assessment={'implementation_required':False},
                evidence_references=[dict(source='p.pdf',page=3,chunk_index=2,chunk_id='p.pdf:p3:c2',quote='Policy must require MFA.')])


@pytest.fixture
def setup(tmp_path):
    rows=[record(),dict(record(),id='q2')]
    batch=tmp_path/'batch.json'
    batch.write_text(json.dumps(rows),encoding='utf-8')
    _,digest=load_batch(batch)
    db=tmp_path/'review.sqlite3'
    store=ReviewStore(db,digest)
    return batch,db,store,rows


def read(content):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(BytesIO(content)) as z:
        assert z.testzip() is None
        for name in z.namelist():
            ET.fromstring(z.read(name))
        tables=[]
        for i in range(1,5):
            root=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
            assert not root.findall('.//s:f',ns)
            assert all(c.get('t')=='inlineStr' for c in root.findall('.//s:c',ns))
            tables.append([[c.find('s:is/s:t',ns).text or '' for c in row] for row in root.findall('s:sheetData/s:row',ns)])
        return tables


def save(store,r,action='edited',revision=0,**extra):
    args=dict(revision=revision,action=action,answer='Incomplete draft',selected=[],reviewer='Alice',note='Need scope review',scope_checked=False)
    args.update(extra)
    return store.save(r,**args)


def test_export_reopens_legacy_crlf_batch_pin(tmp_path):
    rows = [record()]
    batch = tmp_path / 'batch.json'
    data = json.dumps(rows, indent=2).encode()
    legacy_digest = hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest()
    db = tmp_path / 'review.sqlite3'
    store = ReviewStore(db, legacy_digest)
    save(store, rows[0])

    batch.write_bytes(data)
    questions, *_ = read(export_bytes(batch, db))
    assert dict(zip(questions[0], questions[1]))['review_status'] == 'edited'

    batch.write_bytes(data.replace(b'Batch draft', b'Changed draft'))
    with pytest.raises(ValueError, match='different batch'):
        export_bytes(batch, db)


@pytest.mark.parametrize('status',['edited','rejected','needs_evidence'])
def test_incomplete_and_rejected(setup,status):
    batch,db,store,rows=setup
    save(store,rows[0],action=status)
    q,v,c,n=read(export_bytes(batch,db))
    a=dict(zip(q[0],q[1]))
    b=dict(zip(q[0],q[2]))
    assert a['review_status']==status and a['reviewed_answer']=='Incomplete draft'
    assert a['approved_policy_extract']=='' and a['approval_scope']==''
    assert b['review_status']=='unreviewed' and b['reviewed_answer']==''
    assert b['batch_answer_unreviewed']=='Batch draft' and b['revision']=='0'
    assert len(v)==1 and c[1][1]=='UNVERIFIED CANDIDATE'
    assert a['implementation_evidence_missing']==str(status=='needs_evidence').lower()


def test_approval_then_conflicting_save_uses_latest(setup):
    batch,db,store,rows=setup
    r=rows[0]
    selected=['p.pdf:p3:c2']
    save(store,r,action='approved',selected=selected,answer=policy_extract(r,selected),scope_checked=True)
    q,v,c,_=read(export_bytes(batch,db))
    a=dict(zip(q[0],q[1]))
    assert a['review_status']=='approved' and a['approved_policy_extract']==policy_extract(r,selected)
    assert len(v)==1 and c[1][1:3]==['UNVERIFIED CANDIDATE','true']
    save(store,r,revision=1,action='rejected',answer='Rejected by Bob',reviewer='Bob')
    with pytest.raises(ReviewConflict):
        save(store,r,revision=1,answer='Stale approved answer')
    q,*_=read(export_bytes(batch,db))
    a=dict(zip(q[0],q[1]))
    assert a['revision']=='2' and a['review_status']=='rejected' and a['reviewer']=='Bob'
    assert a['approved_policy_extract']=='' and a['reviewed_answer']=='Rejected by Bob'


def test_determinism_fidelity_readonly_and_cli(setup,tmp_path):
    batch,db,store,rows=setup
    before=[p.read_bytes() for p in (batch,db)]
    one=export_bytes(batch,db)
    assert one==export_bytes(batch,db)
    q,v,c,_=read(one)
    a=dict(zip(q[0],q[1]))
    assert json.loads(a['provenance_json'])==rows[0]['provenance']
    assert c[1][3:8]==['p.pdf','3','2','p.pdf:p3:c2','Policy must require MFA.']
    assert json.loads(c[1][8])==rows[0]['evidence_references'][0]
    out=tmp_path/'out.xlsx'
    assert main([str(batch),str(db),str(out)])==0
    assert out.read_bytes()==one
    with pytest.raises(SystemExit):
        main([str(batch),str(db),str(out)])
    assert [p.read_bytes() for p in (batch,db)]==before


@pytest.mark.parametrize('attack',['=1+1','+SUM(1,2)','-1+2','@SUM(A1)','\t=cmd()','\r=cmd()','https://example.com','<>&"\nUnicode ÃƒÂ°Ã…Â¸Ã‚Â¦Ã¢â‚¬Â°'])
def test_every_cell_literal_and_lossless(attack):
    xml=worksheet([['id','question_text','answer','source','reviewer'],[attack]*5])
    root=ET.fromstring(xml)
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    cells=root.findall('.//s:row',ns)[1].findall('s:c',ns)
    assert all(c.get('t')=='inlineStr' and c.find('s:f',ns) is None for c in cells)
    assert [c.find('s:is/s:t',ns).text for c in cells]==[attack]*5


def test_injected_real_fields(setup):
    batch,db,store,rows=setup
    # IDs, questions, references, reviewer fields, metadata, provenance and nested JSON.
    rows[0].update(id='=ID()',question='+QUESTION()',answer='@BATCH()',outcome='-OUTCOME()',provenance={'=key':'=VALUE()'})
    rows[0]['evidence_references'][0].update(source='=SOURCE()',chunk_id='=CHUNK()',quote='=QUOTE()')
    batch.write_text(json.dumps(rows))
    _,digest=load_batch(batch)
    other=db.with_name('injected.sqlite3')
    s=ReviewStore(other,digest)
    save(s,rows[0],answer='=ANSWER()',reviewer='=REVIEWER()',note='=NOTE()',selected=['=CHUNK()'])
    q,v,c,_=read(export_bytes(batch,other))
    a=dict(zip(q[0],q[1]))
    assert a['question_id']=='=ID()' and a['reviewed_answer']=='=ANSWER()' and a['reviewer']=='=REVIEWER()'
    assert c[1][3]=='=SOURCE()' and c[1][6:8]==['=CHUNK()','=QUOTE()']


@pytest.mark.parametrize('text',['a'*32768,'\x00','\x0b'], ids=['overlong','nul','vertical_tab'])
def test_refuse_unrepresentable_text(text):
    with pytest.raises(ValueError):
        worksheet([['header'],[text]])


def test_missing_mismatch_invalid_approval(setup,tmp_path):
    batch,db,store,rows=setup
    with pytest.raises(sqlite3.OperationalError):
        export_bytes(batch,tmp_path/'absent.sqlite3')
    assert not (tmp_path/'absent.sqlite3').exists()
    with pytest.raises(ValueError):
        snapshot(db,'wrong')
    save(store,rows[0])
    with store.connect() as con:
        payload=store.current(rows[0])
        payload.update(status='approved',answer='Yes deployed',citations=[{'source':'fake.pdf'}])
        con.execute('UPDATE events SET payload=?',(json.dumps(payload),))
    q,v,*_=read(export_bytes(batch,db))
    a=dict(zip(q[0],q[1]))
    assert a['review_status']=='invalid_stored_state' and a['approved_policy_extract']=='' and len(v)==1
    assert json.loads(a['stored_citations_unverified_json'])==[{'source':'fake.pdf'}]


def test_unknown_assessment_missing(setup):
    batch,db,store,rows=setup
    rows[0]['evidence_assessment']={}
    batch.write_text(json.dumps(rows))
    _,digest=load_batch(batch)
    other=db.with_name('unknown.sqlite3')
    ReviewStore(other,digest)
    q,*_=read(export_bytes(batch,other))
    assert dict(zip(q[0],q[1]))['implementation_evidence_missing']=='true'


def test_real_batch_opens_with_independent_reader(tmp_path):
    from openpyxl import load_workbook
    root=Path(__file__).resolve().parents[1]
    batch=root/'data/batch_runs/major_deliverable_1/results.json'
    rows,digest=load_batch(batch)
    db=tmp_path/'real.sqlite3'
    ReviewStore(db,digest)
    content=export_bytes(batch,db)
    book=load_workbook(BytesIO(content))
    assert book.sheetnames==['Questions','Verified citations','UNVERIFIED candidates','Export notes']
    assert book['Questions'].max_row==101 and book['UNVERIFIED candidates'].max_row==501
    assert book['Verified citations'].max_row==1
    assert all(c.data_type!='f' for ws in book for row in ws for c in row)
    assert book['Questions'].freeze_panes=='A2'
    questions = book['Questions']
    assert questions.auto_filter.ref == 'A1:AB101'
    headings = [cell.value for cell in questions[1]]
    assert questions.max_column == len(headings) == 28
    assert headings[24:] == ['preflight_disposition', 'preflight_reason_codes_json',
                             'preflight_reasons_json', 'preflight_next_action']
    from trustquery.questionnaire_status import report
    for source, cells, disposition in zip(rows, list(questions)[1:], report(rows), strict=True):
        values = dict(zip(headings, (cell.value for cell in cells), strict=True))
        assert values['question_id'] == source['id']
        assert values['question_text'] == source['question']
        assert values['batch_answer_unreviewed'] == source['answer']
        assert values['reviewed_answer'] == '' and values['approved_policy_extract'] == ''
        assert values['preflight_disposition'] == disposition['disposition']
        assert json.loads(values['preflight_reason_codes_json']) == disposition['reason_codes']
        assert json.loads(values['preflight_reasons_json']) == disposition['disposition_reasons']
        assert values['preflight_next_action'] == disposition['next_action']
        assert json.loads(values['stored_citations_unverified_json']) == []
    for sheet in book:
        assert all(cell.data_type == 's' and cell.hyperlink is None
                   for row in sheet for cell in row)
    # Reopen the serialized artifact a second time with the independent reader.
    assert list(load_workbook(BytesIO(content))['Questions'].values) == list(questions.values)


def test_snapshot_pins_all_revisions(setup,monkeypatch):
    import trustquery.review_export as module
    batch,db,store,rows=setup
    save(store,rows[0],answer='Before snapshot')
    with store.connect() as con:
        con.execute('PRAGMA journal_mode=WAL')
    real_connect=sqlite3.connect
    class Reader:
        def __init__(self,*args,**kwargs): self.db=real_connect(*args,**kwargs)
        def close(self): self.db.close()
        def __enter__(self): return self
        def __exit__(self,*args):
            self.db.rollback()
            self.db.close()
        def execute(self,sql,*args):
            if sql.startswith('SELECT e.question'):
                # A separate writer commits after the export read snapshot is pinned.
                with real_connect(db) as writer:
                    old=store.current(rows[0])
                    old.update(revision=2,answer='After snapshot')
                    writer.execute('INSERT INTO events(question,revision,payload) VALUES (?,?,?)',('q1',2,json.dumps(old)))
            return self.db.execute(sql,*args)
    monkeypatch.setattr(module.sqlite3,'connect',Reader)
    # Avoid store.current using the patched connection inside our writer hook.
    store.current(rows[0])
    q,*_=read(export_bytes(batch,db))
    assert dict(zip(q[0],q[1]))['reviewed_answer']=='Before snapshot'
    monkeypatch.setattr(module.sqlite3,'connect',real_connect)
    assert store.current(rows[0])['revision']==2


def test_unknown_question_and_mismatched_revision_refused(setup):
    batch,db,store,rows=setup
    save(store,rows[0])
    with store.connect() as con:
        payload=store.current(rows[0])
        payload['revision']=99
        con.execute('UPDATE events SET payload=?',(json.dumps(payload),))
    with pytest.raises(ValueError,match='revision'):
        export_bytes(batch,db)
    with store.connect() as con:
        payload['revision']=1
        con.execute('UPDATE events SET question=?,payload=?',('unknown',json.dumps(payload)))
    with pytest.raises(ValueError,match='unknown question'):
        export_bytes(batch,db)


def test_deterministic_with_other_xml_namespace_users(setup):
    batch,db,store,rows=setup
    before=export_bytes(batch,db)
    ET.register_namespace('unrelated','http://schemas.openxmlformats.org/spreadsheetml/2006/main')
    assert export_bytes(batch,db)==before
