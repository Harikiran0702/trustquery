"""Deterministic, read-only Excel export of batch-bound saved reviews."""
import argparse
import copy
import hashlib
from contextlib import closing
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from trustquery.review_store import load_batch, policy_extract

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def export_review_json(state):
    # Long full-document extractions repeat answer/page text in provenance.
    # Keep text losslessly in answer columns and candidate rows; compact only
    # the redundant audit JSON so Excel's per-cell limit is not exceeded.
    out = copy.deepcopy(state)
    if out.get('experimental_policy'):
        meta = out['experimental_policy']
        meta['answer_sha256'] = hashlib.sha256(meta.pop('answer').encode()).hexdigest()
        for ref in meta['references']:
            ref['quote_sha256'] = hashlib.sha256(ref.pop('quote').encode()).hexdigest()
        out['experimental_text_storage'] = 'Full answer in reviewed_answer; full page quotes in UNVERIFIED candidates'
    return packed(out)


def snapshot(path, digest):
    """Read hash binding and latest revisions within one SQLite transaction."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=10)) as db:
        db.execute('BEGIN')
        if db.execute('SELECT hash FROM batch').fetchall() != [(digest,)]:
            raise ValueError('Review database belongs to a different batch')
        result = {}
        for question, revision, payload, created in db.execute(
            'SELECT e.question,e.revision,e.payload,e.created FROM events e '
            'WHERE e.revision=(SELECT MAX(x.revision) FROM events x WHERE x.question=e.question) ORDER BY e.question'):
            state = json.loads(payload)
            if state.get('revision') != revision:
                raise ValueError('Inconsistent stored review revision')
            result[question] = (state, created)
        return result


def column(index):
    name = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def worksheet(rows):
    if len(rows) > 1048576 or any(len(r) > 16384 for r in rows):
        raise ValueError('Excel worksheet size limit exceeded')
    root = ET.Element('worksheet', xmlns=NS)
    views = ET.SubElement(root, 'sheetViews')
    view = ET.SubElement(views, 'sheetView', workbookViewId='0')
    ET.SubElement(view, 'pane', ySplit='1', topLeftCell='A2', activePane='bottomLeft', state='frozen')
    cols = ET.SubElement(root, 'cols')
    for i, heading in enumerate(rows[0], 1):
        width = 55 if any(s in heading for s in ('answer','question_text','quote','json','note')) else 26
        ET.SubElement(cols, 'col', min=str(i), max=str(i), width=str(width), customWidth='1')
    data = ET.SubElement(root, 'sheetData')
    for number, values in enumerate(rows, 1):
        row = ET.SubElement(data, 'row', r=str(number))
        for index, value in enumerate(values, 1):
            text = str(value)
            if len(text.encode('utf-16-le')) // 2 > 32767:
                raise ValueError('Excel cell text limit exceeded; refusing truncation')
            if any(not (ch in '\t\n\r' or 32 <= ord(ch) <= 0xD7FF or 0xE000 <= ord(ch) <= 0xFFFD or 0x10000 <= ord(ch) <= 0x10FFFF) for ch in text):
                raise ValueError('XML-incompatible cell text; refusing data loss')
            # Every cell is literal text: no formula, hyperlink, DDE or macro elements.
            cell = ET.SubElement(row, 'c', r=f'{column(index)}{number}', t='inlineStr')
            inline = ET.SubElement(cell, 'is')
            ET.SubElement(inline, 't', {'{http://www.w3.org/XML/1998/namespace}space':'preserve'}).text = text
    ET.SubElement(root, 'autoFilter', ref=f'A1:{column(len(rows[0]))}{len(rows)}')
    return ET.tostring(root, encoding='utf-8', xml_declaration=True).replace(b'\r', b'&#13;')


def export_bytes(batch_path, state_path):
    records, digest = load_batch(batch_path)
    states = snapshot(state_path, digest)
    if set(states) - {r['id'] for r in records}:
        raise ValueError('Review database contains unknown question IDs')
    questions = [['question_id','question_text','review_status','reviewed_answer','approved_policy_extract','approval_scope',
                  'implementation_evidence_missing','reviewer','reviewer_decision_note','revision','saved_at_utc',
                  'scope_checked','outcome','batch_answer_unreviewed','selected_chunk_ids_json','provenance_json',
                  'assessment_json','errors_json','batch_sha256','operational_proof','stored_status','stored_citations_unverified_json','batch_citations_unverified_json','saved_review_json','preflight_disposition','preflight_reason_codes_json','preflight_reasons_json','preflight_next_action']]
    candidates = [['question_id','evidence_status','selected_for_review','source','page_1_based','chunk_index_0_based','chunk_id','quote','reference_json']]
    verified = [['question_id','source','page_1_based','chunk_index_0_based','chunk_id','quote','verification_status']]
    from trustquery.questionnaire_status import report
    dispositions = {i['id']: i for i in report(records)}
    for r in records:
        disposition = dispositions[r['id']]
        s, created = states.get(r['id'], ({}, ''))
        status = s.get('status', 'unreviewed')
        selected = s.get('selected', [])
        if not isinstance(selected, list) or any(k not in {e['chunk_id'] for e in r['evidence_references']} for k in selected):
            raise ValueError('Invalid stored candidate selection')
        missing = r['evidence_assessment'].get('implementation_required') is not False or r['outcome'] == 'implementation_evidence_required' or status == 'needs_evidence'
        valid = False
        if status == 'approved':
            try:
                experimental = s.get('experimental_policy')
                if experimental is not None:
                    from trustquery.review_policy import validate, SCOPE
                    validate(r, experimental, s.get('answer'), approve=True)
                valid = (s.get('operational_proof') is False and not missing and r['outcome'] != 'processing_error' and s.get('scope_checked') is True
                         and bool(s.get('reviewer','').strip()) and bool(s.get('note','').strip())
                         and ((experimental is not None and s.get('approval_scope') == SCOPE) or
                              (experimental is None and s.get('approval_scope') == 'synthetic_policy_extract_only'
                               and s.get('answer') == policy_extract(r, selected))))
            except (ValueError, OSError, KeyError, IndexError):
                pass
        effective = status if status in ('unreviewed','edited','rejected','needs_evidence') or valid else 'invalid_stored_state'
        questions.append([r['id'],r['question'],effective,s.get('answer','') if s.get('revision',0) else '',
                          s.get('answer','') if valid else '', s.get('approval_scope', '') if valid else '',
                          str(missing).lower(),s.get('reviewer',''),s.get('note',''),s.get('revision',0),created,
                          str(s.get('scope_checked',False)).lower(),r['outcome'],r['answer'],packed(selected),
                          packed(r.get('provenance',{})),packed(r['evidence_assessment']),packed(r.get('errors',[])),digest,'false',
                          status,packed(s.get('citations',[])),packed(r.get('citations',[])),export_review_json(s),disposition['disposition'],packed(disposition['reason_codes']),packed(disposition['disposition_reasons']),disposition['next_action']])
        for e in s.get('experimental_policy', {}).get('references', []):
            candidates.append([r['id'],'UNVERIFIED EXPERIMENTAL POLICY CANDIDATE','true',e['source'],e['page'],'',e['evidence_id'],e['quote'],packed(e)])
        for e in r['evidence_references']:
            candidates.append([r['id'],'UNVERIFIED CANDIDATE',str(e['chunk_id'] in selected).lower(),e['source'],e['page'],e['chunk_index'],e['chunk_id'],e['quote'],packed(e)])
    notes = [['property','value'],['format','TrustQuery review export v1'],['batch_sha256',digest],
             ['review_state','Latest saved revisions from one SQLite read transaction. Unsaved/conflicted browser drafts excluded.'],
             ['approval','Only synthetic policy extracts can be approved; no adoption, implemented controls or compliance established.'],
             ['preflight_disposition','Saved offline preflight snapshot only; not semantic validation or a reviewer decision. Source drift is rechecked in the UI before drafting.'],
             ['verified_citations','Empty: current review model does not verify citations. Stored/batch citations are never promoted.'],
             ['candidates','All remain UNVERIFIED CANDIDATE, even when selected or policy extract approved.'],
             ['missing_evidence','true for implementation-required, unknown implementation assessment, or needs_evidence review. false does not establish implementation.'],
             ['storage','All cells are literal text, including identifiers, locators and JSON. No truncation.'],
             ['integrity','Batch hash binds reviews to input bytes. Restricted UI runs record eight checked policy PDF hashes in provenance; legacy batches may not. Export does not revalidate current PDFs.']]
    sheets = [('Questions',questions),('Verified citations',verified),('UNVERIFIED candidates',candidates),('Export notes',notes)]
    files = {}
    files['[Content_Types].xml'] = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,5)) + '</Types>')
    relns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    files['_rels/.rels'] = f'<Relationships xmlns="{relns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    files['xl/workbook.xml'] = '<workbook xmlns="'+NS+'" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i,(name,_) in enumerate(sheets,1))+'</sheets></workbook>'
    files['xl/_rels/workbook.xml.rels'] = f'<Relationships xmlns="{relns}">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,5))+'</Relationships>'
    for i, (_, rows) in enumerate(sheets,1):
        files[f'xl/worksheets/sheet{i}.xml'] = worksheet(rows)
    output = BytesIO()
    with ZipFile(output,'w',compression=ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info,content)
    return output.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('state',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() != '.xlsx':
            raise ValueError('Output must be .xlsx')
        content = export_bytes(args.batch,args.state)
        with args.output.open('xb') as stream:
            stream.write(content)
    except (ValueError,OSError,sqlite3.Error) as exc:
        parser.exit(2,f'error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
