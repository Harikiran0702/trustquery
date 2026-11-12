"""Read-only, bounded local policy context. Never verifies a candidate citation."""
from pathlib import Path
import re
from pypdf import PdfReader
from trustquery.batch_retrieval import POLICIES

POLICY_DIRECTORY = Path(__file__).resolve().parents[2] / 'data' / 'sample'


def page_context(reference, directory=None):
    """Resolve only known basenames under the trusted corpus, without path aliases.

    No paths or URLs from batch provenance are followed. Read fresh on demand;
    current local text may differ from the retrieval-time corpus.
    """
    source = reference.get('source')
    if source not in {name + '.pdf' for name in POLICIES}:
        raise ValueError('Source is not an allowed policy PDF basename')
    page = reference.get('page')
    if type(page) is not int or page < 1:
        raise ValueError('Invalid one-based PDF page')
    folder = Path(directory) if directory is not None else POLICY_DIRECTORY
    # Reject junction/symlink redirection of the corpus or document.
    if folder.resolve() != folder.absolute():
        raise ValueError('Redirected policy directory is not allowed')
    path = folder / source
    if path.resolve() != path.absolute():
        raise ValueError('Redirected policy PDF is not allowed')
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Policy PDF exceeds the 20 MiB viewer limit')
    with path.open('rb') as stream:
        if stream.read(5) != b'%PDF-':
            raise ValueError('Source is not a PDF')
        stream.seek(0)
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError('Encrypted policy PDFs are not supported')
        count = len(reader.pages)
        if count > 200 or page > count:
            raise ValueError('PDF page is unavailable or document exceeds 200 pages')
        pages = [(n, (reader.pages[n-1].extract_text() or '')[:100000])
                 for n in range(max(1, page-1), min(count, page+1)+1)]
    def normalize(text):
        return re.sub(r'\s+', ' ', text).strip()
    current = next(text for n, text in pages if n == page)
    quote = reference.get('quote', '')
    return dict(source=source, page=page, total_pages=count, pages=pages,
                quote_present=bool(normalize(quote)) and normalize(quote) in normalize(current))
