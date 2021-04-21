import json
import os
import re
import logging

#import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm

CITATION_PATTERN = re.compile(r'class=\"citation\" data-id=\"([0-9]+?)\"')

logger = logging.getLogger(__name__)


def get_pair_id(a: int, b: int) -> tuple:
    """
    Smaller ID always first

    :param a: document ID
    :param b: document ID
    :return: pair ID (document ID tuple)
    """
    if a < b:
        return a, b
    else:
        return b, a


def get_opinions_from_dump_dir(opinions_dump_dir,
                               opinions=None,
                               id2oid=None,
                               oid2id=None,
                               texts=None,
                               cits_by_source=None,  # source_id => target_ids[]
                               cits_by_target=None,
                               cits=None,
                               limit=0,
                               extract_opinions=True,
                               extract_texts=True,
                               extract_citations=True,
                               text_from_html_warning=False,
                               print_progress=False):
    """

    Read court opinions with citations from CourtListener bulk data.

    Download data from: https://www.courtlistener.com/api/bulk-info/

    :param opinions_dump_dir: Directory where JSON-files are located (decompressed .tar.gz file)
    :param opinions: List of opinions
    :param id2oid: mapping
    :param oid2id: mapping
    :param texts: List of plain text
    :param cits_by_source:
    :param cits_by_target:
    :param cits:
    :param limit: Limit the number of files loaded
    :return: opinions, idx2doc_id, doc_id2idx, texts, cits_by_source, cits_by_target, cits
    """
    if cits is None:
        cits = {}
    if cits_by_target is None:
        cits_by_target = {}
    if cits_by_source is None:
        cits_by_source = {}
    if texts is None:
        texts = []
    if oid2id is None:
        oid2id = {}
    if id2oid is None:
        id2oid = {}
    if opinions is None:
        opinions = {}

    files = os.listdir(opinions_dump_dir)

    logger.info(f'Loading {len(files)} files from {opinions_dump_dir}')

    idx_counter = 0

    if print_progress:
        files = tqdm(files, total=len(files))

    for i, fn in enumerate(files):
        if fn.endswith('.json'):
            with open(os.path.join(opinions_dump_dir, fn)) as fp:
                opinion = json.load(fp)

                if extract_opinions:
                    opinions[opinion['id']] = opinion

                # TODO for full corpus remove uncessary opinion infos
                # TODO text preprocessing
                id2oid[idx_counter] = opinion['id']
                oid2id[opinion['id']] = idx_counter

                if extract_texts:
                    text = opinion['plain_text']

                    if len(text) < 1:
                        if text_from_html_warning:
                            logger.warning(f'Empty plain text in {fn} (use HTML instead)')

                        if 'html' in opinion:
                            text = BeautifulSoup(opinion['html'], features="html.parser").get_text()
                        else:
                            logger.error('HTML field is missing')

                    texts.append(text)

                # citations positions from html
                if extract_citations:
                    content = opinion['html_with_citations']

                    for match in CITATION_PATTERN.finditer(content):
                        cit_pos = match.start(1)
                        target_id = int(match.group(1))
                        # cit_id = get_cit_id(opinion['id'], target_id)  # source_id, target_id
                        cit_id = opinion['id'], target_id

                        # Build citation data
                        if cit_id in cits:
                            cits[cit_id].append(cit_pos)
                        else:
                            cits[cit_id] = [cit_pos]

                            if opinion['id'] not in cits_by_source:
                                cits_by_source[opinion['id']] = []

                            cits_by_source[opinion['id']].append(target_id)

                            if target_id not in cits_by_target:
                                cits_by_target[target_id] = []

                            cits_by_target[target_id].append(opinion['id'])

                idx_counter += 1

                if 0 < limit <= i:
                    break

    return opinions, id2oid, oid2id, texts, cits_by_source, cits_by_target, cits
