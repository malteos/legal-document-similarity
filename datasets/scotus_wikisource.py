"""


https://en.wikisource.org/wiki/Category:United_States_Supreme_Court_decisions_by_topic


https://www.mediawiki.org/wiki/API:Categorymembers


https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Physics&cmlimit=20


https://en.wikisource.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:United_States_Supreme_Court_decisions_by_topic&cmlimit=100


https://en.wikipedia.org/w/api.php?action=parse&page=Pet_door&format=json

https://en.wikipedia.org/w/api.php?action=parse&page=Pet_door&format=json


- get category members
- extract case meta data
- find court listener case


"""
import time
from typing import List

import requests
import re
import logging
import json
import os

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


ARTICLE_NS = 0
CATEGORY_NS = 14
OUT_DIR = 'out'
WIKITEXT_DIR = 'out/wikitext'
DOMAIN = 'en.wikisource.org'

main_category = 'Category:United_States_Supreme_Court_decisions_by_topic'
limit = 500


def get_category_members(category_title, domain, limit=500, cmcontinue=None) -> List:
    global ARTICLE_NS, CATEGORY_NS

    members = []
    url = f'https://{domain}/w/api.php?action=query&list=categorymembers&cmtitle={category_title}&cmlimit={limit}&format=json'

    if cmcontinue:
        url += '&cmcontinue=' + cmcontinue

    res = requests.get(url).json()

    if 'query' in res and 'categorymembers' in res['query']:
        for item in res['query']['categorymembers']:  # iterate over members
            if item['ns'] == ARTICLE_NS:
                members.append({
                    'pageid': item['pageid'],
                    'title': item['title'],
                })
            elif item['ns'] == CATEGORY_NS:
                logger.info(f'Sub-category: {item}')
                members += get_category_members(item['title'], domain, limit)
            else:
                logger.warning(f'Unsupported namespace: {item}')
    else:
        logger.warning(f'Empty response: {res}')

    if 'continue' in res:
        logger.warning('Continue on next API page')
        members += get_category_members(category_title, domain, limit, cmcontinue=res['continue']['cmcontinue'])

    return members


def get_case_from_page(page) -> dict:
    page_title = page['title']

    # page_title = 'Austin v. Michigan Chamber of Commerce'
    url = f'https://{DOMAIN}/w/api.php?action=parse&page={page_title}&prop=wikitext&format=json'
    res = requests.get(url).json()

    wikitext = res['parse']['wikitext']['*']
    page['wikitext'] = wikitext

    match = re.search(r'{{CaseCaption(.*?)}}', wikitext, re.MULTILINE + re.DOTALL)

    if not match:
        logger.warning(f'CaseCaption-template could not be found in: #{idx} / {page_title}')

    else:
        meta = {}

        for line in match.group(1).splitlines():
            line_match = re.search(r'^\| ', line)
            if line_match:
                name, value = line[line_match.end(0):].split('=', 1)
                meta[name.strip()] = value.strip()

        page['meta'] = meta

        if len(meta) < 1:
            logger.warning(f'No meta found in CaseCaption-template: #{idx} / {page_title}')

    return page


url = f'https://{DOMAIN}/w/api.php?action=query&list=categorymembers&cmtitle={main_category}&cmlimit={limit}&format=json'
res = requests.get(url).json()

categories = {}

for item in res['query']['categorymembers']:  # iterate over sub-categories
    if item['ns'] == CATEGORY_NS:  # member is a category

        categories[item['title']] = {
            'pageid': item['pageid'],
            'pages': []
        }
    else:
        logger.warning(f'Is not in category namespace: {item}')

if 'continue' in res:
    logger.warning(f'There are more than {limit} categories available!')


# 'Category:United States Supreme Court decisions on the First Amendment'
completed_categories = [
    fn.replace('.json', '') for fn in os.listdir(OUT_DIR) if fn.endswith('.json')
]

logger.info(f'Completed: {len(completed_categories)} / {len(categories)}')

# Retrieve all pages
for i, category_title in enumerate(categories):
    # category_title = 'Category:United States Supreme Court decisions on the First Amendment'
    if category_title in completed_categories:
        continue

    logger.info(category_title)

    # Pages are category members
    pages = get_category_members(category_title, DOMAIN, limit)

    # Save unprocessed
    with open(os.path.join(OUT_DIR, category_title + '.json'), 'w') as f:
        json.dump(pages, f)

    for idx, page in enumerate(pages):
        try:
            pages[idx] = get_case_from_page(page)

            logger.info(f'Completed: {idx} / {len(pages)} pages')

            time.sleep(10)

        except BaseException as e:
            logger.error(f'Something went wrong... {e} {page}')

    # Save processed pages
    with open(os.path.join(OUT_DIR, category_title + '.json'), 'w') as f:
        json.dump(pages, f)

    if i > (len(completed_categories) + 1):
        break