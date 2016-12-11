from lxml import html
from lxml.html.clean import clean_html
from nltk.tokenize import sent_tokenize
import requests
from progress.bar import Bar
import xml
import json

def convert_unicode_char(text):
    text = text.replace(u'\xe2\x80\x9c', "\"");
    text = text.replace(u'\xe2\x80\x9d', "\"");
    text = text.replace(u'\xe2\x80\x99', '\'');
    text = text.replace(u'\xe2\x80\x94', " ");
    return text;

def clean_up_section(tree):
    original = tree.xpath('.//div[@class="original-line"]/text()')
    translation = tree.xpath('.//div[@class="modern-line"]/text()')

    if len(original) == 0 and len(translation) == 0:
        return [None, None];

    for i in range(0, len(original)):
        original[i] = convert_unicode_char(original[i]);
        original[i] = original[i].encode('ascii', 'ignore');

    original = ' '.join([str(x) for x in original]);
    original = original.replace('\n', ' ');
    original = original.split()
    original = ' '.join([str(x) for x in original])

    for i in range(0, len(translation)):
        translation[i] = convert_unicode_char(translation[i]);
        translation[i] = translation[i].encode('ascii', 'ignore');
    
    translation = ' '.join([str(x) for x in translation]);
    translation = translation.replace('\n', ' ');
    translation = translation.split()
    translation = ' '.join([str(x) for x in translation])
    return [original.lower(), translation.lower()];

def get_page(play, page_num):
    site = 'http://nfs.sparknotes.com/' + play + '/' + 'page_' + str(page_num) + '.html';
    page = requests.get(site)
    if page.status_code == 404:
        return None, None;
    tree = html.fromstring(clean_html(page.text.encode('utf-8')))
    
    original = tree.xpath('//td[@class="noFear-left"]');
    translation = tree.xpath('//td[@class="noFear-right"]');
    
    result_original = [];
    result_translation = [];

    for i in range(0, len(original)):
        orig_val = clean_up_section(original[i])[0];
        trans_val = clean_up_section(translation[i])[1];
        if orig_val == None and trans_val == None:
            continue;
        result_original.append(orig_val);
        result_translation.append(trans_val);
    
    return result_original, result_translation;

def scrape_play(play, max_len):
    original_list = [];
    translation_list = [];
    for i in range(2, max_len + 2, 2):
        orig, trans = get_page(play, i);
        if(orig == None and trans == None):
            break;
        original_list.append(orig);
        translation_list.append(trans);
    return original_list, translation_list;

play_names_style_1 = ['antony-and-cleopatra', 'asyoulikeit', 'errors', 'hamlet', 'henry4pt1', 'henry4pt2', 'henryv', 'juliuscaesar', 'lear', 'macbeth', 'merchant', 'msnd', 'muchado', 'othello', 'richardiii', 'romeojuliet', 'shrew', 'tempest', 'twelfthnight']

plays_original = [];
plays_translation = [];

orig_file = open('original_text.txt', 'w');
trans_file = open('translation_text.txt', 'w');

bar = Bar('Processing', max=(len(play_names_style_1)));
for i in range(0, len(play_names_style_1)):
    orig, trans = scrape_play(play_names_style_1[i], 5000)
    plays_original.append(orig);
    plays_translation.append(trans);
    bar.next();
bar.finish();

orig_file.write(json.dumps(plays_original));
trans_file.write(json.dumps(plays_translation));
orig_file.close();
trans_file.close();
