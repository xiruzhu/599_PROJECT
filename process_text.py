import json
from nltk.tokenize import sent_tokenize, word_tokenize

def get_tokens(s):
    '''
    Tokenize into words in sentences.
    
    Returns list of strs
    '''
    retval = []
    sents = sent_tokenize(s)
    
    for sent in sents:
        tokens = word_tokenize(sent)
        retval.extend(tokens)
    return retval

def convert_unicode_char(text):
    text = text.replace(u'\xe2\x80\x9c', "\"");
    text = text.replace(u'\xe2\x80\x9d', "\"");
    text = text.replace(u'\xe2\x80\x99', '\'');
    text = text.replace(u'\xe2\x80\x94', " ");
    return text;
    
def tokenize_sentence(original_text, translation_text):
    orig_range = [];
    trans_range = [];
    for i in range(0, len(original_text)):
        orig_range.append([])
        for j in range(0, len(original_text[i])):
            orig_range[i].append([]);
            for k in range(0, len(original_text[i][j])):
                try:
                    if len(original_text[i][j][k]) > 0:
                        orig_range[i][j].append(get_tokens(original_text[i][j][k]));
                except:
                    print(original_text[i][j][k])

    for i in range(0, len(translation_text)):
        trans_range.append([])
        for j in range(0, len(translation_text[i])):
            trans_range[i].append([]);
            for k in range(0, len(translation_text[i][j])):
                try:
                    if len(translation_text[i][j][k]) > 0:
                        trans_range[i][j].append(get_tokens(translation_text[i][j][k]));
                except:
                    print(translation_text[i][j][k])
    return orig_range, trans_range;

def calculate_range(original_text, translation_text):
    orig_range = [];
    trans_range = [];
    for i in range(0, len(original_text)):
        for j in range(0, len(original_text[i])):
            for k in range(0, len(original_text[i][j])):
                try:
                    orig_range.append(len(original_text[i][j][k]));
                except:
                    print(original_text[i][j][k])

    for i in range(0, len(translation_text)):
        for j in range(0, len(translation_text[i])):
            for k in range(0, len(translation_text[i][j])):
                try:
                    trans_range.append(len(translation_text[i][j][k]));
                except:
                    print(translation_text[i][j][k])

    return orig_range, trans_range;

def read_file():
    orig_file = open('original_text.txt', 'r');
    trans_file = open('translation_text.txt', 'r');

    original_text = json.load(orig_file);
    translation_text = json.load(trans_file);

    orig_file.close();
    trans_file.close();
    return original_text, translation_text;

def get_freq_dict(word_split_text_file):
    frequency = {}
    for play in word_split_text_file:
        for page in play:
            for text in page:
                for word in text:
                    if word in frequency:
                        frequency[word] += 1;
                    else :
                        frequency[word] = 1;
    return frequency;

