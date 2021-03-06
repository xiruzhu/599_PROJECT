import json
from progress.bar import Bar
import process_text
from nltk.tokenize import sent_tokenize

def read_file(orig_file_name, trans_file_name):
    read_file = open(orig_file_name, 'r');
    read_orig_raw = json.load(read_file);
    read_file.close();
    read_file = open(trans_file_name, 'r');
    read_trans_raw = json.load(read_file);
    read_file.close();
    read_orig_list = [];
    read_trans_list = [];
    for i in range(0, len(read_orig_raw)):
        read_orig_list.extend(read_orig_raw[i]);
        read_trans_list.extend(read_trans_raw[i]);
    
    compacted_orig_result = [];
    compacted_trans_result = [];
    bar = Bar('Processing', max=(len(read_orig_list)));
    for i in range(0, len(read_orig_list)):
        compacted_orig_result.extend(read_orig_list[i]);
        compacted_trans_result.extend(read_trans_list[i]);
        bar.next();
    bar.finish();

    final_orig_text = [];
    final_trans_text = [];
    bar = Bar('Processing', max=(len(compacted_orig_result)));
    for i in range(0, len(compacted_orig_result)):
        if compacted_orig_result[i] == None or compacted_trans_result[i] == None:
            continue;
        orig =  process_text.get_tokens(compacted_orig_result[i].encode('ascii', 'ignore'));
        trans = process_text.get_tokens(compacted_trans_result[i].encode('ascii', 'ignore'));

        temp = [];
        for i in range(0, len(orig)):
            if not (ispunct(orig[i]) and len(orig[i]) > 1):
                temp.extend(split_punct(orig[i]));
        orig = temp;
        temp = [];
        for i in range(0, len(trans)):
            if not (ispunct(trans[i]) and len(trans[i]) > 1):
                temp.extend(split_punct(trans[i]));
        trans = temp;


        for i in range(0, len(orig)):
            if len(orig[i]) > 2:
                orig[i].replace('\"', '');
                orig[i].replace('\'', '');
        for i in range(0, len(trans)):
            if len(trans[i]) > 2:
                trans[i].replace('\"', '');
                trans[i].replace('\'', '');
        final_orig_text.append(' '.join([str(x) for x in orig]));
        final_trans_text.append(' '.join([str(x) for x in trans]));
        bar.next();
    bar.finish();
    return final_orig_text, final_trans_text

def split_punct(some_string):
    accumulated_word = [];
    word = "";
    for character in some_string:
        if character.isalnum() or character == '\'':
            word += str(character);
        else:
            accumulated_word.append(word);
            accumulated_word.append(str(character));
            word = "";
    if len(word) > 0:
        accumulated_word.append(word);
    return accumulated_word;

def ispunct(some_string):
    return not any(char.isalnum() for char in some_string);

def build_dictionary_table(orig_raw, trans_raw):
    dictionary_table = {};
    bar = Bar('Processing', max=(len(orig_raw)));
    for i in range(0, len(orig_raw)):
        orig_tokens = process_text.get_tokens(orig_raw[i]);
        trans_tokens = process_text.get_tokens(trans_raw[i]);
        for j in range(0, len(orig_tokens)):
            if ispunct(orig_tokens[j]):
                continue;
            if not orig_tokens[j] in dictionary_table:
                dictionary_table[orig_tokens[j]] = {};
            if trans_raw[i].find(orig_tokens[j]) >= 0:
                if orig_tokens[j] in dictionary_table[orig_tokens[j]]:
                    dictionary_table[orig_tokens[j]][orig_tokens[j]] += 1;
                else:
                    dictionary_table[orig_tokens[j]][orig_tokens[j]] = 1;
            else:
                for k in range(0, len(trans_tokens)):
                    if ispunct(trans_tokens[k]):
                        continue;
                    if trans_tokens[k] in dictionary_table[orig_tokens[j]]:
                        dictionary_table[orig_tokens[j]][trans_tokens[k]] += 1;
                    else:
                        dictionary_table[orig_tokens[j]][trans_tokens[k]] = 1;
        bar.next();
    bar.finish();
    return dictionary_table; 

def calc_prob_table(dictionary_table, min_size):
    bar = Bar('Processing', max=(len(dictionary_table)));
    new_dict_table = {};
    for words in dictionary_table:
        total = 0.0;
        for definition in dictionary_table[words]:
            if dictionary_table[words][definition] > min_size:
                total += dictionary_table[words][definition];

        new_dict_table[words] = {};
        for definition in dictionary_table[words]:
            if dictionary_table[words][definition] > min_size:
                new_dict_table[words][definition] = dictionary_table[words][definition]/float(total);
        bar.next();
    bar.finish();
    return new_dict_table;

def write_dict_to_csv(dict_table, file_name):
    file_str = "";
    bar = Bar('Processing', max=(len(dict_table)));
    for word in dict_table:
        for definition in dict_table[word]:
            file_str += word + ',' + definition + ',' + str(dict_table[word][definition]) + '\n';
        bar.next();
    bar.finish();
    file_id = open(file_name, 'w');
    file_id.write(file_str);
    file_id.close();

def create_sentence_set(orig_raw, trans_raw):
    parallel_corpus = [];
    non_parallel_corpus = [];
    bar = Bar('Processing', max=(len(orig_raw)));
    for i in range(0, len(orig_raw)):
        sent_orig = sent_tokenize(orig_raw[i]);
        sent_trans = sent_tokenize(trans_raw[i]);
        if len(sent_orig) == 1 and len(sent_trans) == 1:
            parallel_corpus.append([sent_orig[0], sent_trans[0]]);
        elif len(sent_orig) == len(sent_trans):
            for i in range(0, len(sent_orig)):
                parallel_corpus.append([sent_orig[i], sent_trans[i]]);
        else:
            non_parallel_corpus.append([sent_orig, sent_trans]);
        bar.next();
    bar.finish();
    return parallel_corpus, non_parallel_corpus;

def create_corpus(parallel_corpus, file_name):
    sentences = [];
    for translation_unit in parallel_corpus:
        sentences.append(translation_unit[0]);
        sentences.append(translation_unit[1]);
    result = '\n'.join([str(x) for x in sentences])
    file_id = open(file_name, 'w');
    file_id.write(result);
    file_id.close();

def create_non_parallel_corpus(non_parallel_corpus, file_name_orig, file_name_trans):
    bar = Bar('Processing', max=(len(non_parallel_corpus)));
    for i in range(0, len(non_parallel_corpus)):
        file_id = open(file_name_orig+'_'+str(i)+'.txt', 'w');
        for j in range(0, len(non_parallel_corpus[i][0])):
            file_id.write((non_parallel_corpus[i][0][j] + '\n').encode('utf-8', 'ignore'));
        file_id.close();
        file_id = open(file_name_trans+'_'+str(i)+'.txt', 'w');
        for j in range(0, len(non_parallel_corpus[i][1])):
            file_id.write((non_parallel_corpus[i][1][j] + '\n').encode('utf-8', 'ignore'));
        file_id.close();
        bar.next();
    bar.finish();

orig_raw, trans_raw = read_file("original_text.txt", "translation_text.txt");
#parallel_corpus, non_parallel_corpus = create_sentence_set(orig_raw, trans_raw);
#create_corpus(parallel_corpus, 'models/corpora.en-sk');

dict_table = build_dictionary_table(orig_raw, trans_raw);
dict_table = calc_prob_table(dict_table, 5);
write_dict_to_csv(dict_table, 'models/dict.csv');
#create_non_parallel_corpus(non_parallel_corpus, 'models/align_model/non_parallel_corpus/orig', 'models/align_model/non_parallel_corpus/trans');
