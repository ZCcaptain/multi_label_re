import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property
import nltk
# nltk.download('punkt')


class Webnlg_selection_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root
        self.relations_path = os.path.join(self.raw_data_root, 'relations.txt')

        if not os.path.exists(self.relations_path):
            raise FileNotFoundError(
                'relations file not found, please check your downloaded data!')
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

    @cached_property
    def relation_vocab(self):
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, 'r'))

    def gen_bio_vocab(self):
        result = {'<pad>': 3, 'B': 0, 'I': 1, 'O': 2}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))

    def gen_relation_vocab(self):
        relation_vocab = {}
        i = 0
        for line in open(self.relations_path, 'r'):
            relation = line.strip("\n")
            if relation not in relation_vocab:
                relation_vocab[relation] = i
                i += 1
        relation_vocab['N'] = i
        json.dump(relation_vocab,
                  open(self.relation_vocab_path, 'w'),
                  ensure_ascii=False)

    def gen_vocab(self, min_freq: int):
        source = os.path.join(self.raw_data_root, self.hyper.train)
        target = os.path.join(self.data_root, 'word_vocab.json')

        cnt = Counter()  # 8180 total
        with open(source, 'r') as s:
            lines = json.loads(s.read())
            lines_len = len(lines)
            for line in lines:
                instance = line
                text = nltk.word_tokenize(instance['sentText'])
                cnt.update(text)
        result = {'<pad>': 0}
        i = 1
        for k, v in cnt.items():
            if v > min_freq:
                result[k] = i
                i += 1
        result['oov'] = i
        json.dump(result, open(target, 'w'), ensure_ascii=False)
    
    def format_sent(self, text, is_set=True):
        spe = [',', "'", "\"", "/", "|", "\\", "[", "]", "{", "}", "<", ">", ":", ";", "?", "(", ")", "!", "~"]
        text_list = list(text)
        i = 0
        while True:
            if text_list[i] in spe:
                text_list.insert(i, ' ')
                text_list.insert(i + 2, ' ')
                i = i + 3
            else:
                i = i + 1
            if i == len(text_list):
                break
        if text_list[i - 1] == '.' and is_set:
            text_list.insert(i-1, ' ')
        return ''.join(text_list)


    def _read_line(self, line: str) -> Optional[str]:
        instance = line
        textorg = self.format_sent(instance['sentText'])
        text = textorg.split()
        bio = None
        selection = None

        if 'triples' in instance:
            spo_list = instance['triples']
            relationMentions = instance['triples']

            if len(text) > self.hyper.max_text_len:
                return None

            if not self._check_valid(textorg, spo_list):
                return None

            spo_list = [{
                'predicate': spo['predicate'],
                'object': self.format_sent(spo['object'], is_set=False).split(),
                'subject': self.format_sent(spo['subject'], is_set=False).split()
            }  for spo in spo_list]


            bio, selection = self.spo_to_bio_selection(text, relationMentions)

        if len(selection) == 0:
            return None
        result = {
            'text': text,
            'spo_list': spo_list,
            'bio': bio,
            'selection': selection
        }
        return json.dumps(result, ensure_ascii=False)



    def _gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        count = 0
        count1 = 0
        with open(source, 'r') as s, open(target, 'w') as t:

            lines = json.loads(s.read())
            lines_len = len(lines)
            for line in lines:
                count += 1
                newline = self._read_line(line)
                if newline is not None:
                    count1+=1
                    t.write(newline)
                    t.write('\n')
        print(count, count1)

    def gen_all_data(self):
        self._gen_one_data(self.hyper.train)
        self._gen_one_data(self.hyper.dev)

    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False

        for t in spo_list:
            if t['subject'] not in text or t['object'] not in text:
                return False
        return True

    # def spo_to_entities(self, text: str,
    #                     spo_list: List[Dict[str, str]]) -> List[str]:
    #     entities = set(t['object'] for t in spo_list) | set(t['subject']
    #                                                         for t in spo_list)
    #     return list(entities)

    # def spo_to_relations(self, text: str,
    #                      spo_list: List[Dict[str, str]]) -> List[str]:
    #     return [t['predicate'] for t in spo_list]

    # def spo_to_selection(self, text, relationMentions) -> List[Dict[str, int]]:

    #     selection = []
    #     for triplet in relationMentions:


    #         object_pos_end = triplet['arg2EndIndex']
    #         subject_pos_end = triplet['arg1EndIndex']
    #         relation_pos = self.relation_vocab[triplet['predicate']]

    #         selection.append({
    #             'subject': subject_pos_end,
    #             'predicate': relation_pos,
    #             'object': object_pos_end
    #         })

    #     return selection


    def spo_to_bio_selection(self, text: str, relationMentions) -> List[str]:
        bio = ['O'] * len(text)
        selections = []
        for e in relationMentions:
            object, subject = self.format_sent(e['object'], is_set=False).split(),  self.format_sent(e['subject'], is_set=False).split()
            len_o = len(object)
            len_s = len(subject)
            o_end = None
            s_end = None
            for i in range(len(text)):
                if text[i] == object[0] and text[i+len_o-1] == object[-1]:
                    begin = i
                    o_end = i + len_o - 1
                    bio[i] = 'B'
                    if begin < o_end:
                        for i in range(begin + 1, o_end+1):
                            bio[i] = 'I'
                    break
            for i in range(len(text)):
                if text[i] == subject[0] and text[i+len_s-1] == subject[-1]:
                    begin = i
                    s_end = i + len_s - 1
                    bio[i] = 'B'
                    if begin < s_end:
                        for i in range(begin + 1, s_end+1):
                            bio[i] = 'I'
                    break
            if o_end and s_end:
                selections.append({
                'subject': s_end,
                'predicate': self.relation_vocab[e['predicate']],
                'object': o_end
            })
        if len(selections) == 0 and len(relationMentions) != 0:
            with open('./log.log', 'a+') as f:
                f.write(" ".join(text))
                f.write('\n')
        return bio, selections

