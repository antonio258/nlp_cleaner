import enum
import spacy
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
from spellchecker import SpellChecker


class PreProcessing:
    def __init__(self, noadverbs=False, noadjectives=False, noverbs=False, noentities=True, language='en', remove_list=False):
        
        if language == 'en':
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
        elif language == 'pt':
            try:
                self.nlp = spacy.load('pt_core_news_sm')
            except OSError:
                os.system('python -m spacy download pt_core_news_sm')
                self.nlp = spacy.load('pt_core_news_sm')

        self.noadverbs = noadverbs
        self.noadjectives = noadjectives
        self.noverbs = noverbs
        self.noentities = noentities
        self.remove_list = remove_list
        self.stopwords = []

    @staticmethod
    def lowercase_unidecode(text):
        if type(text) == str:
            return unidecode(text.lower())
        elif type(text) == list:
            return [unidecode(str(x).lower()) for x in text]
        return ''

    @staticmethod
    def remove_tweet_marking(text):
        if type(text) == str:
            return re.sub('(@|#)\S+', '', text)
        elif type(text) == list:
            return [re.sub('(@|#)\S+', '', str(x)) for x in text]
        return ''

    @staticmethod
    def remove_punctuation(text):
        punctuation = """(!|"|#|\$|%|&|'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|\@|\[|\]|\^|_|`|\{|\|\}|~|\|)"""
        if type(text) == str:
            text = re.sub(punctuation, ' ',text)
            text = re.sub(' {2,}', ' ', text)
            return text
        elif type(text) == list:
            text = [re.sub(punctuation, ' ',str(x)) for x in text]
            return [re.sub(' {2,}', ' ', x) for x in text]
        return ''

    @staticmethod
    def remove_urls(text):
        if type(text) == str:
            text = re.sub(r'http\S+', '', text)
            return text
        elif type(text) == list:
            return [re.sub(r'http\S+', '', x) for x in text]
        return ''

    @staticmethod
    def remove_repetion(text):
        if type(text) == str:
            return re.sub(r'([a-z])\1{2}', '', text)
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series:
            return [re.sub(r'([a-z])\1{2}', '', str(x)) for x in text]
        return ''

    def remove_stopwords(self, text, method='extended'):
        if method == 'extended':
            stopwords = [unidecode(x).lower() for x in list(self.nlp.Defaults.stop_words)]
            if self.remove_list:
                stopwords.extend(self.remove_list)
        elif method == 'replace':
            stopwords = self.remove_list

        #     self.stopwords += stopwords
        if type(text) == list:
            if type(text[0]) == list:
                for i, j in enumerate(text):
                    text[i] = [x for x in j if x not in stopwords]
                return text
            else:
                return [x for x in text if x not in stopwords]
        return ''


    def spacy_processing(self, docs, n_process=-1, lemma=False):
        all_docs = self.nlp.pipe(docs, n_process=n_process)
        pp_docs = []
        language_docs = []
        for doc in tqdm(all_docs):
            pp_doc = [token for token in doc if token.is_ascii]  # remove no ascii
            if self.noadverbs:
                pp_doc = [token for token in pp_doc if token.pos_ != 'ADV']  # remove adverbs
            if self.noadjectives:
                pp_doc = [token for token in pp_doc if token.pos_ != 'ADJ']  # remove adjectives
            if self.noverbs:
                pp_doc = [token for token in pp_doc if token.pos_ != 'VERB']  # remove verbs
            pp_doc = [token for token in pp_doc if not token.is_digit]  # remove digits
            pp_doc = [token for token in pp_doc if not token.is_punct]  # remove punct
            pp_doc = [token for token in pp_doc if not token.is_space]  # remove whitespace
            # pp_doc = [token for token in pp_doc if not token.is_stop]  # remove stopwords
            pp_doc = [token for token in pp_doc if not token.like_num]  # remove numerals
            # pp_doc = [token for token in pp_doc if not token.like_url]  # remove urls
            # pp_doc = [token for token in pp_doc if not token.like_email]  # remove emails

            # Remove Entities
            if self.remove_list and self.noentities:
                for token in pp_doc:
                    if token.ent_type_ in ['MONEY', 'DATE', 'PERSON', 'PERCENT', 'ORDINAL', 'CARDINAL', 'QUANTITY', 'GPE', 'NORP', 'LANGUAGE']:
                        self.stopwords.append(token.lower_)
            elif self.noentities:
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'MONEY']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'DATE']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'PERSON']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'PERCENT']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'ORDINAL']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'CARDINAL']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'QUANTITY']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'GPE']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'NORP']
                pp_doc = [token for token in pp_doc if token.ent_type_ != 'LANGUAGE']
            pp_doc = [self.remove_infinitive(token.lemma_).lower() if token.pos_ == "VERB" else
                      token.lemma_.lower() if lemma else token.lower_ for token in pp_doc]
            pp_docs.append(pp_doc)

        return pp_docs

    def remove_n(self, text, n):
        if type(text) == str:
            if len(text) < n:
                if self.remove_list:
                    self.stopwords.append(text)
                else:
                    return None
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series:
            if type(text[0]) == list:
                for i, j in enumerate(text):
                    if self.remove_list:
                        self.stopwords += [x for x in j if len(x) < n]
                    else:
                        text[i] = [x for x in j if len(x) >= n]
                return text
            else:
                if self.remove_list:
                    self.stopwords += [x for x in text if len(x) < n]
                else:
                    return [x for x in text if len(x) >= n]
        return ''

    @staticmethod
    def remove_numbers(text, mode='filter', language='pt'):
        spell = SpellChecker(language=language)
        if type(text) == str:
            if mode == 'filter':
                if re.search("[0-9]", text):
                    return None
            elif mode == 'spell':
                if re.search("[0-9]", text):
                    return spell.correction(text)
            elif mode == 'replace':
                return re.sub("[0-9]", "", text)
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series:
            if type(text[0]) == list:
                for i, j in enumerate(text):
                    if mode == 'filter':
                        text[i] = [x for x in j if not re.search("[0-9]", x) if x]
                    elif mode == 'spell':
                        text[i] = [x if not re.search("[0-9]", x) else spell.correction(x) for x in j if x]
                    elif mode == 'replace':
                        text[i] = [re.sub("[0-9]", "", x) for x in j if x]
                return text
            else:
                if mode == 'filter':
                    return [x for x in text if not re.search("[0-9]", x)]
                elif mode == 'spell':
                    return [x if not re.search("[0-9]", x) else spell.correction(x) for x in text if x]
                elif mode == 'replace':
                        return [re.sub("[0-9]", "", x) for x in text if x]
        return text

    @staticmethod
    def remove_gerund(text):
        if type(text) == str:
            return re.sub(r"ndo$", "", text)
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series:
            if type(text[0]) == list:
                for i, j in enumerate(text):
                    text[i] = [re.sub(r"ndo$", "", x) for x in j if x]
            else:
                text = [re.sub(r"ndo$", "", x) for x in text if x]
            return text
        return ''

    @staticmethod
    def remove_infinitive(text):
        if type(text) == str:
            return re.sub(r"r$", "", text)
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series:
            if type(text[0]) == list:
                for i, j in enumerate(text):
                    text[i] = [re.sub(r"r$", "", x) for x in j if x]
            else:
                text = [re.sub(r"r$", "", x) for x in text if x]
            return text
        return ''