import spacy
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
from unidecode import unidecode
from spellchecker import SpellChecker


class PreProcessing:
    def __init__(self, noadverbs: bool = False, noadjectives: bool = False, noverbs: bool = False,
                 noentities: bool = False, language: str = 'en', remove_list: bool = False):
        """
        Classe de pré-processamento de bases textuais
        Args:
            noadverbs (bool): Se verdadeiro os advérbios serão removidos no spacy.
            noadjectives (bool):  Se verdadeiro os adjetivos serão removidos no spacy.
            noverbs (bool):  Se verdadeiro os verbos serão removidos no spacy.
            noentities (bool):  Se verdadeiro as entidades serão removidas no spacy.
            language (str): Idioma utilizado, definiará as stopwords e a correção automatica de números.
            remove_list:
        """
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
        elif language == 'es':
            try:
                self.nlp = spacy.load('pt_core_news_sm')
            except OSError:
                os.system('python -m spacy download es_core_news_sm')
                self.nlp = spacy.load('es_core_news_sm')

        self.noadverbs = noadverbs
        self.noadjectives = noadjectives
        self.noverbs = noverbs
        self.noentities = noentities
        self.remove_list = remove_list
        self.stopwords = [unidecode(x).lower() for x in list(self.nlp.Defaults.stop_words)]

    @staticmethod
    def lowercase_unidecode(text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Aplica normalização nos dados, os caracteres serão converidos em minúsculos e a acentuação será removida.
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        if type(text) == str:
            return unidecode(text.lower())
        elif type(text) == list or type(text) == pd.Series or type(text) == np.array:
            return [unidecode(str(x).lower()) for x in text]
        return ''

    @staticmethod
    def remove_urls(text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Aplica a remoção de urls
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        if type(text) == str:
            text = re.sub(r'http\S+', '', text)
            return text
        elif type(text) == list or type(text) == pd.Series or type(text) == np.array:
            return [re.sub(r'http\S+', '', x) for x in text]
        return ''

    @staticmethod
    def remove_tweet_marking(text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Aplica a remoção de marcações do twitter '@'
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        if type(text) == str:
            return re.sub('(@|#)\S+', '', text)
        elif type(text) == list or type(text) == pd.Series or type(text) == np.array:
            return [re.sub('(@|#)\S+', '', str(x)) for x in text]
        return ''

    @staticmethod
    def remove_punctuation(text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Aplica a remoção de pontuação
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        punctuation = """(!|"|#|\$|%|&|'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|\@|\[|\]|\^|_|`|\{|\}|~|\||\r\n|\n|\r|\\\)"""
        if type(text) == str:
            text = re.sub(punctuation, ' ', text)
            text = re.sub(' {2,}', ' ', text)
            return text
        elif type(text) == list or type(text) == pd.Series or type(text) == np.array:
            text = [re.sub(punctuation, ' ', str(x)) for x in text]
            return [re.sub(' {2,}', ' ', x) for x in text]
        return ''

    @staticmethod
    def remove_repetion(text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Aplica remoção de repetição de caracteres consecutivos
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        if type(text) == str:
            return re.sub(r'([a-z])\1{2}', '', text)
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series or type(text) == np.array:
            return [re.sub(r'([a-z])\1{2}', '', str(x)) for x in text]
        return ''

    def append_stopwords_list(self, stopwords: list) -> None:
        """
        Adiciona stopwords a lista de stopwords
        Args:
            stopwords (list): lista de stopwords a serem inseridas

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        self.stopwords.extend(stopwords)

    def remove_stopwords(self, text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Remove as stopwords de uma string ou lista de tokens
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        if type(text) == str:
            text = text.split(' ')

            return ' '.join(
                [x for x in text if x not in self.stopwords]
            )
        elif type(text) == list or type(text) == pd.Series or type(text) == np.array:
            docs = list(map(lambda doc: doc.split(' ') if type(doc) == str else doc, text))
            return list(
                map(
                    lambda doc: ' '.join([x for x in doc if x not in self.stopwords]), docs)
            )
        return ''

    def spacy_processing(self, docs: list, n_process: int = -1, lemma: str = False) -> list:
        """
        Aplica processamento do spacy, possível remoção de classes gramaticais definidas no init, bem como remoção
        de entidades
        Args:
            docs (list): lista de documentos
            n_process (int): número de threads para execução em paralelo
            lemma (bool): se verdadeiro, será feito uma lemmatização dos tokens

        Returns:
            pp_docs (list): retorna lista de documentos limpos
        """
        all_docs = self.nlp.pipe(docs, n_process=n_process)
        pp_docs = []
        for doc in tqdm(all_docs):
            pp_doc = [token for token in doc if token.is_ascii]  # remove no ascii
            if self.noadverbs:
                pp_doc = [token for token in pp_doc if token.pos_ != 'ADV']  # remove adverbs
            if self.noadjectives:
                pp_doc = [token for token in pp_doc if token.pos_ != 'ADJ']  # remove adjectives
            if self.noverbs:
                pp_doc = [token for token in pp_doc if token.pos_ != 'VERB']  # remove verbs
            pp_doc = [token for token in pp_doc if not token.is_space]  # remove whitespace

            # Remove Entities
            if self.noentities:
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

            pp_doc = [self.remove_infinitive(token.lemma_).lower() if token.pos_ == 'VERB' else
                      token.lemma_.lower() if lemma else token.lower_ for token in pp_doc]
            pp_docs.append(' '.join(pp_doc))
        return pp_docs

    def remove_n(self, text: Union[str, list, np.array, pd.Series], n: int) -> Union[str, list]:
        """
        Remove tokens com tamanho menor que n
        Args:
            text(str|list): string ou lista de strings a ser limpa
            n (int): número mínimo de caracteres
        Returns:
            text(str|list): string ou lista de strings limpa
        """
        string = False
        if type(text) == str:
            text = text.split(' ')
            string = True
        if type(text) == list or type(text) == pd.Series or type(text) == np.array:
            text = list(map(
                lambda doc: ' '.join([word for word in doc.split(' ') if len(word.strip()) >= n]),
                text
            ))
            text = [doc for doc in text if doc]
            if string:
                return ' '.join(text)
            return text
        return ''

    @staticmethod
    def remove_numbers(text: Union[str, list, np.array, pd.Series], mode: str = 'filter',
                       language: str = 'pt') -> Union[str, list]:
        """
        Remoção ou correção de tokens com números,
        se mode == 'filter' tokens com números serão removidos
        se mode == 'spell' será feita uma correção automática dos tokens que possuem números
        se mode == 'replace' apenas os números do token será removido. Ex: t3ste = tste
        Args:
            text (str|list): string ou lista de strings a ser limpa
            mode (str): método de ajuste para palavras com números
            language (str): idioma para correção automatica se mode == 'spell'

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        mode_function = {
            'filter': lambda x: '' if re.search('[0-9]', x) else x,
            'spell': lambda x: spell.correction(x) if re.search('[0-9]', x) else x,
            'replace': lambda x: re.sub('[0-9]', '', x)
        }
        spell = SpellChecker(language=language)
        string = False
        if type(text) == str:
            text = text.split(' ')
            string = True
        if type(text) == list or type(text) == pd.Series or type(text) == np.array:
            text = list(map(
                lambda doc: ' '.join([mode_function[mode](word) for word in doc.split(' ')]),
                text
            ))
            text = [doc for doc in text if doc]
            if string:
                return ' '.join(text)
            return text
        return ''

    @staticmethod
    def remove_gerund(text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Remove as terminações ndo (gerúndio em português)
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        if type(text) == str:
            return re.sub(r'ndo$', '', text)
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series:
            return [re.sub(r'ndo$', '', x) for x in text if x]
        return ''

    @staticmethod
    def remove_infinitive(text: Union[str, list, np.array, pd.Series]) -> Union[str, list]:
        """
        Remove o caractere 'r' do final de palavras, geralmente este indica verbos no infinitivo
        Args:
            text(str|list): string ou lista de strings a ser limpa

        Returns:
            text(str|list): string ou lista de strings limpa
        """
        if type(text) == str:
            return re.sub(r'r$', '', text)
        elif type(text) == list or type(text) == np.array or type(text) == pd.Series:
            return [re.sub(r'r$', '', x) for x in text if x]
        return ''
