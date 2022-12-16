# import torch
import cld3
import fasttext
import numpy as np


class fasttextLanguage:
    def __init__(self, pretrained_path):
        pretrained_lang_model = pretrained_path
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        text = text if type(text) == str else ''
        predictions = self.model.predict(text, k=1) # returns top 2 matching languages
        return predictions[0][0].replace('__label__', '')


class googleLanguage:
    def __init__(self):
        pass
    
    def predict_lang(self, text):
        return cld3.get_language(text)[0]