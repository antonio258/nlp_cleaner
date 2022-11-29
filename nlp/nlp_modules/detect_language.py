# import torch
import gcld3
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
        self.gdetector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=10000)
    
    def predict_lang(self, text):
        text = text if type(text) == str else ''
        return self.gdetector.FindLanguage(text).language