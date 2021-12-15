from nlp_reasoning.data_utils import *

def test_parse_data():
    generator = parse_data('data/Sarcasm_Headlines_Dataset_v2.json')
    assert(next(generator) is not None)

class TestCleanText:
    def test_already_clean(self):
        text = 'I like to go to the store'
        assert(clean_text(text) == text)
    
    def test_empty(self):
        text = ''
        assert(clean_text(text) == text)

    def test_none(self):
        text = None
        assert(clean_text(text) == text)

    def test_url(self):
        text = 'Test Text: '
        url = 'https://www.google.com'
        assert(clean_text(text + url) == text)