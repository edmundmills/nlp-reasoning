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

class TestTrimTrailingSentence:
    def test_empty_string(self):
        text = ''
        assert(trim_trailing_sentence(text) == text)

    def test_no_trail(self):
        text = 'A full sentence.'
        assert(trim_trailing_sentence(text) == text)
    
    def test_two_sentences(self):
        text = 'A full sentence. Another.'
        assert(trim_trailing_sentence(text) == text)
    
    def test_trail(self):
        text = 'A full sentence.'
        trail = ' A trail'
        assert(trim_trailing_sentence(text + trail) == text)