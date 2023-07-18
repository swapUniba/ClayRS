import unittest

import ekphrasis.dicts.emoticons

from clayrs.content_analyzer.information_processor.ekphrasis_processor import Ekphrasis


class TestEkphrasis(unittest.TestCase):
    def test_omit_normalize(self):
        # we normalize email and percentages but we omit only emails
        ek = Ekphrasis(omit=['email'], normalize=['email', 'percent'])

        expected = ['this', 'is', 'an', 'email', ':', 'and', 'this', 'is', 'a', 'percent', '<percent>']
        result = ek.process("this is an email: alias@mail.com and this is a percent 23%")

        self.assertEqual(expected, result)

    def test_unpack_contractions(self):
        ek = Ekphrasis(unpack_contractions=True)

        expected = ['i', 'can', 'not', 'do', 'this', 'because', 'i', 'will', 'not', 'and', 'i', 'should', 'not']
        result = ek.process("I can't do this because I won't and I shouldn't")

        self.assertEqual(expected, result)

    def test_unpack_hashtag(self):
        # default corpus, is the english one
        ek = Ekphrasis(unpack_hashtags=True)

        expected = ['next', 'gamedev', 'retrogaming', 'cool', 'photo', 'no', 'unpack']
        result = ek.process("#next #gamedev #retrogaming #coolphoto no unpack")

        self.assertEqual(expected, result)

        # twitter corpus
        ek = Ekphrasis(unpack_hashtags=True, segmenter='twitter')

        expected = ['next', 'game', 'dev', 'retro', 'gaming', 'cool', 'photo', 'no', 'unpack']
        result = ek.process("#next #gamedev #retrogaming #coolphoto no unpack")

        self.assertEqual(expected, result)

    def test_annotate_all_caps_tag(self):
        ek = Ekphrasis(annotate=['allcaps', 'repeated'])

        expected = ['this', 'is', 'good', '!', '<repeated>', 'text', 'and', 'a', '<allcaps>', 'shouted', '</allcaps>',
                    'one']
        result = ek.process("this is good !!! text and a SHOUTED one")

        self.assertEqual(expected, result)

        # change mode for 'allcaps' annotation
        ek = Ekphrasis(annotate=['allcaps', 'repeated'], all_caps_tag='single')

        expected = ['this', 'is', 'good', '!', '<repeated>', 'text', 'and', 'a', 'shouted', '<allcaps>', 'one']
        result = ek.process("this is good !!! text and a SHOUTED one")

        self.assertEqual(expected, result)

    def test_segmentation(self):
        # perform segmentation with the default segmenter corpus ('english')
        ek = Ekphrasis(segmentation=True)

        expected = ['the', 'water', 'cooler', 'exponential', 'backoff', 'no', 'segmentation']
        result = ek.process("thewatercooler exponentialbackoff no segmentation")

        self.assertEqual(expected, result)

        # perform segmentation with 'twitter' as segmenter corpus
        ek = Ekphrasis(segmentation=True, segmenter='twitter')

        expected = ['the', 'watercooler', 'exponential', 'back', 'off', 'no', 'segmentation']
        result = ek.process("thewatercooler exponentialbackoff no segmentation")

        self.assertEqual(expected, result)

        # verify that segmentation segments also hashtags (without removing the #)
        ek = Ekphrasis(segmentation=True, segmenter='twitter')

        expected = ['the', 'watercooler', 'exponential', 'back', 'off', '#', 'cool', 'photo', 'no', 'segmentation']
        result = ek.process("thewatercooler exponentialbackoff #coolphoto no segmentation")

        self.assertEqual(expected, result)

    def test_dicts(self):
        # test additional substitutions with emoticons dict of ekphrasis and a custom dict
        ekphrasis_emoticons_dict = ekphrasis.dicts.emoticons.emoticons
        my_substit_dict = {'you': '<my_sub>'}

        ek = Ekphrasis(dicts=[ekphrasis_emoticons_dict, my_substit_dict])

        expected = ["hello", "<happy>", "<happy>", "how", "are", "<my_sub>", "?", "<sad>", "<sad>", "!"]
        result = ek.process("Hello :-) :) how are you? :( :'(!")

        self.assertEqual(expected, result)

    def test_spell_correction(self):
        # perform spell_correction with the default correction corpus ('english')
        ek = Ekphrasis(spell_correction=True)

        expected = ["this", 'is', 'huuuuge', '.', 'the', 'correct', "way", "of", "doing", "things", "is", 'not', 'the',
                    'following']
        result = ek.process("This is huuuuge. The korrect way of doing tihngs is not the followingt")

        self.assertEqual(expected, result)

        # perform spell_correction with the a different corpus ('twitter')
        ek = Ekphrasis(spell_correction=True, corrector='twitter')
        expected = ["this", 'is', 'huuuuge', 'i', 'the', 'korrect', "way", "of", "doing", "things", "is", 'not', 'the',
                    'following']
        result = ek.process("This is huuuuge. The korrect way of doing tihngs is not the followingt")

        self.assertEqual(expected, result)

        # perform spell_correction also on elongated words
        ek = Ekphrasis(spell_correction=True, spell_correct_elong=True)
        expected = ["this", 'is', 'huge', '.', 'the', 'correct', "way", "of", "doing", "things", "is", 'not', 'the',
                    'following']
        result = ek.process("This is huuuuge. The korrect way of doing tihngs is not the followingt")

        self.assertEqual(expected, result)

    def test_tokenizer(self):
        def my_tokenizer(text):
            return ['test']

        ek = Ekphrasis(tokenizer=my_tokenizer)

        expected = ['test']
        result = ek.process('this is a trial')

        self.assertEqual(expected, result)

    def test_process(self):
        # test all operations
        ek = Ekphrasis(omit=['user'], normalize=['email', 'user'], unpack_contractions=True,
                       unpack_hashtags=True, annotate=['repeated', 'allcaps'], segmenter='twitter', segmentation=True,
                       corrector='english', spell_correction=True, spell_correct_elong=True, all_caps_tag='single')

        expected = ['this', 'should', 'work', 'and', 'this', 'is', 'an', 'email', '<email>', '.',
                    'this', 'can', 'not', 'be', 'and', 'this', 'is', 'as', 'cool', 'test', '.', 'game',
                    'dev', 'must', 'be', 'segmented', 'and', 'this', 'must', 'be', 'corrected', ',', 'even',
                    'elongated', 'words', '.', 'caps', '<allcaps>']

        result = ek.process("This should work @user and this is an email ajeje@mail.com. "
                            "This can't be and this is a #cooltest. gamedev must be segmented and this must "
                            "be korrected, even elongaaaated words. CAPS")

        self.assertEqual(expected, result)
