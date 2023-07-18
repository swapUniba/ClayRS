import os
import shutil
import unittest
from unittest import TestCase

import pandas as pd

from clayrs.content_analyzer import Ratings, Rank
from clayrs.evaluation.metrics.plot_metrics import LongTailDistr, PopRatioProfileVsRecs, PopRecsCorrelation
from clayrs.evaluation.eval_pipeline_modules.metric_evaluator import Split

original_ratings = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                 'u2', 'u2', 'u2', 'u2', 'u2',
                 'u3', 'u3', 'u3', 'u3', 'u3', 'u3',
                 'u4', 'u4', 'u4', 'u4', 'u4', 'u4', 'u4',
                 'u5', 'u5', 'u5', 'u5', 'u5',
                 'u6', 'u6'],
     'item_id': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                 'i1', 'i9', 'i10', 'i11', 'i55',
                 'i1', 'i12', 'i13', 'i3', 'i10', 'i14',
                 'i3', 'i10', 'i15', 'i16', 'i9', 'i17', 'i99',
                 'i10', 'i18', 'i19', 'i20', 'i21',
                 'inew_1', 'inew_2'],
     'score': [5, 4, 4, 1, 2, 3, 3, 1,
               4, 5, 1, 1, 1,
               3, 3, 2, 1, 1, 4,
               4, 4, 5, 5, 3, 3, 3,
               3, 3, 2, 2, 1,
               4, 3]})
original_ratings = Ratings.from_dataframe(original_ratings)

train = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1',  # removed last 2
                 'u2', 'u2', 'u2', 'u2',  # removed last 1
                 'u3', 'u3', 'u3', 'u3',  # removed last 2
                 'u4', 'u4', 'u4', 'u4', 'u4',  # removed last 2
                 'u5', 'u5', 'u5', 'u5',  # removed last 1
                 'u6'],  # removed last 1
     'item_id': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6',
                 'i1', 'i9', 'i10', 'i11',
                 'i1', 'i12', 'i13', 'i3',
                 'i3', 'i10', 'i15', 'i16', 'i9',
                 'i10', 'i18', 'i19', 'i20',
                 'inew_1'],
     'score': [5, 4, 4, 1, 2, 3,
               4, 5, 1, 1,
               3, 3, 2, 1,
               4, 4, 5, 5, 3,
               3, 3, 2, 2,
               4]})
train = Ratings.from_dataframe(train)

truth = pd.DataFrame(
    {'user_id': ['u1', 'u1',
                 'u2',
                 'u3', 'u3',
                 'u4', 'u4', 'u4',
                 'u5',
                 'u6'],
     'item_id': ['i7', 'i8',
                 'i55',
                 'i10', 'i14',
                 'i9', 'i17', 'i99',
                 'i21',
                 'inew_2'],
     'score': [3, 1,
               1,
               1, 4,
               3, 3, 3,
               1,
               3]})
truth = Ratings.from_dataframe(truth)

# u6 is missing, just to test DeltaGap in case for some users recs can't be computed
recs = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                 'u2', 'u2', 'u2', 'u2', 'u2',
                 'u3', 'u3', 'u3', 'u3', 'u3',
                 'u4', 'u4', 'u4', 'u4', 'u4',
                 'u5', 'u5', 'u5', 'u5', 'u5'],
     'item_id': ['i7', 'i10', 'i11', 'i12', 'i13',
                 'i11', 'i20', 'i6', 'i3', 'i4',
                 'i4', 'i5', 'i6', 'i7', 'i10',
                 'i9', 'i2', 'i3', 'i1', 'i5',
                 'i2', 'i3', 'i4', 'i5', 'i6'],
     'score': [500, 400, 300, 200, 100,
               400, 300, 200, 100, 50,
               150, 125, 110, 100, 80,
               390, 380, 360, 320, 200,
               250, 150, 190, 100, 50]})
recs = Rank.from_dataframe(recs)

split = Split(recs, truth)


class TestLongTail(TestCase):
    def test_string_error(self):
        # Test invalid string passed
        with self.assertRaises(ValueError):
            LongTailDistr('.', on='invalid')

    def test_perform(self):
        # Save on same folder
        metric = LongTailDistr(on='truth')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_truth'
        self.assertTrue(os.path.isfile('./long_tail_distr_truth.png'))

        # Save on same folder with different format (svg)
        metric = LongTailDistr(on='truth', format='svg')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_truth'
        self.assertTrue(os.path.isfile('./long_tail_distr_truth.svg'))

        # Save on a not existent folder with a specified file_name
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='truth')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_truth'
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_truth.png')))

        # Save on an existent folder with a specified file_name, long_tail based on pred frame
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='pred')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_pred'
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_pred.png')))

        # Save on an existent folder with a filename already existent for the first time
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='pred')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_pred' + ' (1)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_pred (1).png')))

        # Save on an existent folder with a filename already existent for the second time
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='pred')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_pred' + ' (2)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_pred (2).png')))

    def test_overwrite(self):
        # Save on an existent folder with a specified file_name, long_tail based on pred frame
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_pred'
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred.png')))

        # Save on an existent folder with a filename already existent for the first time
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_pred' + ' (1)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred (1).png')))

        # Save on an existent folder with a filename already existent for the second time
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred')
        metric.perform(split)
        # The graph is created with file_name = file_name + '_pred' + ' (2)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred (2).png')))

        # Save on an existent folder with a filename already existent and overwrite it
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred', overwrite=True)
        metric.perform(split)
        # The graph is created with file_name = file_name + '_pred'
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred.png')))
        self.assertFalse(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred (3).png')))

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('./long_tail_distr_truth.png')
        os.remove('./long_tail_distr_truth.svg')
        shutil.rmtree('test_long')


class TestPopRatioProfileVsRecs(TestCase):
    def test_perform(self):
        # Save on same folder
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings)
        metric.perform(split)
        self.assertTrue(os.path.isfile('./pop_ratio_profile_vs_recs.png'))

        # Save on same folder with a different format (svg)
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       format='svg')
        metric.perform(split)
        self.assertTrue(os.path.isfile('./pop_ratio_profile_vs_recs.svg'))

        # Save on a not existent folder with a specified file_name
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       out_dir='test_prof_recs',
                                       file_name='prof_vs_recs_custom_name')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_custom_name.png')))

        # Save on an existent folder with a specified file_name and a different pop_percentage
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       pop_percentage=0.6,
                                       out_dir='test_prof_recs',
                                       file_name='prof_vs_recs_different_percentage')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_different_percentage.png')))

        # Save on an existent folder with a specified file_name and with a different format (svg)
        # Save also the frame used to build the box_plot
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       out_dir='test_prof_recs',
                                       file_name='prof_vs_recs_svg_format',
                                       format='svg',
                                       store_frame=True)
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_svg_format.svg')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_svg_format.csv')))

    def test_perform_more_splits(self):
        # Save on same folder
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=[train, truth],
                                       original_ratings=original_ratings,
                                       out_dir='test_more_splits')
        metric.perform(split)
        self.assertTrue(os.path.isfile('test_more_splits/pop_ratio_profile_vs_recs.png'))

        # simulate 2 split call
        metric.perform(split)
        self.assertTrue(os.path.isfile('test_more_splits/pop_ratio_profile_vs_recs (1).png'))

        # simulate 3 split call, even if we only passed 2 user profiles frame
        # we will generate again the first plot
        metric.perform(split)
        self.assertTrue(os.path.isfile('test_more_splits/pop_ratio_profile_vs_recs (2).png'))
        self.assertEqual(os.path.getsize('test_more_splits/pop_ratio_profile_vs_recs.png'),
                         os.path.getsize('test_more_splits/pop_ratio_profile_vs_recs (2).png'))

    def test_overwrite(self):
        # Save on an existent folder with a specified file_name
        # Save also the frame used to build the box_plot
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       out_dir='test_prof_recs/overwrite',
                                       file_name='prof_vs_recs',
                                       store_frame=True)
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.csv')))

        # Save on an existent folder with an existent file_name for the first time
        # Save also the frame used to build the box_plot with an existent file_name for the first time
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       out_dir='test_prof_recs/overwrite',
                                       file_name='prof_vs_recs',
                                       store_frame=True)
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (1).png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (1).csv')))

        # Save on an existent folder with an existent file_name for the second time
        # Save also the frame used to build the box_plot with an existent file_name for the second time
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       out_dir='test_prof_recs/overwrite',
                                       file_name='prof_vs_recs',
                                       store_frame=True)
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (2).png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (2).csv')))

        # Save on an existent folder with an existent file_name and overwrite it
        # Save also the frame used to build the box_plot with an existent file_name and overwrite it
        metric = PopRatioProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5},
                                       user_profiles=train,
                                       original_ratings=original_ratings,
                                       out_dir='test_prof_recs/overwrite',
                                       file_name='prof_vs_recs',
                                       store_frame=True,
                                       overwrite=True)
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.csv')))

        self.assertFalse(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (3).png')))
        self.assertFalse(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (3).csv')))

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('pop_ratio_profile_vs_recs.png')
        os.remove('pop_ratio_profile_vs_recs.svg')
        shutil.rmtree('test_prof_recs')
        shutil.rmtree('test_more_splits')


class TestPopRecsCorrelation(TestCase):
    def test_perform(self):
        # Save on same folder
        metric = PopRecsCorrelation(original_ratings=original_ratings)
        metric.perform(split)
        self.assertTrue(os.path.isfile('./pop_recs_correlation.png'))
        self.assertTrue(os.path.isfile('./pop_recs_correlation_no_zeros.png'))

        # Save on same folder with a different format (svg)
        metric = PopRecsCorrelation(original_ratings=original_ratings, format='svg')
        metric.perform(split)
        self.assertTrue(os.path.isfile('./pop_recs_correlation.svg'))
        self.assertTrue(os.path.isfile('./pop_recs_correlation_no_zeros.svg'))

        # Save on a not existent folder with a specified file_name
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs',
                                    file_name='pop_recs_custom_name')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name.png')))
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name_no_zeros.png')))

        # Save on an existent folder with a specified file_name and with a different format (svg)
        # Save also the frame used to build the box_plot
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs',
                                    file_name='pop_recs_custom_name', format='svg')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name.svg')))
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name_no_zeros.svg')))

    def test_perform_both(self):
        # Save both graph when it makes sense (eg. they are different)
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/both_yes',
                                    mode='both')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/both_yes', 'pop_recs_correlation.png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/both_yes', 'pop_recs_correlation_no_zeros.png')))

        truth = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u2', 'u2', 'u2'],
            'to_id': ['i2', 'i1', 'i3', 'i5', 'i4'],
            'score': [5, 3, 3.6, 4, 2.2]}
        )
        truth = Ratings.from_dataframe(truth)

        recs = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
            'to_id': ['i1', 'i2', 'inew1', 'inew2', 'i5', 'i4', 'i3'],
            'score': [300, 250, 200, 400, 350, 300, 100]}
        )
        recs = Ratings.from_dataframe(recs)
        # All items in the truth set have been recommended, so there's no 'zero' recommendation

        split_no_zero_present = Split(recs, truth)

        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/both_identical',
                                    mode='both')
        metric.perform(split_no_zero_present)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/both_identical', 'pop_recs_correlation.png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(
            os.path.isfile(os.path.join('test_pop_recs/both_identical', 'pop_recs_correlation_no_zeros.png')))

    def test_perform_w_zeros(self):
        # Save only 'w-zeros' graph
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/w_zeros',
                                    mode='w_zeros')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/w_zeros', 'pop_recs_correlation.png')))
        # If The 'no-zeros' graph was created, its file_name would be: file_name = file_name + '_no_zeros'
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/w_zeros', 'pop_recs_correlation_no_zeros.png')))

    def test_perform_no_zeros(self):
        # Save only 'no-zeros' graph
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/no_zeros',
                                    mode='no_zeros')
        metric.perform(split)
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/no_zeros', 'pop_recs_correlation.png')))
        # The 'no-zeros' graph is created adding with file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/no_zeros', 'pop_recs_correlation_no_zeros.png')))

    def test_overwrite(self):
        # Save both graph when it makes sense (eg. they are different)
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/overwrite',
                                    mode='both')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation.png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros.png')))

        # Save both graph already existent for the first time
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/overwrite',
                                    mode='both')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (1).png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(
            os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros (1).png')))

        # Save both graph already existent for the second time
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/overwrite',
                                    mode='both')
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (2).png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(
            os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros (2).png')))

        # Save both graph already existent but overwrite them
        metric = PopRecsCorrelation(original_ratings=original_ratings,
                                    out_dir='test_pop_recs/overwrite',
                                    mode='both',
                                    overwrite=True)
        metric.perform(split)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation.png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros.png')))
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (3).png')))
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (3).png')))

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('./pop_recs_correlation.png')
        os.remove('./pop_recs_correlation.svg')
        os.remove('./pop_recs_correlation_no_zeros.png')
        os.remove('pop_recs_correlation_no_zeros.svg')
        shutil.rmtree('test_pop_recs')


if __name__ == '__main__':
    unittest.main()
