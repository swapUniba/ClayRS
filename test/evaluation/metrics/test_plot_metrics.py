import os
from unittest import TestCase

import pandas as pd

from orange_cb_recsys.evaluation.exceptions import StringNotSupported
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.plot_metrics import LongTailDistr, PopProfileVsRecs, PopRecsCorrelation

truth = pd.DataFrame(
    {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2',
                 'u3', 'u3', 'u3', 'u3', 'u4', 'u4', 'u4', 'u5', 'u5', 'u5'],
     'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8',
               'i2', 'i4', 'i3', 'i20', 'i3', 'i1', 'i21', 'i3', 'i5', 'i1'],
     'score': [5, 3, 3.6, 4, 2.2, 1, 1.5, 3.2, 3.6, 4, 5, 3.5,
               2.2, 2.8, 4, 5, 4.5, 3.5, 5, 4, 4.5, 3.3]})

recs = pd.DataFrame(
    {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2', 'u2',
                 'u3', 'u3', 'u3', 'u3', 'u4', 'u4', 'u4', 'u5', 'u5', 'u5', 'u5', 'u5'],
     'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8', 'i35',
               'i2', 'i4', 'i3', 'i20', 'i3', 'i1', 'i3', 'i5', 'i1', 'i9', 'i36', 'i6'],
     'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50, 25,
               500, 400, 300, 200, 350, 100, 50, 800, 600, 500, 400, 300]})

split_i21_missing_in_recs = Split(recs, truth)


class TestLongTail(TestCase):
    def test_string_error(self):
        # Test invalid string passed
        with self.assertRaises(StringNotSupported):
            LongTailDistr('.', on='invalid')

    def test_perform(self):
        # Save on same folder
        metric = LongTailDistr(on='truth')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_truth'
        self.assertTrue(os.path.isfile('./long_tail_distr_truth.png'))

        # Save on same folder with different format (svg)
        metric = LongTailDistr(on='truth', format='svg')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_truth'
        self.assertTrue(os.path.isfile('./long_tail_distr_truth.svg'))

        # Save on a not existent folder with a specified file_name
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='truth')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_truth'
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_truth.png')))

        # Save on an existent folder with a specified file_name, long_tail based on pred frame
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='pred')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_pred'
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_pred.png')))

        # Save on an existent folder with a filename already existent for the first time
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='pred')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_pred' + ' (1)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_pred (1).png')))

        # Save on an existent folder with a filename already existent for the second time
        metric = LongTailDistr('test_long', file_name='long_tail_custom_name', on='pred')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_pred' + ' (2)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long', 'long_tail_custom_name_pred (2).png')))

    def test_overwrite(self):
        # Save on an existent folder with a specified file_name, long_tail based on pred frame
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_pred'
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred.png')))

        # Save on an existent folder with a filename already existent for the first time
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_pred' + ' (1)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred (1).png')))

        # Save on an existent folder with a filename already existent for the second time
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred')
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_pred' + ' (2)' (Since it already exists)
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred (2).png')))

        # Save on an existent folder with a filename already existent and overwrite it
        metric = LongTailDistr('test_long/overwrite', file_name='long_tail_custom_name', on='pred', overwrite=True)
        metric.perform(split_i21_missing_in_recs)
        # The graph is created with file_name = file_name + '_pred'
        self.assertTrue(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred.png')))
        self.assertFalse(os.path.isfile(os.path.join('test_long/overwrite', 'long_tail_custom_name_pred (3).png')))


class TestPopProfileVsRecs(TestCase):
    def test_perform(self):
        # Save on same folder
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5})
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile('./pop_ratio_profile_vs_recs.png'))

        # Save on same folder with a different format (svg)
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, format='svg')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile('./pop_ratio_profile_vs_recs.svg'))

        # Save on a not existent folder with a specified file_name
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_prof_recs',
                                  file_name='prof_vs_recs_custom_name')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_custom_name.png')))

        # Save on an existent folder with a specified file_name and a different pop_percentage
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, pop_percentage=0.6, out_dir='test_prof_recs',
                                  file_name='prof_vs_recs_different_percentage')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_different_percentage.png')))

        # Save on an existent folder with a specified file_name and with a different format (svg)
        # Save also the frame used to build the box_plot
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_prof_recs',
                                  file_name='prof_vs_recs_svg_format', format='svg', store_frame=True)
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_svg_format.svg')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs', 'prof_vs_recs_svg_format.csv')))

    def test_overwrite(self):
        # Save on an existent folder with a specified file_name
        # Save also the frame used to build the box_plot
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_prof_recs/overwrite',
                                  file_name='prof_vs_recs', store_frame=True)
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.csv')))

        # Save on an existent folder with an existent file_name for the first time
        # Save also the frame used to build the box_plot with an existent file_name for the first time
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_prof_recs/overwrite',
                                  file_name='prof_vs_recs', store_frame=True)
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (1).png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (1).csv')))

        # Save on an existent folder with an existent file_name for the second time
        # Save also the frame used to build the box_plot with an existent file_name for the second time
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_prof_recs/overwrite',
                                  file_name='prof_vs_recs', store_frame=True)
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (2).png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (2).csv')))

        # Save on an existent folder with an existent file_name and overwrite it
        # Save also the frame used to build the box_plot with an existent file_name and overwrite it
        metric = PopProfileVsRecs(user_groups={'a': 0.5, 'b': 0.5}, out_dir='test_prof_recs/overwrite',
                                  file_name='prof_vs_recs', store_frame=True, overwrite=True)
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.png')))
        self.assertTrue(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs.csv')))

        self.assertFalse(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (3).png')))
        self.assertFalse(os.path.isfile(os.path.join('test_prof_recs/overwrite', 'prof_vs_recs (3).csv')))


class TestPopRecsCorrelation(TestCase):
    def test_perform(self):

        # Save on same folder
        metric = PopRecsCorrelation()
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile('./pop_recs_correlation.png'))
        self.assertTrue(os.path.isfile('./pop_recs_correlation_no_zeros.png'))

        # Save on same folder with a different format (svg)
        metric = PopRecsCorrelation(format='svg')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile('./pop_recs_correlation.svg'))
        self.assertTrue(os.path.isfile('./pop_recs_correlation_no_zeros.svg'))

        # Save on a not existent folder with a specified file_name
        metric = PopRecsCorrelation('test_pop_recs', file_name='pop_recs_custom_name')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name.png')))
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name_no_zeros.png')))

        # Save on an existent folder with a specified file_name and with a different format (svg)
        # Save also the frame used to build the box_plot
        metric = PopRecsCorrelation('test_pop_recs', file_name='pop_recs_custom_name', format='svg')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name.svg')))
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs', 'pop_recs_custom_name_no_zeros.svg')))

    def test_perform_both(self):
        # Save both graph when it makes sense (eg. they are different)
        metric = PopRecsCorrelation('test_pop_recs/both_yes', mode='both')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/both_yes', 'pop_recs_correlation.png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/both_yes', 'pop_recs_correlation_no_zeros.png')))

        truth = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u2', 'u2', 'u2'],
            'to_id': ['i2', 'i1', 'i3', 'i5', 'i4'],
            'score': [5, 3, 3.6, 4, 2.2]}
        )

        recs = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
            'to_id': ['i1', 'i2', 'inew1', 'inew2', 'i5', 'i4', 'i3'],
            'score': [300, 250, 200, 400, 350, 300, 100]}
        )
        # All items in the truth set have been recommended, so there's no 'zero' recommendation

        split_no_zero_present = Split(recs, truth)

        metric = PopRecsCorrelation('test_pop_recs/both_no', mode='both')
        metric.perform(split_no_zero_present)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/both_no', 'pop_recs_correlation.png')))
        # If The 'no-zeros' graph was created, its file_name would be: file_name = file_name + '_no_zeros'
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/both_no', 'pop_recs_correlation_no_zeros.png')))

    def test_perform_w_zeros(self):
        # Save only 'w-zeros' graph
        metric = PopRecsCorrelation('test_pop_recs/w_zeros', mode='w_zeros')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/w_zeros', 'pop_recs_correlation.png')))
        # If The 'no-zeros' graph was created, its file_name would be: file_name = file_name + '_no_zeros'
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/w_zeros', 'pop_recs_correlation_no_zeros.png')))

    def test_perform_no_zeros(self):
        # Save only 'no-zeros' graph
        metric = PopRecsCorrelation('test_pop_recs/no_zeros', mode='no_zeros')
        metric.perform(split_i21_missing_in_recs)
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/no_zeros', 'pop_recs_correlation.png')))
        # The 'no-zeros' graph is created adding with file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/no_zeros', 'pop_recs_correlation_no_zeros.png')))

    def test_overwrite(self):
        # Save both graph when it makes sense (eg. they are different)
        metric = PopRecsCorrelation('test_pop_recs/overwrite', mode='both')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation.png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros.png')))

        # Save both graph already existent for the first time
        metric = PopRecsCorrelation('test_pop_recs/overwrite', mode='both')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (1).png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros (1).png')))

        # Save both graph already existent for the second time
        metric = PopRecsCorrelation('test_pop_recs/overwrite', mode='both')
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (2).png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros (2).png')))

        # Save both graph already existent but overwrite them
        metric = PopRecsCorrelation('test_pop_recs/overwrite', mode='both', overwrite=True)
        metric.perform(split_i21_missing_in_recs)
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation.png')))
        # If The 'no-zeros' is created, its file_name will be: file_name = file_name + '_no_zeros'
        self.assertTrue(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation_no_zeros.png')))
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (3).png')))
        self.assertFalse(os.path.isfile(os.path.join('test_pop_recs/overwrite', 'pop_recs_correlation (3).png')))
