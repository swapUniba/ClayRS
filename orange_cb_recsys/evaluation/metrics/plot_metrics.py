from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from orange_cb_recsys.evaluation.exceptions import StringNotSupported, PercentageError
from orange_cb_recsys.evaluation.metrics.fairness_metrics import GroupFairnessMetric, Dict, popular_items, pop_ratio_by_user
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.utils.const import logger


class PlotMetric(RankingNeededMetric):

    def __init__(self, out_dir: str = '.', file_name: str = None, format: str = 'png', overwrite: bool = False):
        self.__out_dir = out_dir
        self.__file_name = file_name
        self.__format = format
        self.__overwrite = overwrite

    @property
    def file_name(self):
        return self.__file_name

    @property
    def output_directory(self):
        return self.__out_dir

    @property
    def format(self):
        return self.__format

    @property
    def overwrite(self):
        return self.__overwrite


    def get_valid_filename(self, filename: str, format: str):
        filename_try = "{}.{}".format(filename, format)

        if self.overwrite is False:
            i = 0
            while os.path.isfile(os.path.join(self.output_directory, filename_try)):
                i += 1
                filename_try = "{} ({}).{}".format(filename, i, format)

        return filename_try

    def save_figure(self, fig, file_name: str):
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

        file_name = self.get_valid_filename(file_name, self.format)
        fig.savefig(os.path.join(self.output_directory, file_name))
        fig.clf()
        plt.close(fig)


class LongTailDistr(PlotMetric):
    """
    LongTailDistr

    Args:
        file_name (str): name of the file that the metrics will serialize
        out_dir (str): directory in which the file will be serialized
    """

    def __init__(self, out_dir: str = '.', file_name: str = 'long_tail_distr', on='truth', format: str = 'png',
                 overwrite: bool = False):
        valid = {'truth', 'pred'}
        self.__on = on.lower()

        if self.__on not in valid:
            raise StringNotSupported("on={} is not supported! Long Tail can be calculated only on:\n"
                                     "{}".format(on, valid))
        super().__init__(out_dir, file_name, format, overwrite)

    def __str__(self):
        return "LongTailDistr"

    def perform(self, split: Split) -> pd.DataFrame:
        """
        Plot the long tail distribution for the truth frame
        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users
        """
        if self.__on == 'truth':
            frame = split.truth
        else:
            frame = split.pred

        counts_by_item = Counter(list(frame['to_id']))
        ordered_item_count_pairs = counts_by_item.most_common()

        ordered_counts = []
        labels = []
        for item_count_pair in ordered_item_count_pairs:
            labels.append(item_count_pair[0])
            ordered_counts.append(item_count_pair[1])

        x = [i for i in range(len(labels))]
        y = ordered_counts

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set(xlabel='Recommended items', ylabel='Num of recommendations',
               title='Long Tail Distribution - {}'.format(self.__on))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical')

        ax.plot(x, y)

        file_name = self.file_name + '_{}'.format(self.__on)

        self.save_figure(fig, file_name=file_name)

        return pd.DataFrame()


class PopProfileVsRecs(GroupFairnessMetric, PlotMetric):
    """
    PopProfileVsRecs

    Args:
        file_name (str): name of the file that the metrics will serialize
        out_dir (str): directory in which the file will be serialized
        user_groups (dict<str, float>): specify how to divide user in groups, so
        specify for each group specify:
        - name
        - percentage of users
        store_frame (bool): True if you want to store the frame in a csv file, False otherwise
    """

    def __init__(self, user_groups: Dict[str, float], out_dir: str = '.',
                 file_name: str = 'pop_ratio_profile_vs_recs', pop_percentage: float = 0.2,
                 store_frame: bool = False, format: str = 'png', overwrite: bool = False):

        PlotMetric.__init__(self, out_dir, file_name, format, overwrite)
        GroupFairnessMetric.__init__(self, user_groups)

        if not 0 < pop_percentage <= 1:
            raise PercentageError('Incorrect percentage! Valid percentage range: 0 < percentage <= 1')

        self.__pop_percentage = pop_percentage
        self.__user_groups = user_groups
        self.__store_frame = store_frame

    def __str__(self):
        return "PopProfileVsRecs"

    def perform(self, split: Split) -> pd.DataFrame:
        """
        Perform the comparison between the profile popularity and recommendation popularity and build a boxplot

        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            score_frame (pd.DataFrame): contains 'user_group', 'profile_pop_ratio', 'recs_pop_ratio'
        """
        predictions = split.pred
        truth = split.truth

        most_popular_items = popular_items(score_frame=truth, pop_percentage=self.__pop_percentage)
        user_groups = self.split_user_in_groups(score_frame=predictions, groups=self.user_groups,
                                                pop_items=most_popular_items)

        split_result = {'user_group': [], 'profile_pop_ratio': [], 'recs_pop_ratio': []}
        data_to_plot = []
        labels = []
        for group_name in user_groups:
            truth_group = truth.query('from_id in @user_groups[@group_name]', engine='python')
            pred_group = predictions.query('from_id in @user_groups[@group_name]', engine='python')

            profile_pop_ratios_frame = pop_ratio_by_user(truth_group, most_popular_items)
            recs_pop_ratios_frame = pop_ratio_by_user(pred_group, most_popular_items)

            profile_pop_ratios = list(profile_pop_ratios_frame['popularity_ratio'])
            recs_pop_ratios = list(recs_pop_ratios_frame['popularity_ratio'])

            split_result['user_group'].append(group_name)
            split_result['profile_pop_ratio'].append(profile_pop_ratios)
            split_result['recs_pop_ratio'].append(recs_pop_ratios)

            profile_data = np.array(profile_pop_ratios)
            data_to_plot.append(profile_data)
            labels.append('{}_profile'.format(group_name))
            recs_data = np.array(recs_pop_ratios)
            data_to_plot.append(recs_data)
            labels.append('{}_recs'.format(group_name))

        # agg backend is used to create plot as a .png file
        mpl.use('agg')

        # Create a figure instance
        fig = plt.figure()

        # Create an axes instance
        ax = fig.add_subplot()

        ax.set(title='Popularity ratio Profile vs Recs')

        # add patch_artist=True option to ax.boxplot()
        # to get fill color
        bp = ax.boxplot(data_to_plot, patch_artist=True)

        first_color = '#7570b3'
        second_color = '#b2df8a'
        fill_color_pop = '#004e98'
        fill_color_recs = '#ff6700'

        # change outline color, fill color and linewidth of the boxes
        for i, box in enumerate(bp['boxes']):
            # change outline color
            box.set(color=first_color, linewidth=2)
            # change fill color
            if i % 2 == 0:
                box.set(facecolor=fill_color_pop)
            else:
                box.set(facecolor=fill_color_recs)

        # change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color=first_color, linewidth=2)

        # change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color=first_color, linewidth=2)

        # change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color=second_color, linewidth=2)

        # change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

        # Custom x-axis labels
        ax.set_xticklabels(labels)

        # Remove top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        file_name = self.file_name
        self.save_figure(fig, file_name=file_name)

        score_frame = pd.DataFrame(split_result)
        if self.__store_frame:
            file_name = self.get_valid_filename(file_name, 'csv')
            score_frame.to_csv(os.path.join(self.output_directory, file_name), index=False)

        return pd.DataFrame()


class PopRecsCorrelation(PlotMetric):
    """
    PopRecsCorrelation
    Args:
        file_name (str): name of the file that the metrics will serialize
        out_dir (str): directory in which the file will be serialized
    """

    def __init__(self, out_dir: str = '.', file_name: str = 'pop_recs_correlation', mode: str = 'both',
                 format: str = 'png', overwrite: bool = False):
        valid = {'both', 'no_zeros', 'w_zeros'}
        self.__mode = mode.lower()

        if self.__mode not in valid:
            raise StringNotSupported("Mode {} is not supported! Modes available:\n"
                                     "{}".format(mode, valid))

        super().__init__(out_dir, file_name, format, overwrite)

    def __str__(self):
        return "PopRecsCorrelation"

    def build_plot(self, x: list, y: list, title: str):
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set(xlabel='Popularity', ylabel='Recommendation frequency',
               title=title)

        ax.scatter(x, y, marker='o', s=20, c='orange', edgecolors='black',
                   linewidths=0.05)

        return fig

    def build_w_zeros_plot(self, popularities: list, recommendations: list):
        title = 'Popularity-Recommendations Correlation'
        fig = self.build_plot(popularities, recommendations, title)

        file_name = self.file_name

        self.save_figure(fig, file_name)

    def build_no_zeros_plot(self, popularities: list, recommendations: list):
        title = 'Popularity-Recommendations Correlation (No zeros)'
        fig = self.build_plot(popularities, recommendations, title)

        file_name = self.file_name + '_no_zeros'

        self.save_figure(fig, file_name)

    def perform(self, split: Split):
        """
        Calculates the correlation between the two frames and store
        the correlation plot
        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users
        """

        predictions = split.pred
        truth = split.truth

        # Calculating popularity by item
        items = truth[['to_id']].values.flatten()
        pop_by_items = Counter(items)

        # Calculating num of recommendations by item
        pop_by_items = pop_by_items.most_common()
        recs_by_item = Counter(predictions[['to_id']].values.flatten())
        popularities = list()
        recommendations = list()
        popularities_no_zeros = list()
        recommendations_no_zeros = list()

        at_least_one_zero = False
        for item, pop in pop_by_items:
            num_of_recs = recs_by_item[item]

            popularities.append(pop)
            recommendations.append(num_of_recs)

            if num_of_recs != 0:
                popularities_no_zeros.append(pop)
                recommendations_no_zeros.append(num_of_recs)
            else:
                at_least_one_zero = True

        # Both when possible
        if self.__mode == 'both':
            self.build_w_zeros_plot(popularities, recommendations)
            if at_least_one_zero:
                self.build_no_zeros_plot(popularities_no_zeros, recommendations_no_zeros)
            else:
                logger.warning("There's no zero recommendation!\n"
                               "The graph with 'no-zero' won't be created, it would be identical to the 'w-zero' one!")

        elif self.__mode == 'w_zeros':
            self.build_w_zeros_plot(popularities, recommendations)

        elif self.__mode == 'no_zeros':
            self.build_no_zeros_plot(popularities_no_zeros, recommendations_no_zeros)

        return pd.DataFrame()
