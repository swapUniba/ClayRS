from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os

from orange_cb_recsys.evaluation.metrics.fairness_metrics import GroupFairnessMetric, Dict, popular_items,\
    pop_ratio_by_user
from orange_cb_recsys.evaluation.metrics.metrics import Metric
from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.utils.const import logger


class PlotMetric(Metric):
    """
    A plot metric is a metric which generates a plot and saves it to the directory specified

    The plot file will be saved as *out_dir/file_name.format*

    Since multiple split could be evaluated at once, the *overwrite* parameter comes into play:
    if is set to False, file with the same name will be saved as *file_name (1).format*, *file_name (2).format*, etc.
    so that for every split a plot is generated without overwriting any file previously generated

    Args:
        out_dir (str): Directory where the plot will be saved. Default is '.', meaning that the plot will be saved
            in the same directory where the python script it's being executed
        file_name (str): Name of the plot file. Every plot metric as a default file name
        format (str): Format of the plot file. Could be 'jpg', 'svg', 'png'. Default is 'png'
        overwrite (bool): parameter which specifies if the plot saved must overwrite any file that as the same name
            ('file_name.format'). Default is False
    """

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

    def get_valid_filename(self, filename: str, format: str) -> str:
        """
        Concrete method which gets a valid *filename.format* based on the overwrite parameter

        If overwrite=False and there are existent file named as *filename.format* in the output directory, the method
        checks if there are also files named as *filename (1).format*, *filename (2).format*, *filename (3).format*.
        It stops at the first non existent *filename (x).format* in the output directory specified in the constructor.
        Args:
            filename (str): Name of the file to save
            format (str): Format of the file to save

        Returns:
            A valid 'filename.format' string based on the overwrite parameter
        """
        filename_try = f"{filename}.{format}"

        if self.overwrite is False:
            i = 0
            while os.path.isfile(os.path.join(self.output_directory, filename_try)):
                i += 1
                filename_try = "{} ({}).{}".format(filename, i, format)

        return filename_try

    def save_figure(self, fig, file_name: str):
        """
        Concrete method which given the figure to save and its file name, it saves the figure in the output directory
        and with the format specified in the constructor

        Args:
            fig: figure to save
            file_name (str): name of the file to save
        """
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

        file_name = self.get_valid_filename(file_name, self.format)
        fig.savefig(os.path.join(self.output_directory, file_name))
        fig.clf()
        plt.close(fig)


class LongTailDistr(PlotMetric):
    """
    This metric generates the Long Tail Distribution plot and saves it in the output directory with the file name
    specified. The plot can be generated both for the *truth set* or the *predictions set* (based on
    the *on* parameter):

    - **on = 'truth'**: in this case the long tail distribution is useful to see which are the most popular items (the
      most rated ones)

    - **on = 'pred'**: in this case the long tail distribution is useful to see which are the most recommended items

    The plot file will be saved as *out_dir/file_name.format*

    Since multiple split could be evaluated at once, the *overwrite* parameter comes into play:
    if is set to False, file with the same name will be saved as *file_name (1).format*, *file_name (2).format*, etc.
    so that for every split a plot is generated without overwriting any file previously generated

    Args:
        out_dir (str): Directory where the plot will be saved. Default is '.', meaning that the plot will be saved
            in the same directory where the python script it's being executed
        file_name (str): Name of the plot file. Default is 'long_tail_distr'
        on (str): Set on which the Long Tail Distribution plot will be generated. Values accepted are 'truth' or 'pred'
        format (str): Format of the plot file. Could be 'jpg', 'svg', 'png'. Default is 'png'
        overwrite (bool): parameter which specifies if the plot saved must overwrite any file that as the same name
            ('file_name.format'). Default is False

    Raises:
        ValueError: exception raised when a invalid value for the 'on' parameter is specified
    """

    def __init__(self, out_dir: str = '.', file_name: str = 'long_tail_distr', on='truth', format: str = 'png',
                 overwrite: bool = False):
        valid = {'truth', 'pred'}
        self.__on = on.lower()

        if self.__on not in valid:
            raise ValueError("on={} is not supported! Long Tail can be calculated only on:\n"
                                     "{}".format(on, valid))
        super().__init__(out_dir, file_name, format, overwrite)

    def __str__(self):
        return "LongTailDistr"

    def perform(self, split: Split) -> pd.DataFrame:
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

        return pd.DataFrame(columns=['from_id', 'to_id', 'score'])


class PopProfileVsRecs(GroupFairnessMetric, PlotMetric):
    """
    This metric generates a plot where users are splitted into groups and, for every group, a boxplot comparing
    profile popularity and recommendations popularity is drawn

    Users are splitted into groups based on the *user_groups* parameter, which contains names of the groups as keys,
    and percentage of how many user must contain a group as values. For example::

        user_groups = {'popular_users': 0.3, 'medium_popular_users': 0.2, 'low_popular_users': 0.5}

    Every user will be inserted in a group based on how many popular items the user has rated (in relation to the
    percentage of users we specified as value in the dictionary):
    users with many popular items will be inserted into the first group, users with niche items rated will be inserted
    into one of the last groups

    You could also specify how many *most popular items* must be considered with the 'pop_percentage' parameter. By
    default is set to 0.2 which means that the top 20% items are considered as most popular

    The plot file will be saved as *out_dir/file_name.format*

    Since multiple split could be evaluated at once, the *overwrite* parameter comes into play:
    if is set to False, file with the same name will be saved as *file_name (1).format*, *file_name (2).format*, etc.
    so that for every split a plot is generated without overwriting any file previously generated

    Thanks to the 'store_frame' parameter it's also possible to store a csv containing the calculations done in order
    to build every boxplot. Will be saved in the same directory and with the same file name as the plot itself (but
    with the .csv format):

    The csv will be saved as *out_dir/file_name.csv*


    Args:
        user_groups (Dict<str, float>): Dict containing group names as keys and percentage of users as value, used to
            split users in groups. Users with more popular items rated are grouped into the first group, users with
            slightly less popular items rated are grouped into the second one, etc.
        out_dir (str): Directory where the plot will be saved. Default is '.', meaning that the plot will be saved
            in the same directory where the python script it's being executed
        file_name (str): Name of the plot file. Default is 'pop_ratio_profile_vs_recs'
        pop_percentage (float): How many (in percentage) 'most popular items' must be considered. Default is 0.2
        store_frame (bool): True if you want to store calculations done in order to build every boxplot in a csv file,
            False otherwise. Default is set to False
        format (str): Format of the plot file. Could be 'jpg', 'svg', 'png'. Default is 'png'
        overwrite (bool): parameter which specifies if the plot saved must overwrite any file that as the same name
            ('file_name.format'). Default is False
    """

    def __init__(self, user_groups: Dict[str, float], out_dir: str = '.',
                 file_name: str = 'pop_ratio_profile_vs_recs', pop_percentage: float = 0.2,
                 store_frame: bool = False, format: str = 'png', overwrite: bool = False):

        PlotMetric.__init__(self, out_dir, file_name, format, overwrite)
        GroupFairnessMetric.__init__(self, user_groups)

        if not 0 < pop_percentage <= 1:
            raise ValueError('Incorrect percentage! Valid percentage range: 0 < percentage <= 1')

        self.__pop_percentage = pop_percentage
        self.__user_groups = user_groups
        self.__store_frame = store_frame

    def __str__(self):
        return "PopProfileVsRecs"

    def perform(self, split: Split) -> pd.DataFrame:
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

            profile_data = profile_pop_ratios
            data_to_plot.append(profile_data)
            labels.append('{}_profile'.format(group_name))
            recs_data = recs_pop_ratios
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

        return pd.DataFrame(columns=['from_id', 'to_id', 'score'])


class PopRecsCorrelation(PlotMetric):
    """
    This metric generates a plot which has as the X-axis the popularity and as Y-axis number of recommendations, so
    that it can be easily seen the correlation between popular (niche) items and how many times are being recommended
    by the recsys

    The plot file will be saved as *out_dir/file_name.format*

    Since multiple split could be evaluated at once, the *overwrite* parameter comes into play:
    if is set to False, file with the same name will be saved as *file_name (1).format*, *file_name (2).format*, etc.
    so that for every split a plot is generated without overwriting any file previously generated

    There exists cases in which some items are not recommended even once, so in the graph could appear
    **zero recommendations**. One could change this behaviour thanks to the 'mode' parameter:

    - **mode='both'**: two graphs will be created, the first one containing eventual *zero recommendations*, the
      second one where *zero recommendations* are excluded. This additional graph will be stored as
      *out_dir/file_name_no_zeros.format* (the string '_no_zeros' will be added to the file_name chosen automatically)

    - **mode='w_zeros'**: only a graph containing eventual *zero recommendations* will be created

    - **mode='no_zeros'**: only a graph excluding eventual *zero recommendations* will be created. The graph will be
      saved as *out_dir/file_name_no_zeros.format* (the string '_no_zeros' will be added to the file_name chosen
      automatically)


    Args:
        out_dir (str): Directory where the plot will be saved. Default is '.', meaning that the plot will be saved
            in the same directory where the python script it's being executed
        file_name (str): Name of the plot file. Default is 'pop_recs_correlation'
        mode (str): Parameter which dictates which graph must be created. By default is 'both', so the graph with
            eventual zero recommendations as well as the graph excluding eventual zero recommendations will be created.
            Check the class documentation for more
        format (str): Format of the plot file. Could be 'jpg', 'svg', 'png'. Default is 'png'
        overwrite (bool): parameter which specifies if the plot saved must overwrite any file that as the same name
            ('file_name.format'). Default is False
    """

    def __init__(self, out_dir: str = '.', file_name: str = 'pop_recs_correlation', mode: str = 'both',
                 format: str = 'png', overwrite: bool = False):
        valid = {'both', 'no_zeros', 'w_zeros'}
        self.__mode = mode.lower()

        if self.__mode not in valid:
            raise ValueError("Mode {} is not supported! Modes available:\n"
                                     "{}".format(mode, valid))

        super().__init__(out_dir, file_name, format, overwrite)

    def __str__(self):
        return "PopRecsCorrelation"

    def build_plot(self, x: list, y: list, title: str):
        """
        Method which builds a matplotlib plot given x-axis values, y-axis values and the title of the plot.
        X-axis label and Y-axis label are hard-coded as 'Popularity' and 'Recommendation frequency' respectively.

        Args:
            x (list): List containing x-axis values
            y (list): List containing y-axis values
            title (str): title of the plot

        Returns:
            The matplotlib figure
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set(xlabel='Popularity', ylabel='Recommendation frequency',
               title=title)

        ax.scatter(x, y, marker='o', s=20, c='orange', edgecolors='black',
                   linewidths=0.05)

        return fig

    def build_w_zeros_plot(self, popularities: list, recommendations: list):
        """
        Method which builds and saves the plot containing eventual *zero recommendations*
        It saves the plot as *out_dir/filename.format*, according to their value passed in the constructor

        Args:
            popularities (list): x-axis values representing popularity of every item
            recommendations (list): y-axis values representing number of times every item has been recommended
        """
        title = 'Popularity-Recommendations Correlation'
        fig = self.build_plot(popularities, recommendations, title)

        file_name = self.file_name

        self.save_figure(fig, file_name)

    def build_no_zeros_plot(self, popularities: list, recommendations: list):
        """
        Method which builds and saves the plot **excluding** eventual *zero recommendations*
        It saves the plot as *out_dir/filename_no_zeros.format*, according to their value passed in the constructor.
        Note that the '_no_zeros' string is automatically added to the file_name chosen

        Args:
            popularities (list): x-axis values representing popularity of every item
            recommendations (list): y-axis values representing number of times every item has been recommended
        """
        title = 'Popularity-Recommendations Correlation (No zeros)'
        fig = self.build_plot(popularities, recommendations, title)

        file_name = self.file_name + '_no_zeros'

        self.save_figure(fig, file_name)

    def perform(self, split: Split):
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
                self.build_no_zeros_plot(popularities, recommendations)
                logger.warning("There's no zero recommendation!\n"
                               "The graph with 'no-zero' is identical to the 'w-zero' one!")

        elif self.__mode == 'w_zeros':
            self.build_w_zeros_plot(popularities, recommendations)

        elif self.__mode == 'no_zeros':
            self.build_no_zeros_plot(popularities_no_zeros, recommendations_no_zeros)

        return pd.DataFrame(columns=['from_id', 'to_id', 'score'])
