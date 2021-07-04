import inspect
import re
import os
import sys
from orange_cb_recsys.utils.custom_logger import getCustomLogger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(THIS_DIR, '../../')
contents_path = os.path.join(root_path, 'contents/')
datasets_path = os.path.join(root_path, 'datasets/')


def progbar(it, prefix='', max_value: int = None, file=sys.stderr, substitute_with_current=False):
    def known_length_progbar(count, total, prefix):
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        if percents <= 100:
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
        else:
            bar = '=' * bar_len

        print('\r%s[%s] %s%s' % (prefix, bar, percents, '%'), end='', file=file)

    def unknown_length_progbar(count, prefix):
        filled_len = int((count % (bar_len / 5)) * bar_len / 5)

        bar = '=' * (filled_len) + '-' * (bar_len - filled_len)

        unk_string = 'Unknown %'
        print('\r%s[%s] %s' % (prefix, bar, unk_string), end='', file=file)

    bar_len = 30
    prefix_orig = prefix

    for count, item in enumerate(it, start=1):
        if substitute_with_current:
            prefix = re.sub('{}', str(item), prefix_orig)

        if max_value:
            known_length_progbar(count, max_value, prefix)
        elif max_value is None and not inspect.isgenerator(it):
            known_length_progbar(count, len(it), prefix)
        else:
            unknown_length_progbar(count, prefix)

        yield item

    file.write("\n")


logger = getCustomLogger('custom_logger')

recsys_logger = getCustomLogger('recsys_logger')

eval_logger = getCustomLogger('eval_logger')

utils_logger = getCustomLogger('utils_logger')
