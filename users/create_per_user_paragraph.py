#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import csv

import ijson.backends.yajl2_cffi as ijson
from six import iteritems
from tqdm import tqdm

COMMENTS_DATASET_FILE_PATH = '../data/comments.json'

USER_COMMENTS_FILE_PATH = 'train_balanced_user.csv'


def main():
    users_comments_dict = collections.defaultdict(list)

    with tqdm(desc="Grouping comments by user", total=12704751) as progress_bar:
        inside_comment = False
        comment_text = None
        comment_username = None

        with open(COMMENTS_DATASET_FILE_PATH, 'rb') as file_:
            # As the JSON file is large (2.5GB) and everything is in one line, is better to read it as a stream,
            # using a SAX-like approach.
            for prefix, type_, value in ijson.parse(file_):
                if inside_comment:
                    if prefix.endswith('.text'):
                        comment_text = value
                    elif prefix.endswith('.author'):
                        comment_username = value
                    elif type_ == 'end_map':  # This assumes there are no nested maps inside the comment maps.
                        if comment_text and comment_username and comment_text != 'nan' \
                                and comment_username != '[deleted]':
                            users_comments_dict[comment_username].append(comment_text)

                        inside_comment = False
                        comment_text = None
                        comment_username = None

                        progress_bar.update()
                elif type_ == 'start_map' and prefix:
                    inside_comment = True

    with open(USER_COMMENTS_FILE_PATH, 'w') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
        writer.writerows((user, " <END> ".join(comments_texts))
                         for user, comments_texts in iteritems(users_comments_dict))


if __name__ == '__main__':
    main()
