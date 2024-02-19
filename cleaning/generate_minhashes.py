"""
This script calculates minhashes for all filtered scrape files found using a recursive
search on "*.minscored".

More explicity, we create a set of 5-grams for each document, and generate document 
level minhashes using 10 hash functions with the excellent datasketch library.

A single file "minhashes.pkl" is created in the scrape directory storing a data
structure in the following format:

[(file_name1, [doc0_minhash, doc1_minhash, ...]), (file_name2, [....]), ....]

Arguments
---------
--scrape_directory (-dir)
    Directory containing the minscored scrapes. You could use the overall work directory if you 
    want as we use glob.glob to search recursively.
--process_count (-procs)
    Number of worker processes in the pool. Defaults to 4.
"""
import argparse
import glob
import os
import sys
import math
import time
from functools import reduce
from operator import add
from contextlib import redirect_stdout
import tqdm
import nltk
from nltk.util import ngrams
from datasketch import MinHash, LeanMinHash
from tqdm_multiprocess import TqdmMultiProcessPool

from utils.utils import timed_pickle_dump, setup_logger
from utils.archiver import Reader

import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

million = math.pow(10, 6)


def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]


# Multiprocessed
def process_file(file_path, log_file, tqdm_func=None, global_tqdm=None):
    setup_logger(log_file)
    reader = Reader()
    minhashes = []
    previous_file_position = 0
    start_time = time.time()
    for text, metadata in reader.read_jsonl2(file_path, get_meta=True):
        n_grams = extract_ngrams(text, 5)
        five_gram_set = set(n_grams)
        minhash = MinHash(num_perm=1000)    # 原代码是10，业界几T是9000，我几百G是1000
        for five_gram in five_gram_set:
            minhash.update(five_gram.encode('utf8'))
        minhashes.append(LeanMinHash(minhash))

        # Update Progress Bar   # YAO: 注释掉
        # current_file_position = reader.fh.tell()
        # global_tqdm.update(current_file_position - previous_file_position)
        # previous_file_position = current_file_position
    duration = round((time.time() - start_time) / 60, 2)
    logger.info(f'{file_path}: {os.path.getsize(file_path) / million:.2f} MB | duration: {duration} mins')
    return file_path, minhashes


def generate_minhashes(directory, process_count, log_file):
    files = sorted(glob.glob(directory, recursive=True))
    total_file_size = reduce(add, map(os.path.getsize, files))
    logger.info(f"Total File Size: {(total_file_size / million):.2f} MB")

    # [(file_name1, [doc0_minhash, doc1_minhash, ...]), (file_name2, [....]), ....]
    with tqdm.tqdm(total=total_file_size, dynamic_ncols=True, unit_scale=1) as progress:
        pool = TqdmMultiProcessPool(process_count)
        tasks = []
        for file_path in files:
            task = (process_file, (file_path, log_file, ))
            tasks.append(task)

        on_done = lambda _: None
        on_error = on_done
        result = pool.map(progress, tasks, on_error, on_done)

    return result


parser_description = 'Generate minhashes for all documents found.'
parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument("-p", "--process_count", type=int, default=96)
parser.add_argument("-d", "--directory", default="")
parser.add_argument("-o", "--output", type=str, default="minhashes.pkl")
parser.add_argument("-l", "--log_file", type=str, default="minhashes.log")



if __name__ == '__main__':
    args = parser.parse_args()
    # if not os.path.isdir(args.directory):
    #     print("directory doesn't exist, exiting.")
    #     sys.exit(0)

    with redirect_stdout(open(os.devnull, "w")):
        nltk.download('punkt')

    log_file = "generate_minhashes.log"
    setup_logger_tqdm(log_file)
    
    logger.info("Generating document level minhashes from 5 gram sets")
    minhashes_by_file = generate_minhashes(args.directory, args.process_count, args.log_file)
    timed_pickle_dump(minhashes_by_file, args.output, "minhashes_by_file")
