"""
This script builds a list of all duplicates by file_id & document_id, and then iterates
through all ".minscored" files from the filename lookup, creating a new archive for each 
file in the original containing all documents that were not marked as duplicates during 
the previous step.

So for each original file, a "_final.jsonl.zst" files will be output in the original
directory.

Arguments
------
--batch_directory (-dir)
    Directory containing the "*duplicates.txt" files along with the "file_name_lookup.pkl"
    created during batch slicing. The "_final.jsonl.zst" files will be output in their
    original directories.
"""
import glob
import os
import pickle
import argparse
import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
from utils.archiver import Reader
from utils.utils import Timer
import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)
from singularity_nlp.data.preprocessing import write_jsonl


def file_dedup(file_name, dup_document_ids, tqdm_func, global_tqdm):    # 前2个参数是task传入的，后2个参数是pool.map传入的！
    timer = Timer().start()
    reader = Reader()
    samples = []
    for document_id, line in enumerate(reader.read_jsonl2(file_name, get_all=True)):
        if document_id not in dup_document_ids:
            if 'text' in line:
                line['content'] = line['text']      # 把字段名统一为content
                del line['text']
            samples.append(line)

    final_file_name = file_name.replace(".jsonl", "_dedup.jsonl")
    write_jsonl(samples, final_file_name)
    global_tqdm.update()
    logger.info(f'{final_file_name} created. {timer.stop_string()}')
    return True


def main(batch_directory, process_count):
    file_name_lookup_path = os.path.join(batch_directory, "file_name_lookup.pkl")
    file_name_lookup = pickle.load(open(file_name_lookup_path, "rb"))

    logger.info("Building duplicates dictionary...")
    duplicates_dict = {file_id: set() for file_id in range(len(file_name_lookup))}
    duplicate_files = glob.glob(os.path.join(batch_directory, "*_duplicates.txt"))
    for duplicate_file in duplicate_files:
        with open(duplicate_file, "r") as fh:
            duplicates = fh.read().splitlines()
            for duplicate in duplicates:
                file_id, document_id = tuple(map(int, duplicate.split(" ")))                
                duplicates_dict[file_id].add(document_id)

    logger.info("Deduplicating files...")
    tasks = []
    for file_id, file_name in enumerate(file_name_lookup):
        # file_dedup(file_name, duplicates_dict[file_id])
        task = (file_dedup, (file_name, duplicates_dict[file_id]))
        tasks.append(task)

    pool = TqdmMultiProcessPool(process_count)
    on_error = lambda _: logger.info("error")
    on_done = lambda _: logger.info("done")
    with tqdm.tqdm(total=len(file_name_lookup), dynamic_ncols=True) as progress:
        result = pool.map(progress, tasks, on_error, on_done)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dedupe from provided indexes.')
    parser.add_argument("-p", "--process_count", type=int, default=4)
    parser.add_argument("-d", "--batch_directory", default="")

    logfile_path = "dedupe_from_index.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()
    main(args.batch_directory, args.process_count)
