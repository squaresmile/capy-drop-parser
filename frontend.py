import argparse
import pathlib
import sys
import multiprocessing
import signal
import json
import logging
import imghdr
import time
import uuid

import fgo_mat_counter

TERMINATE = False
SCRIPT_BASE_PATH = pathlib.Path(sys.argv[0]).parent

def signal_handling(signum,frame):
    global TERMINATE
    TERMINATE = True
    print(f'Notice: app may take up to polling frequency time before exting.')

signal.signal(signal.SIGINT,signal_handling)

def get_node_directories():
    node_dirs = []
    for i in (SCRIPT_BASE_PATH / 'input').iterdir():
        if i.is_dir():
            node_dirs.append(i)

    return node_dirs


def check_dirs_for_new_images(dir_list):
    work_items = []
    for dir in dir_list:
        for f in pathlib.Path(dir).iterdir():
            if f.is_file() and imghdr.what(f) is not None:
                new_path = SCRIPT_BASE_PATH / 'output' / f.parts[-2] / (str(uuid.uuid4()) + f.suffix)
                f.replace(new_path)
                work_items.append(new_path)

    return work_items

def normalize_drop_locations(drops):
    y_locations = []
    for drop in drops:
        y_locations.append(drop['y'])

    y_locations.sort()
    for drop in drops:
        drop['y'] = int(y_locations.index(drop['y']) / 7)

    num_rows = int((len(y_locations) / 7) + 1)
    for row in range(0, num_rows):
        x_locations = []
        row_drops = list(filter(lambda drop: drop['y'] == row, drops))
        for drop in row_drops:
            x_locations.append(drop['x'])

        x_locations.sort()
        for drop in row_drops:
            drop['x'] = int(x_locations.index(drop['x']) % 7)

    return drops

def convert_score_to_float_for_json(drops):
    for drop in drops:
        drop['score'] = float(drop['score'])

    return drops

def create_result_json_file(result):
    image_path = pathlib.Path(result['image_path'])
    json_file_path = image_path.parent / (image_path.stem + '.json')
    with open(json_file_path, 'w') as fp:
        json.dump(result, fp)


def handle_success(result):
    result['drops'] = normalize_drop_locations(result['drops'])
    result['drops'] = convert_score_to_float_for_json(result['drops'])
    create_result_json_file(result)


def handle_failure(result):
    raise Exception(f'Failure handler was called, this should not happen: {result}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-j', '--num_processes', required=False, default=1, help='Number of processes to allocate in the process pool')
    arg_parser.add_argument('-p', '--polling_frequency', required=False, default=60, help='how often to check for new images in seconds')
    args = arg_parser.parse_args()

    logging.basicConfig(format='%(relativeCreated)6d %(threadName)s %(message)s', level=logging.ERROR,
                        filename='logfile.log',
                        filemode='w')

    node_directory_list = get_node_directories()
    process_pool = multiprocessing.Pool(processes=int(args.num_processes))

    while not TERMINATE:
        for wi in check_dirs_for_new_images(node_directory_list):
            with open(SCRIPT_BASE_PATH / 'input' / wi.parts[-2] / 'settings.json') as fp: settings = json.load(fp)
            process_pool.apply_async(fgo_mat_counter.analyze_image_for_discord,
                                     [wi, settings, SCRIPT_BASE_PATH / 'input'/ wi.parts[-2] / 'files'],
                                     {}, handle_success, handle_failure)
            # handle_success(fgo_mat_counter.analyze_image_for_discord(wi, settings, SCRIPT_BASE_PATH / 'input'/ wi.parts[-2] / 'files'))

        time.sleep(int(args.polling_frequency))


    print("shutting down....")
    process_pool.close()
    process_pool.join()
    print('done')

