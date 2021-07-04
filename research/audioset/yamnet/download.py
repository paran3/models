# download.py from https://github.com/DantesLegacy/TensorFlow_AudioSet_Example.git

import contextlib
import os
import csv
import sys
import wave
from label_util import LabelUtil

filename = 'eval_segments.csv'
row_num = 0
path = 'audioset/eval/'
label_util = LabelUtil()
# download_list = ['Howl', 'Horse', 'Pig', 'Goat', 'Sheep', 'Fowl', 'Turkey', 'Mouse', 'Frog', 'Owl', 'Bird', 'Squawk',
#                  'Animal', 'Meow', 'Chicken, rooster', 'Gobble', 'Livestock, farm animals, working animals',
#                  'Wild animals', 'Roar', 'Snake', 'Mosquito', 'Buzz', 'Insect', 'Purr', 'Duck', 'Oink', 'Bleat',
#                  'Goose', 'Moo', 'Mouse', 'Hoot', 'Croak', 'Hiss', 'Yip', 'Bark']
download_list = ['Bleat']
download_label_list = label_util.get_codes(download_list)
last_processed_row = 0


def youtube_download_os_call(id, start_time, idx):
    ret = os.system('ffmpeg -n -ss ' + start_time +
                    ' -i $(youtube-dl -i -w --extract-audio '
                    '--audio-format wav --audio-quality 0 '
                    '--get-url https://www.youtube.com/watch?v=' + id + ')'
                                                                        ' -t 10 ' + path + id + '.wav')

    return ret


def get_wav_file_length(path, idx, id):
    sample = path + id + '.wav'
    with contextlib.closing(wave.open(sample, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)
        print(length)

    return length


def create_error_file(id, idx):
    with open(path + 'error/' + id + '_ERROR.wav', 'a'):
        os.utime(path + 'error/' + id + '_ERROR.wav', None)


def youtube_downloader(id, start_time, idx):
    ret = youtube_download_os_call(id, start_time, idx)

    print('ffmpeg -n -ss ' + start_time +
          ' -i $(youtube-dl -i -w --extract-audio '
          '--audio-format wav --audio-quality 0 '
          '--get-url https://www.youtube.com/watch?v=' + id + ')'
                                                              ' -t 10 ' + path + id + '.wav')
    return ret


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(path+'error/'):
        os.makedirs(path + 'error/')


check_dir(path)

with open(filename, newline='') as f:
    reader = csv.reader(f, quotechar='"', delimiter=',',
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
    try:
        for row in reader:
            if row_num <= last_processed_row:
                row_num += 1
                continue
            # Skip the 3 line header
            if row_num >= 3:
                labels = label_util.get_code_names(list(set(row[3].split(',')) & set(download_label_list)))
                if len(labels) > 0:
                    print('row : ' + str(row))
                    print('rownum : ' + str(row_num))
                    ret = youtube_downloader(row[0], str(float(row[1].lstrip())),
                                             str(row_num - 3))
                    # If there was an error downloading the file
                    # This sometimes happens if videos are blocked or taken down
                    if ret != 0:
                        create_error_file(row[0], str(row_num - 3))

            row_num += 1

    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
