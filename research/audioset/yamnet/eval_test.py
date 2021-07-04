from __future__ import division, print_function

import csv
import os
import random
from datetime import datetime

from label_util import LabelUtil
import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import params as yamnet_params
import yamnet as yamnet_model

filename = 'eval_segments.csv'
path = 'audioset/eval/'
result_path = 'result/'
rate_count = dict()

label_list = ['Howl', 'Horse', 'Pig', 'Goat', 'Sheep', 'Fowl', 'Turkey', 'Mouse', 'Frog', 'Owl', 'Bird', 'Squawk',
              'Animal', 'Meow', 'Chicken, rooster', 'Gobble', 'Wild animals', 'Roar', 'Snake', 'Mosquito', 'Buzz',
              'Insect', 'Purr', 'Duck', 'Oink', 'Bleat', 'Goose', 'Moo', 'Mouse', 'Hoot', 'Croak', 'Hiss', 'Yip', 'Bark']


def main(label_names, video_list_size=5):
    label_util = LabelUtil()
    labels = label_util.get_codes(label_names)

    label_logger, video_logger = get_log_writer()

    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

    for label in labels:
        if label_util.get_code_name(label) is not None:
            label_name = label_util.get_code_name(label)
            print('label : ' + label_name)
            videos = get_videos(label, video_list_size)
            count = 0
            total = len(videos)
            exist_total = 0

            for video in videos:
                video_id = video[0]
                video_labels = video[1]
                video_path = path + video_id + '.wav'
                print(video_path)

                if os.path.isfile(video_path):
                    # Decode the WAV file.
                    wav_data, sr = sf.read(video_path, dtype=np.int16)
                    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
                    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
                    waveform = waveform.astype('float32')

                    # Convert to mono and the sample rate expected by YAMNet.
                    if len(waveform.shape) > 1:
                        waveform = np.mean(waveform, axis=1)
                    if sr != params.sample_rate:
                        waveform = resampy.resample(waveform, sr, params.sample_rate)

                    # Predict YAMNet classes.
                    scores, embeddings, spectrogram = yamnet(waveform)
                    # Scores is a matrix of (time_frames, num_classes) classifier scores.
                    # Average them along time to get an overall classifier output for the clip.
                    prediction = np.mean(scores, axis=0)
                    # Report the highest-scoring classes and their scores.
                    top5_i = np.argsort(prediction)[::-1][:5]
                    print(video_id, ':\n' +
                          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                                    for i in top5_i))

                    exist_total += 1
                    top5_index = []
                    for i in top5_i:
                        if label_name == yamnet_classes[i]:
                            count += 1
                        top5_index.append(i)

                    video_logger.writerow([video_id,
                                           label_util.get_code_names(video_labels.split(',')),
                                           yamnet_classes[top5_index[0]], prediction[top5_index[0]],
                                           yamnet_classes[top5_index[1]], prediction[top5_index[1]],
                                           yamnet_classes[top5_index[2]], prediction[top5_index[2]],
                                           yamnet_classes[top5_index[3]], prediction[top5_index[3]],
                                           yamnet_classes[top5_index[4]], prediction[top5_index[4]],
                                           ])

            rate_count[label_name] = count

            rate = 0.000
            if exist_total > 0:
                rate = round(count / exist_total, 3)
            label_logger.writerow(
                [label_util.get_code_name(label), total, exist_total, count, rate])

    print(rate_count)


def get_log_writer():
    t = datetime.now()
    time_string = t.strftime('%Y-%m-%d-%H:%M:%S')
    check_dir(result_path)

    label_fieldnames = ['label', 'total', 'exist_total', 'count', 'rate']
    video_fieldnames = ['video_id', 'labels', 'top1', 'top1_rate', 'top2', 'top2_rate',
                        'top3', 'top3_rate', 'top4', 'top4_rate', 'top5', 'top5_rate']

    label_logger = get_csv_writer(result_path + 'label_result_' + time_string + '.csv', label_fieldnames)
    video_logger = get_csv_writer(result_path + 'video_result_' + time_string + '.csv', video_fieldnames)

    return label_logger, video_logger


def get_csv_writer(file_name, filednames):
    csvfile = open(file_name, 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(filednames)
    return writer


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_videos(label, list_size):
    videos = dict()
    row_num = 0
    with open(filename, newline='') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',',
                            quoting=csv.QUOTE_ALL, skipinitialspace=True)
        try:
            for row in reader:
                if row_num > 2:
                    if label in list(row[3].split(',')):
                        videos[row[0]] = row[3]
                row_num += 1

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

    return random.choices(population=list(videos.items()), k=int(list_size))


if __name__ == '__main__':
    if len(sys.argv[2:]) > 0:
        main(sys.argv[2:], sys.argv[1])

    else:
        main(label_list, sys.argv[1])
