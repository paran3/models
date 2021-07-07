from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from tflite_model_maker import audio_classifier
from scipy.io import wavfile

SAVED_MODEL_PATH = 'test_models/saved_model'
TFLITE_FILE_PATH = 'test_models/test_model.tflite'

data_dir = './audioset/small_birds_dataset/'
file_name = 'audioset/small_birds_dataset/test/azaspi1/XC616298.wav'

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

# loaded_model = tf.saved_model.load(SAVED_MODEL_PATH)
loaded_model = tf.keras.models.load_model(SAVED_MODEL_PATH)

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

test_data = audio_classifier.DataLoader.from_folder(
    spec, data_dir + 'test', cache=True)

sample_rate, audio_data = wavfile.read(file_name, 'rb')

audio_data = np.array(audio_data) / tf.int16.max
input_size = loaded_model.input_shape[1]

splitted_audio_data = tf.signal.frame(audio_data, input_size, input_size, pad_end=True, pad_value=0)

results = []
print('Result of the window ith:  your model class -> score,  (spec class -> score)')
for i, data in enumerate(splitted_audio_data):
    yamnet_output, inference = loaded_model(data)
    results.append(inference[0].numpy())
    result_index = tf.argmax(inference[0])
    spec_result_index = tf.argmax(yamnet_output[0])
    t = spec._yamnet_labels()[spec_result_index]
    result_str = f'Result of the window {i}: ' \
                 f'\t{test_data.index_to_label[result_index]} -> {inference[0][result_index].numpy():.3f}, ' \
                 f'\t({spec._yamnet_labels()[spec_result_index]} -> {yamnet_output[0][spec_result_index]:.3f})'
    print(result_str)

results_np = np.array(results)
mean_results = results_np.mean(axis=0)
result_index = mean_results.argmax()
print(f'Mean result: {test_data.index_to_label[result_index]} -> {mean_results[result_index]}')
