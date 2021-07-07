import matplotlib.pyplot as plt
import seaborn as sns
from tflite_model_maker import audio_classifier
import tflite_model_maker as mm

spec = audio_classifier.YamNetSpec(
    keep_yamnet_and_custom_heads=True,
    frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
    frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

data_dir = './audioset/small_birds_dataset/'

train_data = audio_classifier.DataLoader.from_folder(
    spec, data_dir + 'train', cache=True)
train_data, validation_data = train_data.split(0.8)
test_data = audio_classifier.DataLoader.from_folder(
    spec, data_dir + 'test', cache=True)

batch_size = 128
epochs = 500

print('Training the model')
model = audio_classifier.create(
    train_data,
    spec,
    validation_data,
    batch_size=batch_size,
    epochs=epochs)

print('Evaluating the model')
model.evaluate(test_data)


def show_confusion_matrix(confusion, test_labels):
    """Compute confusion matrix and normalize."""
    confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
    axis_labels = test_labels
    ax = sns.heatmap(
        confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
        cmap='Blues', annot=True, fmt='.2f', square=True)
    plt.interactive(False)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


test_data = audio_classifier.DataLoader.from_folder(
    spec, data_dir + 'test', cache=True)

confusion_matrix = model.confusion_matrix(test_data)
show_confusion_matrix(confusion_matrix.numpy(), test_data.index_to_label)

models_path = './test_models'
print(f'Exporing the TFLite model to {models_path}')

# model.export(models_path, tflite_filename='my_birds_model.tflite')
model.export(models_path, export_format=[mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])
