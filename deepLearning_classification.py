import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
import numpy as np
import pandas as pd

#From Pandas Dataframe to Tensorflow Dataset
def from_dframe_to_dset(dframe):
    dataframe = dframe.copy()
    label = dataframe.pop('label')
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), label))
    tf_dataset = tf_dataset.shuffle(buffer_size=len(dataframe))
    return tf_dataset

#Normalisation of quantitative features
def encode_numerical_feature(feature, name, dataset):
    # Create a Keras Normalization Layer for the input feature passed as argument
    normalizer = Normalization()
    # Prepare a Dataset containing only the feature
    feature_dset = dataset.map(lambda x, y: x[name])
    feature_dset = feature_dset.map(lambda x: tf.expand_dims(x, -1))
    # Learn the statistics of the data and normalise the input feature
    normalizer.adapt(feature_dset)
    encoded_feature = normalizer(feature)
    return encoded_feature

#One-hot encoding of qualitative features
def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for the integer indices of the input feature passed as argument
    encoder = CategoryEncoding(output_mode='binary')
    # Prepare a Dataset containing only the feature
    feature_dset = dataset.map(lambda x, y: x[name])
    feature_dset = feature_dset.map(lambda x: tf.expand_dims(x, -1))
    # Learn the space of possible indices and apply one-hot encoding to them
    encoder.adapt(feature_dset)
    encoded_feature = encoder(feature)
    return encoded_feature

#Access the training and test data
training_dataframe = pd.read_csv('training_data.csv')
test_dataframe = pd.read_csv('songs_to_classify.csv')

#Separate the training data into learning and validation dataframes with 85%-15% ratio, respectively
validation_dframe = training_dataframe.sample(frac=0.15,random_state=54)
learning_dframe = training_dataframe.drop(validation_dframe.index)

print('Using %d samples for training and %d for validation' %(len(learning_dframe), len(validation_dframe)))

#Transforming Dataframes into Dataset objects and setting batch size of 32
learning_dset = from_dframe_to_dset(learning_dframe).batch(32)
validation_dset = from_dframe_to_dset(validation_dframe).batch(32)

# Categorical features encoded as integers
key = keras.Input(shape=(1,), name='key', dtype='int64')
mode = keras.Input(shape=(1,), name='mode', dtype='int64')

# Numerical features
acousticness = keras.Input(shape=(1,), name="acousticness")
danceability = keras.Input(shape=(1,), name="danceability")
duration = keras.Input(shape=(1,), name="duration")
energy = keras.Input(shape=(1,), name="energy")
instrumentalness = keras.Input(shape=(1,), name="instrumentalness")
liveness = keras.Input(shape=(1,), name="liveness")
loudness = keras.Input(shape=(1,), name="loudness")
speechiness = keras.Input(shape=(1,), name="speechiness")
tempo = keras.Input(shape=(1,), name="tempo")
time_signature = keras.Input(shape=(1,), name="time_signature")
valence = keras.Input(shape=(1,), name="valence")

all_inputs = [
    key,
    mode,
    acousticness,
    danceability,
    duration,
    energy,
    instrumentalness,
    liveness,
    loudness,
    speechiness,
    tempo,
    time_signature,
    valence,
]

# Perform one-hot encoding to integer qualitative features
key_encoded = encode_integer_categorical_feature(key, "key", learning_dset)
mode_encoded = encode_integer_categorical_feature(mode, "mode", learning_dset)

# Normalise quantitative features
acousticness_encoded = encode_numerical_feature(acousticness, "acousticness", learning_dset)
danceability_encoded = encode_numerical_feature(danceability, "danceability", learning_dset)
duration_encoded = encode_numerical_feature(duration, "duration", learning_dset)
energy_encoded = encode_numerical_feature(energy, "energy", learning_dset)
instrumentalness_encoded = encode_numerical_feature(instrumentalness, "instrumentalness", learning_dset)
liveness_encoded = encode_numerical_feature(liveness, "liveness", learning_dset)
loudness_encoded = encode_numerical_feature(loudness, "loudness", learning_dset)
speechiness_encoded = encode_numerical_feature(speechiness, "speechiness", learning_dset)
tempo_encoded = encode_numerical_feature(tempo, "tempo", learning_dset)
time_signature_encoded = encode_numerical_feature(time_signature, "time_signature", learning_dset)
valence_encoded = encode_numerical_feature(valence, "valence", learning_dset)

all_features = layers.concatenate(
    [
        key_encoded,
        mode_encoded,
        acousticness_encoded,
        danceability_encoded,
        duration_encoded,
        energy_encoded,
        instrumentalness_encoded,
        liveness_encoded,
        loudness_encoded,
        speechiness_encoded,
        tempo_encoded,
        time_signature_encoded,
        valence_encoded,
    ]
)

#Architecting the network
l1 = layers.Dense(13, activation="relu")(all_features)
l1 = layers.Dropout(0.5)(l1)
l2 = layers.Dense(32, activation="relu")(l1)
l2 = layers.Dropout(0.5)(l2)
l3 = layers.Dense(64, activation="relu")(l2)
l4 = layers.Dense(128, activation="relu")(l3)
l5 = layers.Dense(32, activation="relu")(l4)
l5 = layers.Dropout(0.1)(l5)
output = layers.Dense(1, activation="sigmoid")(l5)

#Compiling the model
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

#keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

#Training the model
model.fit(learning_dset, epochs=60, validation_data=validation_dset)

number_predictions = len(training_dataframe)

#Variables to assess the model based on the confusion matrix performance
true_negative = 0
true_positive = 0
false_negative = 0
false_positive = 0
condition_positive = 0
condition_negative = 0

for index in range(number_predictions):
    sample = {
        "key": training_dataframe['key'][index],
        "mode": training_dataframe['mode'][index],
        "acousticness": training_dataframe['acousticness'][index],
        "danceability": training_dataframe['danceability'][index],
        "duration": training_dataframe['duration'][index],
        "energy": training_dataframe['energy'][index],
        "instrumentalness": training_dataframe['instrumentalness'][index],
        "liveness": training_dataframe['liveness'][index],
        "loudness": training_dataframe['loudness'][index],
        "speechiness": training_dataframe['speechiness'][index],
        "tempo": training_dataframe['tempo'][index],
        "time_signature": training_dataframe['time_signature'][index],
        "valence": training_dataframe['valence'][index],
    }
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    prediction = model.predict(input_dict)

    if prediction >= 0.5:
        prediction = 1
    else:
        prediction = 0

    if training_dataframe['label'][index] == 1:
        condition_positive += 1
        if prediction == 1:
            true_positive += 1
        else:
            false_negative += 1
    else:
        condition_negative += 1
        if prediction == 0:
            true_negative += 1
        else:
            false_positive += 1

accuracy = (true_positive + true_negative)/number_predictions
sensitivity = true_positive/condition_positive
specificity = true_negative/condition_negative

print('Size of dataset: ' + str(number_predictions))
print('True Positive: ' + str(true_positive))
print('True Negative: ' + str(true_negative))
print('False Positive: ' + str(false_positive))
print('False Negative: ' + str(false_negative))
print('Accuracy: ' + str(accuracy))
print('Sensitivity: ' + str(sensitivity))
print('Specificity: ' + str(specificity))


