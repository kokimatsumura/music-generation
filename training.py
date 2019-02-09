import glob, pickle, numpy
from music21 import converter, instrument, note, chord
from keras.models import load_model, Model,Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Flatten, Dropout, Input, LSTM, BatchNormalization, Conv1D, TimeDistributed
from keras.optimizers import Adam, RMSprop

def get_notes_and_durations():
    notes_and_durations = []

    for file in glob.glob("Chopin/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes_and_durations.append((str(element.pitch), element.quarterLength, element.offset))
            elif isinstance(element, chord.Chord):
                notes_and_durations.append((element, element.quarterLength, element.offset))

    return notes_and_durations


def combine_all_notes(notes_and_durations):
    combined_notes = []
    i = 0
    while i < (len(notes_and_durations)-1):
        if notes_and_durations[i][2] == notes_and_durations[i+1][2]:
            step = 2
            combine = chord.Chord([notes_and_durations[i][0], notes_and_durations[i+1][0]])
            quarterLength =  min(float(notes_and_durations[i][1]), float(notes_and_durations[i + 1][1]))
            combined_notes.append(('.'.join(str(n) for n in combine.normalOrder), quarterLength))
        else:
            step = 1
            if isinstance(notes_and_durations[i][0], chord.Chord):
                combined_notes.append(('.'.join(str(n) for n in notes_and_durations[i][0].normalOrder), notes_and_durations[i][1]))
            else:
                combined_notes.append((notes_and_durations[i][0], notes_and_durations[i][1]))
        i = i + step

    with open('experiment/conbined_notes', 'wb') as filepath:
        pickle.dump(combined_notes, filepath)

    return combined_notes


def prepare_sequences(notes_durations, n_vocab):

    sequence_length = 100

    pitchnames = sorted(set(item for item in notes_durations))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []

    for i in range(0, len(notes_durations) - sequence_length, 1):
        sequence_in = notes_durations[i:i + sequence_length]
        sequence_out = notes_durations[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    model = Sequential()

    model.add(Conv1D(256, 5, strides=1, input_shape=(network_input.shape[1], network_input.shape[2]), activation='relu'))
    model.add(Dropout(0.2))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(Conv1D(256, 5, strides=1, activation='relu'))
    model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(256)))
    model.add(Dropout(0.2))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(model, network_input, network_output):

    filepath = "experiment/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=0,save_best_only=True,mode='min')
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=150, batch_size=64, callbacks=callbacks_list)



notes_durations = get_notes_and_durations()
combined_notes = combine_all_notes(notes_durations)
n_vocab = len(set(combined_notes))
network_input, network_output = prepare_sequences(combined_notes, n_vocab)
model = create_network(network_input, n_vocab)
train(model, network_input, network_output)
