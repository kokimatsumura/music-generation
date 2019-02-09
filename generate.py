import pickle, numpy
from music21 import instrument, note, stream, chord
from keras.models import load_model, Model,Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Flatten, Dropout, Input, LSTM, BatchNormalization, Conv1D, TimeDistributed
from keras.optimizers import Adam, RMSprop

def prepare_sequences(notes_durations, pitchnames, n_vocab):

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes_durations) - sequence_length, 1):
        sequence_in = notes_durations[i:i + sequence_length]
        sequence_out = notes_durations[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

def create_network(network_input, n_vocab):
    model = Sequential()

    model.add(Conv1D(256, 5, strides=1, input_shape=(network_input.shape[1], network_input.shape[2]), activation='relu'))
    model.add(Dropout(0.1))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(Conv1D(256, 5, strides=1, activation='relu'))
    model.add(Dropout(0.1))

    model.add(TimeDistributed(Dense(256)))
    model.add(Dropout(0.1))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.load_weights('experiment/weights-improvement-67-0.0649-bigger.hdf5')

    return model



def generate_notes_and_durations(model, network_input, pitchnames, n_vocab):

    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(600):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):

    offset = 0
    output_notes = []
    prediction_notes = []
    prediction_durations = []
    for i in range(len(prediction_output)):
        prediction_notes.append(prediction_output[i][0])
        prediction_durations.append(prediction_output[i][1])

    for i in range(len(prediction_notes)):

        if ('.' in prediction_notes[i]) or prediction_notes[i].isdigit():
            notes_in_chord = prediction_notes[i].split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note), quarterLength=float(prediction_durations[i]))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(prediction_notes[i], quarterLength=float(prediction_durations[i]))
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += prediction_durations[i]
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='experiment/created_song.mid')


with open('experiment/combined_notes', 'rb') as filepath:
    notes_durations = pickle.load(filepath)
pitchnames = sorted(set(item for item in notes_durations))

n_vocab = len(set(notes_durations))
network_input, normalized_input = prepare_sequences(notes_durations, pitchnames, n_vocab)
model = create_network2(normalized_input, n_vocab)
prediction_output = generate_notes_and_durations(model, network_input, pitchnames, n_vocab)
create_midi(prediction_output)
