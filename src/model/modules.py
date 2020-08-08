import torch
import numpy as np
import re


# Modules for harmonic filters
def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # lowest note
    low_midi = note_to_midi('C1')

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i+1)))

    return harmonic_hz, int(level)

def hz_to_midi(hz):
    return 12 * (np.log2(hz) - np.log2(440.0)) + 69

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0)/12.0))

def note_to_midi(note):
    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    acc_map = {"#": 1,"": 0,"b": -1,"!": -1,"♯": 1,"𝄪": 2,"♭": -1,"𝄫": -2,"♮": 0}

    match = re.match(
        r"^(?P<note>[A-Ga-g])"
        r"(?P<accidental>[#♯𝄪b!♭𝄫♮]*)"
        r"(?P<octave>[+-]?\d+)?"
        r"(?P<cents>[+-]\d+)?$",
        note)
    pitch = match.group("note").upper()
    offset = np.sum([acc_map[o] for o in match.group("accidental")])
    octave = match.group("octave")
    cents = match.group("cents")

    if not octave:
        octave = 0
    else:
        octave = int(octave)
    if not cents:
        cents = 0
    else:
        cents = int(cents) * 1e-2

    note_value = 12 * (octave + 1) + pitch_map[pitch] + offset + cents
    return note_value

def midi_to_note(midi, octave=True, cents=False, key="C:maj", unicode=True):
    note_map = key_to_notes(key=key, unicode=unicode)
    note_num = int(np.round(midi))
    note_cents = int(100 * np.around(midi - note_num, 2))
    note = note_map[note_num % 12]

    if octave:
        note = "{:s}{:0d}".format(note, int(note_num / 12) - 1)
    if cents:
        note = "{:s}{:+02d}".format(note, note_cents)

    return note

def hz_to_note(hz):
    return midi_to_note(hz_to_midi(hz))

def key_to_notes(key, unicode=True):
    # Parse the key signature
    match = re.match(
        r"^(?P<tonic>[A-Ga-g])"
        r"(?P<accidental>[#♯b!♭]?)"
        r":(?P<scale>(maj|min)(or)?)$",
        key,
    )
    if not match:
        raise ParameterError("Improper key format: {:s}".format(key))
    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    acc_map = {"#": 1, "": 0, "b": -1, "!": -1, "♯": 1, "♭": -1}

    tonic = match.group("tonic").upper()
    accidental = match.group("accidental")
    offset = acc_map[accidental]

    scale = match.group("scale")[:3].lower()
    major = scale == "maj"

    if major:
        tonic_number = ((pitch_map[tonic] + offset) * 7) % 12
    else:
        tonic_number = ((pitch_map[tonic] + offset) * 7 + 9) % 12

    if offset < 0:
        use_sharps = False

    elif offset > 0:
        use_sharps = True

    elif 0 <= tonic_number < 6:
        use_sharps = True

    elif tonic_number > 6:
        use_sharps = False

    # Basic note sequences for simple keys
    notes_sharp = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
    notes_flat = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]

    sharp_corrections = [
        (5, "E♯"),
        (0, "B♯"),
        (7, "F𝄪"),
        (2, "C𝄪"),
        (9, "G𝄪"),
        (4, "D𝄪"),
        (11, "A𝄪"),
    ]

    flat_corrections = [
        (11, "C♭"),
        (4, "F♭"),
        (9, "B𝄫"),
        (2, "E𝄫"),
        (7, "A𝄫"),
        (0, "D𝄫"),
    ]  # last would be (5, 'G𝄫')

    # Apply a mod-12 correction to distinguish B#:maj from C:maj
    n_sharps = tonic_number
    if tonic_number == 0 and tonic == "B":
        n_sharps = 12

    if use_sharps:
        # This will only execute if n_sharps >= 6
        for n in range(0, n_sharps - 6 + 1):
            index, name = sharp_corrections[n]
            notes_sharp[index] = name

        notes = notes_sharp
    else:
        n_flats = (12 - tonic_number) % 12

        # This will only execute if tonic_number <= 6
        for n in range(0, n_flats - 6 + 1):
            index, name = flat_corrections[n]
            notes_flat[index] = name

        notes = notes_flat

    # Finally, apply any unicode down-translation if necessary
    if not unicode:
        translations = str.maketrans({"♯": "#", "𝄪": "##", "♭": "b", "𝄫": "bb"})
        notes = list(n.translate(translations) for n in notes)
    return notes