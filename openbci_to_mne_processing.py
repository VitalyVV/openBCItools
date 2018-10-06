import numpy as np
import mne
from pathlib import Path


def microvolts_to_volts(value):
    """
    Since openBCI writes data into micro volts and mne works with volts we
    will need to convert the data later.
    :param value: single micro volts value
    :return: same value in volts
    """
    return float(value) / 1000


def load_openbci_file(filename, ch_names=None):
    """
    Load data from OpenBCI file into mne RawArray for later use

    :param filename: filename for reading in form of relative path from working directory
    :param ch_names: dictionary having all or some channels like this:
            {"fp1":1, "fp2":2, "c3":3, "c4":4, "o1":5, "o2":6, "p3":7, "p4":8}
            Key specifies position on head using 10-20 standard and
            Value referring to channel number on Cyton BCI board
    :return: RawArray class of mne.io library
    """
    if ch_names is None:
        ch_names = {"fp1":1, "fp2":2, "c3":3, "c4":4, "o1":5, "o2":6, "p3":7, "p4":8}

    # Converter of BCI file to valuable data
    converter = {i: (microvolts_to_volts if i < 12 else lambda x: str(x).split(".")[1][:-1])
                 for i in range(0, 13)}

    info = mne.create_info(
        ch_names=list(ch_names.keys()),
        ch_types=['eeg' for i in range(0, len(ch_names))],
        sfreq=250,
        montage='standard_1020'
    )
    data = np.loadtxt(filename, comments="%", delimiter=",",
                      converters=converter).T[list(ch_names.values())]
    return mne.io.RawArray(data, info)


def create_epochs(raw_data, duration=1):
    """
    Chops the RawArray onto Epochs given the time duration of every epoch
    :param raw_data: mne.io.RawArray instance
    :param duration: seconds for copping
    :return: mne Epochs class
    """
    events = mne.make_fixed_length_events(raw_data, duration=duration)
    epochs = mne.Epochs(raw_data, events, preload=True)
    return epochs


def get_files(dir='.', pattern='*.txt'):
    """
    Loading files from given directory with specified pattern.
    :param dir: Lookup directory
    :param pattern: Pattern for files. Default *.txt for loading raw BCI files
    :return: array of file paths
    """
    # Specifying files directory, select all the files from there which is txt
    datadir = Path(dir).glob(pattern)
    # Transferring generator into array of file paths
    return [x for x in datadir]
