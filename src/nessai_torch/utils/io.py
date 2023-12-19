"""I/O related utils"""
import h5py
from typing import Any


def encode_for_hdf5(value: Any) -> Any:
    """Encode a value for HDF5 file format.

    Parameters
    ----------
    value
        Value to encode.

    Returns
    -------
    Encoded value.
    """
    if value is None:
        output = "__none__"
    else:
        output = value
    return output


def add_dict_to_hdf5_file(hdf5_file: h5py.File, path: str, dictionary):
    """Save a dictionary to a HDF5 file.

    Based on :code:`recursively_save_dict_contents_to_group` in bilby.

    Parameters
    ----------
    hdf5_file
        HDF5 file.
    path
        Path added to the keys of the dictionary.
    dictionary
        The dictionary to save.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            add_dict_to_hdf5_file(hdf5_file, path + key + "/", value)
        else:
            hdf5_file[path + key] = encode_for_hdf5(value)


def save_dict_to_hdf5(dictionary: dict, filename: str) -> None:
    """Save a dictionary to a HDF5 file.

    Parameters
    ----------
    dictionary
        Dictionary to save.
    filename
        Filename (with the extension) to save the dictionary to. Should include
        the complete path.
    """
    with h5py.File(filename, "w") as f:
        add_dict_to_hdf5_file(f, "/", dictionary)
