import h5py
import numpy as np
import pytest
from unittest.mock import mock_open, patch

from nessai_torch.utils.io import (
    add_dict_to_hdf5_file,
    encode_for_hdf5,
    save_dict_to_hdf5,
)


@pytest.fixture
def data_dict():
    data = dict(
        a=np.array([1, 2, 3]),
        b=np.array([(1, 2)], dtype=[("x", "f4"), ("y", "f4")]),
        cls=object(),
        l=[1, 2, 3],
        dict1={"a": None, "b": 2},
        dict2={"c": [1, 2, 3], "array": np.array([3, 4, 5])},
        s="A string",
        nan=None,
    )
    return data


@pytest.mark.parametrize(
    "value, expected", [(None, "__none__"), ([1, 2], [1, 2])]
)
def test_encode_to_hdf5(value, expected):
    assert encode_for_hdf5(value) == expected


def test_add_dict_to_hdf5_file(tmp_path, data_dict):
    data_dict.pop("cls")
    with h5py.File(tmp_path / "test.h5", "w") as f:
        add_dict_to_hdf5_file(f, "/", data_dict)
        assert list(f.keys()) == sorted(data_dict.keys())
        assert f["/dict1/a"][()].decode() == "__none__"
        np.testing.assert_array_equal(
            f["dict2/array"][:], data_dict["dict2"]["array"]
        )


def test_save_dict_to_hdf5(data_dict):
    f = mock_open()
    filename = "result.h5"
    with patch("h5py.File", f) as mock_file, patch(
        "nessai_torch.utils.io.add_dict_to_hdf5_file"
    ) as mock_add:
        save_dict_to_hdf5(data_dict, filename)
    mock_file.assert_called_once_with(filename, "w")
    mock_add.assert_called_once_with(f(), "/", data_dict)
