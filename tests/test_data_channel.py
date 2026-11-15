"""A wrong channel name must produce an error that says which channels exist."""
import pytest

from sleepnet.data import load_recording, write_synthetic_edf_pair


def test_unknown_channel_lists_the_available_ones(tmp_path):
    psg, hyp, _ = write_synthetic_edf_pair(tmp_path, "SC4001E", 40, 1)
    with pytest.raises(ValueError, match=r"EEG Cz-Oz.*\['EEG Fpz-Cz'\]"):
        load_recording(psg, hyp, channel="EEG Cz-Oz")
