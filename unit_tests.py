from data import BaseDataset, PoisonDataset
from data.data_util import fill_bases_directory

# Check that the base dataset isn't empty
def base_data_exists():
    dataset = BaseDataset()
    assert len(dataset) > 0

# Check that the poison dataset isn't empty
def poison_data_exists():
    dataset = PoisonDataset()
    assert len(dataset) > 0

# Check that the base dataset isn't empty after filling
def base_data_exists_after_filling():
    fill_bases_directory()
    assert base_data_exists()

def run_pre_poison_tests():
    base_data_exists()
    base_data_exists_after_filling()

def run_all_tests():
    base_data_exists()
    poison_data_exists()
    base_data_exists_after_filling()