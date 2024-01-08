from data import BaseDataset, PoisonDataset

def base_data_exists():
    dataset = BaseDataset()
    assert len(dataset) > 0

def poison_data_exists():
    dataset = PoisonDataset()
    assert len(dataset) > 0

def run_pre_poison_tests():
    base_data_exists()

def run_all_tests():
    base_data_exists()
    poison_data_exists()