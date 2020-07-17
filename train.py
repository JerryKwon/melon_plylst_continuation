from models import ICBF_OCC
from data_loader import DATA_LOADER


def main():
    files_dict = {'song_meta': 'song_meta.json', 'train': 'train.json', 'val':'val.json','test': 'test.json'}
    data_dict = dict()
    data_loader = DATA_LOADER()
    for key,item in files_dict.items():
        data_dict[key] = data_loader.load_json(item)

    icbf_occ_model = ICBF_OCC(train_df=data_dict["train"],test_df=data_dict["test"])

if __name__ == '__main__':
    main()
