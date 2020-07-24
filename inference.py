from models import ICBF_OCC
from data_loader import DATA_LOADER
from tqdm import tqdm

def main():
    files_dict = {'song_meta': 'song_meta.json', 'train': 'train.json', 'val':'val.json','test': 'test.json'}
    data_dict = dict()
    data_loader = DATA_LOADER()
    print(f"Load DataFiles: {files_dict.values()}")
    for key,item in tqdm(files_dict.items()):
        data_dict[key] = data_loader.load_json_to_pandas(item)
        print(f"\n{item} is loaded successfully.")

    icbf_occ_model = ICBF_OCC(train_df=data_dict["train"],test_df=data_dict["val"])
    icbf_rcomm_result = icbf_occ_model.execute_recommendation()
    data_loader.write_json(icbf_rcomm_result,'icbf_occ_rcomm_results_2.json')

if __name__ == '__main__':
    main()