import argparse
import warnings

from tqdm import tqdm

from models import ICBF_OCC,HYBRID_CBF_ICBF
from data_loader import DATA_LOADER

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="execute inference.py")
    parser.add_argument('--model_type', type=str, default=None, help='select model type [icbf | hybrid]')
    parser.add_argument('--is_valid', type=str2bool, help='select dataset [True=val.json | False=test.json]')

    args = parser.parse_args()
    model_type = args.model_type
    is_vaild = args.is_valid

    files_dict = {'song_meta': 'song_meta.json', 'train': 'train.json', 'val':'val.json','test': 'test.json'}
    data_dict = dict()
    data_loader = DATA_LOADER()
    print(f"Load DataFiles: {files_dict.values()}")
    for key,item in tqdm(files_dict.items()):
        data_dict[key] = data_loader.load_json_to_pandas(item)
        print(f"\n{item} is loaded successfully.")

    if is_vaild:
        target = data_dict["val"]
        target_str = "valid"
    else:
        target = data_dict["test"]
        target_str = "test"


    if model_type == "icbf":
        icbf_model = ICBF_OCC(train_df=data_dict["train"],test_df=target)
        icbf_rcomm_result = icbf_model.execute_recommendation()
        data_loader.write_json(icbf_rcomm_result, target_str+'_icbf_rcomm_results.json')


    if model_type == "hybrid":
        hybrid_model = HYBRID_CBF_ICBF(train_df=data_dict["train"], test_df=target)
        hybrid_rcomm_result = hybrid_model.execute_recommendation()
        data_loader.write_json(hybrid_rcomm_result, target_str+'_hybrid_rcomm_results.json')


if __name__ == '__main__':
    main()