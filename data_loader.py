import json
import io
import os
import platform
import warnings

import numpy as np
import pandas as pd

class DATA_LOADER:

    def __init__(self):

        warnings.filterwarnings("ignore")

        # https: // stackoverflow.com / questions / 8220108 / how - do - i - check - the - operating - system - in -python / 8220141
        # https://stackoverflow.com/questions/40416072/reading-file-using-relative-path-in-python-project
        os_env= platform.system()

        if os_env == 'Linux':
            self.data_path = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '/data/'
            self.result_path = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '/result/'
        elif os_env == 'Windows':
            self.data_path = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '\data\\'
            self.result_path = os.path.abspath(os.path.dirname(os.path.abspath('__file__'))) + '\\result\\'

    def write_json(self, data, fname):
        def _conv(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            raise TypeError

        with io.open(self.result_path + fname, "w", encoding="UTF-8") as f:
            json_str = json.dumps(data, ensure_ascii=False, default=_conv)
            f.write(json_str)

    def load_json(self, fname):
        file_name = self.data_path + fname
        with open(file_name,"r",encoding='UTF-8') as f:
            json_obj = json.load(f)
        return json_obj

    def load_json_to_pandas(self,fname):
        file_name = self.data_path + fname
        pd_obj = pd.read_json(file_name, encoding='utf-8')

        return pd_obj

    def debug_json(self, r):
        print(json.dumps(r, ensure_ascii=False, indent=4))

