import os
import glob
import pandas as pd

if __name__ == "__main__":
    dirname =  'result'
    # 'result' 'result_sigs'  'result_logis' 
    files = glob.glob('{}/*.csv'.format(dirname))
    for file in files:
        name = os.path.basename(file)
        print(name)
        read_file = pd.read_csv (file)
        read_file.to_excel(file.replace('.csv', '.xlsx').replace('{}'.format(dirname), '{}_x'.format(dirname)), index = None, header=True)