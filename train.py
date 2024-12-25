from main_func import train_for_pcl

from parser import get_parser

import warnings
 
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = get_parser()
    data_url = args.data_url
    train_url = args.train_url
    result_url = args.result_url
    train_for_pcl(data_url, result_url, train_url)
    print("train complete.")
