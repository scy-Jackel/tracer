from main_func import view_for_pcl

from parser import get_parser


if __name__ == '__main__':
    args = get_parser()
    data_url = args.data_url
    train_url = args.train_url
    result_url = args.result_url
    view_for_pcl(data_url, result_url, train_url)
    print("inference complete.")

