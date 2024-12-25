import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='general parser.')
    parser.add_argument('--data_url',
                        help='path to training/inference dataset folder',
                        default='./extractfiles/')

    parser.add_argument('--train_url',
                        help='model folder to save/load',
                        default='./all_ckpts/')

    parser.add_argument('--result_url',
                        help='folder to save inference results',
                        default='./cover_2d_out/')

    parser.add_argument('--data_url2', 
                        type=str, 
                        default=None, 
                        help='Path to dataset2')                    

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        help='input batch size',
                        default=4)

    parser.add_argument('--seed', 
                        type=int, 
                        default=34, 
                        help='Random Seed')

    parser.add_argument('-l', '--learning_rate',
                        type=float, 
                        default=0.0001, 
                        help='Interval of evaluate.')                    

    parser.add_argument('-nc', '--num_classes',
                        type=int,
                        help='input image channels',
                        default=8)

    parser.add_argument('-ngf', '--num_filters_g',
                        type=int,
                        help='number of filters for the first layer of the generator',
                        default=16)

    parser.add_argument('-ndf', '--num_filters_d',
                        type=int,
                        help='number of filters for the first layer of the discriminator',
                        default=16)

    parser.add_argument('-nep', '--num_epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=5000)

    parser.add_argument('--interval_eval',
                        type=int,
                        help='number of discriminator iterations per each generator iter, default=5',
                        default=5)

    parser.add_argument('-f', '--load', 
                        type=str, 
                        default=False,
                        help='Load model from a .pth file')

    parser.add_argument('-s', '--scale', 
                        type=float, 
                        default=1.0,
                        help='Downscaling factor of the images')

    parser.add_argument('-v', '--val', 
                        type=float, 
                        default=30.0,
                        help='Percent of the data that is used as validation (0-100)')
                        
    parser.add_argument('--norm', 
                        action='store_true',
                        help="Normalize training images.",
                        default=False)

    parser.add_argument('--viz', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)

    parser.add_argument('--no-save', action='store_true',
                        help="Do not save the output masks",
                        default=False)

    parser.add_argument('--mask-threshold', 
                        type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()
