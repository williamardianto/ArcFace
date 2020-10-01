import argparse
import sys
from trainer import trainer
from dataset import Dataset
from PIL import ImageFile
import torch
from models.arcface import ArcFace

def main(args):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    dataset = Dataset(args.training_data_dir, args.validation_data_dir, args.batch_size, args.validation_batch_size)

    model = ArcFace(classnum=dataset.train_classnum)

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

        model.to(args.device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate, momentum=0)

    lr_decay_steps = [int(item) for item in args.learning_rate_decay_steps.split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_steps, gamma=args.learning_rate_decay_factor)




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        help='Training using cpu or gpu(cuda)', default='cpu')
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/arcface')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/arcface')
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--training_data_dir', type=str,
                        help='Path to the training data directory containing aligned face patches.',
                        default='~/datasets/casia/casia_aligned_182_160')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=512)
    # parser.add_argument('--keep_probability', type=float,
    #                     help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=5e-4)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'ADAM'],
                        help='The optimization algorithm to use', default='SGD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.01)
    # parser.add_argument('--learning_rate_decay_epochs', type=int,
    #                     help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_steps', type=str,
                        help='Learning rate decay steps in epoch', default='40,60,80')
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.1)


    # Parameters for validation
    parser.add_argument('--validation_data_dir', type=str,
                        help='Path to the validation data directory containing aligned face patches.',
                        default='~/datasets/lfw/lfw_aligned_160')
    parser.add_argument('--validation_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--validation_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--validation_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--validation_distance_metric', type=int,
                        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--validation_use_flipped_images',
                        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--validation_subtract_mean',
                        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))