import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images 
TRAIN_DIR = '../data/output/train'
# validation images 
VALID_DIR = '../data/output/valid'
# annotation file
ANN_DIR = '../data/output/metadata.json'
# classes: 0 index is reserved for background
CLASSES = [
    '0', '1'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True
# location to save model and plots
OUT_DIR = '/result'