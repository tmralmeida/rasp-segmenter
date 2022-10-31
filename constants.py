# data related 
DATASET_NAME = "synthetic"
RAW_DATA_PATH = "dataset_of_synthetic_rgb-d_plants/" # where to load from 
PROC_DATA_PATH = "rasp-segmenter/data" # where to save
CHECK_SAVE_PATH = "rasp-segmenter"
TRAIN_PT = 0.7 # train set %
VAL_PT = 0.2 # val set %
TEST_PT = 0.1 # test set %

# model
MODEL_NAME = "segnet"
INPUT_SHAPE = (224, 224)
NUM_CLASSES = 5


# hyperparams
NUM_EPOCHS = 10
VAL_EVERY = 2
BATCH_SIZE = 8
INPUT_CHANNELS = 3
LR = 1e-3