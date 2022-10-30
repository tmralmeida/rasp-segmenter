# data related 
DATASET_NAME = "synthetic"
RAW_DATA_PATH = "dataset_of_synthetic_rgb-d_plants/" # where to load from 
PROC_DATA_PATH = "rasp-segmenter/data" # where to save
TRAIN_PT = 0.7 # train set %
VAL_PT = 0.2 # val set %
TEST_PT = 0.1 # test set %
NUM_CLASSES = 5
INPUT_SHAPE = (224, 224)


# hyperparams
BATCH_SIZE = 8
INPUT_CHANNELS = 3