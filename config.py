IMG_SIZE = 224

BACKBONE = 'swsl_resnet18'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
LR = 0.0001
BATCH_SIZE = 2

DSET_BASE_DIR = 'data/chihuahua-muffin'
CLASSES_LIST = ['chihuahua', 'muffin']
CLASSES_MAP = {class_:idx for idx, class_ in enumerate(CLASSES_LIST)}
CLASSES_BY_IDX = {idx:class_ for idx, class_ in enumerate(CLASSES_LIST)}
NCLASSES = 1