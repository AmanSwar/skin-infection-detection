TRAIN_DL = DataLoader(train_dataset , batch_size=BATCH_SIZE , shuffle=True , pin_memory=True)
VALID_DL = DataLoader(valid_dataset , batch_size=BATCH_SIZE , shuffle=False , pin_memory=True)


def train_valid():
    return TRAIN_DL , VALID_DL