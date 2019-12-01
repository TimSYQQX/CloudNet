import torch.utils.data as data
import pandas as pd
import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import CloudDataset
import segmentation_models_pytorch as smp
from catalyst.dl.runner import SupervisedRunner
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torch
from torch.utils.data import DataLoader
from utils import rle_decode, get_training_augmentation, get_preprocessing, get_validation_augmentation
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from albumentations import pytorch as AT
from PIL import Image
from tqdm import tqdm
import cv2
from  torch.utils.tensorboard import SummaryWriter
import os
import argparse

def main():
    ######### add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_num', type=int, default=19,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset_path', default='../input/understanding_cloud_organization', help="Path to the dataset directory")
    parser.add_argument('--log_path', default='./logs/segmentation',
                        help="Path to the log directory")
    parser.add_argument('--encoder', default='resnet50', help="Type of encoder in model")
    parser.add_argument('--encoder_weight', default='imagenet', help='Type of encoder weight in model')
    parser.add_argument('--device', default='cuda', help='Training device with default cuda')
    parser.add_argument('--class_num', type=int, default=4, help='Number of classes trying to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    args = parser.parse_args()
    num_epochs = args.epochs_num
    path = args.dataset_path
    logdir = args.log_path
    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')

    ######### should go to logging
    n_train = len(os.listdir(f'{path}/train_images'))
    n_test = len(os.listdir(f'{path}/test_images'))
    train['Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
    train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()
    train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts().value_counts()

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])


#     ######### visualize training image
#     fig = plt.figure(figsize=(25, 16))
#     for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):
#         for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):
#             ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])
#             im = Image.open(f"{path}/train_images/{row['Image_Label'].split('_')[0]}")
#             plt.imshow(im)
#             mask_rle = row['EncodedPixels']
#             try:  # label might not be there!
#                 mask = rle_decode(mask_rle)
#             except:
#                 mask = np.zeros((1400, 2100))
#             plt.imshow(mask, alpha=0.5, cmap='gray')
#             ax.set_title(f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['label']}")



    ######### generating training, validation and test set
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts(). \
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42,
                                            stratify=id_mask_count['count'], test_size=0.1)
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values




    ######### model parameter
    ACTIVATION = None
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weight,
        classes=args.class_num,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weight)



    ######### define train training parameter
    num_workers = 0
    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn), path=path)
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids,
                                 transforms=get_validation_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn), path=path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = torch.nn.BCELoss()
    runner = SupervisedRunner()



    ######### train the model
#     runner.train(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         loaders=loaders,
#         callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
#         logdir=logdir,
#         num_epochs=args.epochs_num,
#         verbose=True
#     )
    model.cuda(1)
    writer = SummaryWriter()
    step = 0
    for epoch in range(num_epochs):
        torch.cuda.empty_cache() 
        print("epoch: ", epoch)
        print("training")
        model.train()
        for img, label in (loaders["train"]):
            img = img.cuda(1)
            label = label.cuda(1)        
            logit = model(img)
            print(logit.shape, label.shape)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("training_loss", loss.item(), step)
            step += 1
            if not step % 40:
                print("training: ", loss.item(), "step:", step)

        print("validating")
        model.eval()
        validation_loss = []
        for img, label in tqdm(loaders["valid"]):
            img = img.cuda(1)
            label = label.cuda(1)
            logit = model(img)
            loss = criterion(logit, label)
            validation_loss.append(loss.item())
        
        print("Validation loss: ", np.mean(validation_loss))
        writer.add_scalar("validation_loss", np.mean(validation_loss), epoch)
        torch.save(model.state_dict(), "../checkpoint/"+str(epoch))
        scheduler.step(np.mean(validation_loss))
    writer.close()

if __name__== "__main__":
    main()








