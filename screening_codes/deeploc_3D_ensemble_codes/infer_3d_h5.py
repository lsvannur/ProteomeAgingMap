import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
import nibabel as nib
from collections import Counter
import pickle
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
import argparse
import os
import monai
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
from utils import *

import h5py
import re
import glob
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Custom3DDataset(Dataset):
    def __init__(self, data, label, ds_type="train"):
        self.data = data
        self.ds_type = ds_type
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming `image_path` and `label` columns in the dataframe
        img = self.data[idx]
        label = self.label[idx]

        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Add channel dimension

        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / (img_std + 1e-8)

        if img.shape != (1, 8, 64, 64):
            raise ValueError(f"Unexpected image shape: {img.shape}")


        #label = int(label)

        sample = {
            'image': torch.tensor(img, dtype=torch.float32),
            'target': torch.tensor(label, dtype=torch.float32)
        }

        return sample


def prepare_data(df, rev_label_map):
    df = df[(df['labels']!='punctate composite') & (df['labels']!='microtubule')]
    df['labels'] = df['labels'].str.replace(' ', '_')
    df = df.reset_index(drop=True)

    features = df['features']
    labels = df['labels']
    labels_encoded = [rev_label_map[i] for i in labels]
    labels_onehot = np.zeros((len(labels_encoded), 17))
    for index, labelid in enumerate(labels_encoded):
        labels_onehot[index][labelid] = 1

    
    dataset_whole = Custom3DDataset(data=features, label=labels_onehot, ds_type=f"testinfer")
    dataset_whole_dl = torch.utils.data.DataLoader(
                            dataset_whole,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                        )
    
    return dataset_whole_dl, labels_onehot



# Define a wrapper class for the MONAI ResNet10 model
class ResNet10Wrapper(nn.Module):
    def __init__(self, original_model):
        super(ResNet10Wrapper, self).__init__()
        self.original_model = original_model
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten if necessary
        return self.original_model.fc(x)

    def penultimate_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten if necessary
        return x




def run_3d(data, model_path, running_batch_name):

    data['labels'] = ['mitochondria' for _ in range(len(data['gfp_image_names']))]
    features = data['gfp_images']
    labels = ['bud_neck' if x == 'bud' else x for x in data['labels']]
    masks = data['masks']
    gfp_image_names = data['gfp_image_names']
    cw_well_names = data['cw_well_names']
    mask_image_names = data['mask_image_names']

    label_num_map = {0: 'ER', 1: 'Golgi', 2: 'actin', 3: 'bud_neck', 4: 'cell_periphery', 5: 'cytoplasm', 6: 'endosome', 7: 'lipid_particle', 8: 'mitochondria', 9: 'none', 10: 'nuclear_periphery', 11: 'nucleolus', 12: 'nucleus', 13: 'peroxisome', 14: 'spindle_pole', 15: 'vacuolar_membrane', 16: 'vacuole'}
    rev_label_map = {label_num_map[i]:i for i in label_num_map}

    temp = pd.DataFrame({'cw_well_names':cw_well_names, 'gfp_image_names':gfp_image_names,'features':features})
    temp = temp.sort_values(by=['gfp_image_names']).reset_index(drop=True)

    temp_mask_df = pd.DataFrame({'mask_image_names':mask_image_names, 'masks':masks})
    temp_mask_df = temp_mask_df.sort_values(by=['mask_image_names']).reset_index(drop=True)
    temp['masks'] = temp_mask_df['masks']
    temp['mask_image_names'] = temp_mask_df['mask_image_names']


    print(temp.isna().sum())

    features = temp['features']
    labels = ['mitochondria' for _ in range(len(features))]
    masks = temp['masks']
    gfp_image_names = temp['gfp_image_names']
    mask_image_names = temp['mask_image_names']
    cw_well_names = temp['cw_well_names']


    print('features:', len(features))
    print('labels:',len(labels)) 
    print('gfp images names',len(gfp_image_names))
    print('cw names:', len(cw_well_names))
    print('mask image names:', len(mask_image_names))
    print('masks', len(masks))
    df = pd.DataFrame({'features':features, 'labels':labels, 'cw_well_names':cw_well_names, 'gfp_image_names': gfp_image_names, 'mask_image_names':mask_image_names, 'masks':masks})



    dataset_whole_dl, labels_onehot = prepare_data(df, rev_label_map)
    num_classes = 17

    # Instantiate your model and wrap it
    original_model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=17)
    model = ResNet10Wrapper(original_model)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.5, last_epoch=-1, verbose=True)




    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda")
    finalized_model = ResNet10Wrapper(original_model).to(device)
    finalized_model.load_state_dict(torch.load(model_path, map_location=device))


    model=finalized_model
    model.to(device)
    with torch.no_grad():
        test_loss = 0.0
        preds = []
        all_pred_probs = []
        true_labels = []
        all_last_layer_output = []
        epoch_iterator_test = tqdm(dataset_whole_dl)
        for step, batch in enumerate(epoch_iterator_test):
            model.eval()
            images, targets = batch["image"].to(device), batch["target"].to(device)

            outputs = model(images)
            penultimate_output = model.penultimate_forward(images).cpu().numpy()
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            epoch_iterator_test.set_postfix(
                batch_loss=(loss.item()), loss=(test_loss / (step + 1))
            )
            pred_probs = torch.softmax(outputs, dim=1)
            all_pred_probs.append(pred_probs.cpu().numpy())
            preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.append(torch.argmax(targets,dim=1).cpu().numpy())
            all_last_layer_output.append(penultimate_output)

        preds = np.concatenate(preds)
        true_labels = np.concatenate(true_labels)
        all_pred_probs = np.concatenate(all_pred_probs)
        acc_score = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='weighted')
        precision = precision_score(true_labels, preds, average=None)
        recall = recall_score(true_labels, preds, average=None)

        print(
            f"TEST: Average loss: {test_loss/(step+1)} + Accuracy = {acc_score} + F1 Score = {f1}"
        )
        print(f"Precision per class: {precision}")
        print(f"Recall per class: {recall}")


    #Penultimate layer features
    all_outs = np.concatenate(all_last_layer_output)
    all_outs = pd.DataFrame(all_outs)
    all_outs['true_labels'] = true_labels
    all_outs['preds'] = preds
    all_outs['gfp_image_names'] = gfp_image_names
    all_outs['mask_image_names'] = mask_image_names
    all_outs['cw_well_names'] = cw_well_names
    all_outs['combined'] = all_outs['cw_well_names'].str.cat(all_outs['gfp_image_names'], sep='_')

    all_outs.to_csv(f"./penult_3d/penult_{running_batch_name}_3d.csv", index=None)



    #Predicted Probabilities
    ans3d = pd.DataFrame(all_pred_probs, columns=list(rev_label_map.keys()))
    ans3d['preds'] = preds
    ans3d['true_labels'] = true_labels
    ans3d['gfp_image_names'] = gfp_image_names
    ans3d['cw_well_names'] = cw_well_names
    ans3d['combined'] = ans3d['cw_well_names'].str.cat(ans3d['gfp_image_names'], sep='_')

    ans3d.to_csv(f'./predictions_3d/{running_batch_name}_predprob_3d.csv', index=None)


    features_2d_train_val = np.array([np.max(feat, axis=0) for feat in features])
    masks_2d_train_val = np.array(masks)
    labels_2d_train_val = labels_onehot

    regex = re.compile(r'\d+')

    # open a hdf5 file and create earrays
    f = h5py.File( f'./hdf5_2d_inputs/{running_batch_name}_2d.hdf5', mode='w')
    train_shape = (len(features_2d_train_val), 64*64*2)

    f.create_dataset("Index1", (len(labels_2d_train_val),17), np.uint8)
    f.create_dataset("data1", train_shape, np.float64)


    f["Index1"][...] = labels_2d_train_val

    for i in range(len(features_2d_train_val)):
        mask_img = masks_2d_train_val[i]
        GFP_img = features_2d_train_val[i]

        f["data1"][i, ...] = np.concatenate((GFP_img.ravel(), mask_img.ravel()))
    f.close()


if __name__ == '__main__':
    os.mkdir("penult_3d")
    os.mkdir("penult_2d")
    os.mkdir("predictions_3d")
    os.mkdir("predictions_2d")
    os.mkdir("predictions_ensemble")

    with open('infer_config_h5.yaml', 'r') as file:
        config = yaml.safe_load(file)

    input_conf = config['input']
    batch_folder_path = input_conf['batch_folder_path']
    running_batch_name = input_conf['running_batch_name']
    modelpath = input_conf['modelpath_3d']

    batch_file_names = os.listdir(batch_folder_path)
    for batch_name in batch_file_names:
        running_batch_name_temp = f'{running_batch_name}_{batch_name.split('.')[0]}'
        print(running_batch_name_temp)
        data = load_dict_from_hdf5(f"{batch_folder_path}{batch_name}")
        gfp_names = []
        gfp_images = []
        masks_names = []
        masks_images = []
        cw_well_names = []
        print(data.keys())
        for key, val in data['GFP_crop'].items():
            for gfp_key,  gfp_val in val.items():
                if gfp_key.endswith('_GFP.tif'):
                    gfp_names.append(gfp_key)
                    gfp_images.append(gfp_val[:8,:,:])
                    cw_well_names.append(key)
                if gfp_key.endswith('_mask.tif'):
                    masks_names.append(gfp_key)
                    masks_images.append(gfp_val)

        data_final = {'cw_well_names': cw_well_names, 'masks': masks_images, 'mask_image_names': masks_names, 'gfp_image_names': gfp_names, 'gfp_images': gfp_images}
        print(len(gfp_images), len(gfp_names), len(masks_images), len(masks_names), len(cw_well_names))


        # scar_names = []
        # scar_images = []
        # scar_well_names = []
        # for key, val in data['scar_z_crop'].items():
        #     for scar_key,  scar_val in val.items():
        #         scar_names.append(scar_key)
        #         scar_images.append(scar_val)
        #         scar_well_names.append(key)

        # final_scar = {'scar_names': scar_names, 'scar_images': scar_images, 'scar_well_names':scar_well_names}
        # print(len(final_scar['scar_images']), len(scar_names), len(scar_well_names))


        # with open(f'{running_batch_name_temp}_scar_crop.pkl', 'wb') as file:
        #         pickle.dump(final_scar, file)

        run_3d(data_final, modelpath, running_batch_name_temp)