import os
import h5py
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import monai
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Custom3DDataset(Dataset):
    def __init__(self, data, ds_type="train"):
        self.data = data
        self.ds_type = ds_type
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Add channel dimension

        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / (img_std + 1e-8)

        if img.shape != (1, 8, 64, 64):
            raise ValueError(f"Unexpected image shape: {img.shape}")

        sample = {'image': torch.tensor(img, dtype=torch.float32)}
        return sample


def prepare_data(df):
    df = df.reset_index(drop=True)

    features = df['features']
    
    dataset_whole = Custom3DDataset(data=features, ds_type=f"testinfer")
    dataset_whole_dl = torch.utils.data.DataLoader(
                            dataset_whole,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                        )
    
    return dataset_whole_dl



# Define a wrapper class for the MONAI ResNet10 model
class ResNet50Wrapper(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Wrapper, self).__init__()
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

    features = data['gfp_images']
    masks = data['masks']
    gfp_image_names = data['gfp_image_names']
    cw_well_names = data['cw_well_names']
    mask_image_names = data['mask_image_names']

    label_num_map = {0: 'ER', 1: 'Golgi', 2: 'actin', 3: 'bud_neck', 4: 'cell_periphery', 5: 'cytoplasm', 6: 'endosome', 7: 'lipid_particle', 8: 'mitochondria', 9: 'none', 10: 'nuclear_periphery', 11: 'nucleolus', 12: 'nucleus', 13: 'peroxisome', 14: 'spindle_pole', 15: 'vacuolar_membrane', 16: 'vacuole'}
    rev_label_map = {label_num_map[i]:i for i in label_num_map}


    temp = pd.DataFrame({
        'cw_well_names': cw_well_names,
        'gfp_image_names': gfp_image_names,
        'features': features
    })

    temp['base_name'] = temp['gfp_image_names'].str.replace('_GFP.tif', '', regex=False)


    temp_mask_df = pd.DataFrame({
        'mask_image_names': mask_image_names,
        'masks': masks
    })

    temp_mask_df['base_name'] = temp_mask_df['mask_image_names'].str.replace('_mask.tif', '', regex=False)


    # Merge using base_name
    temp = temp.merge(
        temp_mask_df,
        on='base_name',
        how='inner'
    )
    temp = temp.drop(columns=['base_name'])

    features = temp['features']
    masks = temp['masks']
    gfp_image_names = temp['gfp_image_names']
    mask_image_names = temp['mask_image_names']
    cw_well_names = temp['cw_well_names']


    print('features:', len(features))
    print('gfp images names',len(gfp_image_names))
    print('cw names:', len(cw_well_names))
    print('mask image names:', len(mask_image_names))
    print('masks', len(masks))
    df = pd.DataFrame({'features':features, 'cw_well_names':cw_well_names, 'gfp_image_names': gfp_image_names, 'mask_image_names':mask_image_names, 'masks':masks})


    dataset_whole_dl = prepare_data(df)
    original_model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=17)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finalized_model = ResNet50Wrapper(original_model).to(device)
    finalized_model.load_state_dict(torch.load(model_path, map_location=device))


    model=finalized_model
    with torch.no_grad():
        preds = []
        all_pred_probs = []
        all_last_layer_output = []
        epoch_iterator_test = tqdm(dataset_whole_dl)
        model.eval()
        for step, batch in enumerate(epoch_iterator_test):
            images = batch["image"].to(device)

            outputs = model(images)
            penultimate_output = model.penultimate_forward(images).cpu().numpy()
            
            pred_probs = torch.softmax(outputs, dim=1)
            all_pred_probs.append(pred_probs.cpu().numpy())
            preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_last_layer_output.append(penultimate_output)

        preds = np.concatenate(preds)
        all_pred_probs = np.concatenate(all_pred_probs)

    #Penultimate layer features
    all_outs = np.concatenate(all_last_layer_output)
    all_outs = pd.DataFrame(all_outs)
    all_outs['preds'] = preds
    all_outs['gfp_image_names'] = gfp_image_names
    all_outs['mask_image_names'] = mask_image_names
    all_outs['cw_well_names'] = cw_well_names
    all_outs['combined'] = all_outs['cw_well_names'].str.cat(all_outs['gfp_image_names'], sep='_')

    all_outs.to_csv(f"./penult_3d/penult_{running_batch_name}_3d.csv", index=None)

    #Predicted Probabilities
    ans3d = pd.DataFrame(all_pred_probs, columns=list(rev_label_map.keys()))
    ans3d['preds'] = preds
    ans3d['gfp_image_names'] = gfp_image_names
    ans3d['cw_well_names'] = cw_well_names
    ans3d['combined'] = ans3d['cw_well_names'].str.cat(ans3d['gfp_image_names'], sep='_')

    ans3d.to_csv(f'./predictions_3d/{running_batch_name}_predprob_3d.csv', index=None)


    features_2d_train_val = np.array([np.max(feat, axis=0) for feat in features])
    masks_2d_train_val = np.array(masks)

    # open a hdf5 file and create earrays
    f = h5py.File( f'./hdf5_2d_inputs/{running_batch_name}_2d.hdf5', mode='w')
    train_shape = (len(features_2d_train_val), 64*64*2)
    f.create_dataset("data1", train_shape, np.float64)

    
    for i in range(len(features_2d_train_val)):
        mask_img = masks_2d_train_val[i]
        GFP_img = features_2d_train_val[i]

        f["data1"][i, ...] = np.concatenate((GFP_img.ravel(), mask_img.ravel()))
    f.close()


if __name__ == '__main__':
    os.makedirs("penult_3d", exist_ok=True)
    os.makedirs("penult_2d", exist_ok=True)
    os.makedirs("predictions_3d", exist_ok=True)
    os.makedirs("predictions_2d", exist_ok=True)
    os.makedirs("predictions_ensemble", exist_ok=True)

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

        run_3d(data_final, modelpath, running_batch_name_temp)