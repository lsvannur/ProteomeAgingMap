import os
import pandas as pd
import numpy as np
from PIL import Image
import skimage
from skimage import io, exposure, measure, morphology, filters, restoration, util
from nd2reader import ND2Reader
import nd2
import time
import matplotlib.pyplot as plt
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *



def process_file(file, image_folder):
    image_path = os.path.join(image_folder, file)
    result = {}
    if not os.path.isdir(image_path):
        im_array = nd2.imread(image_path)
        with ND2Reader(image_path) as images:
            split_file = file.split('_')
            channel_names = images.metadata['channels']
            unit = images.metadata['pixel_microns']
            print(unit)
            print(channel_names)
            

            if channel_names[0] in ['TK_Just change SoRa Triggerd1x', 'Yoo 405(CW)', 'TCW1', 'TCW2']:        
                outputarray = im_array
                CW = np.max(outputarray[:,:,:], axis=0)
                CW = (CW - np.min(CW)) / (np.max(CW) - np.min(CW)) * 255
                p0, p99 = np.percentile(CW, (0, 99.8))
                CW = np.where(CW < p99, CW, CW * 0.6)
                CW = CW.astype(np.uint8)
                CW = exposure.rescale_intensity(CW, in_range=(0, p99))
                result['CW'] = {
                    f'CW_{split_file[4]}_{split_file[6]}_1.tif': CW[0:1024, 0:1024],
                    f'CW_{split_file[4]}_{split_file[6]}_2.tif': CW[1024:2048, 0:1024],
                    f'CW_{split_file[4]}_{split_file[6]}_3.tif': CW[0:1024, 1024:2048],
                    f'CW_{split_file[4]}_{split_file[6]}_4.tif': CW[1024:2048, 1024:2048]
                }

            elif channel_names[0] in ['TK_Just change SoRa Triggerd1GFP', 'Yoo 488 #3', 'TGFP1', 'TGFP2', 'TGFP1_Manual']:       
                outputarray = im_array
                result['GFP'] = {
                    f'GFP_{split_file[4]}_{split_file[6]}_1.tif': outputarray[:, 0:1024, 0:1024],
                    f'GFP_{split_file[4]}_{split_file[6]}_2.tif': outputarray[:, 1024:2048, 0:1024],
                    f'GFP_{split_file[4]}_{split_file[6]}_3.tif': outputarray[:, 0:1024, 1024:2048],
                    f'GFP_{split_file[4]}_{split_file[6]}_4.tif': outputarray[:, 1024:2048, 1024:2048]
                }

            elif channel_names[0] in ['TK_Just change 1x TriggerdRFP', 'TWGA1', 'TWGA2']: 
                outputarray = im_array[:,:,:]
                result['scar'] = {
                    f'scar_{split_file[4]}_{split_file[6]}_1.tif': outputarray[:, 0:1024, 0:1024],
                    f'scar_{split_file[4]}_{split_file[6]}_2.tif': outputarray[:, 1024:2048, 0:1024],
                    f'scar_{split_file[4]}_{split_file[6]}_3.tif': outputarray[:, 0:1024, 1024:2048],
                    f'scar_{split_file[4]}_{split_file[6]}_4.tif': outputarray[:, 1024:2048, 1024:2048]
                }

            elif channel_names[0] in ['TK_Just change SoRa TriggerdGreen', 'Yoo 488 #3_SoRa', 'TSRGFP1']:  
                outputarray = im_array[:,:,:]
                result['GFP_SoRa'] = {
                    f'GFP_{split_file[4]}_{split_file[6]}_1.tif': outputarray
                }

            elif channel_names[0] in ['Yoo 594(Budscar)-SoRa', 'TK_Just change SORA TriggerdRFP', 'TSRWGA1']: 
                outputarray = im_array[:,:,:]
                result['scar_SoRa'] = {
                    f'scar_{split_file[4]}_{split_file[6]}_1.tif': outputarray
                }

            elif channel_names[0] in ['Yoo 405_CW_SoRa', '405 nm', 'TSRCW1']:   
                outputarray = im_array[:,:,:]
                CW = np.std(outputarray[:,:,:], axis=0)
                CW = CW / np.max(CW) * 255
                CW = CW.astype(np.uint8)
                CW = exposure.rescale_intensity(CW, in_range=(0, 98))
                result['CW_SoRa'] = {
                    f'CW_{split_file[4]}_{split_file[6]}_1.tif': CW    
                }
            

    return result


def nd2_load(input_folder_path, input_batch_folder_path, output_folder_path, CPU_COUNT, BATCH_SIZE):
    


   
    start = time.time()
    image_folder = input_folder_path
    all_files = [file for file in os.listdir(image_folder) if file.endswith('nd2')]
    data = pd.DataFrame({'files': sorted(all_files)})
    data.to_csv(os.path.join(output_folder_path, f'all_files.csv'),index=None)


    data['identifier'] = data['files'].apply(lambda x: x.split('__')[1].split('_')[0])

    grouped = data.groupby('identifier')['files'].apply(list).reset_index()

    batches = create_batches(grouped, BATCH_SIZE)


    final_batches = {}
    for id, batch in enumerate(batches):
        batch_wells = batches[id]['identifier'].values.tolist()
        batch_files = np.concatenate(batches[id]['files'].values).ravel().tolist()
        final_batches[id] = {'wells': batches[id]['identifier'].values.tolist(), 
                            'files': batch_files}
        
    pd.DataFrame(final_batches).T.drop(['files'], axis=1).to_csv(os.path.join(output_folder_path,'final_batch.csv'))
    

    for i in range(len(final_batches)):
        batch_files = final_batches[i]['files']
        batch_index = i
        print(f"Processing batch {batch_index}/{len(final_batches)}")

        results = []
        with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
            futures = {executor.submit(process_file, file, image_folder): file for file in batch_files}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        save_results_to_hdf5_after_nd2_load(results, input_batch_folder_path, batch_index)

        end = time.time()
        print(f"==================== Batch {batch_index} processed and loaded nd2(HDF5) in {end-start} seconds! =============")

    end = time.time()
    print(f"================== All Batches processed and loaded nd2(HDF5) in {end-start} seconds! ===================")

if __name__ == '__main__':
    nd2_load('/home/zhoulab/Documents/nikon_raw_data/NG7/20240715_NG7-4_Old_Plate1/',
             '/home/zhoulab/Documents/newversion_codes/input_batches',
             '/run/user/1000/gvfs/smb-share:server=bigrock,share=zhoulab/Raj/cellintegration/outputs_batches',
             CPU_COUNT=12, BATCH_SIZE=1)

