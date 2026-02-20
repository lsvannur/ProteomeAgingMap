import os
import re
import glob
import sys
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import h5py
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore', category=UserWarning)
mpl.rcParams['figure.dpi'] = 300


from ultralytics import YOLO
from cellpose import utils, io, models
from cellpose import models, core
from cellpose.io import logger_setup

from load_nd2_multiprocess import nd2_load
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import *


def main(results_track, model_budscar_path, batch_file_name, CPU_COUNT, local_batch_folder_path, output_batch_folder_path, summary_folder_path, quantification_folder_path):
    """
    call function to create the required directories
    run the segmentation using cellpose
    run postprocess with multiprocessing
    run the YOLO bud scar detection
    """
    #load all the inputs
    chan = [0,0]
    model = models.Cellpose(gpu=True, model_type='cyto')
    model_budscar = YOLO(model_budscar_path)
    seg_result_track = {'mask':{}}
    file_list = list(results_track['CW'].keys())

    #check GPU
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d'%use_GPU)
    logger_setup();

    
    # Run Cellpose
    only_CW_images = list(results_track['CW'].values())
    masks, flows, styles, diams = model.eval(only_CW_images, diameter=80, channels=chan)

    for i, mask in enumerate(masks):
            seg_result_track['mask'][file_list[i]] = mask


    #Run Postprocessing
    results = []
    with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = {executor.submit(post_process_save, 
                                   results_track['CW'][file_list[index]], 
                                   results_track['GFP'][file_list[index].replace('CW','GFP')],
                                   results_track['scar'][file_list[index].replace('CW','scar')], 
                                   seg_result_track['mask'][file_list[index]], 
                                   index, file_list): index for index in range(len(file_list))}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    #Run BudScarDetection
    scar_folders, scar_crop_folder_path = create_scar_input_folder_images(results)
    bud_scars= {'scar_crop_1.5':{}}
    bud_scars = bud_scar_detection(scar_folders, scar_crop_folder_path, bud_scars, model_budscar)

       
    # Process and save GFP Images quantifications
    GFP_process_results = []
    with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = {executor.submit(process_GFP_image, 
                                   results[index]['quantification'][list(results[index]['quantification'].keys())[0]],
                                   results[index]['GFP_crop'][list(results[index]['quantification'].keys())[0][:-4]],
                                   list(results[index]['quantification'].keys())[0],
                                   index,
                                   len(results)):index for index in range(len(results))}
        for future in as_completed(futures):
            result = future.result()
            GFP_process_results.append(result)
    
    # Running Summarization and adding n_scar in Quantification
    gfp_quantification = {
        'quantification':{},
        'GFP_z': {}
    }

    for result in GFP_process_results:
        for key, value in result.items():
            gfp_quantification[key].update(value)

    all_scar_crop_dict = bud_scars['scar_crop_1.5']
    all_quantification_dict = gfp_quantification['quantification']
    GFP_track_quantification = get_final_quantification_files(quantification_folder_path, all_scar_crop_dict, all_quantification_dict)
    run_summarization(GFP_track_quantification, summary_folder_path)

    # Save all the results in h5
    print(f"SAVING {batch_file_name} files locally")
    save_results_to_hdf5(local_batch_folder_path, results, seg_result_track, bud_scars, GFP_process_results, results_track, batch_file_name)

    #Pushing file to Server
    print(f"Pushing {batch_file_name} file to server")
    fast_copy(os.path.join(local_batch_folder_path,batch_file_name), os.path.join(output_batch_folder_path, batch_file_name) )


if __name__ == "__main__":
    start_time = time.time()

    # Load the YAML file
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    input_conf = config['input']
    image_folder = input_conf['image_folder']
    model_budscar_path = input_conf['model_budscar_path']
    CPU_COUNT = input_conf['CPU_COUNT']
    BATCH_SIZE = input_conf['BATCH_SIZE']
    local_batch_folder_path = input_conf['local_batch_folder_path']
    output_batch_folder_path = input_conf['output_batch_folder_path']

    # # Load all nd2 files and convert to CW, GFP, scar, SORA
    nd2_load(image_folder, local_batch_folder_path, output_batch_folder_path, CPU_COUNT, BATCH_SIZE)
    
    files = glob.glob(os.path.join(image_folder, '*'))

    ## Delete each file
    for file_path in files:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")

    # Load the batches created after nd2_load and Run Segmentation, Bud Scar Detection, 
    batches_folder = [file for file in os.listdir(local_batch_folder_path) if file.endswith('h5')]
    

    os.mkdir(os.path.join(output_batch_folder_path,'summaries'))
    os.mkdir(os.path.join(output_batch_folder_path,'quantifications'))
    
    summary_folder_path = os.path.join(output_batch_folder_path, 'summaries')
    quantification_folder_path = os.path.join(output_batch_folder_path, 'quantifications')


    for batch_file in batches_folder:
        filename = os.path.join(local_batch_folder_path, batch_file)
        print(f"======== Loading the {batch_file} File ===========")
        data = load_dict_from_hdf5(filename)
        print("========= Loaded the Batch File ==========")

        main(data, model_budscar_path, batch_file, CPU_COUNT, local_batch_folder_path, output_batch_folder_path, summary_folder_path, quantification_folder_path)
        end_time = time.time()
        print(f"============= Time taken for {batch_file} is: {end_time-start_time} =============")

    end_time = time.time()
    files = glob.glob(os.path.join(local_batch_folder_path, '*'))

    # # Delete each file
    # for file_path in files:
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    #         print(f"Removed: {file_path}")
    # print(f"=========== Total time taken for all Batches: {end_time-start_time} =========")