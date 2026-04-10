import os
import shutil
import gc
import re
import fnmatch
import numpy as np
import pandas as pd
from skimage.transform import hough_circle, hough_circle_peaks # type: ignore
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage.filters.rank import mean
from skimage import (
    data, restoration, util
)
from skimage import exposure,filters, color
import skimage
from skimage.util import img_as_ubyte
from skimage.feature import canny
from skimage.draw import circle_perimeter
import skimage.measure as measure
import h5py

from cellpose import io

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings("ignore", message="Bad rank filter performance is expected due to a large number of bins", module="skimage.filters.rank")

def create_batches(grouped_df, batch_size=5):
    return [grouped_df[i:i + batch_size] for i in range(0, len(grouped_df), batch_size)]




def post_process_save(CW_im, GFP_im, scar_im, masks, index, file_list):
    """
    Find equator for each cell image and combine the either 2 halves together
    """
    file = file_list[index]
    result = {'scar_z_crop': { file[0:-4]: {}},
              'scar_crop': {file[0:-4]:{}},
              'GFP_crop': {file[0:-4]:{}},
              'quantification':{}}
    
    print(f"POST PROCESS: Finding equator for image {index}/{len(file_list)}")
    
    df = pd.DataFrame(columns=['cell_ID', 'mask_area','equator','cx','cy','x1','x2','y1','y2'])
    expand_p = 3
    for i in range(1, int(np.max(masks) + 1)):
        if (np.where(masks==i)[0].size >0 and np.where(masks==i)[1].size >0 and 
        (np.where(masks==i)[0].max()-np.where(masks==i)[0].min())>10 and
        (np.where(masks==i)[1].max()-np.where(masks==i)[1].min())>10):
            cx=( np.where(masks==i)[1].max()-np.where(masks==i)[1].min())//2 + np.where(masks==i)[1].min()
            cy=( np.where(masks==i)[0].max()-np.where(masks==i)[0].min())//2 + np.where(masks==i)[0].min()
            x_left = np.where(masks==i)[1].min()-expand_p 
            x_right = np.where(masks==i)[1].max()+expand_p
            y_left = np.where(masks==i)[0].min()-expand_p 
            y_right = np.where(masks==i)[0].max()+expand_p
            df.loc[i,'cell_ID']=i
            df.loc[i,'cx']=cx
            df.loc[i,'cy']=cy
            df.loc[i,'x1']=x_left
            df.loc[i,'x2']=x_right
            df.loc[i,'y1']=y_left
            df.loc[i,'y2']=y_right
            if (cy-32>0 and cy+32<1024 and cx-32>0 and cx+32<1024 and
            x_left>0 and x_right < 1024 and 
            y_left>0 and y_right < 1024):
            
                mask = np.zeros(masks.shape).astype(np.uint8)
                mask[np.where(masks == i)] = 1
                
                crop_mask = mask[y_left : y_right ,x_left:x_right]
                df.loc[i,'mask_area']=np.sum(crop_mask)
                crop_mask = crop_mask*255
                crop_mask64 = mask[cy-32:cy+32,cx-32:cx+32]*255
                result['GFP_crop'][file[0:-4]][str(i) + '_mask.tif'] = crop_mask64
                result['scar_z_crop'][file[0:-4]][str(i) + '_mask.tif'] = crop_mask
                
               
                crop_CW = CW_im[cy-32:cy+32,cx-32:cx+32]
                crop_CW = np.copy(crop_CW)
                crop_CW[crop_mask64==0]=0
                result['GFP_crop'][file[0:-4]][str(i) + '_CW.tif'] = crop_CW
                
                crop_GFP = GFP_im[:,cy-32:cy+32,cx-32:cx+32]
                result['GFP_crop'][file[0:-4]][str(i) + '_GFP.tif'] = crop_GFP
                
                scar_z_crop = scar_im[:,y_left : y_right ,x_left:x_right]
                result['scar_z_crop'][file[0:-4]][str(i) + '_.tif'] = scar_z_crop
                scar_crop_max = np.max(scar_z_crop,axis=0).astype(np.uint16)

                if np.std(scar_crop_max) > 30:
                    score_max = 0
                    equator=0
                    for j in range(int(scar_z_crop.shape[0]/2)):
                        bs = scar_z_crop[2*j,:,:]
                        if np.std(bs) > 50:
                            p0, p99 = np.percentile(bs, (0, 99))
                            img_rescale = exposure.rescale_intensity(bs, in_range=(p0, p99))
                        else:
                            img_rescale = bs
                    
                        edges = canny(img_rescale, sigma=2, low_threshold=1, high_threshold=50)
        
                        # Detect two radii
                        hough_radii = np.arange(20, 30)
                        hough_res = hough_circle(edges, hough_radii).astype(np.float32)
                        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                           total_num_peaks=2)
            
                        if len(accums) == 0:
                            score = 0
                        else:
                            score = round(10*accums[0]+np.average(radii),2)
               
                        if score > score_max:
                            score_max = score
                            equator =2*j
               
                    df.loc[i,'equator']=equator

                    if equator==0:
                        crop_scar_b = scar_z_crop[0,:,:]
                        crop_scar_t = np.max(scar_z_crop[1:-1,:,:],axis=0)
                    if equator>0 and equator<(scar_z_crop.shape[0]-1):
                        crop_scar_b = np.max(scar_z_crop[0:equator,:,:],axis=0)
                        crop_scar_t = np.max(scar_z_crop[equator:-1,:,:],axis=0)
                    if equator == scar_z_crop.shape[0]-1 :
                        crop_scar_b = np.max(scar_z_crop[0:equator,:,:],axis=0)
                        crop_scar_t = scar_z_crop[scar_z_crop.shape[0]-1,:,:]
                    crop_scar_b = np.copy(crop_scar_b)
                    crop_scar_t = np.copy(crop_scar_t)
                    avg_t = mean(crop_scar_t, square(3))      #smooth image
                    background_t = restoration.rolling_ball(avg_t,radius=10)
                    bs_t=avg_t-background_t
                    bs_t[crop_mask==0]=0
                    if np.std(bs_t) > 30:
                        p0_t, p99_t = np.percentile(bs_t, (0, 99))
                        img_rescale_t = exposure.rescale_intensity(bs_t, in_range=(p0_t, p99_t))
                        result['scar_crop'][file[0:-4]][str(i) + '_t.png'] = img_rescale_t                     
                    
                    avg_b = mean(crop_scar_b, square(3))      #smooth image
                    background_b = restoration.rolling_ball(avg_b,radius=10)
                    bs_b=avg_b-background_b
                    bs_b[crop_mask==0]=0
                    if np.std(bs_b) > 30:
                        p0_b, p99_b = np.percentile(bs_b, (0, 99))
                        img_rescale_b = exposure.rescale_intensity(bs_b, in_range=(p0_b, p99_b))
                        
                        result['scar_crop'][file[0:-4]][str(i) + '_b.png'] = img_rescale_b                     

    result['quantification'][file[0:-4]+'.csv'] = df.to_dict(orient='list')
           
    return result



def process_GFP_image(csv_dict, gfp_crop, file, index, total_len):
    
    print(f"GFP Image process and Save {index}/{total_len}")
    

    folder_name = os.path.splitext(file)[0]
    result = {'GFP_z': {folder_name:{}},
              'quantification':{}}
    
    GFP_quantification = pd.DataFrame(csv_dict)
    for i in range(len(GFP_quantification)):
        if GFP_quantification.loc[i,'mask_area'] > 0:
            cell_id = GFP_quantification.loc[i,'cell_ID']
            GFP_crop_name =  str(cell_id) + '_GFP.tif'
            mask_crop_name =  str(cell_id) + '_mask.tif'
            GFP_crop = gfp_crop[GFP_crop_name]
            mask_crop = gfp_crop[mask_crop_name]

            props = measure.regionprops(mask_crop)
            perimeter = props[0].perimeter
            area = props[0].area
            pi = 3.14
            roundness = 4*pi*area/(perimeter*perimeter)
            GFP_quantification.loc[i,'roundness']=roundness
            GFP_crop_z = np.max(GFP_crop,axis=0)
            GFP_crop_z[mask_crop==0]=0
            GFP_quantification.loc[i,'GFP_sum']=np.sum(GFP_crop_z)
            GFP_crop_z = np.max(GFP_crop,axis=0)
            GFP_crop_z = GFP_crop_z/np.max(GFP_crop_z)*255
            GFP_crop_z=GFP_crop_z.astype(np.uint8)
            result['GFP_z'][folder_name][str(cell_id)+'_GFPz.tif'] = GFP_crop_z
            
    result['quantification'][file[0:-4]+'.csv'] = GFP_quantification.to_dict(orient='list')
    return result

def bud_scar_detection(scar_folders, scar_crop_folder_path, bud_scars, model):
    """
    input the folder that contains images as inputs for Budscar Model
    """
    print("Running bud scar detection")
    for scar_folder in scar_folders:
        if os.path.isdir(os.path.join(scar_crop_folder_path,scar_folder)) == True and scar_folder + '.csv' not in list(bud_scars['scar_crop_1.5'].keys()):
            scar_path = os.path.join(scar_crop_folder_path,scar_folder)
            df = pd.DataFrame(columns = ['file','n_scar'])
            if len(os.listdir(scar_path))>0:
                for file in os.listdir(scar_path):
                    if file.endswith('.png')==0:
                        os.remove(os.path.join(scar_path,file))

                results = model(scar_path + '/' ,save=True,save_crop=False,hide_labels=True,hide_conf=True,save_txt=True,conf=0.15)
                
                for r in results:
                    r_df = pd.DataFrame(columns = ['file','n_scar'])
                    r_df.loc[0,'file'] = r.path
                    r_df.loc[0,'n_scar'] = r.__len__()
                    df=pd.concat([df,r_df],ignore_index=True)
            bud_scars['scar_crop_1.5'][scar_folder + '.csv'] = df.to_dict(orient='list')
    return bud_scars




def get_final_quantification_files(quantification_folder_path, all_scar_crop_dict, all_quantification_dict):
    no_scars_files = []
    regex = re.compile(r'\d+')

    GFP_track_quantification = all_quantification_dict
    GFP_quantification_files = list(GFP_track_quantification.keys())

    for GFP_quantification_file in GFP_quantification_files:
        try:
            GFP_quantification = pd.DataFrame(all_quantification_dict[GFP_quantification_file])
            scar_quantification = pd.DataFrame(all_scar_crop_dict[GFP_quantification_file])
            GFP_quantification['n_scar'] = 0

            for i in range(len(scar_quantification)):
                img_name = os.path.basename(scar_quantification.loc[i,'file'] ) 
                cell_id = regex.findall(str(img_name))[0]
                GFP_quantification.loc[GFP_quantification['cell_ID']==int(cell_id),'n_scar']  = scar_quantification.loc[i,'n_scar'] + GFP_quantification.loc[GFP_quantification['cell_ID']==int(cell_id),'n_scar']
            

            GFP_quantification.to_csv(os.path.join(quantification_folder_path, GFP_quantification_file))
            GFP_track_quantification[GFP_quantification_file] = GFP_quantification.to_dict(orient='list')
        except:
            no_scars_files.append(GFP_quantification_file)
    print(f"{len(no_scars_files)}  files has no bud scars based on threshold set in POST Processing function!!!!! \n {no_scars_files}")
    return GFP_track_quantification




def run_summarization(GFP_track_quantification, summary_folder_path):
    all_summary = {'summary':{}}
    wellNames=[]
    for fileName in list(GFP_track_quantification.keys()):
        wellName = fileName.split('_')[1]
        wellNames.append(wellName)
    well_list = np.unique(np.array(wellNames))


    for well in well_list:
        files = [key for key in list(GFP_track_quantification.keys()) if fnmatch.fnmatch(key, 'CW_'+well+'*')]
        

        df = pd.DataFrame()
        for file in files:
            fileName = os.path.basename(file)
            fieldName = fileName.split('_')[2]
            subImName = fileName.split('_')[3][0]
            dff = pd.DataFrame(GFP_track_quantification[file])
            dff['field'] = str(fieldName)
        
            dff['sub-field'] = subImName
            df = pd.concat([df, dff], ignore_index=True)
        df.to_csv(os.path.join(summary_folder_path, f'{well}.csv'),index=None)
        all_summary['summary'][well+'.csv'] = df.to_dict(orient='list')












def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively save dictionary contents to HDF5 group
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, int, float, list)):
            h5file.create_dataset(path + key, data=item)
        elif isinstance(item, str):
            h5file.create_dataset(path + key, data=item, dtype=h5py.special_dtype(vlen=str))
        elif isinstance(item, dict):
            group = h5file.create_group(path + key)
            recursively_save_dict_contents_to_group(group, '', item)
        elif isinstance(item, pd.DataFrame):
            # Convert DataFrame to a dictionary and save
            h5file.create_dataset(path + key, data=np.array(item))
        else:
            print(f"Warning: Unable to save {key} with value {item} of type {type(item)}")


def save_results_to_hdf5(batch_folder_path, results, seg_result_track, bud_scars, GFP_process_results, results_track, batch_file_name):
    """
    Function to save final results of pipeline in hdf5 format for each batch
    """
    hdf5_structure = {
        'CW': results_track['CW'],
        'GFP': results_track['GFP'],
        'scar': results_track['scar'],
        'GFP_SoRa': results_track['GFP_SoRa'],
        'CW_SoRa': results_track['CW_SoRa'],
        'scar_SoRa': results_track['scar_SoRa'],
        'masks': seg_result_track['mask'],
        'scar_crop_1.5': {},
        'scar_z_crop': {},
        'scar_crop': {},
        'GFP_crop': {},
        'quantification':{},
        'GFP_z': {}
    }

    for result in results:
        for key, value in result.items():
            hdf5_structure[key].update(value)

    for key, value in bud_scars.items():
        hdf5_structure[key] = value

    for result in GFP_process_results:
        for key, value in result.items():
            hdf5_structure[key].update(value)
    
    
    batch_filename = os.path.join(batch_folder_path, batch_file_name)
    with h5py.File(batch_filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', hdf5_structure)





def save_results_to_hdf5_after_nd2_load(results, batch_folder_path, batch_index):
    hdf5_structure = {
        'CW': {},
        'GFP': {},
        'scar': {},
        'GFP_SoRa': {},
        'CW_SoRa': {},
        'scar_SoRa': {}
    }

    for result in results:
        for key, value in result.items():
            hdf5_structure[key].update(value)
    
    
        
    batch_filename = os.path.join(batch_folder_path, f'batch_{batch_index}.h5')
    with h5py.File(batch_filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', hdf5_structure)






def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
        else:
            ans[key] = item[()]
    return ans






def create_scar_input_folder_images(results):
    folder_path = os.path.join(os.getcwd(), 'scar_crop')  # Assuming folder_name is in the current working directory
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Previous Folder scar_crop removed successfully.")
    else:
        print(f"Folder scar_crop does not exist.")
    
    os.mkdir('scar_crop')
    scar_crop_folder_path = os.path.join(os.getcwd(), 'scar_crop')
    scar_folders = []

    for result in results:
        for key, inner_dict in result['scar_crop'].items():
            folder_path = key
            if not os.path.exists(folder_path):
                os.makedirs(os.path.join(scar_crop_folder_path, folder_path))
            
            for filename, image_data in inner_dict.items():
                image_path = os.path.join(scar_crop_folder_path, folder_path, filename)
                io.imsave(image_path,image_data)
                scar_folders.append(folder_path)
    return scar_folders, scar_crop_folder_path


def fast_copy(src, dst, buffer_size=1024 * 1024):
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        shutil.copyfileobj(fsrc, fdst, length=buffer_size)