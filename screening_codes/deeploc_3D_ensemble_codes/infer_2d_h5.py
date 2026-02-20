import tensorflow as tf
import os
import preprocess_images as procIm
import numpy as np
import pandas as pd
import copy

import matplotlib
import matplotlib.pyplot as plt

import cv2
import h5py
tf.compat.v1.disable_eager_execution()
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def getSpecificChannels(flatImageData,channels,imageSize=64):
    return np.hstack(([flatImageData[:,c*imageSize**2:(c+1)*imageSize**2] for c in channels]))

def processBatch(curBatch):
    curImages = getSpecificChannels(curBatch['data'],[0,1])
    curLabels = curBatch['Index'][:]

    cropSize = 60
    stretchLow = 0.0 # stretch channels lower percentile
    stretchHigh = 100.0 # stretch channels upper percentile
    imSize = 64
    numChan = 2
    processedBatch=procIm.preProcessImages(curImages,
                               imSize,cropSize,numChan,
                               rescale=False,stretch=True,
                               means=None,stds=None,
                               stretchLow=stretchLow,stretchHigh=stretchHigh,
                               jitter=True,randTransform=True)
    return {'data':processedBatch,'Index':curLabels}

def processBatchTest(curBatch):
    #print(curBatch)
    curImages = getSpecificChannels(curBatch['data'],[0,1])
    curLabels = curBatch['Index'][:]

    cropSize = 60
    stretchLow = 0.0 # stretch channels lower percentile
    stretchHigh = 100.0 # stretch channels upper percentile
    imSize = 64
    numChan = 2
    processedBatch=procIm.preProcessTestImages(curImages,
                               imSize,cropSize,numChan,
                               rescale=False,stretch=True,
                               means=None,stds=None,
                               stretchLow=stretchLow,stretchHigh=stretchHigh)
    return {'data':processedBatch,'Index':curLabels}


def proccessCropsLoc(processedBatch,predicted_y,inputs,is_training,keep_prob,sess):
    crop_list = np.zeros((len(processedBatch), 5, 17))
    for crop in range(5):
        images = processedBatch[:, crop, :, :, :]

        tmp = copy.copy(sess.run([predicted_y], feed_dict={inputs: images, is_training: False,keep_prob:1.0}))
        #print(tmp)
        crop_list[:, crop, :] = tmp[0]

    mean_crops = np.mean(crop_list, 1)
    return mean_crops



def read_and_combine_datasets(train_path, test_path):
    with h5py.File(train_path, 'r') as trainH5:
        print(trainH5.keys())
        train_data = trainH5['data1'][:]
        train_index = trainH5['Index1'][...]
    return {'data': train_data, 'Index': train_index}


def eval(combined_batch, locNetCkpt, running_batch_name, savepath, hdf5file, miniBatchSize=8):
    localizationTerms = ['ER', 'Golgi', 'actin', 'bud_neck', 'cell_periphery', 'cytoplasm', 'endosome', 'lipid_particle', 'mitochondria', 'none', 'nuclear_periphery', 'nucleolus', 'nucleus', 'peroxisome', 'spindle_pole', 'vacuolar_membrane', 'vacuole']

    # Load the TensorFlow session and model
    locSession = tf.compat.v1.Session()
    loc_saver = tf.compat.v1.train.import_meta_graph(locNetCkpt + '.meta')
    loc_saver.restore(locSession, locNetCkpt)
    loc = tf.compat.v1.get_default_graph()

    # Get the tensors for predictions and penultimate layer
    pred_loc = loc.get_tensor_by_name(u'pred_layer:0')  # Final layer (classification output)
    penultimate_layer = loc.get_tensor_by_name(u'fc_2/activation:0')  # Penultimate layer before classification

    input_loc = loc.get_tensor_by_name(u'inputs:0')
    is_training_loc = loc.get_tensor_by_name(u'is_training:0')
    keep_prob_loc = loc.get_tensor_by_name(u'Placeholder:0')

    # Prepare to store predictions, penultimate layer outputs, and true labels
    allPred = pd.DataFrame(np.zeros((len(combined_batch['data']), len(localizationTerms)), dtype=float), columns=localizationTerms)
    penultimate_layer_outputs = []  # To store outputs of the penultimate layer
    true_labels = []
    preds = []

    # Process data in batches
    num_samples = combined_batch['data'].shape[0]
    for batch_start in range(0, num_samples, miniBatchSize):
        print(f'Batch: {batch_start}/{num_samples}')
        batch_end = min(batch_start + miniBatchSize, num_samples)
        curBatch = {'data': combined_batch['data'][batch_start:batch_end], 'Index': combined_batch['Index'][batch_start:batch_end]}

        # Process the current batch
        testBatchAll = processBatchTest(curBatch)
        processedBatch = testBatchAll['data']

        # Reshape the data to remove the crop dimension (combine crops into the batch size)
        batch_size = processedBatch.shape[0]
        num_crops = processedBatch.shape[1]
        reshaped_data = np.reshape(processedBatch, (batch_size * num_crops, 60, 60, 2))  # Flatten crops into batch size

        # Get predictions and penultimate layer outputs for the reshaped batch
        predictedBatch_Loc, penultimate_batch = locSession.run([pred_loc, penultimate_layer],
                                                               feed_dict={input_loc: reshaped_data,
                                                                          is_training_loc: False,
                                                                          keep_prob_loc: 1.0})

        # Reshape the predictions back to original batch size (average predictions across crops)
        predictedBatch_Loc = np.reshape(predictedBatch_Loc, (batch_size, num_crops, -1))
        predictedBatch_Loc = np.mean(predictedBatch_Loc, axis=1)  # Average predictions over the crops

        # Update predictions and true labels
        allPred.iloc[batch_start:batch_start + len(predictedBatch_Loc), :] = predictedBatch_Loc
        true_labels.extend(np.argmax(np.array(testBatchAll['Index']), axis=1))
        preds.extend(np.argmax(predictedBatch_Loc, axis=1))

        # Collect penultimate layer outputs for this batch (average across crops)
        penultimate_batch = np.reshape(penultimate_batch, (batch_size, num_crops, -1))
        penultimate_layer_outputs.extend(np.mean(penultimate_batch, axis=1))  # Average across crops

    # Compute metrics after processing all batches
    true_labels = np.array(true_labels)
    preds = np.array(preds)

    acc_score = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    precision = precision_score(true_labels, preds, average=None)
    recall = recall_score(true_labels, preds, average=None)

    # Output results
    print(f"Accuracy: {acc_score}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Save the prediction dataframe
    allPred['true_labels'] = true_labels
    allPred['preds'] = preds
    basename_pred = hdf5file.split('.')[0]
    allPred.to_csv(f"{savepath}/predictions_2d/{basename_pred}_predprob_2d.csv", index=None)

    # Save the penultimate layer outputs to a file (optional)
    penultimate_layer_outputs = np.array(penultimate_layer_outputs)
    pen_outputs= pd.DataFrame(penultimate_layer_outputs)
    pen_outputs['true_labels'] = true_labels
    pen_outputs['preds'] = preds
    basename_penult = hdf5file.split('.')[0]
    pen_outputs.to_csv(f'{savepath}/penult_2d/penult_{basename_penult}_2d.csv', index=None)

    locSession.close()
    return



#################################### MAIN ##########################
def main(_):
    with open('infer_config_h5.yaml', 'r') as file:
        config = yaml.safe_load(file)

    input_conf = config['input']
    running_batch_name = input_conf['running_batch_name']
    locNetCkpt = input_conf['modelpath_2d']
    print(locNetCkpt+'.meta')
    if not os.path.exists(locNetCkpt+'.meta'):
        raise NameError('please download pretrained model')

    hdf5dir = './hdf5_2d_inputs/'
    savepath = '../deeploc_3D_emsemble_model/'

    
    for hdf5file in os.listdir(hdf5dir):
        print(hdf5file)
    
        running_batch_name = hdf5file
        data_path = os.path.join(hdf5dir, hdf5file)
        combined_batch = read_and_combine_datasets(data_path, test_path=None)
        eval(combined_batch, locNetCkpt, running_batch_name, savepath, hdf5file)

if __name__ == '__main__':
    tf.compat.v1.app.run()  