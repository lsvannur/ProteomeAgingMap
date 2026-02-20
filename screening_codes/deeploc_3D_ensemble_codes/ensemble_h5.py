import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



dir_2d_preds = './predictions_2d/'
dir_3d_preds = './predictions_3d/'

pred_2d_files = sorted(os.listdir(dir_2d_preds))
pred_3d_files = sorted(os.listdir(dir_3d_preds))


for i in range(len(pred_2d_files)):
  print(i)
  print(pred_2d_files[i], pred_3d_files[i])
  df_2d = pd.read_csv(os.path.join(dir_2d_preds, pred_2d_files[i]))
  df_3d = pd.read_csv(os.path.join(dir_3d_preds, pred_3d_files[i]))
  running_batch_name = '_'.join(pred_3d_files[i].split('_')[:-1])

  mylabels = ['ER', 'Golgi', 'actin', 'bud_neck', 'cell_periphery', 'cytoplasm', 'endosome', 'lipid_particle', 'mitochondria', 'none', 'nuclear_periphery', 'nucleolus', 'nucleus', 'peroxisome', 'spindle_pole', 'vacuolar_membrane','vacuole']
  ensemble_df = pd.DataFrame()
  for l in mylabels:
    ensemble_df[l] = 0.5*df_2d[l] + 0.5*df_3d[l]

  ensemble_preds = np.argmax(np.array(ensemble_df), axis=1)
  true_labels=df_2d['true_labels']


  acc_score = accuracy_score(true_labels, ensemble_preds)
  f1 = f1_score(true_labels, ensemble_preds, average='weighted')
  precision = precision_score(true_labels, ensemble_preds, average=None)
  recall = recall_score(true_labels, ensemble_preds, average=None)


  print(f"Accuracy: {acc_score}")
  print(f"F1 Score: {f1}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")


  ensemble_df['ensemble_pred'] = ensemble_preds
  ensemble_df['true_labels'] = true_labels
  ensemble_df['gfp_image_names'] = df_3d['gfp_image_names']
  ensemble_df['cw_well_names'] = df_3d['cw_well_names']
  ensemble_df['combined'] = df_3d['combined']


  ensemble_df.to_csv(f'../predictions_ensemble/{running_batch_name}_ensemble.csv', index=None)


  df_2d['gfp_image_names'] = df_3d['gfp_image_names']
  df_2d['cw_well_names'] = df_3d['cw_well_names']
  df_2d['combined'] = df_3d['combined']
  df_2d.to_csv(f'{dir_2d_preds}/{pred_2d_files[i]}', index=None)