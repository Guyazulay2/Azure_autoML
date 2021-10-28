#!/usr/bin/env python
# coding: utf-8

# In[47]:
from enum import auto
from azureml.core import Workspace, Dataset
from azureml.data.dataset_factory import DataType
import argparse
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.widgets import RunDetails
import os


ti = InteractiveLoginAuthentication(tenant_id='2e3734a5-13a8-4988-9236-1c8efdf5cdca')
subscription_id = "def7fc39-4b18-4efa-a893-680f4efebe2c"
resource_group = 'New_subscription'
workspace_name = 'automl_happy'
workspace = Workspace(subscription_id, resource_group, workspace_name,auth=ti)

parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str, help='Enter CSV file ')
parser.add_argument('-name', type=str, help='Enter Dataset name')
args = parser.parse_args()

data_dir = 'C:/Users/User/Desktop/vscode/Azure_autoML/data'
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

datastore = workspace.get_default_datastore()
datastore.upload(src_dir='Azure_autoML/data', target_path=f'Azure_autoML/data/{args.file}')
dataset = Dataset.Tabular.from_delimited_files(path = [(datastore, (f'Azure_autoML/data/{args.file}'))])
file_ds = dataset.register(workspace=workspace, name=args.name, description='New Dataset Uploaded', create_new_version = True)


# dataset = Dataset.get_by_name(workspace, name='new_csv_test')
# dataset.to_pandas_dataframe()

# "C:\Users\User\Desktop\vscode\Azure_autoML"
workspace.write_config(path="C:/Users/User/Desktop/vscode/Azure_autoML", file_name="config.json")
print("** Finisht Upload the data **")


# In[7]:


from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
import pandas as pd
import datetime


d_date = datetime.datetime.now()
reg_format_date = d_date.strftime("%Y-%m-%d--%H:%M:%S")

ws = Workspace.from_config(path="C:/Users/User/Desktop/vscode/Azure_autoML")
experiment_name = f'{args.file[:-4]}'

experiment = Experiment(ws, experiment_name)

output = {}
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index= [''])
outputDf.T


# In[8]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cpu_cluster_name = "CPU-compute"

try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found exsist cluster, Using it ! ")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDART_D2_V2', max_nodes=6)

    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
print("** Finish Check  cpu_cluster**")


# In[12]:
print("in Stage 12")


from azureml.train.automl import AutoMLConfig
import logging


automl_settings = {
    "experiment_timeout_hours": 3.0,
    "enable_early_stopping": False,
    "iteration_timeout_minutes": 45,
    "max_concurrent_iterations": 4,
    "max_cores_per_iteration": -1,
    "featurization": 'auto',
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'classification',
                             primary_metric= "AUC_weighted",
                             model_explainability= True,
                             compute_target=compute_target,
                             blocked_models=['XGBoostClassifier'],
                             label_column_name = '0',
                             training_data = dataset,
                             n_cross_validations=10,
                             **automl_settings
                            )
print("** Started now The train, wait 3 Hours **")


# In[13]:
print("in Stage 13")

from time import sleep

remote_run = experiment.submit(automl_config, show_output = False)
sleep(30)

# In[15]:
print("in Stage 15")

from azureml.widgets import RunDetails
remote_run.wait_for_completion(show_output=True)
print("** wait_for_completion **")

best_run, fitted_model = remote_run.get_output()
best_run_metrics = best_run.get_metrics()


# In[18]:
print("in Stage 18")

from azureml.interpret import ExplanationClient
from azureml.core.run import Run

print("Wait for the best model explanation run to complete")
model_explainability_run_id = remote_run.id + "_" + "ModelExplain"
#print(model_explainability_run_id)
model_explainability_run = Run(experiment=experiment, run_id=model_explainability_run_id)
model_explainability_run.wait_for_completion()
best_run, fitted_model = remote_run.get_output()


# In[19]:
print("in Stage 19")

from azureml.interpret import ExplanationClient
from azureml.widgets import RunDetails



client = ExplanationClient.from_run(best_run)
engineered_explanations = client.download_model_explanation(raw=True)
exp_data = engineered_explanations.get_feature_importance_dict()
exp_data


# In[20]:
print("in Stage 20")

# import datetime
import os

# d_date = datetime.datetime.now()
# reg_format_date = d_date.strftime("%Y-%m-%d--%H:%M:%S")
best_run, fitted_model = remote_run.get_output()

metrics_dir = 'C:/Users/User/Desktop/vscode/Azure_autoML/models_metrics'
if not os.path.isdir(metrics_dir):
    os.mkdir(metrics_dir)

fileee = open(f"C:/Users/User/Desktop/vscode/Azure_autoML/models_metrics/{args.file[:-4]}.txt", "w")
fileee.write(f'{fitted_model}\n')
fileee.close()

best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    file1 = open(f"C:/Users/User/Desktop/vscode/Azure_autoML/models_metrics/{args.file[:-4]}.txt", "a") 
    file1.write(f"\n-----------\n{metric_name}\n{metric}") 
    file1.close()


print("--- Saved all model metrics in txt file ---")

# In[21]:
print("in Stage 21")

import datetime
import os

print("** Exporting The model **\n------------")
d_date = datetime.datetime.now()
reg_format_date = d_date.strftime("%Y-%m-%d--%H:%M:%S")

model_name = best_run.properties['model_name']
script_file_name = 'inference/score.py'
best_run.download_file('outputs/scoring_file_v_1_0_0.py', f'C:/Users/User/Desktop/vscode/Azure_autoML/inference/{args.file[:-4]}score.py')
model_dir='Models'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

best_run.download_file('outputs/model.pkl', f'Models/{args.file[:-4]}.pkl')


# In[22]:
print("in Stage 22")


description = 'AutoML Model trained'
tags = None
model = remote_run.register_model(model_name = model_name, description = description, tags = tags)
print("Finished.")
