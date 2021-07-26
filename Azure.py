#!/usr/bin/env python
# coding: utf-8

# In[47]:
from azureml.core import Workspace, Dataset
from azureml.data.dataset_factory import DataType
import argparse
from azureml.widgets import RunDetails


parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str, help='Enter CSV file ')
parser.add_argument('-name', type=str, help='Enter Dataset name')
args = parser.parse_args()


subscription_id = '05ed07e3-df05-46ff-8687-8555449022bb'
resource_group = 'ML-resources'
workspace_name = 'ML-resources'
workspace = Workspace(subscription_id, resource_group, workspace_name)


datastore = workspace.get_default_datastore()


datastore.upload(src_dir='data', target_path='data')
dataset = Dataset.Tabular.from_delimited_files(path = [(datastore, (f'data/{args.file}'))])

file_ds = dataset.register(workspace=workspace, name=args.name, description='New Dataset Uploaded')


# dataset = Dataset.get_by_name(workspace, name='new_csv_test')
# dataset.to_pandas_dataframe()


workspace.write_config(path="/home/yossi/Desktop/new", file_name="config.json")
print("** Finisht Upload the data **")


# In[7]:


from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
import pandas as pd

ws = Workspace.from_config()
experiment_name = 'test'

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

cpu_cluster_name = "GPU-compute"

try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found exsist cluster, Using it ! ")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDART_D2_V2', max_nodes=6)

    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
print("** Finish Check  Gpu_cluster**")


# In[12]:


from azureml.train.automl import AutoMLConfig
import logging


automl_settings = {
    "experiment_timeout_hours": 0.3,
    "enable_early_stopping": True,
    "iteration_timeout_minutes": 5,
    "max_concurrent_iterations": 1,
    "max_cores_per_iteration": 1,
    "featurization": 'auto',
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'classification',
                             primary_metric= 'AUC_weighted',
                             compute_target=compute_target,
                             blocked_models=['TensorFlowLinearClassifier'],
                             label_column_name = '0',
                             training_data = dataset,
                             n_cross_validations=2,
                             **automl_settings
                            )
print("** Started now The train, whit 30 min **")


# In[13]:


remote_run = experiment.submit(automl_config, show_output = False)


# In[15]:

from azureml.widgets import RunDetails
remote_run.wait_for_completion(show_output=True)
print("** wait_for_completion **")


# In[16]:

best_run, fitted_model = remote_run.get_output()
print(best_run)
print(fitted_model)


# In[18]:

from azureml.interpret import ExplanationClient
from azureml.core.run import Run

print("Wait for the best model explanation run to complete")
model_explainability_run_id = remote_run.id + "_" + "ModelExplain"
print(model_explainability_run_id)
model_explainability_run = Run(experiment=experiment, run_id=model_explainability_run_id)
model_explainability_run.wait_for_completion()

# Get the best run object
best_run, fitted_model = remote_run.get_output()


# In[19]:

from azureml.interpret import ExplanationClient

client = ExplanationClient.from_run(best_run)
engineered_explanations = client.download_model_explanation(raw=True)
exp_data = engineered_explanations.get_feature_importance_dict()
exp_data

# In[20]:

best_run, fitted_model = remote_run.get_output()


# In[21]:
import datetime
import os

print("** Exporting The model **")
d_date = datetime.datetime.now()
reg_format_date = d_date.strftime("%Y-%m-%d--%H:%M:%S")

model_name = best_run.properties['model_name']
script_file_name = 'inference/score.py'
best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'inference/score.py')
model_dir = 'Model' # Local folder where the model will be stored temporarily
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
    
best_run.download_file('outputs/model.pkl', f'{model_dir}/{args.file}-{reg_format_date}.pkl')


# In[22]:


description = 'AutoML Model trained'
tags = None
model = remote_run.register_model(model_name = model_name, description = description, tags = tags)
print(remote_run.model_id)

