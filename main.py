### Need some extra paths
import sys, os
sys.path.append('modules')
sys.path.append('dense_correspondence/dataset')

### Set a few environment variables that would've been normally set in Docker
os.environ["DC_SOURCE_DIR"] = os.getcwd()
os.environ["DC_DATA_DIR"] = "/home/michelism/Data/pdc"

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import logging

#utils.set_default_cuda_visible_devices()
#utils.set_cuda_visible_devices([0]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
logging.basicConfig(level=logging.INFO)

from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation


import numpy as np
np.random.seed(42)  # Even this doesn't help... It's absolute chaos random.


config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', 'caterpillar_upright.yaml')
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'training', 'training.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)
dataset = SpartanDataset(config=config)

logging_dir = "trained_models/tutorials"
num_iterations = 10000
d = 3 # the descriptor dimension
name = "caterpillar_%d" %(d)
train_config["training"]["logging_dir_name"] = name
train_config["training"]["logging_dir"] = logging_dir
train_config["dense_correspondence_network"]["descriptor_dimension"] = d
train_config["training"]["num_iterations"] = num_iterations

TRAIN = True
EVALUATE = True



# All of the saved data for this network will be located in the
# code/data/pdc/trained_models/tutorials/caterpillar_3 folder

if TRAIN:
    #print "training descriptor of dimension %d" %(d)
    train = DenseCorrespondenceTraining(dataset=dataset, config=train_config)
    train.run()
    #print "finished training descriptor of dimension %d" %(d)


model_folder = os.path.join(logging_dir, name)
model_folder = utils.convert_data_relative_path_to_absolute_path(model_folder)

if EVALUATE:
    DCE = DenseCorrespondenceEvaluation
    num_image_pairs = 100
    DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs)    

    from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluationPlotter as DCEP

    import matplotlib.pyplot as plt

    dc_data_dir = utils.get_data_dir()

    folder_name = "tutorials"
    net_to_plot = os.path.join(folder_name, "caterpillar_3")

    network_name = net_to_plot
    path_to_csv = os.path.join(dc_data_dir, "trained_models", network_name, "analysis/train/data.csv")
    fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, save=False)

    path_to_csv = os.path.join(dc_data_dir, "trained_models", network_name, "analysis/train/data.csv")
    fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, previous_fig_axes=fig_axes, save=False)

    plt.savefig("Training.png")


    path_to_csv = os.path.join(dc_data_dir, "trained_models", network_name, "analysis/test/data.csv")
    fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, save=False)

    path_to_csv = os.path.join(dc_data_dir, "trained_models", network_name, "analysis/test/data.csv")
    fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, previous_fig_axes=fig_axes, save=False)

    plt.savefig("Test.png")


    path_to_csv = os.path.join(dc_data_dir, "trained_models", network_name, "analysis/cross_scene/data.csv")
    fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, save=False)

    path_to_csv = os.path.join(dc_data_dir, "trained_models", network_name, "analysis/cross_scene/data.csv")
    fig_axes = DCEP.run_on_single_dataframe(path_to_csv, label=network_name, previous_fig_axes=fig_axes, save=False)

    plt.savefig("CrossScene.png") 




    config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 
                               'dense_correspondence', 'evaluation', 'evaluation.yaml')
    config = utils.getDictFromYamlFilename(config_filename)
    default_config = utils.get_defaults_config()

    # utils.set_cuda_visible_devices([0])
    dce = DenseCorrespondenceEvaluation(config)
    DCE = DenseCorrespondenceEvaluation

    network_name = "caterpillar_3"
    dcn = dce.load_network_from_config(network_name)
    dataset = dcn.load_training_dataset()
    DenseCorrespondenceEvaluation.evaluate_network_qualitative(dcn, dataset=dataset, randomize=True)