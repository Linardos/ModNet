# ModNet

ModNet was an attempt at a neural network that learns semantical hierarchies from the ImageNet dataset. The purpose of this experiment was to find out if implicitly indicating the semantic hierarchy yields improved performance. The network would first learn to simple recognize animals (canines and felines) contrast to random pictures (flora, rock, geo formations, fungi, construction sites) and after that it would be extended to learn more specific, complex classes from the same dataset.

After the network was trained on animals, two separate extensions were trained on canines and felines and finally a forgetting test was applied to assess the model's performance on the simple task of recognizing animals.



# How to Use

Task_1 : separating animals (canines and felines) from random pictures (flora, rock, geo formations, fungi, construction sites) 

Use set_directories.py to create the directories train, val which split into the two classes 0 and 1. To make a new task change the following parameters within the code:

    path_0 = path_not_animals
    path_1 = path_animals
    PathToTask = './Task_1_Animals/'

To make the number of examples even between the 2 classes, use balance_classes.sh after manually changing it to point to the task directory you want

    Task_1_Animals -> Task_2_Canines etc


Use train.py to run the model on the task, change these parameters:

    TASK_TARGET = [1, 'Animals'] #level of task depth, target class
    prev_model_loader = "Module_1_Animals.pt"
    MODULAR = False
    FREEZE = True
    pooling_num = int(28 / ((TASK_TARGET[0] - 1)*2)) if TASK_TARGET[0] > 1 else 28


Use estimator.py to shuffle between val and train folders and to change validation to train percentage (it will also run train)


