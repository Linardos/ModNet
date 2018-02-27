# ModNet
A Modular Neural Network applied on hierarchical classification of ImageNet data

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


