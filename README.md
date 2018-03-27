# ModNet

ModNet was an attempt at a neural network that learns semantical hierarchies from the ImageNet dataset. The purpose of this experiment was to find out if implicitly indicating the semantic hierarchy yields improved performance. The network would first learn to simple recognize animals (canines and felines) contrast to random pictures (flora, rock, geo formations, fungi, construction sites) and after that it would be extended to learn more specific, complex classes from the same dataset.

After the network was trained on animals, two separate extensions were trained on canines and felines and finally a forgetting test was applied to assess the model's performance on the simple task of recognizing animals.

# Performance on Animals
![alt text](https://github.com/Linardos/ModNet/blob/master/Results/plot1.png)

Figure 1: Training Results on the first level task with target class Animals. 5950 samples
were loaded from each class, split 70/30 between train and test set. So far this a small
ResNet architecture, this step is for the network to learn the highest class in the hierarchy:
Animals, nothing novel was used here.

# Performance on Canines and Felines
![alt text](https://github.com/Linardos/ModNet/blob/master/Results/plot2.png)

Figure 2: Training Results on a second level task with target class Canines the pretrained
layers were loaded from the previous task on all layers. 3829 samples were loaded from
each class, split 80/20 between train and test set. a) Different initial learning rates were
used per layer, the closer to the start the smaller the learning rate (learning rate for layer3
was 0.1, for layer2 0.1*0.65, for layer1 0.1*0.4). There is quite clearly overfit in both cases

![alt text](https://github.com/Linardos/ModNet/blob/master/Results/plot3.png)

Figure 3: I used the preprocessing transformations this time on both the train and test set.
These transformations are Random Resized Crop and Random Horizontal Flip. Overfit is
drastically reduced. Training on the Frozen model was highly unstable so I increased the
epoch number for this one

![alt text](https://github.com/Linardos/ModNet/blob/master/Results/plot4.png)

Figure 4: This table demonstrates the results from a ”forgetting” test. After training the
network on Task 2 I decided to look at whether the original layers remember the first task.
So I loaded the first layers of the network to predict classes from Task 1. I gathered all of
the data from the previous task and had the networks try and predict whether they are
seeing an animal or not. The fully connected layer from the previous module was loaded
and the convolutional layers from the new module. Interestingly decreasing the learning
rates of the first layers seems to worsen forgetting of the first task. Also I was expecting
the layers that were frozen when learning task 2 to have the same performance as the
original model as it is essentially the same. I can only think that some layer was missed
during freezing to explain that.

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


