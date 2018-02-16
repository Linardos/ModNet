import pickle
import matplotlib.pyplot as plt

Task_Target = '2_Canines'

# To Load a file

"""
with open('model_config_and_performance_{}.pkl'.format(task_number), 'rb') as handle:

    hyperparameters, to_plot = pickle.load(handle)
"""
with open('model_config_and_performance_{}.pkl'.format(Task_Target), 'rb') as handle:

    hyperparameters, to_plot = pickle.load(handle)


plt.subplot(221)
plt.plot(to_plot['epoch_ticks'], to_plot['train_losses'], label = "Training Loss")
plt.plot(to_plot['epoch_ticks'], to_plot['test_losses'], label = "Test Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(to_plot['epoch_ticks'])

plt.subplot(222)
plt.plot(to_plot['epoch_ticks'], to_plot['train_accuracies'], label = "Training Accuracy")
plt.plot(to_plot['epoch_ticks'], to_plot['test_accuracies'], label = "Test Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(to_plot['epoch_ticks'])

plt.subplot(223)
plt.plot(to_plot['epoch_ticks'], to_plot['train_sensitivities'], label = "Training Sensitivity")
plt.plot(to_plot['epoch_ticks'], to_plot['test_sensitivities'], label = "Test Sensitivity")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Sensitivity")
plt.xticks(to_plot['epoch_ticks'])

plt.subplot(224)
plt.plot(to_plot['epoch_ticks'], to_plot['train_specificities'], label = "Training Specificity")
plt.plot(to_plot['epoch_ticks'], to_plot['test_specificities'], label = "Test Specificity")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Specificity")
plt.xticks(to_plot['epoch_ticks'])

plt.tight_layout()

plt.savefig('Performance_{}'.format(Task_Target))


for key in to_plot.keys():
    print("Final value in {} is {} \n".format(key, to_plot[key][-1]))
