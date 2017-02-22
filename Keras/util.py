import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
from sklearn.metrics import confusion_matrix
import numpy as np
import os.path

class Util(object):
    
    def plot_image(self, image, img_shape=(28,28)):
        plt.imshow(image.reshape(img_shape),
                   interpolation='nearest',
                   cmap='binary')

        plt.show()
    
    def plot_some_images(self, images, img_shape=(28,28), grid_x=3, grid_y=3):
        fig, axes = plt.subplots(grid_y, grid_x)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].reshape(img_shape),
                       interpolation='nearest',
                       cmap='binary')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    
    def plot_images(self, images, cls_true, cls_pred=None, img_shape=(28,28), interpolation='none', with_labels=False, labels=[]):
        if not (len(images) == len(cls_true) == 9):
            print("\nI cannot plot the graph because there less than 9 points or because the dimension of images and cls_true mismatch")
            return
        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary', interpolation=interpolation)
            # Show true and predicted classes.
            if cls_pred is None:
                if not with_labels:
                    xlabel = "Class: {0}".format(cls_true[i])
                else: 
                    xlabel = "Class: {0}".format(labels[cls_true[i]])
            else:
                if not with_labels:
                    xlabel = "Class: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
                else: 
                    xlabel = "Class: {0}, Pred: {1}".format(labels[cls_true[i]], labels[cls_pred[i]])
            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def maybe_save_network(self, model, model_image_path='images/temp.png'):
        if not os.path.exists(model_image_path):
            plot(model, to_file=model_image_path)
        return model_image_path
    
    def plot_history(self, history, metric='acc', loc='lower right'): 
        # list all data in history
        # print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric])
        if metric == 'acc': 
            metric = 'accuracy'
        plt.title('model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc=loc)
        plt.show()
        
    def plot_confusion_matrix(self, y_true, num_classes, cls_pred, with_labels=False, labels=[], text=True):
        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=y_true,
                              y_pred=cls_pred)
        # Plot the confusion matrix as an image.
        plt.matshow(cm)
        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        if with_labels:
            plt.xticks(range(num_classes), labels, rotation='vertical')
            plt.yticks(range(num_classes), labels)
        else:
            plt.xticks(tick_marks, range(num_classes))
            plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        # Print the confusion matrix as text.
        if text:
            print(cm)