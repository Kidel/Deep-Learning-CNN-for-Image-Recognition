# code refactored from Magnus Erik Hvass Pedersen tutorials

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import time
from datetime import timedelta

class UtilTransfer(object):
    
    def random_batch(self, transfer_values_train, train_batch_size, labels_train):
        num_images = len(transfer_values_train)
        # Create a random index.
        idx = np.random.choice(num_images,
                               size=train_batch_size,
                               replace=False)
        # Use the random index to select random x and y-values.
        x_batch = transfer_values_train[idx]
        y_batch = labels_train[idx]
        return x_batch, y_batch
    
    def optimize(self, num_iterations, transfer_values_train, train_batch_size, labels_train, session, global_step, optimizer, accuracy):
        start_time = time.time()

        for i in range(num_iterations):
            # Get a batch of training examples.
            x_batch, y_true_batch = self.random_batch(transfer_values_train, train_batch_size, labels_train)
            # Put the batch into a dict with the proper names
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
            # Run the optimizer using this batch of training data.
            # We also want to retrieve the global_step counter.
            i_global, _ = session.run([global_step, optimizer],
                                      feed_dict=feed_dict_train)
            # Print status to screen
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                batch_acc = session.run(accuracy,
                                        feed_dict=feed_dict_train)
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))

        end_time = time.time()
        time_dif = end_time - start_time
        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        
    def plot_example_errors(self, cls_pred, correct, images_test, cls_test, plot_images, images):
        # This function is called from print_test_accuracy() below.
        incorrect = (correct == False)
        images = images_test[incorrect]
        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]
        # Get the true classes for those images.
        cls_true = cls_test[incorrect]
        # Plot the first 9 images.
        plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])
       
    def plot_confusion_matrix(self, cls_pred, cls_test, num_classes, class_names):
        # This is called from print_test_accuracy() below.
        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                              y_pred=cls_pred)  # Predicted class.
        # Print the confusion matrix as text.
        for i in range(num_classes):
            # Append the class-name to each line.
            class_name = "({}) {}".format(i, class_names[i])
            print(cm[i, :], class_name)
        # Print the class-numbers for easy reference.
        class_numbers = [" ({0})".format(i) for i in range(num_classes)]
        print("".join(class_numbers))
        # Plot the confusion matrix as an image.
        plt.matshow(cm)
        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()
        
    def predict_cls(self, transfer_values, labels, cls_true, num_images, batch_size, session, y_pred_cls):
        # Number of images.
        num_images = len(transfer_values)
        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        # Now calculate the predicted classes for the batches.
        i = 0
        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_images)
            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {x: transfer_values[i:j],
                         y_true: labels[i:j]}
            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
            # Set the start-index for the next batch to the end-index of the current batch.
            i = j
        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)
        return correct, cls_pred

    def predict_cls_test(self, transfer_values_test, labels_test, cls_test, num_images, batch_size, session, y_pred_cls):
        return self.predict_cls(transfer_values_test, labels_test, cls_test, num_images, batch_size, session, y_pred_cls)
    
    def classification_accuracy(self, correct):
        return correct.mean(), correct.sum()
    
    def print_test_accuracy(self, show_example_errors, show_confusion_matrix, transfer_values_test, labels_test, cls_test, num_images, batch_size, session, y_pred_cls):
        # For all the images in the test-set,
        # calculate the predicted classes and whether they are correct.
        correct, cls_pred = self.predict_cls_test(transfer_values_test, labels_test, cls_test, num_images, batch_size, session, y_pred_cls)
        # Classification accuracy and the number of correct classifications.
        acc, num_correct = self.classification_accuracy(correct)
        # Number of images being classified.
        num_images = len(correct)
        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, num_correct, num_images))
        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct)
        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred)