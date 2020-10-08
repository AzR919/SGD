import numpy as np
import pickle
import os
from datetime import datetime


class model():
    def __init__(self, args, num_features):
        """
        @brief:= initialize the model
        @in:= args: the cmd args
              num_features: the number of features
        """

        self.num_features = num_features
        self.reg = args.regulizer
        self.epochs = args.num_epochs
        self.eta = args.step_size
        self.lambdda = args.lambdda

        self.weights = np.zeros(num_features)
        self.bias = 0 

        self.save_file = (os.path.join(args.save_dir, "saved_weights_{}_{}_{}_{}.pkl".format(self.reg, self.epochs, self.eta, self.lambdda))) if args.save_dir is not None else None


    def sgd_train(self, train_data):
        """
        @brief:= the train loop with trains the weights
        @in:= train_data: the data to be trained on
        @out:= list of loss per epoch
        """

        print("-----------------------------------------")
        print("Begin Train")
        print("-----------------------------------------")


        train_rows = train_data.index.values

        losses = []

        for epoch in range(self.epochs):
            
            for index in train_rows:
                true_val = train_data["Sales"][index]
                pred_val = self.sgd_predict(train_data.values[index][1:])

                loss_index = self.sgd_loss(true_val, pred_val)
                losses.append(loss_index)

                if epoch % 10 == 0 and index == 0:
                    print("Epoch num: {}/{}".format(epoch, self.epochs))
                    print("Loss of this index: {}".format(loss_index))


                self.sgd_step(train_data.values[index][1:], true_val, pred_val)



        self.sgd_save()

        print("-----------------------------------------")
        print("End Train")
        print("-----------------------------------------")

        return losses


    def sgd_predict(self, features):
        return sum(self.weights * features) + self.bias

    def sgd_loss(self, true, pred, eval=False):
        """
        @brief:= calculates the loss
        @in:= true: the true labels
              pred: the predicted labels
              eval: whether this is to calculate loss for test set or not
        """
        if eval:
            return (true-pred)**2

        if self.reg == "l":
            return (true-pred)**2
        if self.reg == "l1":
            return (true-pred)**2 + self.lambdda * sum(np.abs(self.weights))
        if self.reg == "l2":
            return (true-pred)**2 + self.lambdda * sum(self.weights*self.weights)


    def sgd_step(self, data, true, pred):
        """
        @brief:= step in the gradient descent
        @in:= data: the x values
              true: the true labels
              pred: the predicted labels
        """

        if self.reg == "l":
            self.weights = self.weights - (self.eta * 2 * (pred - true) * data)
        if self.reg == "l2":
            self.weights = self.weights - (self.eta * (2 * (pred - true) * data + self.lambdda * 2 * self.weights))
        if self.reg == "l1":
            self.weights = self.weights - (self.eta * (2 * (pred - true) * data + self.lambdda * np.array([0.0 if x == 0 else 1.0 if x > 0 else -1.0 for x in self.weights])))
        
        self.bias = self.bias - (self.eta * 2 * (pred - true))

    def sgd_save(self):

        """
        @brief:= saves the weights if asked
        """

        if self.save_file == None:
            return
        print("-----------------------------------------")
        print("Saving params")
        print("-----------------------------------------")
        pickle.dump((self.weights, self.bias), open(self.save_file, 'wb'))

    def sgd_load(self):
        """
        @brief:= loads the weights if asked
        """
        if self.save_file == None:
            return
        print("-----------------------------------------")
        print("Loading params")
        print("-----------------------------------------")
        (self.weights, self.bias) = pickle.load(open(self.save_file, 'rb'))


    def sgd_eval(self, test_data):
        """
        @brief:= evaluates the accuracy on the test data
        @in:= the test data
        """

        test_rows = test_data.index.values
        loss_total = 0
        for index in test_rows:
            true_val = test_data["Sales"][index]
            pred_val = self.sgd_predict(test_data.values[index][1:])

            loss_index = self.sgd_loss(true_val, pred_val, eval=True)
            loss_total += loss_index


        loss_test = loss_total/len(test_rows)

        print("-----------------------------------------")
        print("Eval Loss: {}".format(loss_test))
        print("-----------------------------------------")


