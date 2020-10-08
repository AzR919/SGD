import sys
import os
import numpy as np
import pandas
import argparse
import util
import model
import matplotlib.pyplot as plt

import ipdb



if __name__=="__main__":

    parser = argparse.ArgumentParser("Argument parser for SGD")
    parser.add_argument("--train_root", type=str, default='../carseats_train.csv', help="path to the training data")
    parser.add_argument("--test_root", type=str, default='../carseats_test.csv', help="path to the test data")
    parser.add_argument("--regulizer", type=str, default='l', help="which regulizer to use")
    parser.add_argument("--save_dir", type=str, default=None, help='save path for model weights')

    parser.add_argument("--num_epochs", type=int, default='50', help='number of epochs to train for')
    parser.add_argument("--step_size", type=float, default='0.01', help='step size')
    parser.add_argument("--lambdda", type=float, default='0.1', help='regulization term')

    args = parser.parse_args()

    if (not os.path.exists(args.train_root)):
            print("Train data path does not exist")
            exit(1)
    if (not os.path.exists(args.test_root)):
            print("Test data path does not exist")
            exit(1)

    #ipdb.set_trace()

    print("-----------------------------------------")
    print("Start Loading Data")
    print("-----------------------------------------")

    train_raw = pandas.read_csv(args.train_root)
    test_raw = pandas.read_csv(args.test_root)
    train_pre, test_pre = util.pre_process(train_raw, test_raw)
    num_features = len(test_pre.columns) - 1

    print("-----------------------------------------")
    print("Finish Loading Data")
    print("-----------------------------------------")


    my_model = model.model(args, num_features)

    
    losses = my_model.sgd_train(train_pre)

    reg = "" if args.regulizer == "l" else "reg: {}, ".format(args.regulizer)
    lambddaa = "" if args.regulizer == "l" else "lambda: {}".format(args.lambdda)

    title = "Training Loss, num_epochs: {}, {}step_size: {}, {}".format(args.num_epochs, reg, args.step_size, lambddaa)

    plt.plot(losses)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()


    my_model.sgd_eval(test_pre)

    

    print("-----------------------------------------")
    print("All Done")
    print("-----------------------------------------")

