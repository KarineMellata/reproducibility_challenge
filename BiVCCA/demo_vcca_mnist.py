import numpy as np
import math
import os
import tensorflow as tf
import vcca_IM as vcca
from myreadinput import read_mnist


import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--Z", default=10, help="Dimensionality of features", type=int)
parser.add_argument("--H1", default=10, help="Dimensionality of private variables for view 1", type=int) 
parser.add_argument("--H2", default=10, help="Dimensionality of private variables for view 2", type=int)
parser.add_argument("--IM", default=0.0, help="Regularization constant for the IM penalty", type=float)
parser.add_argument("--dropprob", default=0.0, help="Dropout probability of networks.", type=float) 
parser.add_argument("--checkpoint", default="./vcca_mnist", help="Path to saved models", type=str) 
args=parser.parse_args()


def main(argv=None):

    # Set random seeds.
    np.random.seed(0)
    tf.set_random_seed(0)
    
    # Obtain parsed arguments.
    Z=args.Z
    print("Dimensionality of shared variables: %d" % Z)
    H1=args.H1
    print("Dimensionality of view 1 private variables: %d" % H1)
    H2=args.H2
    print("Dimensionality of view 2 private variables: %d" % H2)
    IM_penalty=args.IM
    print("Regularization constant for IM penalty: %f" % IM_penalty)
    dropprob=args.dropprob
    print("Dropout rate: %f" % dropprob)
    checkpoint=args.checkpoint
    print("Trained model will be saved at %s" % checkpoint)

    # Some other configurations parameters for mnist.
    losstype1=0
    losstype2=1
    learning_rate=0.0001
    l2_penalty=0
    latent_penalty=1.0
    
    # Define network architectures.
    network_architecture=dict(
        n_input1=784, # MNIST data input (img shape: 28*28)
        n_input2=784, # MNIST data input (img shape: 28*28)
        n_z=Z,  # Dimensionality of shared latent space
        F_hidden_widths=[1024, 1024, 1024, Z],
        F_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
        n_h1=H1, # Dimensionality of individual latent space of view 1
        G1_hidden_widths=[1024, 1024, 1024, H1],
        G1_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
        n_h2=H2, # Dimensionality of individual latent space of view 2
        G2_hidden_widths=[1024, 1024, 1024, H2],
        G2_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
        H1_hidden_widths=[1024, 1024, 1024, 784],
        H1_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid],
        H2_hidden_widths=[1024, 1024, 1024, 784],
        H2_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid]
        )
    
    # First, build the model.
    model=vcca.VCCA(network_architecture, losstype1, losstype2, learning_rate, l2_penalty, latent_penalty)
    saver=tf.train.Saver()
    
    # Second, load the saved moded, if provided.
    if checkpoint and os.path.isfile(checkpoint):
        print("loading model from %s " % checkpoint)
        saver.restore(model.sess, checkpoint)
        epoch=model.sess.run(model.epoch)
        print("picking up from epoch %d " % epoch)
        tunecost=model.sess.run(model.tunecost)
        print("tuning cost so far:")
        print(tunecost[0:epoch])
    else:
        print("checkpoint file not given or not existent!")

    # File for saving classification results.
    classfile=checkpoint + '_classify.mat'
    if os.path.isfile(classfile):
        print("Job is already finished!")
        return
    
    # Third, load the data.
    trainData,tuneData,testData=read_mnist()
    
    # Traning.
    model=vcca.train(model, trainData, tuneData, saver, checkpoint, batch_size=100, max_epochs=250, save_interval=5, keepprob=(1.0-dropprob))
    
    # SVM linear classification.
    print("Performing linear SVM!")
    trainData,tuneData,testData=read_mnist()
    best_error_tune=1.0
    from sklearn import svm
    for c in [0.1, 1.0, 10.0]:
        lin_clf=svm.SVC(C=c, kernel="linear")
        # train
        svm_x_sample=trainData.images1[::]
        svm_y_sample=np.reshape(trainData.labels[::], [svm_x_sample.shape[0]])
        svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
        # svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
        # svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
        lin_clf.fit(svm_z_sample, svm_y_sample)   
        # dev
        svm_x_sample=tuneData.images1
        svm_y_sample=np.reshape(tuneData.labels, [svm_x_sample.shape[0]])
        svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
        # svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
        # svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
        pred=lin_clf.predict(svm_z_sample)
        svm_error_tune=np.mean(pred != svm_y_sample)
        print("c=%f, tune error %f" % (c, svm_error_tune))
        if svm_error_tune < best_error_tune:
            best_error_tune=svm_error_tune
            bestsvm=lin_clf
    
    # test
    svm_x_sample=testData.images1
    svm_y_sample=np.reshape(testData.labels, [svm_x_sample.shape[0]])
    svm_z_sample, svm_z_std=model.transform_shared(1, svm_x_sample)
    # svm_z_sample_private, _=model.transform_private(1, svm_x_sample)
    # svm_z_sample=np.concatenate([svm_z_sample, svm_z_sample_private], 1)
    pred=bestsvm.predict(svm_z_sample)
    best_error_test=np.mean(pred != svm_y_sample)
    print("tuneerr=%f, testerr=%f" % (best_error_tune, best_error_test))

    
    # TSNE visualization and clustering.
    print("Visualizing shared variables!")
    trainData,tuneData,testData=read_mnist()
    # z_train, _=model.transform_shared(1, trainData.images1)
    z_tune, _=model.transform_shared(1, tuneData.images1)
    z_test, _=model.transform_shared(1, testData.images1)
    from sklearn.manifold import TSNE
    tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
    z_tsne=tsne.fit_transform( np.asfarray(z_test[::2], dtype="float") )
    
    if H1>0:
        print("Visualizing private variables!")
        h_test, _= model.transform_private(1, testData.images1)
        tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=3000)
        h_tsne=tsne.fit_transform( np.asfarray(h_test[::2], dtype="float") )
    else:
        h_test=-1
        h_tsne=-1
    
    
    import scipy.io as sio
    sio.savemat(classfile, {
        'tuneerr':best_error_tune,  'testerr':best_error_test,
        'z_tune':z_tune,  'z_test':z_test,  'z_tsne':z_tsne,
        'h_test':h_test,  'h_tsne':h_tsne
        })


if __name__ == "__main__":
    tf.app.run()

        
