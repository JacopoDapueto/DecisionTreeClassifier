from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as la
import scipy.linalg as linalg


# ...functions to plot and manipulate data..

# data must be in the form [X, Y], where X is a (n,d) ndarray and Y is a (n,1) ndarray. they are stored in outX and outY
def separatingXY(data, outX, outY): 
        for y in data:
            for pred in y[1]:
                outY.append(pred)
        for x in data:
            for pred in x[0]:
                outX.append(pred)
        return outX, outY

# sum the values in data and distribuctionDictionary with the same key
def dictDistribution(data, distribuctionDictionary): # data = Dictionary
        for i in data.items():
            distribuctionDictionary[i[0]] = distribuctionDictionary.get(i[0], 0.0) + i[1]
        return distribuctionDictionary

def plotErrorBestDepthMethod(Vm, Vs, Tm, Ts, depth_list):
    fig = plt.figure(figsize=(10,8)) 
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)

    ax0.set_title('Mean error')
    ax0.semilogx(depth_list, Tm,'-o' , label='Training error: MEAN')
    ax0.semilogx(depth_list, Vm, '-o',label='Validation error: MEAN')
    ax0.set_xlabel("max_depth")
    ax0.set_ylabel("Error")
    ax0.legend()
    ax0.grid()
    
    ax1.set_title('Variance error')
    ax1.semilogx(depth_list, Ts,'-o' , label='Training error: VARIANCE')
    ax1.semilogx(depth_list, Vs, '-o',label='Validation error: VARIANCE')
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax1.grid()

def plotErrorBestTreeMethod(Pe, Te):
    tree_list = range(Pe.size)
    fig = plt.figure(figsize=(10,8)) 
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)

    ax0.set_title('Pruning set error')
    ax0.semilogx(tree_list, Pe,'-o' )
    ax0.set_xlabel("nTree")
    ax0.set_ylabel("Error")
    ax0.grid()
    
    ax1.set_title('Training set error')
    ax1.semilogx(tree_list, Te,'-o')
    ax1.set_xlabel("nTree")
    ax1.set_ylabel("Error")
    ax1.grid()

def Area( DT, Xtr, Ytr, Xte, Yte, cm = plt.cm.RdBu, cm_bright = ListedColormap(['#FF0000', '#0000FF'])):
    xi = np.arange(Xtr[:, 0].min() - 0.5, Xtr[:, 0].max() + 0.5, 0.02 ) 
    yi = np.arange(Xtr[:, 1].min() - 0.5, Xtr[:, 1].max() + 0.5, 0.02 ) 
    X, Y = np.meshgrid(xi,yi)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Xt_p, Y_pred = DT.prediction(np.c_[X.ravel(),Y.ravel()])
    Y_pred = Y_pred.reshape(X.shape)
    plt.contourf(X, Y, Y_pred, cmap=cm, alpha=.8)
    
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=Ytr, cmap=cm_bright, edgecolors='k', alpha = 0.5, label= 'Training set')
    plt.scatter(Xte[:, 0], Xte[:, 1], c=Yte, cmap=cm_bright, edgecolors='k', alpha = 1.0, label= 'Test set')

    plt.title("Decision surface")
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.tight_layout()
    plt.legend()

def separatingFLR(Xtr, Ytr, Ypred, method_type, nLevels):
    
    xi = np.arange(Xtr[:, 0].min(), Xtr[:, 0].max(), 0.02 ) #np.linspace(Xtr[:, 0].min(), Xtr[:, 0].max(), 500)
    yi = np.arange(Xtr[:, 1].min(), Xtr[:, 1].max(), 0.02 ) #np.linspace(Xtr[:, 1].min(), Xtr[:, 1].max(), 500)
    X, Y = np.meshgrid(xi,yi)
    
    #zi = griddata(Xtr, Ypred, (X,Y), method='linear')
    zi = griddata(Xtr, Ypred, (X,Y), method=method_type)

    plt.contour(xi, yi, zi, 300, linewidths=2, colors='k', levels=list(range(nLevels)))
    #plt.contour(xi, yi, zi )
    # plot data points.
    plt.scatter(Xtr[:,0], Xtr[:,1], c=Ytr, marker='o', s=100, zorder=10, alpha=0.8)
    plt.xlim(Xtr[:,0].min(), Xtr[:,0].max())
    plt.ylim(Xtr[:,1].min(), Xtr[:,1].max())
    
def flipLabels(Y, perc):
    
    if perc < 1 or perc > 100:
        print("p should be a percentage value between 0 and 100.")
        return -1

    if any(np.abs(Y) != 1):
        print("The values of Ytr should be +1 or -1.")
        return -1

    Y_noisy = np.copy(np.squeeze(Y))
    if Y_noisy.ndim > 1:
        print("Please supply a label array with only one dimension")
        return -1

    n = Y_noisy.size
    n_flips = int(np.floor(n * perc / 100))
    idx_to_flip = np.random.choice(n, size=n_flips, replace=False)
    Y_noisy[idx_to_flip] = -Y_noisy[idx_to_flip]

    return Y_noisy

def splitData(X, Y, P):
    
    labels = np.unique(Y).astype(int)
    
    first = 1
    
    for L in labels:
        
        ntr = int(np.ceil(len(Y[Y==L])*P))
        nte = len(Y[Y==L])-ntr
        
        idx = np.random.permutation(len(Y[Y==L]))
        
        #print(np.shape(Y==L))
        #print(np.shape(L))
        #print(np.shape(X))
        
        Lidx = np.where(Y==L)[0]
        
        XL = X[Lidx,:]
        YL = Y[Y==L]
    
        X1 = XL[idx[:ntr],:]
        Y1 = YL[idx[:ntr]]
    
        X2 = XL[idx[ntr:],:]
        Y2 = YL[idx[ntr:]]
        
        if(first==1):
            Xtr = X1
            Ytr = Y1
            Xte = X2
            Yte = Y2
            first = 0
        else:
            Xtr = np.append(Xtr, X1, axis=0)
            Ytr = np.append(Ytr, Y1, axis=0)
            Xte = np.append(Xte, X2, axis=0)
            Yte = np.append(Yte, Y2, axis=0)
            
    
    return Xtr, Ytr, Xte, Yte

def centering(X, Xtr):
    mean = Xtr.mean(axis=0)
    X_z = X - mean
    return X_z

def explainedVariance(eig_vals):
    
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in eig_vals]
    cum_var_exp = np.cumsum(var_exp)
    
    return cum_var_exp

def PCA(X, k):
       
    # standardize the data
    mean = X.mean(axis=0)
    X_z = X - mean
    
    # Compute the covariance matrix of X_z
    cov_mat = np.dot(X_z.T,X_z)
    
    # compute eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvec = la.eig(cov_mat)
    # sort the eigenvalues in decreasing order and obtain the corresponding indexes
    
    indexes = np.argsort(eigvals,kind='mergesort')[::-1]
    
    # select the first k eigenvectors (Principal Components)
    
    PC = eigvec[indexes[0:k]]
   
    # compute the Cumulative Explained Variance
    
    expl_var= explainedVariance(eigvals[indexes])
    
    return PC, expl_var

def PCA_Project(X, PC):
    
    # standardize the data
    mean = X.mean(axis=0)
    X_z = X - mean
    
    # obtain the projected points
    
    X_proj = np.dot(X,PC.T)
    
    
    return X_proj

def calcError(Ypred, Yte):
    misclassified = list(Ypred==Yte).count(False)
    return misclassified/Ypred.shape[0] if Ypred.shape[0]>0 else 0.0 # if there are no tuples in the test set it returns 0.0

