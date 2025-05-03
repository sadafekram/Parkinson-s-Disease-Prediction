import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from scipy.spatial import distance_matrix

class LeastLinearSquares:
    
    def __init__(self, X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=30):
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.mean_X = mean_X
        self.std_X = std_X
        self.mean_Y = mean_Y
        self.std_Y = std_Y
                    
    def run(self):
        
        tic = time.time()
        
        self.W = np.dot(np.dot(np.linalg.inv(np.dot(self.X_train.T, self.X_train)), self.X_train.T), self.y_train)
        
        self.y_hat_tr_norm = np.dot(self.X_train, self.W)
        self.y_hat_te_norm = np.dot(self.X_test, self.W)
        self.y_tr_denorm = self.y_train*self.std_Y+self.mean_Y
        self.y_te_denorm = self.y_test*self.std_Y+self.mean_Y
        self.y_hat_tr_denorm = self.y_hat_tr_norm*self.std_Y+self.mean_Y
        self.y_hat_te_denorm = self.y_hat_te_norm*self.std_Y+self.mean_Y
        
        self.err_tr = (self.y_tr_denorm-self.y_hat_tr_denorm)
        self.err_te = (self.y_te_denorm-self.y_hat_te_denorm)
        
        toc = time.time()
        print(f"Finished Sucessfully in {toc-tic} ms !")
        return
    
    def MSE(self):
        
        self.MSE = np.zeros((1,2), dtype=float)
        self.MSE[0, 0] = np.linalg.norm(self.y_train - np.dot(self.X_train, self.W))**2/self.X_train.shape[0]
        self.MSE[0, 1] = np.linalg.norm(self.y_test - np.dot(self.X_test, self.W))**2/self.X_test.shape[0]
        
        return self.MSE
    
    def get_weights(self):
        
        return self.W
    
    def plot_weights(self, regressors):
        
        sns.set_theme(style="white")
        plt.figure(figsize=(16,4))
        plt.plot(np.arange(self.X_train.shape[1]), self.W, '-o', markersize=8, c='r', mfc='black', linestyle='-', linewidth=3)
        plt.xticks(np.arange(self.X_train.shape[1]), regressors, rotation=45, fontsize=15)
        plt.yticks(np.arange(-1, 1.5, 0.5), fontsize=15)
        plt.ylabel(r'$\^w(n)$', fontsize=15)
        plt.grid(linestyle='--', linewidth=0.75)
        plt.tight_layout()
        plt.savefig('./LLS_Weights.png', dpi=400)
        plt.show()
        
    def plot_histogram(self):
        
        max_bins = np.max([np.max(self.err_tr),np.max(self.err_te)])
        min_bins = np.min([np.min(self.err_tr),np.min(self.err_te)])
        common_bins = np.arange(min_bins, max_bins, (max_bins - min_bins) / 50)
        errors = [self.err_tr, self.err_te]
        sns.set_theme(style="white")
        plt.figure(figsize=(6,4))
        plt.hist(self.err_tr, bins=common_bins, density=True, histtype='bar',label='Training', alpha=0.5, color='blue')
        plt.hist(self.err_te, bins=common_bins, density=True, histtype='bar', label='Test', alpha=0.5, color='red')
        sns.kdeplot(self.err_tr, color='blue', alpha=0.5, linewidth=4)
        sns.kdeplot(self.err_te, color='red', alpha=0.5, linewidth=4)
        plt.xlabel(r'$e=y-\^y$', fontsize=15)
        plt.ylabel(r'$P(e$ in bin$)$', fontsize=15)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('./LLS-hist.png', dpi=400)
        plt.show()
        
    def plot_regplot(self):
        
        sns.set_theme(style='white')
        plt.figure(figsize=(4,4))
        sns.relplot(x=self.y_te_denorm, y=self.y_hat_te_denorm, color='g')
        v = plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]], 'k',linewidth=3)
        plt.xlabel(r'$y$', fontsize=15)
        plt.axis('square')
        plt.ylabel(r'$\^y$', fontsize=15)
        plt.grid()
        plt.tight_layout()
        plt.savefig('./LLS-yhat_vs_y.png', dpi=400)
        plt.show()

    def stats(self):
        
        self.err_tr_max=self.err_tr.max()
        self.err_tr_min=self.err_tr.min()
        self.err_tr_mu=self.err_tr.mean()
        self.err_tr_sig=self.err_tr.std()
        self.err_tr_MSE=np.mean(self.err_tr**2)
        self.R2_tr=1-self.err_tr_MSE/(np.std(self.y_tr_denorm)**2)
        self.c_tr=np.mean((self.y_tr_denorm-self.y_tr_denorm.mean())*(self.y_hat_tr_denorm-self.y_hat_tr_denorm.mean()))/(self.y_tr_denorm.std()*self.y_hat_tr_denorm.std())
        self.err_te_max=self.err_te.max()
        self.err_te_min=self.err_te.min()
        self.err_te_mu=self.err_te.mean()
        self.err_te_sig=self.err_te.std()
        self.err_te_MSE=np.mean(self.err_te**2)
        self.R2_te=1-self.err_te_MSE/(np.std(self.y_te_denorm)**2)
        self.c_te=np.mean((self.y_te_denorm-self.y_te_denorm.mean())*(self.y_hat_te_denorm-self.y_hat_te_denorm.mean()))/(self.y_te_denorm.std()*self.y_hat_te_denorm.std())
        cols=['min','max','mean','std','MSE','R^2','corr_coeff']
        rows=['Training','test']
        p=np.array([
            [self.err_tr_min,self.err_tr_max,self.err_tr_mu,self.err_tr_sig,self.err_tr_MSE,self.R2_tr,self.c_tr],
            [self.err_te_min,self.err_te_max,self.err_te_mu,self.err_te_sig,self.err_te_MSE,self.R2_te,self.c_te],
                    ])

        results=pd.DataFrame(p,columns=cols,index=rows)
        return results
        
        
class LinearRegression:
    
    def __init__(self, X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=30):
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.mean_X = mean_X
        self.std_X = std_X
        self.mean_Y = mean_Y
        self.std_Y = std_Y
            
    def SteepestDescent(self):
        
        if self.opt_mode=="Global":
            self.Model = np.zeros((self.epochs, 3), dtype=float)
            self.weights = np.zeros((self.epochs, self.X_train.shape[1]), dtype=float)
            self.W = np.random.rand(self.X_train.shape[1])
            Hessian = 2*self.X_train.T@self.X_train
            
            for iteration in range(self.epochs):
                gradient = 2*self.X_train.T @ (self.X_train@self.W - self.y_train)
                G = (np.linalg.norm(gradient)**2) / (gradient.T@Hessian@gradient)
                self.W = self.W - G*gradient
                self.y_hat_train = self.X_train@self.W
                self.y_hat_test = self.X_test@self.W
                self.MSE_train = np.linalg.norm(self.X_train@self.W - self.y_train)**2/self.X_train.shape[0]
                self.MSE_test = np.linalg.norm(self.X_test@self.W - self.y_test)**2/self.X_test.shape[0]
                self.Model[iteration ,0] = iteration
                self.Model[iteration ,1] = self.MSE_train
                self.Model[iteration ,2] = self.MSE_test
                self.weights[iteration ,:] = self.W
                if iteration%100==0:
                    print(f">>> It-{iteration} finished")
                
            self.y_hat_tr_norm = np.dot(self.X_train, self.weights[-1 ,:])
            self.y_hat_te_norm = np.dot(self.X_test, self.weights[-1 ,:])
            self.y_tr_denorm = self.y_train*self.std_Y+self.mean_Y
            self.y_te_denorm = self.y_test*self.std_Y+self.mean_Y
            self.y_hat_tr_denorm = self.y_hat_tr_norm*self.std_Y+self.mean_Y
            self.y_hat_te_denorm = self.y_hat_te_norm*self.std_Y+self.mean_Y
        
            self.err_tr = (self.y_tr_denorm-self.y_hat_tr_denorm)
            self.err_te = (self.y_te_denorm-self.y_hat_te_denorm)
                
        elif self.opt_mode=="Local":
            
            d_matrix = distance_matrix(self.X_train, self.X_test)
            self.Model = np.zeros((self.X_test.shape[0], self.epochs, 3), dtype=float)
            self.weights = np.zeros((self.X_test.shape[0], self.epochs, self.X_train.shape[1]), dtype=float)
            self.W = np.random.rand(self.X_train.shape[1])
            
            for i in range(self.X_test.shape[0]):
                sorted_temp = np.argsort(d_matrix[:,i], axis=0, kind='quicksort')
                temp_Xtrain = self.X_train[sorted_temp[0:self.b_size],:]
                temp_ytrain = self.y_train[sorted_temp[0:self.b_size]]
                Hessian = 2*temp_Xtrain.T@temp_Xtrain
                for iteration in range(self.epochs):
                    gradient = 2*temp_Xtrain.T @ (temp_Xtrain@self.W - temp_ytrain)
                    G = (np.linalg.norm(gradient)**2) / (gradient.T@Hessian@gradient)
                    self.W = self.W - G*gradient
                    self.y_hat_train = temp_Xtrain@self.W
                    self.y_hat_test = self.X_test[i]@self.W
                    self.MSE_train = np.linalg.norm(temp_Xtrain@self.W - temp_ytrain)**2/temp_Xtrain.shape[0]
                    self.MSE_test = np.linalg.norm(self.X_test[i]@self.W - self.y_test[i])**2
                    self.Model[i, iteration ,0] = iteration
                    self.Model[i, iteration ,1] = self.MSE_train
                    self.Model[i, iteration ,2] = self.MSE_test
                    self.weights[i, iteration ,:] = self.W
                    if iteration%100==0:
                        print(f"Iterations Over Test {i+1}")
                        
            self.y_hat_te_norm = np.diag(np.dot(self.X_test, self.weights[:, -1 ,:].T))
            self.y_te_denorm = self.y_test*self.std_Y+self.mean_Y
            self.y_hat_te_denorm = self.y_hat_te_norm*self.std_Y+self.mean_Y

            self.err_te = (self.y_te_denorm-self.y_hat_te_denorm)
     
    def compiler(self, optimizer="SteepestDescent", mode="Global", **kwargs):
        
        self.opt = optimizer
        self.opt_mode = mode
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        self.b_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10
        
        print("{ Linear Regression : {Optimizer: "+ str(self.opt)+"}"+", {Mode: "+ str(self.opt_mode) + "}"
             + ", {Epochs: " + str(self.epochs) + "}" + ", {Batch Size: " + str(self.b_size) + "} }")
        
    def run(self):
        
        tic = time.time()
        self.SteepestDescent()    
        toc = time.time()
        print(f"Finished Sucessfully in {(toc-tic)*1000} ms !")
            
    
    def get_weights(self):
        return self.weights
    
    def MSE(self):
        if self.opt_mode=="Global":   
            self.MSE = np.zeros((1,2), dtype=float)
            self.MSE[0, 0] = self.Model[-1 ,1]
            self.MSE[0, 1] = self.Model[-1 ,2]
            
        elif self.opt_mode=="Local":
            self.MSE = np.zeros((1,2), dtype=float)
            self.MSE[0, 0] = self.Model[:, -1 ,1].mean()
            self.MSE[0, 1] = np.linalg.norm(np.diag(self.X_test@self.weights[:, -1 ,:].T) - self.y_test)**2/self.X_test.shape[0]
            
        return self.MSE
            
    def plot_weights(self, regressors):
        
        if self.opt_mode=="Global":
            sns.set_theme(style="white")
            plt.figure(figsize=(16,4))
            plt.plot(np.arange(self.X_train.shape[1]), self.weights[-1 ,:], '-o', markersize=8, c='r', mfc='black', linestyle='-', linewidth=3)
            plt.xticks(np.arange(self.X_train.shape[1]), regressors, rotation=45, fontsize=15)
            plt.yticks(np.arange(-1, 1.5, 0.5), fontsize=15)
            plt.ylabel(r'$\^w(n)$', fontsize=15)
            plt.grid(linestyle='--', linewidth=0.75)
            plt.tight_layout()
            plt.savefig('./GSD_Weights.png', dpi=400)
            plt.show()
        elif self.opt_mode=="Local":
            print("NO WEIGHTS IN LOCAL MODE!")
        
    def plot_histogram(self):
        
        if self.opt_mode=="Global": 
            max_bins = np.max([np.max(self.err_tr),np.max(self.err_te)])
            min_bins = np.min([np.min(self.err_tr),np.min(self.err_te)])
            common_bins = np.arange(min_bins, max_bins, (max_bins - min_bins) / 50)
            errors = [self.err_tr, self.err_te]
            sns.set_theme(style="white")
            plt.figure(figsize=(6,4))
            plt.hist(self.err_tr, bins=common_bins, density=True, histtype='bar',label='Training', alpha=0.5, color='blue')
            plt.hist(self.err_te, bins=common_bins, density=True, histtype='bar', label='Test', alpha=0.5, color='red')
            sns.kdeplot(self.err_tr, color='blue', alpha=0.5, linewidth=4)
            sns.kdeplot(self.err_te, color='red', alpha=0.5, linewidth=4)
            plt.xlabel(r'$e=y-\^y$')
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig('./GSD-hist.png', dpi=400)
            plt.show()
            
        elif self.opt_mode=="Local":
            max_bins = np.max(self.err_te)
            min_bins = np.min(self.err_te)
            common_bins = np.arange(min_bins, max_bins, (max_bins - min_bins) / 50)
            sns.set_theme(style="white")
            plt.figure(figsize=(6,4))
            plt.hist(self.err_te, bins=common_bins, density=True, histtype='bar', label='Test', alpha=0.5, color='red')
            sns.kdeplot(self.err_te, color='red', alpha=0.5, linewidth=4)
            plt.xlabel(r'$e=y-\^y$')
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig('./LSD-hist.png', dpi=400)
            plt.show()
            
    def plot_regplot(self):
        
        sns.set_theme(style='white')
        plt.figure(figsize=(4,4))
        sns.relplot(x=self.y_te_denorm, y=self.y_hat_te_denorm, color='g')
        v = plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]], 'k',linewidth=3)
        plt.xlabel(r'$y$', fontsize=15)
        plt.axis('square')
        plt.ylabel(r'$\^y$', fontsize=15)
        plt.grid()
        plt.tight_layout()
        if self.opt_mode=="Global":
            plt.savefig('./GSD-yhat_vs_y.png', dpi=400)
        elif self.opt_mode=="Local":
            plt.savefig('./LSD-yhat_vs_y.png', dpi=400)       
        plt.show()
        
    def stats(self):
        
        if self.opt_mode=="Global": 
            self.err_tr_max=self.err_tr.max()
            self.err_tr_min=self.err_tr.min()
            self.err_tr_mu=self.err_tr.mean()
            self.err_tr_sig=self.err_tr.std()
            self.err_tr_MSE=np.mean(self.err_tr**2)
            self.R2_tr=1-self.err_tr_MSE/(np.std(self.y_tr_denorm)**2)
            self.c_tr=np.mean((self.y_tr_denorm-self.y_tr_denorm.mean())*(self.y_hat_tr_denorm-self.y_hat_tr_denorm.mean()))/(self.y_tr_denorm.std()*self.y_hat_tr_denorm.std())
            self.err_te_max=self.err_te.max()
            self.err_te_min=self.err_te.min()
            self.err_te_mu=self.err_te.mean()
            self.err_te_sig=self.err_te.std()
            self.err_te_MSE=np.mean(self.err_te**2)
            self.R2_te=1-self.err_te_MSE/(np.std(self.y_te_denorm)**2)
            self.c_te=np.mean((self.y_te_denorm-self.y_te_denorm.mean())*(self.y_hat_te_denorm-self.y_hat_te_denorm.mean()))/(self.y_te_denorm.std()*self.y_hat_te_denorm.std())
            cols=['min','max','mean','std','MSE','R^2','corr_coeff']
            rows=['Training','test']
            p=np.array([
                [self.err_tr_min,self.err_tr_max,self.err_tr_mu,self.err_tr_sig,self.err_tr_MSE,self.R2_tr,self.c_tr],
                [self.err_te_min,self.err_te_max,self.err_te_mu,self.err_te_sig,self.err_te_MSE,self.R2_te,self.c_te],
                        ])

            results=pd.DataFrame(p,columns=cols,index=rows)
            return results
            
        elif self.opt_mode=="Local":
            
            self.err_te_max=self.err_te.max()
            self.err_te_min=self.err_te.min()
            self.err_te_mu=self.err_te.mean()
            self.err_te_sig=self.err_te.std()
            self.err_te_MSE=np.mean(self.err_te**2)
            self.R2_te=1-self.err_te_MSE/(np.std(self.y_te_denorm)**2)
            self.c_te=np.mean((self.y_te_denorm-self.y_te_denorm.mean())*(self.y_hat_te_denorm-self.y_hat_te_denorm.mean()))/(self.y_te_denorm.std()*self.y_hat_te_denorm.std())
            cols=['min','max','mean','std','MSE','R^2','corr_coeff']
            rows=['Test']
            p=np.array([
                [self.err_te_min,self.err_te_max,self.err_te_mu,self.err_te_sig,self.err_te_MSE,self.R2_te,self.c_te],
                        ])

            results=pd.DataFrame(p,columns=cols,index=rows)
            return results