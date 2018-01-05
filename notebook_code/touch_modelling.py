import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy



def subplot_touch_data(ax, touches):
    plt.xlim((0, 1))
    plt.ylim((1, 0))    
    for i in range(touches.shape[0]):
        xVals = (touches[i, 2], touches[i, 0])
        yVals = (touches[i, 3], touches[i, 1])         
        plt.plot(xVals, yVals, color='k', lw=1)
        plt.scatter(touches[i, 2], touches[i, 3], color='k', s=4)
    
def plot_touch_data(touches, touches2):
    plt.figure(figsize=(7,7))
    ax = plt.subplot(121, aspect=16./9)
    plt.title('User 1', fontsize=20)
    subplot_touch_data(ax, touches)
    ax2 = plt.subplot(122, aspect=16./9)
    plt.title('User 2', fontsize=20)
    subplot_touch_data(ax2, touches2)
    plt.tight_layout()
    plt.show()

'''
def plot_user_identification_results(results):
    plt.figure(figsize=(7,7))
    ax = plt.subplot(111)
    title = 'Continuous user identification'
    subplot_user_identification_results(ax, results,title)
    plt.tight_layout()
    plt.show()
'''

def subplot_user_identification_results(ax, results, title, mark=None):
    plt.title(title, fontsize=20)
    plt.xlabel('Time (number of touches)', fontsize=18)
    plt.ylabel('User probability', fontsize=18)
    plt.ylim((-0.01,1.1))
    plt.plot(results[:,0], label='User 1', lw=4, alpha=.8)
    plt.plot(results[:,1], label='User 2', lw=4, alpha=.8)
    if mark is not None:
        plt.plot([mark, mark], [0, 1], lw=1, ls='--', c='k', label='True switch')
    plt.legend(loc='best', fontsize=18)

def plot_user_identification_results(results, results2, 
        titles=['Ground truth: user 1 is touching', 'Ground truth: user 2 is touching'],
        marks=None):
    plt.figure(figsize=(12,7))
    ax = plt.subplot(121)
    title = titles[0]
    subplot_user_identification_results(ax, results, title, marks[0] if marks is not None else None)
    ax2 = plt.subplot(122)
    title2 = titles[1]
    subplot_user_identification_results(ax2, results2, title2, marks[1] if marks is not None else None)
    plt.tight_layout()
    plt.show()




def createRBFKernel(M, M2, gamma):
    
    n, m = np.shape(M)
    n2, m2 = np.shape(M2)

    K = np.matrix(np.zeros((n, n2)), dtype="float64")

    for i in xrange(n):
        for j in xrange(n2):
            v1 = M[i, :]
            v2 = M2[j, :]
            diff = v1 - v2
            d = np.sum(diff * np.transpose(diff))
            K[i, j] = np.exp(-gamma * d) 
    return K


def plot_gp_example(observed,  N_draws = 20, figsize=(14,7)):

    gamma = 3
    
    points = np.linspace(0, 1, 20)
    points = np.matrix(points, dtype="float64")
    points_T = np.transpose(points)

    obs = np.matrix(observed)
    
    K_points_points = createRBFKernel(points_T, points_T, gamma) 
    K_points_obs = createRBFKernel(points_T, obs[:,0], gamma)  
    K_obs_obs = createRBFKernel(obs[:,0], obs[:,0], gamma) 
    
    M = K_points_obs * scipy.linalg.inv(K_obs_obs + 1e-6*np.identity(K_obs_obs.shape[0]))
    posterior_mean = M * obs[:,1]
    posterior_cov = K_points_points - M * K_points_obs.T 
      
   
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.title('Prior', fontsize=24)
    plt.xlabel('Input value (e.g. touch x)', fontsize=18)
    plt.ylabel('Output value (e.g. touch offset)', fontsize=18)
    plt.xlim((0,1))
    plt.ylim((-2,2))
    for i in xrange(N_draws):
        sample = np.random.multivariate_normal(np.zeros(points.shape[1]),
                                               K_points_points + 1e-6*np.identity(K_points_points.shape[0]), 1)
        plt.plot(np.array(points).flatten(), np.array(sample).flatten(), c='k', alpha=0.25)
        
    plt.subplot(122)
    plt.title('Posterior', fontsize=24)
    plt.xlabel('Input value (e.g. touch x)', fontsize=18)
    plt.ylabel('Output value (e.g. touch offset)', fontsize=18)
    plt.xlim((0,1))
    plt.ylim((-2,2))
    for i in xrange(N_draws):
        sample = np.random.multivariate_normal(np.array(posterior_mean.T).flatten(), 
                                               posterior_cov + 1e-6*np.identity(posterior_cov.shape[0]), 1)
        plt.plot(np.array(points).flatten(), np.array(sample).flatten(), c='k', alpha=0.25)
    plt.scatter(np.array(obs[:,0]).flatten(), np.array(obs[:,1]).flatten(), c='orange', lw=0, s=128, zorder=3)
    plt.tight_layout()
    plt.show()



def touch_data_preparation():
    import touchML.data.DBLoader as tdb

    touchData = tdb.loadForUserAndTask('./touchML/data/touchesDB.sqlite', 
                                    1, 1, 0, returnTrialData=False)
    touchData = np.hstack((0*np.ones_like(touchData[:,0:1]),touchData))


    touchData2 = tdb.loadForUserAndTask('./touchML/data/touchesDB.sqlite', 
                                    4, 1, 0, returnTrialData=False)
    touchData2 = np.hstack((1*np.ones_like(touchData2[:,0:1]),touchData2))

    touchDataAll = np.vstack((touchData, touchData2))

    touch_df = pd.DataFrame(data=touchDataAll, 
                        columns=['user','touch_x', 'touch_y', 'target_x', 'target_y'])

    touch_df.to_csv('./data/touches.csv', index=False)