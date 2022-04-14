#%%
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
import numpy as np
from statsmodels.tsa.stattools import kpss
import math
import seaborn as sns
from scipy.signal import dlsim
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# %%
def Cal_rolling_mean_var(column):
    """
    To calculate and plot rolling mean and rolling variance of a column
        Parameter:
            coulmn (list): list of coulmn values
        Variables:
            rolling_mean (list): a list containing the rolling means
            rolling_var (list): a list containing the rolling variances
        returns:
            None
    """

    rolling_mean = list()
    rolling_var = list()

    for i in range(1,len(column)+1):
        mean=np.mean(column[:i])
        rolling_mean.append(mean)

        var=np.var(column[:i])
        rolling_var.append(var)

    print(f'Final rolling mean: {rolling_mean[-1]:.4f}')
    print(f'Final rolling variance: {rolling_var[-1]:.4f}')

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))

    axes[0].plot(rolling_mean, color='r')
    axes[0].set_title('\nMean')
    axes[0].set(xlabel='Time', ylabel='Mean USD($)')

    axes[1].plot(rolling_var, color='b')
    axes[1].set_title('Variance')
    axes[1].set(xlabel='Time', ylabel='Variance USD($)')

    plt.tight_layout()
    plt.suptitle(f'Dependant Variable vs Time')
    plt.show()


def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# print('\nThe null hypothesis of the ADF is that there is a unit root, with\
#  the alternative that there is no unit root.The p-value below a threshold\
#  (1% or 5%) suggests we reject the null hypothesis (stationary) and a p-value\
#  above the threshold suggests we fail to reject the null hypothesis (non-stationary).\n')

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

# print('The null and alternate hypothesis for the KPSS test are opposite that of the ADF test.\
#  The p-value below a threshold (1% or 5%) suggests we reject the null hypothesis (non-stationary)\
#  and a p-value above the threshold suggests we fail to reject the null hypothesis (stationary).')


def pearson_correlation_coeff(x, y):
    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'])
    x_bar = df['x'].mean()
    y_bar = df['y'].mean()

    x_xbar = df['x'] - x_bar
    y_ybar = df['y'] - y_bar

    x_xbar_sq = x_xbar ** 2
    y_ybar_sq = y_ybar ** 2

    corr =(sum(x_xbar * y_ybar))/(math.sqrt(sum(x_xbar_sq) * sum(y_ybar_sq)))
    return corr

# acf (R_of_y, acf_plot, acf)
def R_of_y(y: list, tau: int):
    """[summary]
    to calculate r(y)
    Args:
        y (list): [list of values to calculate acf]
        tau (int): [number of lags to calculate]

    Returns:
        [acf]: [acf of yt value on lag = tau ]
    """
    # y = y.astype(float)
    y_bar = np.nanmean(y)
    numerator_0 = []
    numerator_1 = []
    denominator = []
    for i, n in enumerate(y):
        denominator.append((n - y_bar)**2) 
        if i >= abs(tau):
            numerator_0.append(n-y_bar)
            if tau < 0:
                numerator_1.append(y[i+tau]-y_bar)    
            else:
                numerator_1.append(y[i-tau]-y_bar)
    denominator=np.nan_to_num(denominator)  
           
    numerator_0=np.nan_to_num(numerator_0)
    numerator_1=np.nan_to_num(numerator_1)
    acf = np.dot(numerator_0, numerator_1)/np.nansum(denominator)
    return acf

def acf_plot(acf, tau, y, title='Autocorrelation of White Noise'):
    """ to plot acf
    Args:
        acf (list): [list of calculated acf values]
        tau (list): [range of tau]
        y (list): [original sample]
        title (string): [Title for the plot; default -> 'Autocorrelation of White Noise']
    Returns:
        NONE
    """
    fig, ax = plt.subplots(figsize=(12,6))
    markerline, stemline, baseline = plt.stem(acf, markerfmt='C3o', basefmt='C0-')
    plt.setp(markerline, markersize = 6)
    plt.ylabel('Magnitute')
    plt.xlabel('Lags')
    if max(tau) <= 10:
        plt.xticks(ticks=range(0,len(acf))[::1], labels = tau[::1])
    if max(tau) > 10 and max(tau) < 100:
        plt.xticks(ticks=range(0,len(acf))[::5], labels = tau[::5])
    elif max(tau) >= 100 and max(tau) < 200:
        plt.xticks(ticks=range(0,len(acf))[::20], labels = tau[::20])
    elif max(tau) >= 200 and max(tau) < 600:
        plt.xticks(ticks=range(0,len(acf))[::50], labels = tau[::50])
    else:
        plt.xticks(ticks=range(0,len(acf))[::200], labels = tau[::200])
    plt.title(f'{title}')
    # ax.fill_between(range(0,len(acf)),confint[0],confint[1], alpha=0.25)
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(ymin=-m,ymax=m,alpha=0.2,color='b')
    plt.tight_layout()
    plt.show()

def acf(y: list, taus: int, plot: bool, title='Autocorrelation of White Noise'):
    """ to calculated all the acf values
    Args:
        y (list): [list of numbers to calculate acf]
        taus (list): [lags range]
        plot (bool): [to call plot functions]
        title (string): [Title for the plot; default -> 'Autocorrelation of White Noise']
        
    Returns:
        list: [list of acf values]
    """
    if taus > 0:
        taus = list(range(-taus,taus+1))
    else:
        taus = list(range(taus, abs(taus)+1))
    
    acf_list = []
    for t in taus:
        acf_list.append(R_of_y(y,t))
    
    if taus[-1] <0:
        acf_list = acf_list[::-1]

    if plot == True:
        acf_plot(acf_list,taus,y, title)
    else:
        return acf_list

def Cal_GPAC(acf: list, len_j: int, len_k: int):
    ''' [Calculate and Plot GPAC Table]
    Args:
        acf (list): list of acf values
        len_j (int): number of rows of the GPAC table
        len_k (int): number of columns of the GPAC table

    Returns:
        NONE
    '''
    len_k = len_k + 1
    gpac = np.empty(shape=(len_j, len_k))

    for k in range(1, len_k):
        num = np.empty(shape=(k, k))
        den = np.empty(shape=(k, k))
        for j in range(0, len_j):
            for row in range(0, k):
                for col in range(0, k):
                    if col < k - 1:
                        num[row][col] = acf[np.abs(j+(row-col))]
                        den[row][col] = acf[np.abs(j+(row-col))]
                    else:
                        num[row][col] = acf[np.abs(j+row+1)]
                        den[row][col] = acf[np.abs(j+(row-col))]

            num_determinant = round(np.linalg.det(num),5)
            denom_determinant = round(np.linalg.det(den),5)

            if denom_determinant == 0:
                gpac[j][k] = float('inf')
            else:
                gpac[j][k] = round(num_determinant/denom_determinant,3)

    gpac = pd.DataFrame(gpac[:, 1:])
    gpac.columns = [i for i in range(1, len_k)]

    print("GPAC TABLE: \n",gpac)
    plt.figure(figsize=(8,6))
    sns.heatmap(gpac, annot=True, fmt=".3f")
    plt.xlabel('k')
    plt.ylabel('j')
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.show()

def rolling_mean(y:list):
    mean=[]
    s = pd.Series(y)
    for i in range(len(s)):
        mean.append(np.mean( s.head(i) ))
    return mean

def rolling_variance(y:list):
    var=[]
    s = pd.Series(y)
    for i in range(len(s)):
        var.append(np.var( s.head(i) ))
    return var

def q_value(y: list,lags: int, t:int):
    r=acf(y,lags, plot= False)
    rk=np.square(r[lags+2:])
    return t*(np.sum(rk))

# Average Prediction
def avg_pred(t: list, yt: list):
    """
    to calculate average predictions
    Args:
        t (list): [list of times]
        yt (list): [list of y values]

    Returns:
        [forecast]: [list of forecasts]
    """
    forecast = []
    
    for i,v in zip(t, yt):
        if i == 0:
            forecast.append(np.nan)
        else:
            forecast.append(round(np.nanmean(yt[:i]),2))
    return forecast
# for h step -> yhat = average of trainset yt[1:]

# Naive 
def naive_forecast(t: list, yt: list):
    """
    to calculate naive forecast
    Args:
        t (list): [list of times]
        yt (list): [list of y values]

    Returns:
        [forecast]: [list of forecasts]
    """
    forecast = []
    forecast.append(np.nan)
    for i,v in zip(t, yt):
        if i >= 0:
            forecast.append(yt[i])
    forecast.pop(-1)
    return forecast
# for h step -> yhat = last trainset yt

# Drift
def drift_predict(t: list, yt: list, h):
    """
    to calculate drift predict
    Args:
        t (list): [list of times]
        yt (list): [list of y values]
        h (int): [step]

    Returns:
        [forecast]: [list of predicts]
    """
    forecast = []
    for i,v in zip(t, yt):
        if i == 0:
            forecast.append(np.nan)
        elif i == 1:
            forecast.append(np.nan) 
        else:
            forecast.append(round(yt[i-1]+((h*(yt[i-1]-yt[0]))/(i-1)),2))
    return forecast

def drift_forecast(y_begin, y_end, t, h):
    """
    to calculate drift forecast
    Args:
        y_begin (number): [1st value of yt series]
        y_end (number): [last value of yt series]
        t (int): [time of y_end]
        h (int): [lag of prediction]

    Returns:
        [forecast]: [list of forecasts]
    """
    forecast = []
    for i in range(1,h+1):
        forecast.append(round(y_end+((i*(y_end-y_begin))/(t-1)),2))
    return forecast

# Simple Exponential Smoothing
def ses_predict(t, yt, alpha, initial):
    """
    to calculate Simple Exponential Smoothing forecast
    Args:
        alpha (float): [damping factor -> 0 ≤ alpha ≤ 1]
        initial (number): [1st value of yt series or initial condition]
        t (list): [list of times]
        yt (list): [list of values at t times]

    Returns:
        [predict]: [list of predictions]
    """
    predict = []
    

    for t, v in zip(t,yt):
        if t == 0:
            predict.insert(t,initial)
        elif t > 0:
            y_hat = predict[-1]
            predict.insert(t,round(alpha*yt[t-1] + (1-alpha)* y_hat,2))

    return predict
# for h step -> yhat = predict using last trainset row

def moving_avg(y):
    m = int(input("Enter the value of m:"))
    result = pd.DataFrame()
    
    def calculate_mavg(m,y):
        m_avg_list = pd.DataFrame(columns=['t', f'{m}MA'])
        k = int(np.floor((m - 1) / 2))
        
        for t in range(k, (len(y) - k)):
            s = 0
            for j in range(-k, k + 1):
                s = s + y[t + j]
            m_avg = round(s / m, 2)
            m_avg_list.loc[len(m_avg_list)] = [t, m_avg]
        return m_avg_list

    def even_mavg(y,m):
        
        cumsum = np.cumsum(np.insert(y, 0, 0)) 
        return (cumsum[m:] - cumsum[:-m]) / float(m)
        
    if m < 0:
        print("Invalid value of m")
    elif m == 1 | m == 2:
        print('Not acceptable m value. Please enter more than 2.')

    elif m%2 != 0:    
        result = calculate_mavg(m,y)
        # print(result)
    if m % 2 == 0:
        mv1 = even_mavg(y,m)
        m2 = int(input("Enter the value of second order m:"))
        mv2 = even_mavg(mv1,m2)
        tstart=((m-1)/2+(m2-1)/2)+1
        result['t'] = np.arange(tstart,len(mv2)+tstart)
        result[f'{m}MA'] = mv2
        
    return result

def b_hat_calc(x,y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)

def AR():
    
    N = int(input('Enter number of samples: '))
    na = int(input("Enter order of the AR process: "))

    input_n_string = input("Enter a list of numerators separated by space: ")
    num = list(map(float,input_n_string.split()))
    input_d_string = input("Enter a list of denominators separated by space: ")
    den = list(map(float,input_d_string.split()))
    
    np.random.seed(123)
    e = np.random.normal(0,1,N)

    system = (num,den,1)
    _,y = dlsim(system,e)
    T = len(y)- na -1

    vars = []
    for a in range(na, 0, -1):
        vars.append(y[a-1:a+T])

    X = np.hstack(vars)
    Y = np.array(y[na:(na+T)+1])

    def b_hat_calc(x,y):
        return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
        
    b_hat4 = b_hat_calc(-X,Y)

    print(f"Sample {N}: The unknown coefficients are: \n {pd.DataFrame(b_hat4, columns=['b_k'])}")    

def ARMA():
    T = int(input('Enter the number of data samples: '))
    mean = int(input('Enter the mean of white noise: '))
    var = int(input('Enter the variance of white noise: ')) 
    ar_ord = int(input('Enter AR order: '))
    ma_ord = int(input('Enter MA order: '))

    input_d_string = input("Enter the coefficients of AR: hint:- Enter a list separated by space: ")
    an = list(map(float,input_d_string.split())) # denominators
    input_n_string = input("Enter the coefficients of MA: hint:- Enter a list separated by space: ")
    bn = list(map(float,input_n_string.split())) # numerators
    

    arparams = np.array(an)
    maparams = np.array(bn)

    ar = np.r_[arparams]
    ma = np.r_[maparams]

    arma_process = sm.tsa.ArmaProcess(ar,ma)

    print('Is this a stationary process: ', arma_process.isstationary)

    mean_y = mean * (1 + np.sum(bn)) / (1 + np.sum(an)) 
    y = arma_process.generate_sample(T, scale = np.sqrt(var) + mean_y)

    lags = int(input('Enter Lag size for ACF: '))
    ry = arma_process.acf(lags = lags+1)
    return y,ry

def Cal_GPAC(acf: list, len_j: int, len_k: int):
    ''' [Calculate and Plot GPAC Table]
    Args:
        acf (list): list of acf values
        len_j (int): number of rows of the GPAC table
        len_k (int): number of columns of the GPAC table

    Returns:
        NONE
    '''
    len_k = len_k + 1
    gpac = np.empty(shape=(len_j, len_k))

    for k in range(1, len_k):
        num = np.empty(shape=(k, k))
        den = np.empty(shape=(k, k))
        for j in range(0, len_j):
            for row in range(0, k):
                for col in range(0, k):
                    if col < k - 1:
                        num[row][col] = acf[np.abs(j+(row-col))]
                        den[row][col] = acf[np.abs(j+(row-col))]
                    else:
                        num[row][col] = acf[np.abs(j+row+1)]
                        den[row][col] = acf[np.abs(j+(row-col))]

            num_determinant = round(np.linalg.det(num),5)
            denom_determinant = round(np.linalg.det(den),5)

            if denom_determinant == 0:
                gpac[j][k] = float('inf')
            else:
                gpac[j][k] = round(num_determinant/denom_determinant,3)

    gpac = pd.DataFrame(gpac[:, 1:])
    gpac.columns = [i for i in range(1, len_k)]

    print("GPAC TABLE: \n",gpac)
    plt.figure(figsize=(8,6))
    sns.heatmap(gpac, annot=True, fmt=".3f")
    plt.xlabel('k')
    plt.ylabel('j')
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.show()

def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags = lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    fig = plt.figure(figsize=(14,8))
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax = plt.gca(), lags =lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags =lags)
    fig.tight_layout(pad = 3)
    plt.show()

# LM ALGORITHM
def cal_e(teta,na,y):
    numerator=[1]+list(teta[na:])
    denominator=[1]+list(teta[:na])
    
    if len(numerator)!=len(denominator):
        while len(numerator)<len(denominator):
            numerator.append(0)
        while len(denominator)<len(numerator):
            denominator.append(0)
    system=(denominator,numerator,1)
    _,e=dlsim(system,y)
    e=[i[0] for i in e]
    return np.array(e)

def step0(na,nb):
    teta_o=np.zeros(shape=(na+nb,1))
    return teta_o.flatten()

def step1(delta,na,nb,teta,y):
    x=[]
    e_teta=cal_e(teta,na,y)
    SSE_0=np.dot(e_teta.T,e_teta)
    for i in range(na+nb):
        teta_delta = teta.copy()
        teta_delta[i]=teta[i]+delta
        en=cal_e(teta_delta,na,y)
        xi=(e_teta-en)/delta
        x.append(xi)
    X=np.transpose(x)
    A=np.dot(X.T,X)
    G=np.dot(X.T,e_teta)
    return A,G,SSE_0

def step2(A,G,mu,na,nb,teta,y):
    n=na+nb
    I=np.identity(n)
    dteta1=A+(mu*I)
    dteta_inv=np.linalg.inv(dteta1)
    delta_teta=np.dot(dteta_inv,G)
    teta_new=teta+delta_teta
    e=cal_e(teta_new,na,y)
    SSE_new=np.dot(e.T,e)
    if np.isnan(SSE_new):
        SSE_new=10**10
    return SSE_new,delta_teta,teta_new

def step3(max_iter, mu, delta, epsilon, mu_max, na, nb, y):
    iter=0
    teta=step0(na,nb)
    SSE=[]
    while iter<max_iter:
        A,G,SSE_0=step1(delta, na, nb, teta, y)
        if iter == 0:
            SSE.append(SSE_0)
        SSE_new,delta_teta,teta_new=step2(A, G, mu, na, nb, teta,y)
        SSE.append(SSE_new)
        if SSE_new<SSE_0:
            if np.linalg.norm(delta_teta)<epsilon:
                teta_hat=teta_new
                var=SSE_new/(len(y)-A.shape[0])
                A_inv=np.linalg.inv(A)
                cov=var*A_inv
                return SSE,cov,teta_hat,var
            else:
                teta=teta_new
                mu=mu/10
        while SSE_new>=SSE_0:
            mu = mu * 10
            if mu > mu_max:
               print('Mu limit exceeded')
               return None, None, None, None
            SSE_new, delta_teta, teta_new = step2(A, G, mu, na,nb, teta, y)
        iter += 1
        teta = teta_new
        if iter > max_iter:
            print('Maximum iterations exceeded')
            return None,None,None,None

def LMA(y, na, nb):
    SSE,cov,teta_hat,var = step3(100,0.01,1e-6,1e-3,1e10,na,nb,y)
    return SSE,cov,teta_hat,var

def conf_int(cov,teta,na,nb):
    print("Confidence Interval:")
    for i in range(na):
        right = teta[i] + 2*np.sqrt(cov[i][i])
        left = teta[i] - 2*np.sqrt(cov[i][i])
        print(f'{left:.6f} <a{i+1}< {right:.6f}')
    for i in range(nb):
        right = teta[na+i] + 2*np.sqrt(cov[na+i][na+i])
        left = teta[na+i] - 2*np.sqrt(cov[na+i][na+i])
        print(f'{left:.6f} <b{i+1}< {right:.6f}')

def zero_pole(teta,na):
    y_den=[1]+list(teta[:na])
    e_num=[1]+list(teta[na:])
    zeros=np.roots(e_num)
    poles=np.roots(y_den)
    print("The roots of numerator(poles): \n",zeros)
    print("The roots of denominator(zero): \n",poles)

def plot_SSE(SSE):
    iter=np.arange(0,len(SSE))
    plt.plot(iter,SSE)
    plt.xlabel('Number of iterations')
    plt.ylabel('SSE')
    plt.title('Sum square error vs. No. of iterations')
    plt.show()

#%%