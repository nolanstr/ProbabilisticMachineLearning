import numpy as np
import matplotlib.pyplot as plt

def gen_data():
    x = np.linspace(0,5,100).reshape((-1,1))
    a = np.random.normal(5,5,size=x.shape)
    b = np.random.normal(10,2,size=x.shape)
    c = np.random.normal(4,2,size=x.shape)
    
    y_true = 5*x + 10*x + 4
    y_noisy = a*pow(x,2) + b*x + c
    y_noisy = y_noisy + np.random.normal(loc=0, scale=1, size=y_noisy.shape)
    
    data = np.hstack((x,y_noisy))

    np.save('noisy_data', data)
    plt.scatter(x, y_true, c=np.array(plt.cm.Pastel1(3)).reshape((1,-1)), 
                                                        label='true Model')
    plt.plot(x, y_noisy, c=plt.cm.Pastel1(4), label='noisy data')
    plt.title('plot of noisy data and noiseless (true) model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('data_check', dpi=1000)

if __name__ == '__main__':
    gen_data()

