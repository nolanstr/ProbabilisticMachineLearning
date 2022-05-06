import numpy as np

def gen_data():
    a = 5
    b = 7
    x = np.linspace(0,5,100).reshape((-1,1))
    y = np.sin(a * x) + b
    y_noisy = y + np.random.normal(loc=0, scale=1, size=y.shape)
    
    data = np.hstack((x,y_noisy))

    np.save('noisy_data', data)


if __name__ == '__main__':
    gen_data()

