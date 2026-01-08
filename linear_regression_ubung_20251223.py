import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def test_data_creator(data_points=100) -> np.array:
    rnd = np.random.RandomState(20000102)
    x = rnd.rand(data_points)*18
    y = np.sin(x)*0.5+0.1*rnd.randn(data_points)
    plt.scatter(x, y)

    arr = np.array([x, y])
    arr = arr.reshape(arr.shape[1], arr.shape[0])
    return arr

def plot_data(data_intercept, data_coef, func="poly7") -> None:
    def poly(x, data_coef, intercept):
        func = intercept
        for coef in range(len(data_coef)):
            print(coef)
            term = data_coef[coef] * (x**coef)
            func += term
        return func

    x = np.linspace(0, 18, 100)
    y = poly(x=x, data_coef=data_coef, intercept=data_intercept)

    plt.plot(x, y, linewidth=2.0)
    plt.show()
    return None


def main():
    test_data = test_data_creator()
    poly_model = PolynomialFeatures(7)
    poly_model_fitter = LinearRegression()

    # Pipeline Stept 1: Date extention with polynominal features
    test_data_step1 = poly_model.fit_transform(X=test_data[:, 0:1], y=test_data[:, 1])
    #print(test_data_step1)

    # Pipeline Step 2:
    test_data_step2 = poly_model_fitter.fit(X=test_data_step1, y=test_data[:, 1])
    print(test_data_step2.coef_)
    print(test_data_step2.intercept_)

    plot_data(data_intercept=test_data_step2.intercept_, data_coef=test_data_step2.coef_, func="poly7")

    #

    return None







if __name__ == "__main__":
    main()