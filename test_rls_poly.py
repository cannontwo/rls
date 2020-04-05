import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from sklearn.preprocessing import PolynomialFeatures

from rls_filter import RLSFilterAnalyticIntercept

def plot_model(model):
    xx = np.linspace(-1, 1)
    yy = [model.predict(np.array([[x]]))[0] for x in xx]

    plt.plot(xx, yy)

def calc_model_preds(model, poly_dim):
    transform = PolynomialFeatures(poly_dim, include_bias=False)
    xx = np.linspace(-2, 2, num=100)
    yy = [model.predict(transform.fit_transform(np.array([[x]])).transpose())[0] for x in xx]

    return xx, yy

def plot_true_func(A, b):
    xx = np.linspace(-1, 1)
    yy = [(A.dot(np.array([[x]])) + b)[0] for x in xx]

    plt.plot(xx, yy)

def calc_true_func(A, b, poly_dim):
    transform = PolynomialFeatures(poly_dim, include_bias=False)
    xx = np.linspace(-2, 2, num=100)
    yy = [(A.dot(transform.fit_transform(np.array([[x]])).transpose()) + b)[0] for x in xx]

    return xx, yy

def calc_mse(model, x_data, y_data, poly_dim):
    transform = PolynomialFeatures(poly_dim, include_bias=False)
    total_err = 0.0
    for x, y in zip(x_data, y_data):
        pred = model.predict(transform.fit_transform(x).transpose())
        total_err += np.linalg.norm(pred - y) ** 2

    return total_err / float(len(x_data))

def run(alpha=1e4):
    model = RLSFilterAnalyticIntercept(1, 1, alpha=alpha)

    A = np.random.randn(1, 1)
    b = np.random.randn(1, 1)

    print("A is {}".format(A))
    print("b is {}".format(b))

    x_data = []
    y_data = []

    y_max = max(A.dot(np.array([[-1.0]])) + b, A.dot(np.array([[1.0]])) + b)[0][0]
    y_min = min(A.dot(np.array([[-1.0]])) + b, A.dot(np.array([[1.0]])) + b)[0][0]

    for i in range(100):
        x_new = np.random.uniform(-1, 1, size=(1, 1))
        y_new = A.dot(x_new) + b + np.random.randn(1, 1) * 1e-1

        x_data.append(x_new)
        y_data.append(y_new)

        model.process_datum(x_new, y_new)

        err = calc_mse(model, x_data, y_data, 1)
        print("MSE is now {}".format(err))

        plt.xlim(-1, 1)
        plt.ylim(y_min, y_max)
        plt.scatter(x_data, y_data)
        plot_model(model)
        plot_true_func(A, b)
        plt.legend(['model', 'true'])
        plt.title('Iteration {}'.format(i))
        plt.show()
        plt.clf()

def run_animate():
    poly_dim = 20
    true_poly_dim = 20

    model = RLSFilterAnalyticIntercept(poly_dim, 1, alpha=1.0)
    high_alpha_model = RLSFilterAnalyticIntercept(poly_dim, 1, alpha=1e2)
    low_alpha_model = RLSFilterAnalyticIntercept(poly_dim, 1, alpha=1e-2)

    transform = PolynomialFeatures(poly_dim, include_bias=False)
    true_transform = PolynomialFeatures(true_poly_dim, include_bias=False)

    test_in = true_transform.fit_transform(np.array([[1.0]]))

    A = np.random.randn(1, test_in.shape[1])
    b = np.random.randn(1, 1)

    y_max = 10.0
    y_min = -10.0

    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(y_min, y_max))
    model_line, = ax.plot([], [], lw=3, c='k')
    high_a_model_line, = ax.plot([], [], lw=3, c='r')
    low_a_model_line, = ax.plot([], [], lw=3, c='m')
    true_line, = ax.plot([], [], lw=3, c='g')
    scatter = ax.scatter([], [], alpha=0.5)
    title = ax.text(0.85, 0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, 
            transform=ax.transAxes, ha='center')

    x_data = []
    y_data = []
    

    def init():
        model_line.set_data([], [])
        high_a_model_line.set_data([], [])
        low_a_model_line.set_data([], [])
        true_line.set_data([], [])
        scatter.set_offsets(np.zeros((0, 2)))
        title.set_text("")

        return model_line, high_a_model_line, low_a_model_line, true_line, scatter, title

    def animate(i):
        x_new = np.random.uniform(-1, 1, size=(1, 1))
        y_new = A.dot(transform.fit_transform(x_new).transpose()) + b + np.random.randn(1, 1) * 1e-1

        x_data.append(x_new)
        y_data.append(y_new)

        transformed_x = transform.fit_transform(x_new.transpose()).transpose()
        model.process_datum(transformed_x, y_new)
        high_alpha_model.process_datum(transformed_x, y_new)
        low_alpha_model.process_datum(transformed_x, y_new)

        err = calc_mse(model, x_data, y_data, poly_dim)
        print("\nAlpha=1.0 MSE is now {}".format(err))
        err = calc_mse(high_alpha_model, x_data, y_data, poly_dim)
        print("Alpha=1e2 MSE is now {}".format(err))
        err = calc_mse(low_alpha_model, x_data, y_data, poly_dim)
        print("Alpha=1e-2 MSE is now {}".format(err))

        xx, preds = calc_model_preds(model, poly_dim)
        model_line.set_data(xx, preds)
        model_line.set_label('alpha=1.0 model')

        xx, preds = calc_model_preds(high_alpha_model, poly_dim)
        high_a_model_line.set_data(xx, preds)
        high_a_model_line.set_label('alpha=1e2 model')

        xx, preds = calc_model_preds(low_alpha_model, poly_dim)
        low_a_model_line.set_data(xx, preds)
        low_a_model_line.set_label('alpha=1e-2 model')

        xx, yy = calc_true_func(A, b, true_poly_dim)
        true_line.set_data(xx, yy)
        true_line.set_label("true")

        offsets = np.block([np.array(x_data).reshape((-1, 1)), np.array(y_data).reshape((-1, 1))])
        scatter.set_offsets(offsets)

        legend = plt.legend()

        title.set_text("Iteration {}".format(len(x_data)))

        return model_line, high_a_model_line, low_a_model_line, true_line, scatter, title, legend
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    #run()
    run_animate()
