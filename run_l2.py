import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import copy

learning_rate = 0.001
nb_epochs = 5
colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
lmbd_list = [0, 0.001, 0.01, 0.1, 1, 10, 100]


def create_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)


def print_data():
    for x in range(len(data['x'])):
        print("Salty: " + str(data['x'][x]) + " Temperature: " + str(data['y'][x]) + "\n")


def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    # matricno mnozenje x sa w kolonom
    return hyp


def loss(x, y, w, b, lmbd, reg=None):
    # racuna hipotezu
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    # racunanje loss u odnosu na dobijeni y i pravi y

    mse = tf.reduce_mean(tf.square(prediction - y_col))
    l2_reg = lmbd * tf.reduce_mean(tf.square(w))
    return tf.add(mse, l2_reg)


def calc_grad(x, y, w, b, lmbd):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, lmbd)

    w_grad, b_grad = tape.gradient(loss_val, [w, b])
    return w_grad, b_grad, loss_val


def train_step(x, y, w, b, optimizer, lmbd):
    w_grad, b_grad, loss = calc_grad(x, y, w, b, lmbd)
    # Umesto prethodnog resenja koje je moglo da nas dovede cesto do ulaska u lokalni minimum, mada i ovaj ne garantuje
    # da nece uci u lokalni minimum
    optimizer.apply_gradients(zip([w_grad, b_grad], [w, b]))
    return loss


if __name__ == "__main__":
    file_name = 'bottle.csv'
    # Koristimo genfromtxt umesto loadtxt zbog toga sto u imamo prazne ćelije
    all_data = np.genfromtxt(file_name, delimiter=',', usecols=(5, 6), dtype='float32', skip_footer=864163,
                             invalid_raise=False, skip_header=1)

    nan_mask = np.any(np.isnan(all_data), axis=1)
    filtered_data = all_data[~nan_mask]

    data = dict()
    data['x'] = filtered_data[:, 1]
    data['y'] = filtered_data[:, 0]

    nb_samples = data['x'].shape[0]
    indices = np.random.permutation(nb_samples)
    data['x'] = data['x'][indices]
    data['y'] = data['y'][indices]

    # Normalizujemo vrednosti oko nule, da budu između -1 i 1. Feature sa različitim veličinima npr. prvi od 0 do 10,
    # drugi npr. 10 hiljada. Ovime feature će nam biti istog reda veličine. Skaliramo feature.
    # Racunamo kao (vrednost - (srednja vrednost svih x)) / (podelimo sa standardnom devijacijom)
    data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
    data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

    print_data()
    cost_values = []

    for i, x in enumerate(range(0, 7)):

        # Idemo do 4 stepena polinoma
        nb_features = 4
        # print(data['x'][:3])
        dataTraining = dict()
        dataTraining['x'] = copy.deepcopy(create_feature_matrix(data['x'], nb_features))
        dataTraining['y'] = copy.deepcopy(data['y'])
        # print(data['x'][:3])

        plt.scatter(dataTraining['x'][:, 0], dataTraining['y'])
        plt.xlabel('Salty')
        plt.ylabel('Temperature')

        # Vektor duzine koliko imamo featura w
        # Na pocetku 0, samo prava y = 0 -> x1*w1 + x2*w2
        w = tf.Variable(tf.zeros(nb_features))
        b = tf.Variable(0.0)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        epoch_losses = []
        for epoch in range(nb_epochs):

            epoch_loss = 0
            for sample in range(nb_samples):
                x = dataTraining['x'][sample].reshape((1, nb_features))
                # dobijamo vektor koji ce za x imati n stepena i dobijamo vektor y
                y = dataTraining['y'][sample]
                # racunamo loss u odnosu na trenutnu krivu koju imamo
                cur_loss = train_step(x, y, w, b, optimizer, lmbd_list[i])
                # racunamo loss
                epoch_loss += cur_loss

            epoch_loss /= nb_samples
            epoch_losses.append(epoch_loss)

            print(f'Epoch: {epoch + 1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

        cost_values.append(loss(dataTraining['x'], dataTraining['y'], w, b, lmbd_list[i]))

        print(f'w = {w.numpy()}, bias = {b.numpy()}')
        xs = create_feature_matrix(np.linspace(-2, 4, 100, dtype='float32'), nb_features)
        hyp_val = pred(xs, w, b)
        plt.plot(xs[:, 0].tolist(), hyp_val.numpy().tolist(), color=colors[i])
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])

    plt.show()
    plt.figure()
    plt.plot(lmbd_list, cost_values, marker='o')
    # Iscrtavamo za sve vrednosti lambda
    plt.xlabel('Lambda')
    plt.ylabel('Funkcija troska')
    plt.title('Zavisnost funkcije troška od parametra lambda')
    plt.legend()
    plt.grid(True)
    plt.show()
