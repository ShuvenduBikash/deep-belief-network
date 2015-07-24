import numpy as np

from RBM import RBM


class DBN():
    def __init__(self, hidden_layers_structure, optimization_algorithm='sgd', learning_rate=0.3, num_epochs=10):
        self.RBM_layers = [RBM(num_hidden_units=num_hidden_units, optimization_algorithm=optimization_algorithm,
                               learning_rate=learning_rate, num_epochs=num_epochs) for num_hidden_units in
                           hidden_layers_structure]
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def fit(self, data, labels=None):
        input_data = data
        for rbm in self.RBM_layers:
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)

        if labels is not None:
            self.__fine_tuning(data, labels)
        return self

    def transform(self, data):
        input_data = data
        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
        return input_data

    def predict(self, data):
        return

    def __compute_activations(self, sample):
        input_data = sample
        layers_activation = list()

        for rbm in self.RBM_layers:
            input_data = rbm.transform(input_data)
            layers_activation.append(input_data)

        # Computing activation of output layer
        input_data = RBM.sigmoid(np.dot(self.W, input_data) + self.b)
        layers_activation.append(input_data)

        return layers_activation

    def __stochastic_gradient_descent(self, _data, _labels):
        data = np.copy(_data)
        labels = np.copy(_labels)
        for iteration in range(1, self.num_epochs + 1):
            idx = np.random.permutation(len(data))
            data = data[idx]
            labels = labels[idx]
            matrix_error = np.zeros(data.shape)
            i = 0
            for sample, label in zip(data, labels):
                delta_W, delta_bias, error_vector = self.__backpropagation(sample, label)
                # TODO update parameters
                # self.W += learning_rate * delta_W
                # self.b += learning_rate * delta_b
                # self.c += learning_rate * delta_c
                matrix_error[i, :] = error_vector
                i += 1
            error = np.sum(np.linalg.norm(matrix_error, axis=1))
            print ">> Epoch %d finished \tPrediction error %f" % (iteration, error)

    def __backpropagation(self, input_vector, label):
        x, y = input_vector, label
        deltas = list()
        list_layer_weights = list()
        for rbm in self.RBM_layers:
            list_layer_weights.append(rbm.W)
        list_layer_weights.append(self.W)

        # Forward pass
        layers_activation = self.__compute_activations(input_vector)

        # Backward pass: computing deltas
        activation_output_layer = layers_activation[-1]
        delta_output_layer = (activation_output_layer - y)
        deltas.append(delta_output_layer)
        layer_idx = range(len(self.RBM_layers))
        layer_idx.reverse()
        delta_previous_layer = delta_output_layer
        for layer in layer_idx:
            neuron_activations = layers_activation[layer]
            W = list_layer_weights[layer + 1]
            delta = np.dot(delta_previous_layer, W) * (neuron_activations * (1 - neuron_activations))
            deltas.append(delta)
            delta_previous_layer = delta
        deltas.reverse()

        # Computing gradients
        layers_activation.pop()
        layers_activation.insert(0, input_vector)
        layer_gradient_weights = list()
        layer_gradient_bias = list()
        for layer in range(len(list_layer_weights)):
            W = list_layer_weights[layer]
            gradient_W = np.zeros(W.shape)
            neuron_activations = layers_activation[layer]
            for j in range(len(neuron_activations)):
                gradient_W[:, j] = deltas[layer] * neuron_activations[j]
            gradient_W2 = np.outer(deltas[layer], neuron_activations)  # TODO check this
            layer_gradient_weights.append(gradient_W)
            layer_gradient_bias.append(deltas[layer])

        return layer_gradient_weights, layer_gradient_bias, delta_output_layer

    def __fine_tuning(self, data, _labels):
        self.num_classes = len(np.unique(_labels))  # TODO Possible bug here
        self.W = np.random.randn(self.num_classes, self.RBM_layers[-1].num_hidden_units)
        self.b = np.random.randn(self.num_classes)

        labels = self.__transform_labels_to_network_format(_labels)

        if self.optimization_algorithm is 'sgd':
            self.__stochastic_gradient_descent(data, labels)
        return

    def __transform_labels_to_network_format(self, labels):
        new_labels = np.zeros([len(labels), self.num_classes])
        i = 0
        for label in labels:
            new_labels[i][label] = 1
            i += 1
        return new_labels