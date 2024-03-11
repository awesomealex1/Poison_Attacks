import matplotlib.pyplot as plt
import os

def get_network_results(path):
    '''
    Returns the results of a network training.
    Args:
        path: Path to the results file.
    Returns:
        List of tuples of (epoch, loss, fake_correct, fake_incorrect, real_correct, real_incorrect)
    '''
    results = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            epoch = int(line[0].split(': ')[1])
            loss = float(line[1].split(': ')[1])
            fake_correct = int(line[2].split(': ')[1])
            fake_incorrect = int(line[3].split(': ')[1])
            real_correct = int(line[4].split(': ')[1])
            real_incorrect = int(line[5].split(': ')[1])
            results.append((epoch, loss, fake_correct, fake_incorrect, real_correct, real_incorrect))
    return results

def get_accuracy(results):
    '''
    Returns the accuracy of a network.
    Args:
        results: Results of a network training
    Returns:
        List of tuples of (epoch, accuracy)
    '''
    accuracies = []
    for result in results:
        epoch = result[0]
        fake_correct = result[2]
        fake_incorrect = result[3]
        real_correct = result[4]
        real_incorrect = result[5]
        accuracy = (fake_correct + real_correct)/(fake_correct + fake_incorrect + real_correct + real_incorrect)
        accuracies.append((epoch, accuracy))
    return accuracies

def get_f1_real(results):
    '''
    Returns the F1 score of a network.
    Args:
        results: Results of a network training
    Returns:
        List of tuples of (epoch, f1)
    '''
    f1s = []
    for result in results:
        epoch = result[0]
        fake_correct = result[2]
        fake_incorrect = result[3]
        real_correct = result[4]
        real_incorrect = result[5]
        precision = real_correct/(real_correct + fake_incorrect)
        recall = real_correct/(real_correct + real_incorrect)
        f1 = 2 * (precision * recall)/(precision + recall)
        f1s.append((epoch, f1))
    return f1s

def get_f1_fake(results):
    '''
    Returns the F1 score of a network for fake images.
    Args:
        results: Results of a network training
    Returns:
        List of tuples of (epoch, f1)
    '''
    f1s = []
    for result in results:
        epoch = result[0]
        fake_correct = result[2]
        fake_incorrect = result[3]
        real_correct = result[4]
        real_incorrect = result[5]
        precision = fake_correct/(fake_correct + real_incorrect)
        recall = fake_correct/(fake_correct + fake_incorrect)
        f1 = 2 * (precision * recall)/(precision + recall)
        f1s.append((epoch, f1))
    return f1s

def plot_accuracy_train_val(results_train, results_val, name):
    '''
    Plots the accuracy of a network.
    Args:
        results: Results of a network training
        name: Name of the network
    '''
    accuracies_train = get_accuracy(results_train)
    accuracies_val = get_accuracy(results_val)
    epochs = [accuracy[0] for accuracy in accuracies_train]
    accuracy_train = [accuracy[1] for accuracy in accuracies_train]
    accuracy_val = [accuracy[1] for accuracy in accuracies_val]
    plt.xticks(epochs)
    plt.plot(epochs, accuracy_train, label='train')
    plt.plot(epochs, accuracy_val, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy of {name}')
    plt.legend()
    plt.show()

def plot_f1_real_train_val(results_train, results_val, name):
    '''
    Plots the F1 score of a network.
    Args:
        results: Results of a network training
        name: Name of the network
    '''
    f1s_train = get_f1_real(results_train)
    f1s_val = get_f1_real(results_val)
    epochs = [f1[0] for f1 in f1s_train]
    f1_train = [f1[1] for f1 in f1s_train]
    f1_val = [f1[1] for f1 in f1s_val]
    plt.xticks(epochs)
    plt.plot(epochs, f1_train, label='train')
    plt.plot(epochs, f1_val, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title(f'F1 of {name}')
    plt.legend()
    plt.show()

def plot_f1_fake_train_val(results_train, results_val, name):
    '''
    Plots the F1 score of a network for fake images.
    Args:
        results: Results of a network training
        name: Name of the network
    '''
    f1s_train = get_f1_fake(results_train)
    f1s_val = get_f1_fake(results_val)
    epochs = [f1[0] for f1 in f1s_train]
    f1_train = [f1[1] for f1 in f1s_train]
    f1_val = [f1[1] for f1 in f1s_val]
    plt.xticks(epochs)
    plt.plot(epochs, f1_train, label='train')
    plt.plot(epochs, f1_val, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title(f'F1 of {name}')
    plt.legend()
    plt.show()

def plot_loss_train_val(results_train, results_val, name):
    '''
    Plots the loss of a network.
    Args:
        results: Results of a network training
        name: Name of the network
    '''
    losses_train = [result[1] for result in results_train]
    losses_val = [result[1] for result in results_val]
    epochs = [result[0] for result in results_train]
    plt.xticks(epochs)
    plt.plot(epochs, losses_train, label='train')
    plt.plot(epochs, losses_val, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss of {name}')
    plt.legend()
    plt.show()

def get_predictions(target_results_file):
    '''
    Returns the predictions of a network.
    Args:
        target_results_file: Path to the results file of the target network
    Returns:
        List of tuples of (epoch, prediction)
    '''
    predictions = []
    with open(target_results_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'prediction' in line:
                prediction = line[line.find('[[')+2:line.find(']]')]
                prediction_vals = prediction.split(', ')
                predictions.append(float(prediction_vals[0]), float(prediction_vals[1]))

    return predictions

def get_prediction_change(target_results_file):
    predictions = get_predictions(target_results_file)
    return predictions[-2][0]/predictions[-1][0]

def plot_prediction_change(poison_numbers):
    poison_predictions = []
    experiment_path = 'results_for_diss/attack_xception_full_baseline'
    for poison_number in poison_numbers:
        experiment_n_path = os.path.join(experiment_path, f'{poison_number}_poisons')
        for folder in os.listdir(experiment_n_path):
            if True:
                poison_predictions.append(get_prediction_change(f'poison_{poison_number}_results.txt'))

def extract_target_results(file):
    predictions_clean = []
    predictions_poisoned = []
    poison_line = True  # Switch between reading poisoned, clean
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            prediction = line[line.find('[[')+2:line.find(']]')]
            prediction_vals = prediction.split(', ')
            if poison_line:
                predictions_poisoned.append((float(prediction_vals[0]), float(prediction_vals[1])))
            else:
                predictions_clean.append((float(prediction_vals[0]), float(prediction_vals[1])))
            poison_line = not poison_line
    return predictions_clean, predictions_poisoned

def graph_poison_distances(file):
    target_dists = get_poison_target_feature_dist(file)
    base_dists = get_poison_base_dist(file)

    plt.plot(target_dists, label='Target')
    plt.plot(base_dists, label='Base')
    plt.xlabel('Iteration')
    plt.ylabel('Distances')
    plt.title('Feature distances')
    plt.legend()
    plt.show()

def get_poison_target_feature_dist(file):
    dists = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Poison-target' in line:
                dist = float(line[line.find(':')+2:])
                dists.append(dist)
    return dists

def get_poison_base_dist(file):
    dists = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Poison-base' in line:
                dist = float(line[line.find(':')+2:])
                dists.append(dist)
    return dists
    