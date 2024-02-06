import os

def save_training_epoch(model_name, epoch, loss, fake_correct, fake_incorrect, real_correct, real_incorrect):
    '''Saves the results of a training epoch to a file.'''
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
    if not os.path.exists(f'experiment_results/{model_name}'):
        os.makedirs(f'experiment_results/{model_name}')
    with open(f'experiment_results/{model_name}/training_results.txt', 'a+') as f:
        f.write(f'Epoch: {epoch}, Loss: {loss}, Fake Correct: {fake_correct}, Fake Incorrect: {fake_incorrect}, Real Correct: {real_correct}, Real Incorrect: {real_incorrect} \n')

def save_validation_epoch(model_name, epoch, loss, fake_correct, fake_incorrect, real_correct, real_incorrect):
    '''Saves the results of a training epoch to a file.'''
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
    if not os.path.exists(f'experiment_results/{model_name}'):
        os.makedirs(f'experiment_results/{model_name}')
    with open(f'experiment_results/{model_name}/validation_results.txt', 'a+') as f:
        f.write(f'Epoch: {epoch}, Loss: {loss}, Fake Correct: {fake_correct}, Fake Incorrect: {fake_incorrect}, Real Correct: {real_correct}, Real Incorrect: {real_incorrect} \n')