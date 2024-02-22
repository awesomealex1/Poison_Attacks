import os

def save_training_epoch(model_name, epoch, loss, fake_correct, fake_incorrect, real_correct, real_incorrect):
    '''Saves the results of a training epoch to a file.'''
    create_model_directory()
    with open(f'experiment_results/{model_name}/training_results.txt', 'a+') as f:
        f.write(f'Epoch: {epoch}, Loss: {loss}, Fake Correct: {fake_correct}, Fake Incorrect: {fake_incorrect}, Real Correct: {real_correct}, Real Incorrect: {real_incorrect} \n')

def save_validation_epoch(model_name, epoch, loss, fake_correct, fake_incorrect, real_correct, real_incorrect):
    '''Saves the results of a training epoch to a file.'''
    create_model_directory()
    with open(f'experiment_results/{model_name}/validation_results.txt', 'a+') as f:
        f.write(f'Epoch: {epoch}, Loss: {loss}, Fake Correct: {fake_correct}, Fake Incorrect: {fake_incorrect}, Real Correct: {real_correct}, Real Incorrect: {real_incorrect} \n')

def save_test(model_name, loss, fake_correct, fake_incorrect, real_correct, real_incorrect):
    '''Saves the results of a training epoch to a file.'''
    create_model_directory()
    with open(f'experiment_results/{model_name}/test_results.txt', 'a+') as f:
        f.write(f'Loss: {loss}, Fake Correct: {fake_correct}, Fake Incorrect: {fake_incorrect}, Real Correct: {real_correct}, Real Incorrect: {real_incorrect} \n')

def save_target_results(model_name, score, prediction):
    '''Saves the results of a training epoch to a file.'''
    create_model_directory()
    with open(f'experiment_results/{model_name}/target_results.txt', 'a+') as f:
        f.write(f'Score: {score[0][0].item()} {score[0][1].item()}, Prediction: {prediction[0]} {prediction[1][0][0].item()} {prediction[1][0][1].item()} \n')

def create_model_directory(model_name):
    '''Creates a directory for a model if it does not exist.'''
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
    if not os.path.exists(f'experiment_results/{model_name}'):
        os.makedirs(f'experiment_results/{model_name}')