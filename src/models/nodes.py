import os
import time

import config.local as config
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.deeplog import DeepLog
from torch.utils.tensorboard import SummaryWriter


def make_sequences(name, sample):
    sequences = []
    ln = sample + [-1] * (config.WINDOW_SIZE + 1 - len(sample))
    sequences.append(tuple(ln))
    print('Number of sequences({}): {}'.format(name, len(ln)))
    return sequences


def predict(num_classes, model_path):
    # Hyper-Parameters
    device = torch.device(config.DEVICE)
    input_size = config.INPUT_SIZE
    num_layers = config.NUM_LAYERS
    hidden_size = config.HIDDEN_SIZE
    window_size = config.WINDOW_SIZE
    num_candidates = config.NUM_CANDIDATES

    model = DeepLog(input_size, hidden_size, num_layers, num_classes, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = make_sequences('normal', config.NORMAL_SAMPLE)
    test_abnormal_loader = make_sequences('abnormal', config.ABNORMAL_SAMPLE)
    true_positive = 0
    false_positive = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    false_positive += 1
                    break
    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    true_positive += 1
                    break
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    false_negative = len(test_abnormal_loader) - true_positive
    precision = 100 * true_positive / (true_positive + false_positive)
    recall = 100 * true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            false_positive, false_negative, precision, recall, f1))
    print('Finished Predicting')


def train(dataloader, num_classes):
    model = DeepLog(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, num_classes, config.DEVICE)
    model_name = 'Adam_batch_size={}_epoch={}'.format(str(config.BATCH_SIZE), str(config.NUM_EPOCHS))
    writer = SummaryWriter(log_dir=os.path.join(config.MODEL_DIR, model_name))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(config.NUM_EPOCHS):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, config.WINDOW_SIZE, config.INPUT_SIZE).to(config.DEVICE)
            output = model(seq)
            loss = criterion(output, label.to(config.DEVICE))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, config.NUM_EPOCHS, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    model_file = config.MODEL_DIR + '/' + model_name + '.pt'
    torch.save(model.state_dict(), model_file)
    writer.close()
    print('Finished Training')
    return model_file
