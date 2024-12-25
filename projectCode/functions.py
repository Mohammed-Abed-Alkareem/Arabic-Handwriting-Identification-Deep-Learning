

import torch
import torch.nn as nn



def trainModelGPU(net, train_loader, val_loader, optimizer, criterion, epochs=100, patience=6, min_delta=0.001):
    '''
    Function to train a PyTorch model on the GPU with early stopping
    :param net: PyTorch model
    :param train_loader: DataLoader for the training dataset
    :param val_loader: DataLoader for the validation dataset
    :param optimizer: PyTorch optimizer
    :param criterion: Loss function
    :param epochs: Number of epochs to train the model (default=100)
    :param patience: Number of epochs to wait for improvement before stopping training (default=6)
    :param min_delta: Minimum change in validation loss to qualify as an improvement (default=0.001)
 
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    net.to(device)  # Move the model to the GPU


   # Initialize lists to store average loss, validation loss, and accuracy for each epoch
    epoch_loss_history = []
    epoch_val_loss_history = []
    epoch_train_acc_history = []
    epoch_val_acc_history = []

    # Early stopping parameters
    best_val_loss = float('inf')  # Initialize the best validation loss to a very large number
    increase_counter = 0  # Counter for consecutive increases in validation loss

    net.train()  # Set the model to training mode
    for epoch in range(epochs):  # Loop over the dataset multiple times
        running_loss = 0.0  # Initialize running loss for the training data
        val_running_loss = 0.0  # Initialize running loss for the validation data
        correct_train = 0  # Correct predictions during training
        total_train = 0  # Total samples during training
        correct_val = 0  # Correct predictions during validation
        total_val = 0  # Total samples during validation
        epoch_losses = []  # Store the losses for the current epoch
        print(f'Epoch {epoch + 1}')

        # Training loop
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the parameters

            running_loss += loss.item()  # Accumulate the running loss
            epoch_losses.append(loss.item())  # Store the loss for the current batch

            # Calculate accuracy for the current batch
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            total_train += labels.size(0)  # Update total number of samples
            correct_train += (predicted == labels).sum().item()  # Update correct predictions

        average_loss = running_loss / len(train_loader)  # Calculate average loss for the epoch
        train_accuracy = correct_train / total_train  # Calculate training accuracy for the epoch
        epoch_loss_history.append(average_loss)  # Store the average loss
        epoch_train_acc_history.append(train_accuracy)  # Store the training accuracy

        # Validation phase
        net.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)  # Forward pass
                
                val_loss = criterion(outputs, labels)  # Calculate validation loss

                val_running_loss += val_loss.item()  # Accumulate validation loss

                # Calculate accuracy for the current batch
                _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
                total_val += labels.size(0)  # Update total number of samples
                correct_val += (predicted == labels).sum().item()  # Update correct predictions

        average_val_loss = val_running_loss / len(val_loader)  # Calculate average validation loss for the epoch
        val_accuracy = correct_val / total_val  # Calculate validation accuracy for the epoch
        epoch_val_loss_history.append(average_val_loss)
        epoch_val_acc_history.append(val_accuracy)  # Store the validation accuracy

        print(f'Epoch {epoch + 1} Training Loss: {average_loss:.3f}, Validation Loss: {average_val_loss:.3f}')
        print(f'Epoch {epoch + 1} Training Accuracy: {train_accuracy:.3f}, Validation Accuracy: {val_accuracy:.3f}')

        # Early stopping logic
        if average_val_loss < best_val_loss - min_delta:
            best_val_loss = average_val_loss
            increase_counter = 0  # Reset increase counter if there is an improvement
        else:
            if average_val_loss > best_val_loss:
                increase_counter += 1  # Increment increase counter if validation loss increases
            else:
                increase_counter = 0  # Reset increase counter if no increase

        # Early stopping condition: Stop training if validation loss increases for patience consecutive epochs
        if increase_counter >= patience:
            print(f"Early stopping due to validation loss increasing for {patience} consecutive epochs at epoch {epoch + 1}")
            break

    print('='*50)
    print('Finished Training')

    return epoch_loss_history, epoch_val_loss_history, epoch_train_acc_history, epoch_val_acc_history

