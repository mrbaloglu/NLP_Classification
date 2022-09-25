import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Union, Optional, Tuple
from tqdm import tqdm
import mlflow
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if device == torch.device("cpu"):
    print("Using cpu.")
else:
    print("Using gpu.")

class PytorchModelTrainer:
    def __init__(self, name: str = "PytorchModelTrainer"):
        self.name = name

    def __str__(self):
        return self.name

    def train_pytorch_model(self, model: nn.Module,
                            train_dataloader: DataLoader,
                            val_dataloader: DataLoader,
                            optimizer: torch.optim.Optimizer,
                            loss_fn: torch.nn.modules.loss,
                            n_epochs: int,
                            use_mlflow: bool = False,
                            use_bert_tokens: bool = False):
        """
        Train a given pytorch model using given optimizer and loss function, for given number of epochs.

        Arguments:
            model -- Pytorch model to be trained.
            train_dataloader -- Dataloader of the training data.
            val_dataloader -- Dataloader of the validation data.
            optimizer -- The optimizer of the training scheme.
            loss_fn -- The loss function to be used for training.
            n_epochs -- Number of epochs for the training.

        Keyword Arguments:
            use_mlflow --  Whether to use mlflow to log model stats and parameters. (default: {False})
                        !!! The function MUST be called inside an mlflow run 
                        (between mlflow.start_run() ... mlflow.end_run() or after with mlflow.start_run())
                        for mlflow to properly log params and metrics. 
            use_bert_tokens -- Should be True when using pretrained bert tokenizations, as the x's in dataloaders will include 
                        masks as well. (default: {False})
        """
        
        model = model.to(device)
        
        def train_one_epoch(train_dataloader: DataLoader, use_bert_tokens: bool = False):
            running_loss = 0.
            last_loss = 0.

            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            for i, data in enumerate(tqdm(train_dataloader)):
                # Every data instance is an input + label pair
                inputs, labels = data
                input_id = None
                mask = None
                if use_bert_tokens:
                    mask = inputs[0]['attention_mask'].to(device)
                    input_id = inputs[0]['input_ids'].squeeze(1).to(device)
                    labels = labels.to(device)
                else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = None
                if use_bert_tokens:
                    outputs = model(input_id, mask)
                else:
                    outputs = model(inputs)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 1000 == 999:
                    last_loss = running_loss / 1000  # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.

            return last_loss


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_vloss = 1e+6

        for epoch in range(n_epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(train_dataloader, use_bert_tokens=use_bert_tokens)

            # We don't need gradients on to do reporting
            model.train(False)

            running_vloss = 0.0
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata

                vinput_id = None
                vmask = None
                if use_bert_tokens:
                    vmask = vinputs[0]['attention_mask'].to(device)
                    vinput_id = vinputs[0]['input_ids'].squeeze(1).to(device)
                    vlabels = vlabels.to(device)
                else:
                    vinputs = vinputs.to(device)
                    vlabels = vlabels.to(device)

                voutputs = None
                if use_bert_tokens:
                    voutputs = model(vinput_id, vmask)
                else:
                    voutputs = model(vinputs)

                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print(type(avg_vloss))
            
            if use_mlflow:
                mlflow.log_metric("validation_loss", float(avg_vloss.detach().numpy()))


            # Track best performance, and save the model's state
            """if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)"""

            epoch_number += 1
