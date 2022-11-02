import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Union, Optional, Tuple
from tqdm import tqdm
import mlflow
from datetime import datetime
import gc
from typing import Union, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # regression metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve # classificaiton metrics
import numpy as np
import platform

if platform.system() == "Windows":
    import winsound
else:
    import subprocess

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""if device == torch.device("cpu"):
    print("Using cpu.")
else:
    print("Using gpu.")
"""
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
                            use_bert_tokens: bool = False,
                            device: torch.device = torch.device("cpu")):
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
            device -- device to be used for training. (default: {torch.device("cpu")})
        """
        
        
        
        def train_one_epoch(train_dataloader: DataLoader, use_bert_tokens: bool = False):
            running_loss = 0.
            last_loss = 0.

            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            for i, data in enumerate(tqdm(train_dataloader, desc=f"Loss: {running_loss}")):
                # Every data instance is an input + label pair
                inputs, labels = data
                input_id = None
                mask = None
                if use_bert_tokens:
                    mask = inputs[0]['attention_mask'].to(device)
                    input_id = inputs[0]['input_ids'].squeeze(1).to(device)
                else:
                    inputs = inputs.to(device)

                labels = labels.to(device)
                
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = None
                if use_bert_tokens:
                    outputs = model(input_id, mask)
                    del input_id, mask
                    gc.collect()
                else:
                    outputs = model(inputs)
                    del inputs
                    gc.collect()

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                del outputs, labels
                gc.collect()

                # Gather data and report
                running_loss += loss.cpu().item()
                del loss
                gc.collect()

                if i % 1000 == 999:
                    last_loss = running_loss / 1000  # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.
                # clear the memory of gpu
                if device == torch.device("cuda:0"):
                    torch.cuda.empty_cache()


            if platform.system() == "Windows":
                winsound.PlaySound("./beep-sound-sound-effect.wav", winsound.SND_FILENAME)
            else:
                subprocess.call(["afplay", "/Users/emrebaloglu/Documents/NLP/NLP_Classification/beep-sound-sound-effect.wav"])
            return last_loss


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_vloss = 1e+6
        model = model.to(device)

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
                else:
                    vinputs = vinputs.to(device)

                vlabels = vlabels.to(device)

                voutputs = None
                if use_bert_tokens:
                    voutputs = model(vinput_id, vmask)
                    del vinput_id, vmask
                    gc.collect()
                else:
                    voutputs = model(vinputs)
                    del vinputs
                    gc.collect()

                vloss = loss_fn(voutputs, vlabels).cpu().item()
                running_vloss += vloss

                del vlabels, voutputs, vloss
                gc.collect()
                if device == torch.device("cuda:0"):
                    torch.cuda.empty_cache()
                
                

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print(type(avg_vloss))
            
            if use_mlflow:
                mlflow.log_metric("validation_loss", avg_vloss)


            # Track best performance, and save the model's state
            """if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)"""

            epoch_number += 1
        if platform.system() == "Windows":
            winsound.PlaySound("./done-sound-effect.wav", winsound.SND_FILENAME)
        else:
            subprocess.call(["afplay", "/Users/emrebaloglu/Documents/NLP/NLP_Classification/done-sound-effect.wav"])

    def test_pytorch_model_w_loader(self, model: nn.Module,
                            test_dataloader: DataLoader,
                            model_obj: str,
                            using_bert_tokens: bool = False,
                            device: torch.device = torch.device("cpu")) -> Union[Tuple[float, float], Tuple[float, float, float, float]]:
        """Test the pytorch model with given test dataloader.

        Args:
            model (nn.Module):  Model to be tested.
            test_dataloader (DataLoader): Dataloder for test data.
            model_obj (str): Objective for the model. Must be either 'classification' or 'regression'.
            using_bert_tokens (bool): Whether using bert tokens in the model or not. (default: {False})
        Returns:
            Union[Tuple[float, float], Tuple[float, float, float, float]]: Metrics of the model depending on objective.
            Will return [mean_squared_error, R2_score] if regression,
                        [accuracy, precision, recall, f1_score] if classification.
        """

        assert model_obj == "classification" or model_obj == "regression", \
            f"Model objective must be either 'classification' or 'regression', got {model_obj}."

        if model_obj == "regression":
            ss_res = 0.
            r2_ = 0.
            n_samples_total = 0.
            ss_tot = 0.
            for i, tdata in enumerate(test_dataloader):
                tinputs, tlabels = tdata
                n_samples = len(tinputs)
                n_samples_total += n_samples
                tinput_id = None
                tmask = None
                if using_bert_tokens:
                    tmask = tinputs[0]['attention_mask'].to(device)
                    tinput_id = tinputs[0]['input_ids'].squeeze(1).to(device)
                else:
                    tinputs = tinputs.to(device)

                tlabels = tlabels.to(device)
                toutputs = None
                if using_bert_tokens:
                    toutputs = model(tinput_id, tmask)
                    del tinput_id, tmask
                    gc.collect()
                else:
                    toutputs = model(tinputs)
                    del tinputs
                    gc.collect()

                tloss = nn.MSELoss(toutputs, tlabels).cpu().item()
                tloss2 = nn.MSELoss(toutputs, torch.mean(tlabels)).cpu().item() # for ss_total in r2 calculation

                del toutputs, tlabels
                gc.collect()

                ss_res += tloss*n_samples
                ss_tot += tloss2*n_samples
                
                if device == torch.device("cuda:0"):
                    torch.cuda.empty_cache()

            return ss_res/n_samples_total, 1 - (ss_res/ss_tot) # mse and r2

                    

        else:
            n_samples_total = 0.
            correct_preds = 0.
            for i, tdata in enumerate(test_dataloader):
                tinputs, tlabels = tdata
                n_samples = len(tinputs)
                n_samples_total += n_samples
                tinput_id = None
                tmask = None
                if using_bert_tokens:
                    tmask = tinputs[0]['attention_mask'].to(device)
                    tinput_id = tinputs[0]['input_ids'].squeeze(1).to(device)
                else:
                    tinputs = tinputs.to(device)

                tlabels = tlabels.to(device)
                toutputs = None
                if using_bert_tokens:
                    toutputs = model(tinput_id, tmask)
                    del tinput_id, tmask
                    gc.collect()
                else:
                    toutputs = model(tinputs)
                    del tinputs
                    gc.collect()

                correct_preds += (toutputs == tlabels).int()

    def test_pytorch_model_np(self, model: nn.Module, xtest: np.ndarray, ytest: np.ndarray, 
        model_obj: str, clf_mode: Optional[str] = "binary") -> Union[Tuple[float, float, float], 
                                                                    Tuple[float, float, float, float]]:      

        model = model.to(torch.device("cpu"))
        ypred = model(torch.Tensor(xtest)).cpu().detach().numpy()
        if model_obj ==  "regression":
            r2 =  r2_score(ytest, ypred)
            mae = mean_absolute_error(ytest, ypred)
            mse = mean_squared_error(ytest, ypred)

            return mse, mae, r2
        else:
            if clf_mode == "binary":
                ypred = (ypred > 0.5).astype(int)
            else: 
                ypred = np.argmax(ypred, axis=1)
            
            acc = accuracy_score(ytest, ypred) 
            prec = 0.
            recall = 0.
            f1 = 0.
            if clf_mode == "binary":
                prec = precision_score(ytest, ypred)
                recall = recall_score(ytest, ypred)
                f1 = f1_score(ytest, ypred)
            else:
                prec = precision_score(ytest, ypred, average="micro")
                recall = recall_score(ytest, ypred, average="micro")
                f1 = f1_score(ytest, ypred, average="micro")
            
            return acc, prec, recall, f1
        