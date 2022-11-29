import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Union, Optional, Tuple, Dict
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

def calculate_stats_from_cm(confusion_matrix: np.ndarray, macro_avg: bool = True) -> Dict[str, float]:
    """Calculate accuracy, precision, recall and F1 Score from a given confusion matrix.
       By default, macro avg. of precision, recall and F1 Score is calculated.
    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        macro_avg (bool): Whether to calculate macro avg. or not. Set it false to calculate micro avg.
    Returns:
        Dict[str, float]: Dictionary of stats -- {"accuracy": acc_val, "precision": prec_val, "recall": rec_val, "f1": f1_val}
    """
    if macro_avg:
        precisions = np.zeros((confusion_matrix.shape[0],))  # per class precision scores
        recalls = np.zeros((confusion_matrix.shape[0],)) # per class recall scores
        f1s = np.zeros((confusion_matrix.shape[0],)) # per class f1 scores
        for j in range(confusion_matrix.shape[0]):
            if np.sum(confusion_matrix[j, :]) != 0:
                precisions[j] = confusion_matrix[j, j] / np.sum(confusion_matrix[j, :])
                
            if np.sum(confusion_matrix[:, j]) != 0:
                recalls[j] = confusion_matrix[j, j] / np.sum(confusion_matrix[:, j])
            
            if precisions[j] + recalls[j] != 0:
                f1s[j] = 2*precisions[j]*recalls[j] / (precisions[j]+recalls[j])

            
        macro_f1 = np.average(f1s)
        macro_precision = np.average(precisions)
        macro_recall = np.average(recalls)
        accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)

        return {"accuracy": accuracy, "precision": macro_precision, "recall": macro_recall, "f1": macro_f1}
    else: 
        raise NotImplementedError


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

            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            for i, data in enumerate(tqdm(train_dataloader, desc=f"Loss: {running_loss}")):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs = inputs.to(device)

                labels = labels.to(device)
                
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
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

                # clear the memory of gpu
                if device == torch.device("cuda:0"):
                    torch.cuda.empty_cache()



            if platform.system() == "Windows":
                winsound.PlaySound("./beep-sound-sound-effect.wav", winsound.SND_FILENAME)
            else:
                subprocess.call(["afplay", "/Users/emrebaloglu/Documents/NLP/NLP_Classification/beep-sound-sound-effect.wav"])
            return running_loss / (i+1)


        epoch_number = 0
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
                vinputs = vinputs.to(device)

                vlabels = vlabels.to(device)

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
                            device: torch.device = torch.device("cpu")) -> Union[Tuple[float, float], Dict[str, float]]:
        """Test the pytorch model with given test dataloader.

        Args:
            model (nn.Module):  Model to be tested.
            test_dataloader (DataLoader): Dataloder for test data.
            model_obj (str): Objective for the model. Must be either 'classification' or 'regression'.
        Returns:
            Union[Tuple[float, float], Tuple[float, float, float, float]]: Metrics of the model depending on objective.
            Will return [mean_squared_error, R2_score] if regression,
                        dict[accuracy, precision, recall, f1_score] if classification.
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
                tinputs = tinputs.to(device)

                tlabels = tlabels.to(device)
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
            n_labels = model.get_output_dim()
            binary = False
            if n_labels == 1:
                n_labels = 2
                binary = True

            cm = np.zeros((n_labels, n_labels))
            for i, tdata in enumerate(tqdm(test_dataloader, desc="Testing...")):
                tinputs, tlabels = tdata
                
                toutputs = model(tinputs)
                preds = None
                if binary:
                    preds = (toutputs > 0.5).int()
                else:
                    preds = torch.argmax(toutputs, dim=1) # here check the labels as well for onehot encoding
                
                for j in range(len(preds)):
                    cm[preds[j], tlabels[j].int()] += 1
            

                del tinput_id, tmask
                gc.collect()
            return calculate_stats_from_cm(cm)

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
        
class BertModelTrainer:
    def __init__(self, name: str = "BertModelTrainer"):
        self.name = name

    def __str__(self):
        return self.name

    def train_model(self, model: nn.Module,
                            train_dataloader: DataLoader,
                            val_dataloader: DataLoader,
                            optimizer: torch.optim.Optimizer,
                            loss_fn: torch.nn.modules.loss,
                            n_epochs: int,
                            use_mlflow: bool = False,
                            device: torch.device = torch.device("cpu")):
        """
        Train a given pytorch model with bert using given optimizer and loss function, for given number of epochs.

        Arguments:
            model -- Pytorch model to be trained. (taken from transformers library of hugging face)
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
            
            device -- device to be used for training. (default: {torch.device("cpu")})
        """
        
        
        
        def train_one_epoch(train_dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                            loss_fn: torch.nn.modules.loss, device: torch.device = torch.device("cpu")):
            running_loss = 0.

            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            for i, data in enumerate(tqdm(train_dataloader, desc=f"Loss: {running_loss}")):
                # Every data instance is an input + label pair
                inputs, labels = data
                mask = inputs['attention_mask'].to(device)
                input_id = inputs['input_ids'].squeeze(1).to(device)
                
                labels = labels.to(device)
                
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(input_id, mask)
                # del input_id, mask
                # gc.collect()
                
                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # del outputs, labels
                # gc.collect()

                # Gather data and report
                running_loss += loss.cpu().item()
                # del loss
                # gc.collect()

                # clear the memory of gpu
                # if device == torch.device("cuda:0"):
                #     torch.cuda.empty_cache()



            if platform.system() == "Windows":
                winsound.PlaySound("./beep-sound-sound-effect.wav", winsound.SND_FILENAME)
            else:
                subprocess.call(["afplay", "/Users/emrebaloglu/Documents/NLP/NLP_Classification/beep-sound-sound-effect.wav"])
            return running_loss / (i+1)


        epoch_number = 0
        model = model.to(device)

        for epoch in range(n_epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(train_dataloader, optimizer, loss_fn, device)

            # We don't need gradients on to do reporting
            model.train(False)

            running_vloss = 0.0
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                vmask = vinputs['attention_mask'].to(device)
                vinput_id = vinputs['input_ids'].squeeze(1).to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinput_id, vmask)
                # del vinput_id, vmask
                # gc.collect()

                vloss = loss_fn(voutputs, vlabels).cpu().item()
                running_vloss += vloss

                # del vlabels, voutputs, vloss
                # gc.collect()
                # if device == torch.device("cuda:0"):
                #     torch.cuda.empty_cache()
                
                

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

    def test_model_w_loader(self, model: nn.Module,
                            test_dataloader: DataLoader,
                            model_obj: str,
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
                        [accuracy, precision, recall, f1_score] if classification (macro avgs).
        """
        assert model_obj == "classification" or model_obj == "regression", \
            f"Model objective must be either 'classification' or 'regression', got {model_obj}."

        model.train(False)
        if model_obj == "regression":
            ss_res = 0.
            r2_ = 0.
            n_samples_total = 0.
            ss_tot = 0.
            for i, tdata in enumerate(tqdm(test_dataloader, desc="Testing...")):
                tinputs, tlabels = tdata
                n_samples = len(tinputs)
                n_samples_total += n_samples
                
                tmask = tinputs['attention_mask'].to(device)
                tinput_id = tinputs['input_ids'].squeeze(1).to(device)
                

                tlabels = tlabels.to(device)
                
                toutputs = model(tinput_id, tmask)
                del tinput_id, tmask
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
            n_labels = model.get_output_dim()
            binary = False
            if n_labels == 1:
                n_labels = 2
                binary = True

            cm = np.zeros((n_labels, n_labels))
            for i, tdata in enumerate(tqdm(test_dataloader, desc="Testing...")):
                tinputs, tlabels = tdata
                
                tmask = tinputs['attention_mask'].to(device)
                tinput_id = tinputs['input_ids'].squeeze(1).to(device)
            

                tlabels = tlabels.to(device)
                toutputs = model(tinput_id, tmask)
                preds = None
                if binary:
                    preds = (toutputs > 0.5).int()
                else:
                    preds = torch.argmax(toutputs, dim=1) # here check the labels as well for onehot encoding
                
                for j in range(len(preds)):
                    cm[preds[j], tlabels[j].int()] += 1
            

                del tinput_id, tmask
                gc.collect()
            return calculate_stats_from_cm(cm)


                

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
        