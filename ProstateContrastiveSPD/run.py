import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from spdnetwork.optimizers import  MixOptimizer 
from spdnetwork.nn import LogEig
from Utils import get_fold_of_data
from DatasetManagement import DatasetManagement
from Models import Contrastive_CB3, SPDnet, SPDnet1Bire
from spdnetwork.optimizers import MixOptimizer


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = str(1)


def load_data(fold, name, percentage):
    train_embeddings_path = f'/data/{percentage}/{name}/fold_{fold}/train/embeddingsSPD_train.pt'
    train_labels_path = f'/data/{percentage}/{name}/fold_{fold}/train/labelsSPD_train.pt'
    
    val_embeddings_path = f'/data/{percentage}/{name}/fold_{fold}/val/embeddingsSPD_val.pt'
    val_labels_path = f'/data/{percentage}/{name}/fold_{fold}/val/labelsSPD_val.pt'
    
    train_embeddings = torch.load(train_embeddings_path)
    train_labels = torch.load(train_labels_path).view(-1)
    val_embeddings = torch.load(val_embeddings_path)
    val_labels = torch.load(val_labels_path).view(-1)
    
    return train_embeddings, train_labels, val_embeddings, val_labels

name_embeddings = ['embeddingsSPD', 'embeddingsTripletSPD']
percentages_path = ['Embeddings60%', 'Embeddings80%']


for percentage in percentages_path:
    print(f'---------------------Using percentage: {percentage}---------------------')
    for name in name_embeddings:
        print(f'---------------------Using embeddings: {name}---------------------')

        # Configuración de entrenamiento
        num_folds = 5
        batch_size = 32
        num_epochs = 1500
        lr = 0.001
        lr_others = 0.0001
        momentum = 0.6

        save_path = f'/data/{percentage}/results/{name}'
        os.makedirs(save_path, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        accuracy_list = []
        auc_list = []

        # Entrenamiento y validación en cada fold
        for fold in range(1, num_folds + 1):
            print(f'Starting fold {fold}')
            
            # Cargar datos
            train_embeddings, train_labels, val_embeddings, val_labels = load_data(fold, name, percentage)
            
            train_dataset = TensorDataset(train_embeddings, train_labels)
            val_dataset = TensorDataset(val_embeddings, val_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            model = SPDnet(device).to(device)
            print("------------Entrenando con la SPDnet------------")
            # criterion = nn.BCEWithLogitsLoss()
            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            optimizer_class = torch.optim.RMSprop
            optimizer = MixOptimizer(parameters = model.parameters(),    
                                    optimizer = optimizer_class,                                
                                    lr = lr,
                                    lr_others = lr_others,
                                    momentum = momentum                               
                                )
            best_auc = 0
            best_model = None
            
            train_losses = []
            val_losses = []
            
            # Entrenamiento
            # Parámetros para Early Stopping
            patience = num_epochs * 0.05
            best_val_loss = float('inf')
            early_stopping_counter = 0
            
            for epoch in range(num_epochs):
                model.train()
                running_train_loss = 0.0
                
                for batch_embeddings, batch_labels in train_loader:
                    batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
                    optimizer.zero_grad()
                    batch_embeddings = batch_embeddings.unsqueeze(1)  # Ajustar el tamaño del batch para el modelo
                    outputs = model(batch_embeddings)
                    loss = criterion(outputs, batch_labels.long())#Long
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()
                    
                train_loss = running_train_loss / len(train_loader)
                train_losses.append(train_loss)
            
                # Validación
                model.eval()
                running_val_loss = 0.0

                all_preds = []
                all_labels = []
            
                with torch.no_grad():
                        
                    model.eval()
                    model.to(device)
                    for batch_embeddings, batch_labels in val_loader:
                        batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
                        batch_embeddings = batch_embeddings.unsqueeze(1)  # Ajustar el tamaño del batch para el modelo
                        outputs = model(batch_embeddings)
                        loss = criterion(outputs, batch_labels.long()) #Long
                        running_val_loss += loss.item()
                        
                        probs = torch.softmax(outputs, dim=1)  # Usar softmax para obtener probabilidades de clases
                        preds = torch.argmax(probs, dim=1).cpu().numpy()  # Obtener las predicciones de clase
                        all_preds.extend(preds)
                        all_labels.extend(batch_labels.cpu().numpy())
            
                val_loss = running_val_loss / len(val_loader)
                val_losses.append(val_loss)
                
                accuracy = accuracy_score(all_labels, all_preds)
                auc = roc_auc_score(all_labels, all_preds)
                
                #Comprobar si la perdida de valicación es mejora
                if(val_loss < best_val_loss):
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
            
                if auc > best_auc:
                    best_auc = auc
                    best_model = model.state_dict()
                
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val AUC: {auc:.4f}')
                
                if early_stopping_counter >= patience:
                    print(f'Early stopping at epoch: {epoch + 1}' )
                    break
            
            accuracy_list.append(accuracy)
            auc_list.append(auc)
            
            print(f'Fold {fold}, Accuracy: {accuracy}, AUC: {auc}')

            # Crear la carpeta para guardar el modelo del pliegue correspondiente
            fold_save_path = os.path.join(save_path, f'fold_{fold}')
            os.makedirs(fold_save_path, exist_ok=True)
            
            # Guardar el mejor modelo
            best_model_path = os.path.join(fold_save_path, 'best_model.pth')
            torch.save(best_model, best_model_path)
            
            epochs_range = range(1, len(train_losses) + 1)

            plt.figure()
            plt.plot(epochs_range, train_losses, label='Train Loss')
            plt.plot(epochs_range, val_losses, label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Fold {fold} Loss')
            plt.legend()
            plt.savefig(os.path.join(fold_save_path, 'loss_plot.png'))
            plt.close()

        # Resultados promedio
        print(f'Mean Accuracy: {sum(accuracy_list) / num_folds}')
        print(f'Mean AUC: {sum(auc_list) / num_folds}')
