class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
                
def get_fold_of_data(kfold_number, train_data_percentage):
    import json
    import os
    import numpy as np


    if train_data_percentage is not None:
        FOLD_IDEX_PATH = '/data/json_index_shuffle.json'
        print(f'-------Cargando los datos del JSON revuelto con el {train_data_percentage*100} % en el fold {kfold_number+1}-------')
    else:
        FOLD_IDEX_PATH = '/data/picai_folds_indexes.json'
        print("^^^^^^^^^^Entrenando con todo el Dataset^^^^^^^^^^")
    
    fold_indexes = open(FOLD_IDEX_PATH, 'r')
    indexdes = json.load(fold_indexes)

        
    training_target_kfold = 'Fold_{}_train'.format(kfold_number)
    validation_target_kfold = 'Fold_{}_val'.format(kfold_number)

    BASE_PATH = '/data/'
    JSON_PATH = os.path.join(BASE_PATH, 'info-12x32x32.json')
    IMAGES_PATH = os.path.join(BASE_PATH, 'size-12x32x32')
    file = open(JSON_PATH, 'r')

    metadata = json.load(file)
    file_names = os.listdir(IMAGES_PATH)

    X_train = []
    Y_train = []
    
    ids_train = []
    ids_val = []

    X_validation = []
    Y_validation = []
    
    idx_count = 0 #Contador para ir sabiendo cuantos voy cargando
    
    if train_data_percentage is not None:
        target_stop = int(train_data_percentage * len(indexdes[training_target_kfold])) #Hasta que idx voy a entrenar
        print(train_data_percentage)
    
    if train_data_percentage is not None:   
        #print("Entrenando con porcentajes")
        for patient_id in indexdes[training_target_kfold]:
            idx_count+=1
            file_name = '{}.npy'.format(patient_id)
            img = np.load(os.path.join(IMAGES_PATH, file_name))
            X_train.append(img)
            ids_train.append(patient_id)
            #print("Entrenando con porcentajes")
    
            y = metadata[patient_id]['label']
            Y_train.append(y)
            
            if idx_count == target_stop:
                break;
    else:  
        print("Entrenando sin porcentajes") 
        for patient_id in indexdes[training_target_kfold]:
            
            file_name = '{}.npy'.format(patient_id)
            img = np.load(os.path.join(IMAGES_PATH, file_name))
            X_train.append(img)
            ids_train.append(patient_id)
    
            y = metadata[patient_id]['label']
            Y_train.append(y)
        
        
    for patient_id in indexdes[validation_target_kfold]:
        file_name = '{}.npy'.format(patient_id)
        img = np.load(os.path.join(IMAGES_PATH, file_name))
        X_validation.append(img)
        ids_val.append(patient_id)

        y = metadata[patient_id]['label']
        Y_validation.append(y)
        
    return np.array(X_train), np.array(X_validation), np.array(Y_train), np.array(Y_validation), ids_train, ids_val, indexdes