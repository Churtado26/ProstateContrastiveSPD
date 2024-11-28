import pandas as pd
import torch
import numpy as np

class ProstatexDataset():
    
    def __init__(self, mode = 'train'):
        """Constructor of the class, this method reads from the base data directory to obtain four pandas dataframes
        with all the data and metadata available from the patients, and their respective annotations provided by expert
        radiologists in the PROSTATEx dataset."""
        
        self.__mode = mode
        
        if self.__mode == 'train':
            base_path = '/data/PROSTATEx/dataframes_processed_training/'
        else:
            base_path = '/data/PROSTATEx/dataframes_processed_testing/'
        print('===========Loading modalities dataframes==================')
        self.__ktrans_df__ = pd.read_pickle(base_path+'ktrans.pkl')
        self.__adc_df__ = pd.read_pickle(base_path+'adc.pkl')
        self.__bval_df__ =  pd.read_pickle(base_path+'bval.pkl')
        self.__training_t2_tra_df__ = pd.read_pickle(base_path+'t2_tra.pkl')
        print('==================Completed===============================')
        
    def get_raw_dataframes(self):
        return [
            self.__adc_df__,
            self.__bval_df__,
            self.__ktrans_df__,
            self.__training_t2_tra_df__,
        ]
        
    def __str__(self):
        return '{} images for t2_tra, {} images for ktrans, {} images for bval, {} images for ADC.'.format(
            len(self.__training_t2_tra_df__),
            len(self.__ktrans_df__),
            len(self.__bval_df__),
            len(self.__adc_df__)
            
        )
        
    def to_categorical(self, y, num_classes):
        """ Returns a numpy array with a 1-hot encoded version of the original data. """
        return np.eye(num_classes, dtype='uint8')[y]
    
    def subsampling_data(self, labels, train_indexes, factor, strategy='stratified'):
        #get indexes by class
        pos_indexes = np.where(labels==1)[0].tolist()
        neg_indexes = np.where(labels==0)[0].tolist()
        #Filter the indexes using only the training_samples provided
        neg_indexes = list(set(neg_indexes) & set(train_indexes))
        pos_indexes = list(set(pos_indexes) & set(train_indexes))
        #sort the positives and negatives.
        target_pos = int( len(pos_indexes)*factor )
        target_neg = int( len(neg_indexes)*factor )
        pos_indexes.sort()
        neg_indexes.sort()
        if strategy == 'stratified':
            neg_indexes = neg_indexes[:target_neg]
            pos_indexes = pos_indexes[:target_pos]
        if strategy == 'reduce_positives':
            pos_indexes = pos_indexes[:target_pos]
            neg_indexes = neg_indexes[:]
        if strategy == 'reduce_negatives':
            pos_indexes = pos_indexes[:]
            neg_indexes = neg_indexes[:target_neg]
        #Join both classes and sort.
        subsampling_indexes = neg_indexes + pos_indexes
        subsampling_indexes.sort()
        excluded_indexes = list(set(train_indexes).difference(set(subsampling_indexes)))
        excluded_indexes.sort()
        result = {
            'subsampling_indexes': subsampling_indexes,
            'excluded_indexes': excluded_indexes
        }
        return result
    
    
    
    def get_lesion_roi(self, target_size, baseline='mertash'):
        """Returns a tuple (X,y) with the original data and the labes respectively. X is a list of the available sequences
        and/or modalities available, y is the array of labels that indicate if the lesion is clinically significant or not.
        
        Keyword arguments:
            
            target_size: Tuple of dimensions (C, H, W), represents the desired shape of the images.
            baseline: String that represents the strategy to collect the data, Default: mertash, currently available ['mertash']
        """
        
        z = target_size[0]
        y = target_size[1]
        x = target_size[2]
        ijks = self.__training_t2_tra_df__.ijk.values
        if baseline == 'xmasnet':
            rois = []
            ijks = self.__training_t2_tra_df__.ijk.values
            if self.__mode == 'train':
                labels = self.__training_t2_tra_df__.ClinSig.values*1
            else:
                labels = np.zeros(len(self.__training_t2_tra_df__))
            t2_imgs = self.__training_t2_tra_df__.img.values
            ktrans_imgs = self.__ktrans_df__.img_ktrans.values
            adc_imgs = self.__adc_df__.img_adc.values
            bval_imgs = self.__bval_df__.img_bval.values
            for index in range(len(ijks)):
                full_shape = (3, t2_imgs[index][0,...].shape[0], t2_imgs[index][0,...].shape[1] )
                current_image = np.zeros(full_shape)
                current_ijk = ijks[index].split(' ')
                current_ijk = [ int(x) for x in current_ijk ]
                target_adc_slice = adc_imgs[index][current_ijk[2], ...]
                target_bval_slice = bval_imgs[index][current_ijk[2], ...]
                target_ktrans_slice = ktrans_imgs[index][current_ijk[2], ...]

                current_image[0, ...] = target_bval_slice
                current_image[1, ...] = target_adc_slice
                current_image[2, ...] = target_ktrans_slice

                current_image = current_image.take(
                    axis=2,
                    indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                    mode='wrap'
                )
                current_image = current_image.take(
                    axis=1,
                    indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                    mode='wrap'
                )
                rois.append(current_image)

            return np.array(rois), np.array(labels).tolist()
        
        elif baseline == 'mertash':
            adc_imgs_rois = []
            bval_imgs_rois = []
            ktrans_imgs_rois = []
            t2_imgs_rois = []
            
            if self.__mode == 'train':
                labels = self.__training_t2_tra_df__.ClinSig.values*1
            else:
                labels = np.zeros(len(self.__training_t2_tra_df__))
            zones = self.__training_t2_tra_df__.zone.values.tolist()
            t2_imgs = self.__training_t2_tra_df__.img.values.tolist()
            ktrans_imgs = self.__ktrans_df__.img_ktrans.values.tolist()
            adc_imgs = self.__adc_df__.img_adc.values.tolist()
            bval_imgs = self.__bval_df__.img_bval.values.tolist()
                
            
            for index in range(len(ijks)):
                #pre-process zones
                zone = zones[index]
                if zone == 'PZ':
                    zones[index] = 0
                elif zone == 'TZ':
                    zones[index] = 1
                elif zone == 'AS':
                    zones[index] = 2
                else: 
                    zones[index] = 2
                #adc_imgs
                
                current_ijk = ijks[index].split(' ')
                current_ijk = [ int(x) for x in current_ijk ]
                current_image = adc_imgs[index]
                current_image = current_image.take(
                    axis=2,
                    indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                    mode='wrap'
                )
                current_image = current_image.take(
                    axis=1,
                    indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                    mode='wrap'
                )
                if z % 2 == 0:
                    #print('Los slices son pares, no se agregará nada.')
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                        mode='wrap'
                    )
                else:
                    #print('Los slices son impares, se agregará 1 slice')
                    range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2)+1)
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2)+1),
                        mode='wrap'
                    )
                adc_imgs_rois.append(current_image)
                
                #bval images
                current_ijk = ijks[index].split(' ')
                current_ijk = [ int(x) for x in current_ijk ]
                current_image = bval_imgs[index]
                current_image = current_image.take(
                    axis=2,
                    indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                    mode='wrap'
                )
                current_image = current_image.take(
                    axis=1,
                    indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                    mode='wrap'
                )
                if z % 2 == 0:
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                        mode='wrap'
                    )
                else:
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2)+1),
                        mode='wrap'
                    )
                bval_imgs_rois.append(current_image)
                
                #t2 images
                current_ijk = ijks[index].split(' ')
                current_ijk = [ int(x) for x in current_ijk ]
                current_image = t2_imgs[index]
                current_image = current_image.take(
                    axis=2,
                    indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                    mode='wrap'
                )
                current_image = current_image.take(
                    axis=1,
                    indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                    mode='wrap'
                )
                if z % 2 == 0:
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                        mode='wrap'
                    )
                else:
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2) +1),
                        mode='wrap'
                    )
                t2_imgs_rois.append(current_image)

                #ktrans images
                current_ijk = ijks[index].split(' ')
                current_ijk = [ int(x) for x in current_ijk ]
                current_image = ktrans_imgs[index]
                current_image = current_image.take(
                    axis=2,
                    indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                    mode='wrap'
                )
                current_image = current_image.take(
                    axis=1,
                    indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                    mode='wrap'
                )
                if z % 2 == 0:
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                        mode='wrap'
                    )
                else:
                    current_image = current_image.take(
                        axis=0,
                        indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2) +1),
                        mode='wrap'
                    )
                ktrans_imgs_rois.append(current_image)

            return [
                np.expand_dims(np.array(adc_imgs_rois), axis=1), 
                np.expand_dims(np.array(bval_imgs_rois), axis=1),
                np.expand_dims(np.array(ktrans_imgs_rois), axis=1),
                np.expand_dims(np.array(t2_imgs_rois), axis=1),
                self.to_categorical(np.array(zones), num_classes=3)
                ], np.array(labels)
        
    def get_weekly_roi(self, target_size, baseline='mertash'):
        """Returns a tuple (X,y) with pseudodata and the pseudo-label respectively. X is a list of the available sequences
        and/or modalities available, y is the array of labels that indicate if the lesion is clinically significant or not.
        In this particular case, we pick a center of a lesion and labeled it as a 1 and the neighboring regions will be 
        marked as 0, in other words, we use the radiologist annotations as groundtruth and the neighboring lesions as false
        samples.
        Keyword arguments:
            
            target_size: Tuple of dimensions (C, H, W), represents the desired shape of the images.
            baseline: String that represents the strategy to collect the data, Default: mertash, currently available ['mertash']
        """
        
        z = target_size[0]
        y = target_size[1]
        x = target_size[2]
        ijks = self.__training_t2_tra_df__.ijk.values
        if baseline == 'mertash':
            adc_imgs_rois = []
            bval_imgs_rois = []
            ktrans_imgs_rois = []
            t2_imgs_rois = []
            
            labels = []
            zones = []
            zones_df = self.__training_t2_tra_df__.zone.values.tolist()
            ktrans_imgs = self.__ktrans_df__.img_ktrans.values.tolist()
            adc_imgs = self.__adc_df__.img_adc.values.tolist()
            bval_imgs = self.__bval_df__.img_bval.values.tolist()
            t2_imgs = self.__training_t2_tra_df__.img.values.tolist()
                
            
            for index in range(len(ijks)):
                current_ijk = ijks[index].split(' ')
                current_ijk = [ int(x) for x in current_ijk ]
                for lesion_coordinate in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
                    current_ijk = ijks[index].split(' ')
                    current_ijk = [ int(x) for x in current_ijk ]
                    lesion_coordinates_dict = {
                        'a': [current_ijk[0] - int(x/2), current_ijk[1] - int(y/2), current_ijk[2]],
                        'b': [current_ijk[0], current_ijk[1] - int(y/2), current_ijk[2]],
                        'c': [current_ijk[0] + int(x/2), current_ijk[1] - int(y/2), current_ijk[2]],
                        'd': [current_ijk[0] - int(x/2), current_ijk[1], current_ijk[2]],
                        'e': [current_ijk[0], current_ijk[1], current_ijk[2]],
                        'f': [current_ijk[0] + int(x/2), current_ijk[1], current_ijk[2]],
                        'g': [current_ijk[0] - int(x/2), current_ijk[1] + int(y/2), current_ijk[2]],
                        'h': [current_ijk[0], current_ijk[1] + int(y/2), current_ijk[2]],
                        'i': [current_ijk[0] + int(x/2), current_ijk[1] + int(y/2), current_ijk[2]],
                    }
                    current_ijk = lesion_coordinates_dict[lesion_coordinate]
                    #1 if the lesion matches, which is case 'e', 0 for neighbors
                    labels.append(1) if lesion_coordinate == 'e' else labels.append(0)
                    #pre-process zones
                    zone = zones_df[index]
                    if zone == 'PZ':
                        zones.append(0)
                    elif zone == 'TZ':
                        zones.append(1)
                    elif zone == 'AS':
                        zones.append(2)
                    else: 
                        zones.append(3)
                    
                    #adc_imgs
                    current_image = adc_imgs[index]
                    current_image = current_image.take(
                        axis=2,
                        indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                        mode='wrap'
                    )
                    current_image = current_image.take(
                        axis=1,
                        indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                        mode='wrap'
                    )
                    if z % 2 == 0:
                        #print('Los slices son pares, no se agregará nada.')
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                            mode='wrap'
                        )
                    else:
                        #print('Los slices son impares, se agregará 1 slice')
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2)+1),
                            mode='wrap'
                        )
                    adc_imgs_rois.append(current_image)

                    #bval images
                    current_image = bval_imgs[index]
                    current_image = current_image.take(
                        axis=2,
                        indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                        mode='wrap'
                    )
                    current_image = current_image.take(
                        axis=1,
                        indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                        mode='wrap'
                    )
                    if z % 2 == 0:
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                            mode='wrap'
                        )
                    else:
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2)+1),
                            mode='wrap'
                        )
                    bval_imgs_rois.append(current_image)

                    #ktrans images
                    current_image = ktrans_imgs[index]
                    current_image = current_image.take(
                        axis=2,
                        indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                        mode='wrap'
                    )
                    current_image = current_image.take(
                        axis=1,
                        indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                        mode='wrap'
                    )
                    if z % 2 == 0:
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                            mode='wrap'
                        )
                    else:
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2) +1),
                            mode='wrap'
                        )
                    ktrans_imgs_rois.append(current_image)

                    #t2 images
                    current_image = t2_imgs[index]
                    current_image = current_image.take(
                        axis=2,
                        indices=range(current_ijk[0]-int(x/2),current_ijk[0]+int(x/2)),
                        mode='wrap'
                    )
                    current_image = current_image.take(
                        axis=1,
                        indices=range(current_ijk[1]-int(y/2),current_ijk[1]+int(y/2)),
                        mode='wrap'
                    )
                    if z % 2 == 0:
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2), current_ijk[2]+int(z/2)),
                            mode='wrap'
                        )
                    else:
                        current_image = current_image.take(
                            axis=0,
                            indices=range(current_ijk[2]-int(z/2),current_ijk[2]+int(z/2) +1),
                            mode='wrap'
                        )
                    t2_imgs_rois.append(current_image)

            return [
                np.expand_dims(np.array(adc_imgs_rois), axis=1),
                np.expand_dims(np.array(bval_imgs_rois), axis=1),
                np.expand_dims(np.array(ktrans_imgs_rois), axis=1),
                np.expand_dims(np.array(t2_imgs_rois), axis=1),
                self.to_categorical(np.array(zones), num_classes=3)
            ], np.array(labels)


