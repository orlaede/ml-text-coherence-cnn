import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt
from preprocessor import Preprocessor
from similarity_matrix import SimilarityMatrix
from sklearn.metrics import (auc, average_precision_score,
                             plot_precision_recall_curve,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import (KFold, LeaveOneOut, StratifiedKFold,
                                     train_test_split)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class Model:  
    
    def __init__(self):
      
        '''
          draw the ROC curve
            
          -- inputs: 
              None
          -- returns:
              None
        '''
        self.num_of_folds = int(5)
        self.dataset = None
        self.data = None
        self.model = None
        self.test_data = None
        
        self.preprocessor = Preprocessor()
        self.preprocessor.load_tokenizer(file_path = dataset_path + '/tokenizer.pkl')
        self.preprocessor.make_embeddings(path_to_embeddings=dataset_path + '/glove.6B.100d.txt')
    
    def make_dataset(self):

        '''
          Make tensorflow eager dataset object from the loaded data to model

          -- inputs: 
              None
          -- returns:
              None
        '''

        if self.data == None:
            print('cannot create dataset from empty data object, please load data first then create the dataset iterator')
        
        else:
            X_data, y_data = self.data[0], self.data[1]
            
            def generator():
                for train_index, test_index in KFold(n_splits=self.num_of_folds).split(X_data):
                    X_train, X_test = X_data[train_index], X_data[test_index]
                    y_train, y_test = y_data[train_index], y_data[test_index]
                    yield X_train,y_train,X_test,y_test

            self.dataset =  tf.data.Dataset.from_generator(generator, (tf.string,tf.int64,tf.string,tf.int64))
    
    

    def make_model(self):

        '''
          Make keras CNN model

          -- inputs: 
              None
          -- returns:
              None
        '''

        X_input =  tf.keras.Input(shape=(3, 100), name="input-sentences")
        
        
        embedding_layer = tf.keras.layers.Embedding(input_dim= len(self.preprocessor.tokenizer.word_index)+1, 
                                                    output_dim=self.preprocessor.embedding_dim, 
                                                    input_length=self.preprocessor.max_sequence_length,
                                                    trainable = False,
                                                    name='glove-embedding-layer')
        embedding_layer.build((None,))
        embedding_layer.set_weights([self.preprocessor.embeddings_matrix])
        
        first_sentence =  embedding_layer(X_input[:,0,:])
        second_sentence =  embedding_layer(X_input[:,1,:])
        third_sentence =  embedding_layer(X_input[:,2,:])
        
        convolutional_filters_map = tf.keras.layers.Conv1D(100,kernel_size=(3), activation='relu', use_bias=True, name='features-map')
        
        Xf = convolutional_filters_map(first_sentence)
        Xs = convolutional_filters_map(second_sentence)         
        Xt = convolutional_filters_map(third_sentence)   

        Xf = tf.keras.layers.MaxPool1D(98, name='first-sentence-pool')(Xf)
        Xs = tf.keras.layers.MaxPool1D(98, name='second-sentence-pool')(Xs)
        Xt = tf.keras.layers.MaxPool1D(98, name='third-sentence-pool')(Xt)

        similarity_fnc = SimilarityMatrix((100,100))

        sim_fs = similarity_fnc([Xf, Xs])
        sim_st = similarity_fnc([Xs, Xt])

        X = tf.keras.layers.concatenate([Xf, sim_fs, Xs, sim_st, Xt])

        ## TODO: this architecture requires grad search hyper-parameters tuning
        X = tf.keras.layers.Dense(256, activation='relu', name='fc1', use_bias=True)(X)
        X = tf.keras.layers.Dropout(0.333)(X)

        X = tf.keras.layers.Dense(512, activation='relu', name='fc2', use_bias=True)(X)
        X = tf.keras.layers.Dropout(0.333)(X)

        X = tf.keras.layers.Dense(512, activation='relu', name='fc3', use_bias=True)(X)
        X = tf.keras.layers.Dropout(0.333)(X)

        X = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(X)

        model = tf.keras.Model(inputs=[X_input], outputs=[X])

        self.model = model
    
    def load_data_from_csv(self, data_path, separator=',', split_train_test=False, make_balanced=False):
        
        '''
            load data from CSV file into dataframe
            
            -- inputs:
                data_path: path to file where data is saved
                separator (optional): value seprator to the file, default is comma
        '''
        
        self.data = pd.read_csv(f'{data_path}', sep=',')
        # separa um sequência de três orações de cada texto do dadtaset
        self.data['data'] = self.data['text'].apply(lambda x: x.strip().split('.')[:3])
        # para eliminar eventuais linhas com menos de 3 sentenças
        self.data = self.data.drop(self.data.loc[[(len(x) < 3) for x in self.data['data']]].index)
        self.data['data'] = self.data['data'].apply(lambda x: self.preprocessor.tokenize_data(x))

        if make_balanced:
          freq = list(df['label'].value_counts())
            #   freq = freq[0]//freq[1]-1
          min_freq = min(freq)

          df_coherent = self.data.loc[self.data['label'] == 1].sample(min_freq, replace=True)
          df_coherent_neg = self.data.loc[self.data['label'] == 0].sample(min_freq, replace=True)
          self.data = pd.concat([df_coherent, df_coherent_neg], ignore_index=True)
          
        #   df_coherent = self.data.loc[self.data['label'] == 1]
        #   df_coherent_replecated = pd.concat([df_coherent]*freq, ignore_index=True)
        #   self.data = pd.concat([df_coherent_replecated, self.data], ignore_index=True)
        
        if split_train_test:

          X_train, X_test, y_train, y_test = train_test_split(np.array(self.data['data'].values.tolist()), np.array(self.data['label'].values.tolist()).reshape(-1,1), test_size=0.2, random_state=42)
          self.data = (X_train, y_train)
          self.test_data = ( X_test, y_test)  
        else:
          self.data = (np.array(self.data['data'].values.tolist()), np.array(self.data['label'].values.tolist()).reshape(-1,1))

      