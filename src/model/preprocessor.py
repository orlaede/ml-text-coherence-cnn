

class Preprocessor:
    
    def __init__(self, embidding_dims= 100, max_sequence_length = 100, padding_type='post'):
        
        self.tokenizer = None
        self.embeddings_matrix = None
        self.embedding_dim = embidding_dims
        self.max_sequence_length = max_sequence_length
        self.padding_type = padding_type
        
    def __make_tokenizer(self, text, oov_token=''):
        '''
            make tokenizer from sentences
            
            -- inputs:
                sentences:text to fit tokenizer on
                oov_token: out of vocabulary token 
            -- returns:
                None
        '''
        self.tokenizer = Tokenizer(oov_token=oov_token)
        self.tokenizer.fit_on_texts(text)    
    
    def load_tokenizer(self, file_path):
        '''
            load tokenizer from pickle file
            
            -- inputs:
                file_path: path to the file of tokenizer 
            -- returns:
                None
        '''
        if  not (os.path.isfile(file_path)):
            # self.make_tokenizer(dataset_path, save_tokenizer=True, tokenizer_file_path='tokenizer.pkl')
            self.make_tokenizer(dataset_path, save_tokenizer=True, tokenizer_file_path=file_path)
            print('Criating tokenizer ...')

        try:
            self.tokenizer = pkl.load(open(file_path,'rb'))
            print(f'Tokenizer ready from {file_path}. word count len: {len(self.tokenizer.word_counts)}')
        except  Exception  as e :
            print(f'Could not load tokenizer from path: {file_path}\n{e}')
    
    
    def make_tokenizer(self, dataset_path, save_tokenizer=False, tokenizer_file_path=None):
        '''
            make a tokenizer from separate files
            
            -- inputs:
                file_path: path to the file of tokenizer 
            -- returns:
                None
        '''
        df = pd.read_csv(f'{dataset_path}/dataset.csv', sep=',')
        # self.tokenizer = 
        self.__make_tokenizer(np.array(df['text']))

        if save_tokenizer == True:
            try:
                pkl.dump(self.tokenizer, open(f'{tokenizer_file_path}', 'wb'))
            except Exception  as e :
                print(f'could not save tokenizer to path: {tokenizer_file_path}\n{e}')
    
    def make_embeddings(self, path_to_embeddings=dataset_path + '/glove.6B.100d.txt'):
        print(f'Making embedding from file: {path_to_embeddings}')
        if self.tokenizer == None:
            print('could not create embeddings matrix from empty tokenizer')
        
        else:  
            embeddings_index = {}
            vocab_size=len(self.tokenizer.word_index)

            with open(f'{path_to_embeddings}', encoding="utf8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs

            embedding_index = np.zeros((vocab_size+1, self.embedding_dim))
            for word, i in self.tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_index[i] = embedding_vector

            self.embeddings_matrix = embedding_index
        
    def tokenize_data(self, paragraphs):          
        sequence = self.tokenizer.texts_to_sequences(paragraphs)
        padded_sequence = pad_sequences(sequence, padding=self.padding_type,maxlen=self.max_sequence_length)
        return padded_sequence        
        