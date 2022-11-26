from random import shuffle


class DataTransformer:
    
    def __init__(self):
        pass
    
    def __permute_file(self, file_lines, size=10):
        '''
            returns all permutations of a file

            -- inputs:
                file_lines: list of file lines
                size: number of permutations to return, other than the original one
            -- returns:
                all permutations of the list element where the original ordering is the first element
        '''
        file_lines = np.array(file_lines)
        indeces = [i for i in range(file_lines.shape[0])]
        shuffled_set = set() 
        while len(shuffled_set) <size:
            shuffle(indeces)
            shuffled_set.add(tuple(indeces))

        permuted_lines = [file_lines]
        for idx in shuffled_set:
            permuted_lines.append(file_lines[list(idx)])

        return permuted_lines

    def __remove_non_ascii(self, text):
        return ''.join([i if ord(i) < 128 and i not in ['.', '\n', ','] else '' for i in text])

    def __read_file(self, file_path, skip_first_token=False):
        '''
            reads the lines from file

            -- inputs: 
                file_path: full path to the file to be permuted
                skip_first_token: boolean used to sanitize the inputs from the DUC dataset
            -- returns:
                list of the file lines
        '''
        #try:
        with open(file_path, 'r') as file:
            if skip_first_token == True:
                return [ self.__remove_non_ascii(line[line.find(' ') + 1:]) for line in file.readlines() ]
            return [ self.__remove_non_ascii(line) for line in file.readlines()]
        #except:
            '''file is either not found or path is wrong
            print(f"file {file_path} was not found!")
            return []'''

    def __get_file_labeled_permutations(self, file_path):
        '''
            generate all permutations of the file, label the original one as coherent, all other permutations are non-coherent

            -- inputs: 
                file_path: full path to the file to be permuted
            -- returns: 
                list of tuples of structure: (lines permutation, label: being 1 for coherent, 0 for non-coherent)
        '''
        file_lines = self.__read_file(file_path, skip_first_token=True)
        permuted_lines = self.__permute_file(file_lines, size=20)
        labels = [1] + [0 for i in range(len(permuted_lines)-1)]
        return zip(permuted_lines, labels)

    @staticmethod
    def generate_file_cliques(file_lines, size=3):

        '''
        divides the file into cliques of similar size

        -- inputs: 
            file_lines: list of file lines to be permuted
            size: size of cliques 
        -- returns:
            list of cliques generated
        '''
        cliques = []
        for idx in range(len(file_lines)-size):
            current_clique = []
            for increment in range(size):
                current_clique.append(file_lines[idx+increment])
            cliques.append(current_clique)

        return cliques
    
    def generate_separate_files_method(self, dataset_path, clique_size = 3):
        '''
            file structure is: n+1 lines where the first line is either 1 for coherent documents, 0 for non coherent
                followed by n-lines of the document.
        '''
        count = 0
        print(f'reading files from {dataset_path}')
        for (_, _, file_names) in walk(dataset_path):
            for file_name in file_names:
                for file_lines, label in self.__get_file_labeled_permutations(dataset_path+file_name):
                    for clique in DataTransformer.generate_file_cliques(file_lines, size=clique_size):
                        with open(f'./processed data/separate-files/{count}.txt','w') as file:
                            file.write(f'{label}. ')
                            for line in clique:
                                file.write(f'{line}. ')
                        count += 1
                        
    def generate_csv_dataset_from_separate_files (self, dataset_path, file_name):
        
        with open(f'{dataset_path}{file_name}', 'a') as csv_file:
            csv_file.write('data,label\n')

            for (_, _, file_names) in walk(dataset_path):
                for file_name in file_names:
                    file_lines = open(f'{dataset_path}{file_name}', 'r').readline().split('.')[:-1]
                    if len(file_lines) != 4:
                        print(file_name)
                    else:
                        for line in file_lines[1:]:
                            csv_file.write(f'{line}. ')
                        csv_file.write(f',{file_lines[0]}\n')