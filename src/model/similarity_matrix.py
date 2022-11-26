class SimilarityMatrix(tf.keras.layers.Layer):

    def __init__(self,dims, **kwargs):
        self.dims_length, self.dims_width = dims
        super(SimilarityMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        
        # Create a trainable weight variable for this layer.
        self._m = self.add_weight(name='M', 
                                    shape=(self.dims_length,self.dims_width),
                                    initializer='uniform',
                                    trainable=True)
        super(SimilarityMatrix, self).build(input_shape)  # Be sure to call this at the end

    def call(self, y): 
        xf, xs = y
        sim1=tf.matmul(xf, self._m)
        transposed = tf.reshape(K.transpose(xs),(-1, 100, 1))
        sim2=tf.matmul(sim1, transposed)
        return sim2

    def compute_output_shape(self, input_shape):
        return (1)


    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'dims_length': self.dims_length, 
            'dims_width': self.dims_width
        })
        return config