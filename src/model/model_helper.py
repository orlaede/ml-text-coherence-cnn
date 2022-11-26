class ModelHelper:
  
    @staticmethod
    def negative_log_likelihood(y_true, y_pred):
        
        '''
          Calculates negative log likelihood

          -- inputs: 
              y_true: ground truth  values
              y_predictions: non categorical predicted values
              y_pred: (optional) title for the plot
          -- returns:
              negative likelihood total loss 
        '''
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    @staticmethod
    def plot_ROC(y_true, y_predictions, title=''):

        '''
          Plot ROC curve
          
          -- inputs: 
              y_true: ground truth  values
              y_predictions: non categorical predicted values
              title: (optional) title for the plot
          -- returns:
              None
        '''

        ## calculate the FPR, TPR, Thresholds and AUC value
        false_pos_rate, true_pos_rate, thresholds = roc_curve(y_true, y_predictions)
        auc_val = auc(false_pos_rate, true_pos_rate)

        ## plot ROC curve
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(false_pos_rate, true_pos_rate, label=f'{title}' +' (area = {:.3f})'.format(auc_val))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()


    @staticmethod
    def plot_precision_recall(model, X_data, y_true):
      
        '''
          Plot precision-recall curve

          -- inputs: 
              model: model for which curve is plotted
              X_data: features to predict 
              y_true: ground truth  values
          -- returns:
              None
        '''
        pred = model.predict(X_data).ravel()

        average_precision = average_precision_score(y_true, pred)

        disp = plot_precision_recall_curve(model, X_data, y_true)
        disp.ax_.set_title('binary Precision-Recall curve: ' + 'AP={0:0.2f}'.format(average_precision))

    @staticmethod
    def compile_model(model , loss_func, monitor_metrics = ['acc'], optimizer='adam'):
      
        '''
            Compile model
            
            -- inputs: 
                model: model to compile
                loss_func: loss function to be used
                monitor_metrics: (optional) metrics to be monitored 
                optimizer: (optional) optimizer to use, adam is the default
            -- returns:
                None
        '''
        model.compile(optimizer=optimizer, loss=loss_func, metrics=monitor_metrics)   

    @staticmethod
    def train_model_kfolds(data, model_class, loss_func, num_of_folds, verbose=2, batch_size=128, plot_roc = False, plot_prec_recall = False ):
        
        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
            #tf.keras.callbacks.TensorBoard(log_dir=r'.\ogs') 
        ]
        
        # get x and y date (array tokenizer index, labels (0 , 1))
        X_data, y_data = data[0].astype(np.float32), data[1].astype(np.float32)
        
        count = 0

        for train_index, test_index in StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=42).split(X_data, y_data):
            
            X_train, X_test = X_data[train_index], X_data[test_index]
            
            y_train, y_test = y_data[train_index], y_data[test_index]
            
            model = model_class()
            model.make_model()
            model = model.model

            ModelHelper.compile_model(model, ModelHelper.negative_log_likelihood)

            model.fit(X_train,y_train,validation_data=(X_test,y_test),verbose=verbose,epochs=20, batch_size=batch_size, callbacks=model_callbacks)
            

            pred = model.predict(X_test).ravel()
            
            loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

            print(f'fold #{count+1} test loss: {loss}, test acc: {acc}')

            average_precision = average_precision_score(y_test, pred)

            print('Average precision-recall score: {0:0.2f}'.format(average_precision))
            
            if plot_roc:
              ModelHelper.plot_ROC(y_test, pred, 'test data')
            if plot_prec_recall:
              ModelHelper.plot_precision_recall(model, X_test, y_test)  

            count += 1
            gc.collect()

    @staticmethod
    def train_model(data, model, loss_func, epoches=100, verbose=1, batch_size=128, early_stop = False):
        
        model_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=dataset_path+'models/model.{epoch:03d}-{val_loss:.4f}.h5'),
            #tf.keras.callbacks.TensorBoard(log_dir=r'.\ogs') 
        ]

        if early_stop:
          model_callbacks.append( 
            tf.keras.callbacks.EarlyStopping(patience=2)
          )
        
        X_data, y_data = data[0].astype(np.float32), data[1].astype(np.float32)
        
        ModelHelper.compile_model(model, loss_func)

        model.fit(X_data,y_data,verbose=verbose,epochs=epoches, batch_size=batch_size, callbacks=model_callbacks, validation_split=0.15)