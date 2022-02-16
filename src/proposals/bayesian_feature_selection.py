import numpy as np

from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score

class BayesianFeatureSelector :

    """
        
        Divides the spectrum of available wavelengths into intervals and considers the selection of a specific
        interval as binary variable in a Bayesian Optimization problem.
        The algorithm proceeds iteratively, discarding at the end of an iteration all intervals not selected by the 
        optimal solution of this iteration.
        For the next iteration the retained intervals are ranked and decomposed into smaller intervals as long as 
        the dimensional limit of the Bayesian Optimization proposed in literature is not reached.
        The algorithm terminates, if the optimization retains all intervals or a maximum number of itertions have been calculated.
        
        Further development option:
        * inclusion of beam search
        * option to force further reduction in wavelengths
    
    """
    
    def __init__(self, model, scorer) :
        
        self.Model = model
        self.scorer = scorer
        self.record = []
        
    def _log_run(self, optimizer) :
    
        #optimal_target = optimizer.max['target']
        
        quality_timeseries = np.array([ob['target'] for ob in optimizer.res])
        #sampled_masks = np.array([list(ob['params'].values()) for ob in optimizer.res],dtype='int')
        
        #
        # record the hamming distance between consecutive samples
        #
        #time_shifted_masks = np.roll(masks,-1,axis=0)
        #diff = np.logical_xor(masks,time_shifted_masks)
        #hamming_distances = np.sum(diff,axis=1)[:-1]
        
        #self.record.append((wavelengths, optimal_target, quality_timeseries, sampled_masks, hamming_distances))
        self.record.append(quality_timeseries)
        
    def _rank_intervals(self, mask, optimizer) :
        
        """
            Scores intervals selected in mask by the mean cross-validation performance across all samples
            of the current optimization run.
            
        """


        #
        # Retrieve the interval masks from all the samples of the optimization run
        #

        interval_masks = np.array([
                                
                                    list(sample['params'].values()) for sample in optimizer.res
                                  ],
                                  dtype='int')
        #
        # Retrieve the cross-validation scores of all runs
        #
        cv_scores = np.array([sample['target'] for sample in optimizer.res])
    
        #
        # Retrieve all intervals of the final mask
        #

        interval_scores = np.array([
                               np.sum(
                                   np.compress(interval_masks[:,i],cv_scores)
                               ) for i in np.nonzero(mask)[0]
                              ])
        #sort decendingly
        interval_ranks = np.flip(np.argsort(interval_scores))
        
        return interval_ranks
        
    def _random_ranking(self,mask) :
    
        return np.random.permutation(np.arange(len(np.nonzero(mask)[0])))
        
    def _decompose_intervals(self, 
                          global_feature_id, 
                          repeating_vector, 
                          mask, 
                          bayesian_optimizer) :
        
        """
            Produces
            - a list of the involved features by their global identifier
            - the repeating vector,
            - the mask
            
            The intervals are always halfed
        """
        
        #
        # Construct the subset of wavelengths selected by mask
        #
        
        expanded_mask = np.repeat(mask,repeating_vector)
        
        retained_wavelenghts = np.compress(expanded_mask,global_feature_id)
        
        #
        # Get repeating values of retained intervals
        #
        
        repeating_vector = np.compress(mask,repeating_vector)
        
        interval_ranks = self._rank_intervals(mask, bayesian_optimizer)
        
        #
        # For each discarded interval, another interval can be split
        #
        repeating_vector = np.insert(repeating_vector,np.arange(1,repeating_vector.shape[0] + 1),0)

        splits = np.count_nonzero(~mask)
        
        for i in range(interval_ranks.shape[0]):
            
            #the currently considered interval can be split 
            if repeating_vector[2 * interval_ranks[i]] >= 2 :
                
                repeating_vector[2 * interval_ranks[i] + 1] = np.floor(repeating_vector[2 * interval_ranks[i]] / 2)
                repeating_vector[2 * interval_ranks[i]] = np.ceil(repeating_vector[2 * interval_ranks[i]] / 2)
                
                splits -= 1
            
            if splits == 0:
                break
        #
        # Remove unnecessarily expanded elements in the repeating vector
        #
        return repeating_vector[np.nonzero(repeating_vector)[0]], retained_wavelenghts
    
    def _produce_evaluation_function(self, train_x, train_y, repeating_vector) :
        
        
        def _evaluation(**kwargs) :
        
            #
            # Convert the sample provided by the Bayesian Optimizer into a binary mask
            #
            selections = np.array(list(kwargs.values()),dtype='int')
        
            #
            # Catch the case of no wavelength interval being selected
            #
            if not np.any(selections) :
                return -1000
        
            #
            # expand the interval selection to wavelength granularity
            #
            mask = np.repeat(selections, repeating_vector)
       
            #
            # Perform cross validation on the model for the selected data
            #
            model = self.Model()
            res = cross_val_score(model, np.compress(mask, train_x, axis=1), train_y, scoring=self.scorer)
        
            return np.mean(res)
            
        return _evaluation
    
    def _fit(self,train_x,train_y, max_runs, samples_per_run = 100, intervals = 20, verbose = False, seed = 1000000) :
            
        """
            Assume for now train_x.shape[1] is divisible by intervals
                
            Current strategy: search as long as improvements can be made or max_runs is reached
        """
        
        #
        # Initialize variable wavelengths to contain all available wavelengths
        # Initialize variable repeating_vector to expand each interval to uniform size : #wavelengths/#intervals
        #
        
        repeating_vector = ((train_x.shape[1] / intervals ) * np.ones((intervals,))).astype('int')
        wavelengths = np.arange(0,train_x.shape[1],dtype='int')
                    
        for i in range(max_runs) :
             
           reduced_train = train_x[:,wavelengths]
           intervals = repeating_vector.shape[0]
           
           optimizer = BayesianOptimization(
                            f=self._produce_evaluation_function(reduced_train,train_y,repeating_vector),
                            pbounds=dict([(f'i{i}',(0,1.9)) for i in range(intervals)]), #binary 
                            verbose=verbose,
                            random_state=seed,
                        ) 
            
           optimizer.maximize( n_iter=samples_per_run)
            
           #
           # Get the best interval selection found
           #
           print(optimizer.max)
           
           mask = np.array(list(optimizer.max['params'].values()),dtype='int')
            
           #all intervals have been selected : no further interval decomposition possible without exceeding dimensional 
           #limit of Bayesian Optimization
           if np.count_nonzero(mask) == mask.shape[0] :  
               
               break
                
           #
           # Decompose intervals
           #
           repeating_vector, wavelengths = self._decompose_intervals(wavelengths,
                                                                    repeating_vector,
                                                                    mask,
                                                                    optimizer)
                                                                    
           #
           # Log progress
           #
           self._log_run(optimizer)
           
           print(f'Reduced dataset to {wavelengths.shape[0]} wavelengths in run {i}')
        
        #
        # Return the best found wavelengths and optimization record
        #
        
        return wavelengths, self.record
        