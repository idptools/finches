"""
Class to build Interation Matrix from Mpipi forcefield By


values : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""
import numpy as np
import math

from .data.forcefield_dependencies import  get_null_interaction_baseline, get_charge_prefactor

from .parsing_aminoacid_sequences import get_charge_weighted_mask, get_charge_weighted_FD_mask
from .parsing_aminoacid_sequences import get_aliphatic_weighted_mask

from .PDB_structure_tools import build_column_mask_based_on_xyz

from .utils import matrix_manipulation

# -------------------------------------------------------------------------------------------------
class Interaction_Matrix_Constructor:
    
    def __init__(self,
                 parameters,
                 sequence_converter=False,
                 charge_prefactor=None,
                 null_interaction_baseline=None, 
                 compute_forcefield_dependencies=False):
        
        """
        The Interaction_Matrix_Constructor is a class which houses user-facing
        functions for calculating inter-residue interactions. 

        Philosophically speaking, the goal here here is to separate 
        the biophysical model used to determine interaction parameters into 
        a set of classes housed in finches.forcefields, and then the class        
        Interaction_Matrix_Constructor takes one of those models in as initializing
        input and provides a standardized way to calculate the same types of 
        interaction properties derived from many different biophysical models.

        For example, using Mpipi_GGv1, one might do:

            # import key modules
            from finches.forcefields.mPiPi import mPiPi_model
            from finches.epsilon_calculation import Interaction_Matrix_Constructor
            
            # Initialize a finches.forcefields.mPiPi.mPiPi_model object
            mPiPi_GGv1_params = mPiPi_model(version = 'mPiPi_GGv1')

            # initialize an Interaction_Matrix_Constructor
            IMC = Interaction_Matrix_Constructor(parameters = mPiPi_GGv1_params)

        The Interaction_Matrix_Constructor object IMC then provides functionality 
        for calculating inter-residue interactions using the energetics in the 
        underlying model. Moreover, the IMC object can also then be updated in a
        variety of ways.

        
        Parameters
        -----------
        parameters : finches.forcefield.<model>.<model_object>
            Instance of one of the forcefield objects found in finches.forcefields 
            module. This object contains all of the parameters for the model, such
            that the Interaction_Matrix_Constructor calls on a series of functions
            that a model presents to calculate the associated interactions in
            a consistent way.

            In this way, different interaction models can be distributed but the 
            same analysis code (suing an Interaction_Matrix_Constructor) can always
            be used.

            This parameters object has two key functions that are required to be
            implemented.

            * parameters.ALL_RESIDUES_TYPES : list of lists which defines residues
              which are allowed to occur in the same type of sequence; e.g. we expect
              a protein list, an RNA list, a DNA list etc. Note that EVERY residue in
              all lists must have a pair-wise interaction parameter calculatable via
              the compute_interaction_parameter() function [described be
                        
            * parameters.compute_interaction_parameter(r1,r2) : function which takes
              two valid residues (i.e. any pair from the residues defined in 
              ALL_RESIDUES_TYPES) and returns a value that reports on the relative
              preferential interaction between those residues.

            * parameters.CONFIGS : dictionary with some general default 
              precomputed values for the forcefield, including: 'charge_prefactor' 
              and , 'null_interaction_baseline', which will get used if these are
              not explicitly passed into this constructor.
                      
        sequence_converter : function 
            A function that takes in a sequence and converts the sequence to 
            alternative  sequnence that matches the acceptable residue types of the 
            parameters object. If no function is provided a default function that 
            simply returns the input sequences is provided. This can be useful if 
            we need to mask sequences in a certain way.

        charge_prefactor : float 
            Model specific value that defines how the charge weighting is scaled. 
            Charge weighting is implememnted in a way that runs of oppositely 
            charged residues are less repulsive for one another than they might 
            otherwise be, mimicking the fact that charged sidechains can point away 
            from one another and/or be influenced by pKa shifts. The 
            charge_prefactor is a scalar which in effect scales the strength this 
            scaling has, and typically needs to be tuned on a per forcefield basis.


        null_interaction_baseline : float  
            Model specific threshold to differentiate between attractive and repulsive
            interactions By default, this baseline is parameterized based on the attractive/
            repulsive interaction associated with a poly(GS) sequence, which we expect
            to behave as a Gaussian chain. 
            

            NOTE -  null_interaction_baseline is specific to the parameter version. 
                    The null_interaction_baseline is the value used to split matrix. 
                    This has been built such that this value recapitulates PolyGS for                    
                    the specific input model being used. Precomputed null_interaction 
                    baseline canbe found in the CONFIGS dictionary of the parameters
                    object.

                    To compute a new null_interaction_baseline for a new forcefield
                    see:

                        data.forcefield_dependencies.get_null_interaction_baseline

        compute_forcefield_dependencies : bool 
            Flag to specify whether to recompute the model specific 
            charge_prefactor and null_interaction_baseline used when computing epsilon if
            these are not available at runtime. 

        """

        # initialize the valid_residue_groups
        # this parameter should be in all finches.methods.forcefields. Note we shoul
        # in the future have a stand-alone validation function for this...
        try:
            self.valid_residue_groups = parameters.ALL_RESIDUES_TYPES
        except AttributeError:
            raise AttributeError('Tried to call ALL_RESIDUE_TYPES from the parameters object but no such variable is found. Please ensure the passed forcefield object possesses a .ALL_RESIDUES_TYPES object variable')

        
        # initialize core class variables; we do this here mainly for convenince in the
        # code, so all the class variables are clearly defined in one place
        
        self.parameters                = None # finches.forcefield.<forcefield_module>.<forcefield_class> object
        self.sequence_converter        = None # function converts a sequence automatically
        self.charge_prefactor          = None # prefactor that scales how much charge weighting matters
        self.null_interaction_baseline = None # fixed value acts as an offset for deliniating attractive/repulsive interactions
        self.lookup = {}                      # dictionary which provides lookup[r1][r2] => interaction value

        
        ## Set up sequence converter
        if not sequence_converter:
            self.sequence_converter = lambda a: a 
        else:
            self.sequence_converter = sequence_converter

        ## Set up charge_prefactor and null_interaction_baseline based on
        ## passed balues
        self.charge_prefactor = charge_prefactor
        self.null_interaction_baseline = null_interaction_baseline

        # this is the main function which sets up the self.lookup table and ensures self.parameters maps to
        # a valid object
        self._update_parameters(parameters)
        
        # check null_interaction_baseline
        if self.null_interaction_baseline == None: 
            try:
                self.null_interaction_baseline = self.parameters.CONFIGS['null_interaction_baseline']
            except Exception as e: 
                if compute_forcefield_dependencies: 
                    self._update_null_interaction_baseline(min_len=10, max_len=500, verbose=True)
                else:
                    print(f'''WARNING: null_interaction_baseline NOT found or defined for the parameter version "{parameters.version}". 
    Update this forcefield model or set compute_forcefield_dependencies = True. 
    Precomputed null_interaction_baseline values can be added to the relevant CONFIGS dictionary at the top of the forcefield module. 

    To compute a new null_interaction_baseline see: data.forcefield_dependencies.null_interaction_baseline\n''')

        # charge charge_prefactor 
        if self.charge_prefactor == None: 
            try:                
                self.charge_prefactor = self.parameters.CONFIGS['charge_prefactor']
            except Exception as e: 
                if compute_forcefield_dependencies:
                    self._update_charge_prefactor(reference_data='DAS_KAPPA_RG_MPIPI', verbose=True)
                else:
                    print(f'''WARNING: Charge_prefactor in NOT found or defined for the parameter version "{parameters.version}".
    Update or set compute_forcefield_dependencies = True. 
    Precomputed charge_prefactor values can be added to the relevant CONFIGS dictionary at the tope of the forcefield module.
                    
    To compute a new charge_prefactor see data.forcefield_dependencies.get_charge_prefactor\n''')


                    
    ## ................................................................................... ##
    ##
    ##
    def _update_lookup_dict(self, unknown_set_to_zero=False):
        """
        Function which, if called, wipes the previous inter-residue interaction
        parameters (the 'lookup_dict') and recalculates using the self.parameter 
        object, which should be a finches.forcefield.<model>.<model_object> object.

        Parameters
        -------------
        unknown_set_to_zero : bool
            Flag to specify whether to set unknown interactions to zero.
            This is useful for models that lack specific amino acids/residues we don't
            want to throw an error for.

        Returns
        -------------
        None

        """
        # make sure all resigroup pairs can be calculated based on passed parameter 
        #  and build reference dictionary
        ## NOTE NEED TO ADD CHECKERS FOR UNFOUND PAIRS ##

        # reset the lookup table so we don't hold over any previous
        # pairs
        self.lookup = {}

        # extract out all residues into a single list. Note the list/set nested operation
        # ensures valid aa is a non-redundant list. 
        valid_aa = list(set([res for sublist in self.valid_residue_groups for res in sublist]))

        # note this cycles through all possible residue pairs as defined by all of the valid
        # amino acids in the valid residue groups        
        for r1 in valid_aa:

            self.lookup[r1] = {}            
            for r2 in valid_aa:
                # this parameter function (compute_interaction_parameters()) is a required
                # function in a valid finches.forcefield.

                # use a try/except statement to zero out missing pairs IF we
                # request this to happen
                try:
                    self.lookup[r1][r2] = self.parameters.compute_interaction_parameter(r1,r2)[0]

                # take a gamble on error handling that an invalid parameter will trigger
                # a KeyError, and handle it IF unknown_set_to_zero is True.
                except KeyError as e:
                    if unknown_set_to_zero:

                        # we may remove this at some point but for now we're going to print
                        # the raw error before printing the specific pair being set to zero
                        # as well.
                        print(e)
                        print(f'WARNING: Unknown residue pair {r1}-{r2}, setting to zero.')
                        self.lookup[r1][r2] = 0.0
                    else:
                        raise Exception(f'ERROR: {e} for {r1} and {r2}.')


    ## ................................................................................... ##
    ##
    ##                
    def _update_parameters(self, new_parameters):
        """

        Function which wipes the current parameters in the model and updates with a
        new set. This 

        This does NOT update

        self.charge_prefactor 
        self.sequence_converter
        self.null_interaction_baseline

        each of which can be updated by their own stand-alone functions defined below.

        Parameters
        ------------
        new_parameters : finches.forcefield.<model>.<model_object>
            Instance of one of the forcefield objects found in finches.forcefields 
            module. This object contains all of the parameters for the model, such
            that the Interaction_Matrix_Constructor calls on a series of functions
            that a model presents to calculate the associated interactions in
            a consistent way. See the constructor for a more complete description.

        Returns
        ----------
        None
            No return but updates this object appropriately        

        """

        # update the parameters object
        self.parameters = new_parameters

        # update the valid residue groups 
        self.valid_residue_groups = new_parameters.ALL_RESIDUES_TYPES 

        # update the valid amino acid interaction lookup table
        # based on the updated self.valid_residue_group and the
        # updated parameters object
        self._update_lookup_dict()

        
    ## ................................................................................... ##
    ##
    ##                
    def _update_null_interaction_baseline(self, min_len=10, max_len=500, verbose=True):
        """
        Function to compute and update the null interaction baseline 
        for specific passed self.parameters model. This works by calling: 

            data.forcefield_dependencies.get_null_interaction_baseline

        The self.null_interaction_baseline parameter is then updated.

        *thanks Alex K. for the recomendation on how to organize this feature 
        
        Parameters
        ---------------
        model : obj
            An instantiation of the one of the forcefield class object 
            IE self.parameters 

        min_len : int 
            The minimum length of a polyGS sequence used 

        max_len : int 
            The minimum length of a polyGS sequence used
        """
        if verbose:
            # remind user that the null_interaction_baseline is being updated
            print(f'Recomputing the null_interaction_baseline for {self.parameters.version}...')

        # return the theretical baseline (ibl) where the slope of epsilon vs polyGS(n) == 0 
        null_interaction_baseline = get_null_interaction_baseline(self, min_len=min_len, max_len=max_len)

        if verbose:
            # remind user that the null_interaction_baseline is being updated
            print(f' the new baseline = {null_interaction_baseline}')

        self.null_interaction_baseline = null_interaction_baseline


    ## ................................................................................... ##
    ##
    ##                        
    def _update_charge_prefactor(self, reference_data='DAS_KAPPA_RG_MPIPI', prefactor_range=None, verbose=True):
        """
        Function to compute and update the charge prefactor
        for specific passed self.parameters model. This works by calling: 
            data.forcefield_dependencies.get_charge_prefactor

        The self.charge_prefactor parameter is then updated.

        *thanks Alex K. for the recomendation on how to organize this feature 
        
        Parameters
        ---------------
        model : obj
            An instantiation of the one of the forcefield class object 
            IE self.parameters 

        reference_data : list 
            dataset to be used for computing the charge prefactor, this dataset 
            should be organized as a list of tuples where the tuples contain 
            three values in the order of (sequence, Rg(y-value), Kappa(x-value)) 
            where the slope of the computed epsion vs x-value will be matched to 
            that of Rg(y-value) vs Kappa(x-value). 

            The defult here is DAS_KAPPA_RG_MPIPI stored: 
                finches.data.reference_sequence_info.DAS_KAPPA_RG_MPIPI 

        prefactor_range : tuple 
            A two position tuple defining the minimum and maximum prefactors 
            to itterate over (min, max). If None, defult is hard coded in as (0,2)

        """
        if verbose:
            # remind user that the charge_prefactor is being updated
            print(f'Recomputing the charge_prefactor for {self.parameters.version}...')

        # return the charge_prefactor where the slope of:
        #    epsilon vs reference_data(x) == reference_data(y) vs reference_data(x) 
        charge_prefactor = get_charge_prefactor(self, reference_data=reference_data, prefactor_range=prefactor_range)

        if verbose:
            # remind user that the charge_prefactor is being updated
            print(f' the new charge prefactor = {charge_prefactor}')

        self.charge_prefactor = charge_prefactor

    ## ................................................................................... ##
    ##
    ##                        
    def _check_sequence(self, sequence):
        """
        Function that checks that passed sequence contains (1) 
        Residues found in ONE of the self.valid_residue_groups 
        and (2) a single sequene ONLY has residues that originate
        from a single residue group.

        Parameters
        --------------
        sequence : str
            Sequence being tested

        Returns
        ----------------
        None
        
        """

        
        
        found_lists = set()
        unique_residues = set(sequence)


        # this keeps tracks of how many DIFFERENT residue groups
        # we found residue in sequence that intersected with. It SHOULD
        # be only 1
        residue_group_count = 0

        # keep track of how many of the uniqu_residues we've seen
        hits = 0
        for residue_group in self.valid_residue_groups:

            # find amino acids in both the unique_residue set and
            # in the current residue_group
            common_values = unique_residues.intersection(residue_group)

            # if our set had one or more overlaps
            if common_values:
                residue_group_count += 1
                hits = hits + len(common_values)
                
        if len(unique_residues) > 0:
            if residue_group_count > 1:
                raise Exception(f'INVALID SEQUENCE - input sequence below contains a mix of valid residues from different groups: \n {sequence}')
            elif hits < len(unique_residues):
                raise Exception(f'INVALID SEQUENCE - unknow residue found in input sequence below: \n {sequence}')
        else:
            raise Exception(f'INVALID SEQUENCE - no input sequence passed')

    ## ------------------------------------------------------------------------------
    ##
    def get_converted_sequence(self, sequence):
        """
        Function to return sequence passed through the sequence converter

        NB: use to be called get_custom_sequence

        Parameters
        ----------
        sequence : str
            Amino acid sequence of interest
        
        Returns
        ----------
        str
            converted sequence from self.sequence_converter(sequence)
        
        """

        # convert if appropriate
        if self.sequence_converter:
            outseq = self.sequence_converter(sequence)
        else:
            outseq = sequence

        # check sequence is valid
        self._check_sequence(outseq)
        
        return outseq


    ######################################################################
    ##                                                                  ##
    ##                                                                  ##
    ##              FUNCTIONS FOR MATRIX CALCULATION                    ##
    ##                                                                  ##
    ##                                                                  ##
    ######################################################################

    ## ------------------------------------------------------------------------------
    ##
    def calculate_pairwise_homotypic_matrix(self, sequence, convert_to_custom=True):
        
        """
        Interaction_Matrix_Constructor.calculate_pairwise_homotypic_matrix

        Note in reality you should just do all pairwise-residues ONCE 
        at the start and then look 'em up, but this code below does the
        dynamic non-redudant on-the-fly calculation of unique pairwise
        residues.

        Parameters
        ---------------
        sequence : str
            Amino acid sequence of interest

        Returns
        ------------------
        np.array
            Returns an (n x n) matrix with pairwise interactions; recall
            that negative values are attractive and positive are repulsive!

        """
        return self.calculate_pairwise_heterotypic_matrix(sequence, sequence, convert_to_custom=convert_to_custom)
        
    ## ------------------------------------------------------------------------------
    ## 
    def calculate_pairwise_heterotypic_matrix(self, sequence1, sequence2, convert_to_custom=True, use_cython=True):
        
        """
        Interaction_Matrix_Constructor.calculate_pairwise_heterotypic_matrix

        This function takes two sequences and returns a sequence1 x sequence2
        2D np.array

        Note in reality you should just do all pairwise-residues ONCE 
        at the start and then look 'em up, but this code below does the
        dynamic non-redudant on-the-fly calculation of unique pairwise
        residues.

        Note we default to using a cython implementation which is 3.5x faster
        than the pure python implementation.

        Parameters
        ---------------
        sequence : str
            Amino acid sequence of interest

        Returns
        ------------------
        np.array
            Returns an (len(s1) x len(s2)) matrix with pairwise interactions; 
            recall that negative values are attractive and positive are 
            repulsive!

        """

        # note we have the if/else statement here because sequence_converter
        # does its own checking post conversion
        if convert_to_custom:
            sequence1 = self.sequence_converter(sequence1)
            sequence2 = self.sequence_converter(sequence2)
        else:
            self._check_sequence(sequence1)
            self._check_sequence(sequence2)


        if use_cython:
            return matrix_manipulation.dict2matrix(sequence1, sequence2, self.lookup)
            
        else:                
            matrix = []
            for r1 in sequence1:
                tmp = []
            
                for r2 in sequence2:                
                    tmp.append(self.lookup[r1][r2])
                matrix.append(tmp)
            
            return np.array(matrix)


    
    ## ------------------------------------------------------------------------------
    ## 
    def calculate_weighted_pairwise_matrix(self,
                                           sequence1,
                                           sequence2,
                                           convert_to_custom=True, 
                                           charge_prefactor=None,
                                           CHARGE=True,
                                           ALIPHATICS=True):
        
        """
        Interaction_Matrix_Constructor.calculate_pairwise_heterotypic_matrix


        Calculate heterotypic matrix and weight the matrix.

        Parameters
        ---------------
        sequence1 : str
            Amino acid sequence of interest

        sequence2 : str
            Amino acid sequence of interest
        
        prefactor : float 
            Model specific value to plug into the local charge weighting of 
            the matrix 

        CHARGE : bool
            Flag to select whether weight the matrix by local sequence charge 

        ALIPHATICS : bool
            Flag to select whether weight the matrix by local patches of aliphatic 
            residues

        Returns
        ------------------
        np.array
            Returns an (n x n) matrix with pairwise interactions; recall
            that negative values are attractive and positive are repulsive, 
            this matrix is weighted by local sequence context of the two 
            inputed sequences.

        """
        
        # compute matrix - note if s1 and s2 are the same that's fine
        matrix = self.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)

        # weight the matrix by local sequence charge
        if CHARGE:

            if charge_prefactor == None:
                charge_prefactor = self.charge_prefactor 

            # the w_mask adds a weighting factor for cross-residues where
            # both residues are of the same type (and surrounded by residues
            # of the same type) it means those repulsive interactions are
            # supressed
            w_mask = get_charge_weighted_mask(sequence1, sequence2)
            try:

                # for the matrix*w_mask calculation, MOST elements in that w_mask array
                # are zero, so most values get zeroed out other than charge residues. Then
                # multiplying by the charge_prefactor which is a scalar gives you a forcefield
                # sepecific weighting factor for charge residues which gets subtracted off
                # the actual matrix to give you a weighted matrix
                w_matrix = matrix - (matrix*w_mask*charge_prefactor)
            except Exception as e:
                print(e)
                raise Exception('Possible issue: INVALID charge_prefactor, check to ensure self.charge_prefactor is defined.')
                print(e)
        else:
            w_matrix = matrix

        # weight the matrix by local patches of aliphatic residues
        if ALIPHATICS:
            w_ali_mask = get_aliphatic_weighted_mask(sequence1, sequence2)
            w_matrix = w_matrix*w_ali_mask

        # note that additional weights or corrections could be added here as needed

        return w_matrix


    ######################################################################
    ##                                                                  ##
    ##                                                                  ##
    ##              FUNCTIONS FOR EPSILON CALCULATION                   ##
    ##                                                                  ##
    ##                                                                  ##
    ######################################################################

    def calculate_epsilon_vectors(self,
                            sequence1,
                            sequence2,
                            CHARGE=True,
                            ALIPHATICS=True):
        """
        Function that returns the attractive and repulsive epsilon vectors for
        the two sequences passed.

        This is a wrapper around the stateless get_sequence_epsilon_vectors().

        Parameters
        --------------
        sequence1 : str
            First sequence to compare

        sequence2 : str
            Second sequence to compare

        CHARGE : bool
            Flag to select whether weight the matrix by local sequence charge 

        ALIPHATICS : bool
            Flag to select whether weight the matrix by local patches of 
            aliphatic residues

        Returns
        ---------------
        tuple
            Returns a tuple of two lists where element 1 is the attractive
            vector and element 2 is the repulsive vector. 
       

        """

        # note this runs charge_preactor and null_interaction_baseline
        # as null which means the default values associated with this
        # object will be used
        return get_sequence_epsilon_vectors(sequence1,
                                            sequence2,
                                            self,
                                            CHARGE=CHARGE,
                                            ALIPHATICS=ALIPHATICS)

    ## ------------------------------------------------------------------------------
    ## 
    def calculate_epsilon_value(self,
                          sequence1,
                          sequence2,
                          CHARGE=True,
                          ALIPHATICS=True):

        """
        Function that returns the overall epsilon value associated with a 
        pair of sequences, as calculated using this
        This is a wrapper around the stateless get_sequence_epsilon_value().

        Parameters
        --------------
        sequence1 : str
            First sequence to compare

        sequence2 : str
            Second sequence to compare

        CHARGE : bool
            Flag to select whether weight the matrix by local sequence charge 

        ALIPHATICS : bool
            Flag to select whether weight the matrix by local patches of 
            aliphatic residues

        Returns
        ---------------
        float
            Single value reporting on the average sequence:sequence interaction
            

        """
            
        # note this runs charge_prefactor and null_interaction_baseline
        # as null which means the default values associated with this
        # object will be used
        return get_sequence_epsilon_value(sequence1,
                                          sequence2,
                                          self,
                                          CHARGE=CHARGE,
                                          ALIPHATICS=ALIPHATICS)

    ## ------------------------------------------------------------------------------
    ## 
    def calculate_sliding_epsilon(self,
                                  sequence1,
                                  sequence2,
                                  window_size=31,
                                  CHARGE=True,
                                  ALIPHATICS=True,
                                  use_cython=True):
        """
        Function that returns the sliding epsilon value associated with a
        pair of sequences, as calculated using this object.

        Specifically, this returns a matrix that rather than using individual
        inter-residue distance to calculate interaction parameters, it takes
        a sliding window of size window_size and calculates the average epsilon
        between the two subsequences sequences for each window. This provides a
        matrix that reports on the average interaction between residues smoothed
        over a window of size window_size.

        Note that we don't pad the sequence here, so the edges of the matrix start
        and end at indices that depend on the window size. To avoid confusion, the
        function also returns the indices for sequence1 and sequence2.

        Note that to plot the returned matrix you can use the following code:

            B = X.calculate_sliding_epsilon(s1, s2)

            fig, ax = plt.subplots(figsize=(10,10))

            plt.imshow(B[0], extent=[B[1][0], B[1][-1], B[2][0], B[2][-1]], aspect='auto', vmax=4, vmin=-4, cmap='seismic', origin='lower')

        A few important points here:
        
        1) setting aspect='auto' is important to ensure the matrix is plotted
        with the correct aspect ratio.

        2) the vmax and vmin are set to -4 and 4 to give reasonable values for
        the MpipiGG forcefield. If you're using a different forcefield you may
        want to adjust these values.

        3) the origin='lower' is important to ensure the matrix is plotted
        with the correct orientation (i.e. 0,0 is in the lower left corner)

        4) the extent=[B[1][0], B[1][-1], B[2][0], B[2][-1]] is important to
        ensure the matrix is plotted with the correct sequence indices. 


        Parameters
        --------------
        sequence1 : str
            First sequence to compare

        sequence2 : str
            Second sequence to compare

        window_size : int
            Size of the sliding window to use. Note that this should be an odd
            number, and if it's not it will be rounded up to the next odd number.


        CHARGE : bool
            Flag to select whether weight the matrix by local sequence charge

        ALIPHATICS : bool
            Flag to select whether weight the matrix by local patches of
            aliphatic residues

        use_cython : bool
            Flag to select whether to use the cythonized version of the code
            or the python version. The cythonized version is reduces the time
            to about 5-10% of the python version. 

        Returns
        ---------------
        tuple
            Returns a tuple of 3 elements. The first is the matrix of sliding
            epsilon values, and the second and 3rd are the indices that map
            sequence position from sequence1 and sequence2 to the matrix

        Raises
        ---------------
        Exception
            If the window size is larger than either of the sequences


        """

        # INTERNAL FUNCTION; this is written as an internal function because
        # going forward we'll probably cythonize this for better performance
        # -----------------------------------------------------
        def __matrix2eps(in_matrix):
            """
            Local function to calculate the epsilon value for a single matrix

            Parameters
            ---------------
            in_matrix : np.array
                Matrix to calculate epsilon value for

            Returns
            ----------------
            float
                Epsilon value for the matrix

            """
            attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(in_matrix, self.null_interaction_baseline)

            attractive_matrix = attractive_matrix - self.null_interaction_baseline
            repulsive_matrix  = repulsive_matrix  - self.null_interaction_baseline

        
            return np.sum(np.mean(attractive_matrix, axis=1)) + np.sum(np.mean(repulsive_matrix, axis=1))
        # -----------------------------------------------------

        # check that windowsize is odd, and if not make it odd
        if window_size % 2 == 0:
            print(f"Warning: window size is even, rounding up to next odd number {window_size+1}")
            window_size = window_size + 1
                
        
        # calculate weight pairwise matrix
        w_matrix = self.calculate_weighted_pairwise_matrix(sequence1,
                                                           sequence2,
                                                           CHARGE=CHARGE,
                                                           ALIPHATICS=ALIPHATICS)
                                                           

        # default to cython version, which is MUCH faster
        if use_cython:
            return  matrix_manipulation.matrix_scan(w_matrix, window_size, self.null_interaction_baseline)
                                                        

        # get dimensions of matrix
        l1 = w_matrix.shape[0]
        l2 = w_matrix.shape[1]

        # check for window size larger than matrix size
        if l1 < window_size or l2 < window_size:
            raise Exception('Window size is larger than matrix size, cannot calculate sliding epsilon')

        
        # calculate sliding epsilon for all possible intermolecular windows. 
        everything = []
        for i in range(0,(l1-window_size)+1):
            tmp = []
            for j in range(0, (l2-window_size)+1):        
                tmp.append(__matrix2eps(w_matrix[i:i+window_size,j:j+window_size]))

            everything.append(tmp)

        everything = np.array(everything)
            
        # finally, determine indices for sequence1 
        start = int((window_size-1)/2)
        end   = l1 - start
        seq1_indices = np.arange(start,end)

        # and sequence2
        start = int((window_size-1)/2)
        end   = l2 - start
        seq2_indices = np.arange(start,end)
                
        return (everything, seq2_indices, seq1_indices)


    

######################################################################
##                                                                  ##
##                                                                  ##
##              FUNCTIONS FOR MATRIX MANIPULATION                   ##
##                                                                  ##
##                                                                  ##
######################################################################

## ---------------------------------------------------------------------------
##
def get_attractive_repulsive_matrixes(matrix, null_interaction_baseline):
    """
    Take interaction array, descritize it by above or below interaction baseline,
    Return two shaped matched matrixes for attractive and repulsive values
    
    The null_interaction_baseline = value to split matrix. This has been built such 
        that this value recapitulates PolyGS for the specific input model being use 
        to see more on how to compute a null_interaction_baseline see... 

        NEED TO UPDATE HERE 
    
    Parameters
    ---------------
    matrix : np.array
        array returned by a function in the Interaction_Matrix_Constructor class

    null_interaction_baseline : float
        Value to specify where to split the matrix for attractive vs repulsive interactions
    
    Returns
    ------------------
    attractive_matrix : np.array
        An array the same shape of the input matrix with only values below the 
        null_interaction_baseline

    repulsive_matrix : np.array
        An array the same shape of the input matrix with only values below the 
        null_interaction_baseline

    """
    attractive_matrix = (matrix < null_interaction_baseline)*matrix
    repulsive_matrix =  (matrix > null_interaction_baseline)*matrix

    return attractive_matrix, repulsive_matrix


## ---------------------------------------------------------------------------
##
def mask_matrix(matrix, column_mask):
    """
    Function to take matrix and multipy it by a mask. This 
    also check to make sure the mask is the same shape. 

    Parameters
    ---------------
    matrix : array 
        A 2D matrix as an array with the shape of (seqence1, seqence2)

    column_mask : array 
        A 2D array with the shape of the inputed matrix

    Returns
    ------------------
    out_matrix : array 
        A 2D matrix with the same shape of the inputed matrix
        where the out_matrix = matrix*column_mask

    """
    # check to ensure matrix and mask are same shape 
    if matrix.shape == column_mask.shape:
        return matrix*column_mask
    else:
        raise Exception('column_mask and matrix are not the same shape')

    
## ---------------------------------------------------------------------------
##

# commented out and to remove
"""
def flatten_matrix_to_vector(matrix, orientation=[0,1]):

    Function to convert matrix into interaction vectors.

    Parameters
    ---------------
    matrix : array 
        A 2D matrix as an array with the shape of (seqence1, seqence2)
    
    orientation : int
        Flag to specify whether to flatten to matrix along the X or Y axis. 
        1 refers to X axis (mean of rows) (vector relative to sequence1)
        0 refers to Y axis (mean of columns) (vector relative to sequence2)


    Returns
    ------------------
    vector : array 
        A 1D array the length of the columns in input matrix. This vector of 
        ist the mean along the vertical axis for each column in the matrix and 
        is normalized by the model specific null_interaction_baseline.

    return np.mean(matrix, axis=orientation)
    """

######################################################################
##                                                                  ##
##                                                                  ##
##               BUILDING VECTORS & COMPUTING EPSILON               ## 
##                                                                  ##
##                                                                  ##
######################################################################

## ---------------------------------------------------------------------------
##
def get_sequence_epsilon_vectors(sequence1,
                                 sequence2,
                                 X,
                                 charge_prefactor=None,
                                 null_interaction_baseline=None,
                                 CHARGE=True,
                                 ALIPHATICS=True):
    """
    Function to epsilon vectors between a pair of passed sequences
    returned vectors are relative to sequence1 such that len(sequence1) equals 
    the len(returned_vectors)

    NOTE this code was previously : get_weighted_sequence_epsilon_value 
        It is now UPDATED to get_sequence_epsilon_vectors and  get_sequence_epsilon_value
        all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        The first sequence to compare 

    sequence2 : str
        The second sequence to compare 

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise interactions 
        and modelspecific parameters 

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

    ALIPHATICS : bool
        Flag to select whether weight the matrix by local patches of aliphatic 
        residues
    
    Returns
    --------
    attractive_vector : list 
        attractive epsilon vector of sequence1 relative to sequence2

    repulsive_vector : list 
        repulsive epsilon vector of sequence1 relative to sequence2 
    
    """

    # check for baseline 
    if not null_interaction_baseline:
        null_interaction_baseline = X.null_interaction_baseline

    # get interaction matrix for said sequence
    w_matrix = X.calculate_weighted_pairwise_matrix(sequence1,
                                                    sequence2,
                                                    convert_to_custom=True, 
                                                    charge_prefactor=charge_prefactor,
                                                    CHARGE=CHARGE,
                                                    ALIPHATICS=ALIPHATICS)
    
    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(w_matrix, null_interaction_baseline)
    
    attractive_matrix = attractive_matrix - null_interaction_baseline
    repulsive_matrix = repulsive_matrix - null_interaction_baseline

    # take average over the matrix rows to get attractive and repulsive values
    attractive_vector = np.mean(attractive_matrix, axis=1) 
    repulsive_vector = np.mean(repulsive_matrix, axis=1)   

    # return attractive and repulsive vectors
    return attractive_vector, repulsive_vector


## ---------------------------------------------------------------------------
##
def get_sequence_epsilon_value(sequence1,
                               sequence2,
                               X,
                               charge_prefactor=None,
                               null_interaction_baseline=None,
                               CHARGE=True,
                               ALIPHATICS=True):
    """
    Function to epsilon value between a pair of passed sequences

    NOTE this was previously : get_weighted_sequence_epsilon_value 
        It is now UPDATED to get_sequence_epsilon_value and all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        The first sequence to compare 

    sequence2 : str
        The second sequence to compare 

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise interactions 
        and modelspecific parameters 

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

    ALIPHATICS : bool
        Flag to select whether weight the matrix by local patches of aliphatic 
        residues
    
    Returns
    --------
    epsilon : float 
        sequence epsilon value as computed between sequence1 and sequence2 
    
    """

    # get attractive and repulsive vectors 
    attractive_vector, repulsive_vector = get_sequence_epsilon_vectors(sequence1,
                                                                       sequence2,
                                                                       X,
                                                                       charge_prefactor=charge_prefactor,
                                                                       null_interaction_baseline=null_interaction_baseline,
                                                                       CHARGE=CHARGE,
                                                                       ALIPHATICS=ALIPHATICS)
    
    # sum vectors to get attractive and repulsive values
    attractive_value = np.sum(attractive_vector)
    repulsive_value = np.sum(repulsive_vector)

    # sum attractive and repulsive values
    return attractive_value + repulsive_value


## ---------------------------------------------------------------------------
##
def get_interdomain_epsilon_vectors(sequence1,
                                    sequence2,
                                    X,
                                    SAFD_cords,
                                    charge_prefactor=None,
                                    null_interaction_baseline=None,
                                    CHARGE=True,
                                    IDR_positon=['Cterm','Nterm','CUSTOM'],
                                    origin_index=None, 
                                    sequence_of_ref='sequence1'):
    """
    Function to epsilon vectors between the surface of a folded domain 
    and a directly ajoining attached IDR sequence. This epsilon value is weighted by
    the local sequence context and the likly reachable Solvent Accessable esidues on 
    the surface of the folded domain.

    NOTE this was previously part of : get_XYZ_weighted_sequence_epsilon_value 
        It is now UPDATED to get_interdomain_epsilon_value and all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        the first sequence (FOLDED DOMAIN) and only SAFD residues.
        everyresidue in this sequence should be SA and in FD. 
        
        To generate this from a PDB see:
            PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs 
        
        sequence1 is the first output of the above function.

    sequence2 : str
        The second sequence to compare (the IDR)

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise
        interactions and modelspecific parameters 

    SAFD_cords : list 
        Sequence mask of sequence1 containing the solvent accessable folded domain (SAFD)
        residue cordinates where len(SAFD_cords) == len(sequence1) 

        This list should be organized such that: 
          values that are NOT solvent accessable and NOT in a folded domain = 0 
          values that are solvent accessable and NOT in a folded domain = [x, y, z]
        
        This SAFD_cords can be returned by PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs

        SAFD_cords is the third output of the above function in PDB_structure_tools.

    IDR_positon : str 
        Flag to denote whether the IDR sequence (sequence2) is directly 'C-terminal' or 'N-terminal'
        of the inputed Folded Domain (sequence1). If 'CUSTOM' the origin_index flag must be set to 
        a specific index in SAFD_cords.

    origin_index : int 
        Optional value formated like on of indexes in the SAFD_cords list that will be used as the 
        point of origin for where the IDR is attached to the fold domain. Defult here is None.  

        NOTE - IF THIS IS PASSED IDR_positon must be set to CUSTOM)

    sequence_of_ref : str 
        Flag to denote whether to build the interaction vectors relative to 'sequence1' or 'sequence2'

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

        NOTE - NO weighting of aliphatics is conducted here because aliphatic weighting is 
        only performed between groups of local aliphatic residues, and no groups are caluculated 
        on the surface of folded domains, therefor all aliphatics who still be treated as if they 
        they are in isolation. 
    
    Returns
    --------
    epsilon : float 
        sequence epsilon value as computed between sequence1 and sequence2 
    
    """
    if IDR_positon not in ['Cterm','Nterm','CUSTOM']:
        raise Exception(f'INVALID IDR_positon passed')
    
    if sequence_of_ref not in ['sequence1','sequence2']:
        raise Exception(f'INVALID sequence_of_ref passed')

    # check for charge_prefactor  
    if not charge_prefactor:
        charge_prefactor = X.charge_prefactor

    # check for baseline 
    if not null_interaction_baseline:
        null_interaction_baseline = X.null_interaction_baseline

    # check origin location 
    if IDR_positon == 'CUSTOM':
        if not origin_index:
            raise Exception('When IDR_positon is set to CUSTOM the origin_index must be set an index in (X,Y,Z) the cordinate')
        try:
            list(map(lambda a: float(a)), origin_index)
        except:
            raise Exception('origin_index must be set to (X,Y,Z) cordinate where XYZ can be floats')

    # parse sequence_of_ref flag 
    orientation = {'sequence1':1,'sequence2':0}[sequence_of_ref] 

    # check to make sure sequence1 is the Folded Domain
    if len(sequence1) != len(SAFD_cords):
        raise Exception('Length of sequence1 does not match length of SAFD cordinates \n sequence1 should be the fold domain sequence')

    # get interaction matrix for said sequence
    matrix = X.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)
    
    # get mask for IDR residues relative resisdues on FD
    # just returns bionary mask
    w_xyz_mask = build_column_mask_based_on_xyz(matrix, SAFD_cords, IDR_positon=IDR_positon, origin_index=origin_index)

    # NOTE - NO weighting of aliphatics is conducted here because aliphatic weighting is 
    #   only performed between groups of local aliphatic residues, and no groups are caluculated 
    #   on the surface of folded domains, therefor all aliphatics who still be treated as if they 
    #   they are in isolation.  

    #  only occurs between on surface residue and IDR window.
    w_mask = get_charge_weighted_FD_mask(sequence1, sequence2) 
    w_matrix = matrix - (matrix*w_mask*charge_prefactor)

    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(w_matrix, null_interaction_baseline)

    # NOTE - filtering the matrix is done after the matrix is split give filtering just multiplys 
    #        by 0 and null_interaction_baseline may fluctuate and is likly not zero 

    # multiply matrix by 1&0s to screen out IDRs residues that cant reach
    wXYZ_attractive_matrix = mask_matrix(attractive_matrix - null_interaction_baseline, w_xyz_mask) 
    wXYZ_repulsive_matrix = mask_matrix(repulsive_matrix - null_interaction_baseline, w_xyz_mask) 

    # original code left here...
    #attractive_vector = flatten_matrix_to_vector(wXYZ_attractive_matrix, orientation=orientation)
    #repulsive_vector = flatten_matrix_to_vector(wXYZ_repulsive_matrix, orientation=orientation)
    attractive_vector = np.mean(wXYZ_attractive_matrix, axis=orientation)
    repulsive_vector = np.mean(wXYZ_repulsive_matrix, axis=orientation)

    # return attractive and repulsive vectors
    return attractive_vector, repulsive_vector


## ---------------------------------------------------------------------------
##
def get_interdomain_epsilon_value(sequence1,
                                  sequence2,
                                  X,
                                  SAFD_cords,
                                  charge_prefactor=None,
                                  null_interaction_baseline=None,
                                  CHARGE=True,
                                  IDR_positon=['Cterm','Nterm', 'CUSTOM'],
                                  origin_index=None,
                                  sequence_of_ref='sequence1'):
    """
    Function to compute epsilon value between the surface of a folded domain 
    and a directly ajoining attached IDR sequence. This epsilon value is weighted by
    the local sequence context and the likly reachable Solvent Accessable esidues on 
    the surface of the folded domain.

    NOTE this was previously : get_XYZ_weighted_sequence_epsilon_value 
        It is now UPDATED to get_interdomain_epsilon_value and all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        the first sequence (FOLDED DOMAIN) and only SAFD residues.
        everyresidue in this sequence should be SA and in FD. 
        
        To generate this from a PDB see:
            PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs 
        
        sequence1 is the first output of the above function.

    sequence2 : str
        The second sequence to compare (the IDR)

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise
        interactions and modelspecific parameters 

    SAFD_cords : list 
        Sequence mask of sequence1 containing the solvent accessable folded domain (SAFD)
        residue cordinates where len(SAFD_cords) == len(sequence1) 

        This list should be organized such that: 
          values that are NOT solvent accessable and NOT in a folded domain = 0 
          values that are solvent accessable and NOT in a folded domain = [x, y, z]
        
        This SAFD_cords can be returned by PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs

        SAFD_cords is the third output of the above function in PDB_structure_tools.

    IDR_positon : str 
        Flag to denote whether the IDR sequence (sequence2) is directly 'C-terminal' or 'N-terminal'
        of the inputed Folded Domain (sequence1). If 'CUSTOM' the origin_index flag must be set to 
        a specific index in SAFD_cords.

    origin_index : int 
        Optional value formated like thes indexes in the SAFD_cords list that will be used as the 
        point of origin for where the IDR is attached to the fold domain. Defult here is None. 

        NOTE - IF THIS IS PASSED IDR_positon must be set to CUSTOM

    sequence_of_ref : str 
        Flag to denote whether to build the interaction vectors relative to 'sequence1' or 'sequence2'

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

        NOTE - NO weighting of aliphatics is conducted here because aliphatic weighting is 
        only performed between groups of local aliphatic residues, and no groups are caluculated 
        on the surface of folded domains, therefor all aliphatics who still be treated as if they 
        they are in isolation. 
    
    Returns
    --------
    epsilon : float 
        sequence epsilon value as computed between sequence1 and sequence2 
    
    """
    # get attractive and repulsive matrixes 
    attractive_vector, repulsive_vector = get_interdomain_epsilon_vectors(sequence1,
                                                                          sequence2,
                                                                          X,
                                                                          SAFD_cords, 
                                                                          charge_prefactor=charge_prefactor,
                                                                          origin_index=origin_index,
                                                                          null_interaction_baseline=null_interaction_baseline,
                                                                          CHARGE=CHARGE,
                                                                          IDR_positon=IDR_positon, 
                                                                          sequence_of_ref=sequence_of_ref)

    # itegerate under vectors to get attractive and repulsive values
    attractive_value = np.sum(attractive_vector)
    repulsive_value = np.sum(repulsive_vector)

    # sum attractive and repulsive vectors to get sequence1 centric vector
    return attractive_value + repulsive_value

