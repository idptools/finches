"""
Class to build Interation Matrix from Mpipi forcefield By


values : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""
import numpy as np
import math
from itertools import product

from .data import forcefield_dependencies
from . import parsing_aminoacid_sequences

from .PDB_structure_tools import build_column_mask_based_on_xyz

from .utils import matrix_manipulation
from . import epsilon_stateless

# -------------------------------------------------------------------------------------------------
class InteractionMatrixConstructor:
    
    def __init__(self,
                 parameters,
                 sequence_converter=False,
                 charge_prefactor=None,
                 null_interaction_baseline=None, 
                 compute_forcefield_dependencies=False):
        
        """
        The InteractionMatrixConstructor is a class which houses user-facing
        functions for calculating inter-residue interactions. This is appropriate
        if you want more fine-grained control over epsilon-associated calculations.
        Alternatively, using one of the finches.frontend modules may be an easier 
        route if you're just doing "standard" types of analysis.

        Philosophically speaking, the goal here is to separate the biophysical
        model used to determine interaction parameters into a set of classes 
        housed in finches.forcefields, and then the class InteractionMatrixConstructor       
        takes one of those models in as initializing input and provides a 
        standardized way to calculate the same types of interaction properties 
        derived from many different biophysical models. In software engineering, this
        is known as an "interface" design pattern.

        For example, using Mpipi_GGv1, one might do:

            # import key modules
            from finches.forcefields.mpipi import Mpipi_model
            from finches.epsilon_calculation import InteractionMatrixConstructor
            
            # Initialize a finches.forcefields.Mpipi.Mpipi_model object
            Mpipi_GGv1_params = Mpipi_model(version = 'Mpipi_GGv1')

            # initialize an InteractionMatrixConstructor
            IMC = InteractionMatrixConstructor(parameters = Mpipi_GGv1_params)

        The InteractionMatrixConstructor object (aka IMC) then provides functionality 
        for calculating inter-residue interactions using the energetics in the 
        underlying model. Moreover, the IMC object can also then be updated in a
        variety of ways.
        
        Parameters
        -----------
        parameters : finches.forcefield.<model>.<model_object>
            Instance of one of the forcefield objects found in finches.forcefields 
            module. This object contains all of the parameters for the model, such
            that the InteractionMatrixConstructor calls on a series of functions
            that a model presents to calculate the associated interactions in
            a consistent way.

            In this way, different interaction models can be distributed but the 
            same analysis code (using an InteractionMatrixConstructor) can always
            be used.

            This parameters object has two key functions that are required to be
            implemented.

            * parameters.ALL_RESIDUES_TYPES : list of lists which define residues
              which are allowed to occur in the same type of sequence; e.g., we expect
              a protein list, an RNA list, a DNA list etc. Note that EVERY residue in
              all lists must have a pair-wise interaction parameter calculatable via
              the compute_interaction_parameter() function (described below)
                        
            * parameters.compute_interaction_parameter(r1,r2) : function which takes
              two valid residues (i.e. any pair from the residues defined in 
              ALL_RESIDUES_TYPES) and returns a value that reports on the relative
              preferential interaction between those residues.

            * parameters.CONFIGS: dictionary with some general default 
              precomputed values for the forcefield, including: 'charge_prefactor' 
              and, 'null_interaction_baseline', which will be used if these are
              not explicitly passed into this constructor.
                      
        sequence_converter : function 
            A function that takes in a sequence and converts the sequence to 
            an alternative sequnence that matches the acceptable residue types of the 
            parameters object. If no function is provided, a default function that 
            returns the input sequences is provided. This can be useful if 
            we need to mask sequences in a certain way.

        charge_prefactor : float 
            Model-specific value that defines how the charge weighting is scaled. 
            Charge weighting is implemented in a way that runs oppositely 
            charged residues are less repulsive for one another than they might 
            otherwise be, mimicking the fact that charged sidechains can point away 
            from one another and/or be influenced by pKa shifts. The 
            charge_prefactor is a scalar which in effect, scales the strength of this 
            scaling has and typically needs to be tuned on a per forcefield basis. 
            Note that this value must be between 0 and 1.

        null_interaction_baseline : float  
            Model-specific threshold to differentiate between attractive and repulsive
            interactions. By default, this baseline is parameterized based on the attractive/
            repulsive interaction associated with a poly(GS) sequence, which we expect
            to behave as a Gaussian chain, however, we can over-ride the default value. 
            

            NOTE -  null_interaction_baseline is specific to the parameter version. 
                    The null_interaction_baseline is the value used to split matrix. 
                    This has been built such that this value recapitulates PolyGS for                    
                    the specific input model being used. Precomputed null_interaction 
                    baseline can be found in the CONFIGS dictionary of the parameters
                    object.

                    To compute a new null_interaction_baseline for a new forcefield
                    see:

                        data.forcefield_dependencies.get_null_interaction_baseline

        compute_forcefield_dependencies : bool 
            Flag to specify whether to recompute the model-specific 
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

        # This is the main function that sets up the self.lookup table and ensures self.parameters maps to
        # a valid object
        self._update_parameters(parameters)
        
        # check null_interaction_baseline
        if self.null_interaction_baseline == None: 
            try:
                self.null_interaction_baseline = self.parameters.CONFIGS['null_interaction_baseline']
            except Exception as e: 
                if compute_forcefield_dependencies:
                    print(f'Recomputing the null_interaction_baseline for {self.parameters.version}...')
                    self.null_interaction_baseline = forcefield_dependencies.get_null_interaction_baseline(self)
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
                print("A charge prefactor must be provided to use charge weighting")
                raise Exception("A charge prefactor must be provided to use charge weighting and one is not defined in the forcefield CONFIGS dictionary.")


                    
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
            that the InteractionMatrixConstructor calls on a series of functions
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
        Function that calculates the homotypic matrix WITHOUT any weighting factors.
        
        Note - the matrix here is the raw unsliding-windowed matrix so will be
        len(seq) by len(seq) long. For visualizing local regions of inter-residue
        interaction use the calculate_sliding_epsilon() function.
        
        In reality, you should just do all pairwise-residues ONCE 
        at the start and then look 'em up, but this code below does the
        dynamic non-redundant on-the-fly calculation of unique pairwise
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
        This function takes two sequences and returns a sequence1 x sequence2
        2D np.array.
        Note - the matrix here is the raw unsliding-windowed matrix so will be
        len(seq) by len(seq) long. For visualizing local regions of inter-residue
        interaction use the calculate_sliding_epsilon() function.

        In reality, you should just do all pairwise-residues ONCE 
        at the start and then look 'em up, but this code below does the
        dynamic non-redundant on-the-fly calculation of unique pairwise
        residues.

        Note we default to using a Cython implementation which is 3.5x faster
        than the pure Python implementation. For now we're leaving in the 
        option to fall back to a Python implementation but that may be 
        removed at some point...

        Parameters
        ---------------
        sequence : str
            Amino acid sequence of interest

        sequence2: str
            Second amino acid sequence of interest

        use_cython : bool
            Flag to select whether to use the cythonized version of the code
            or the Python version. The cythonized version reduces the time
            to about 5-10% of the Python version. 

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
        
        
    def calculate_pairwise_heterotypic_matrix_structnidr_smoothed(self, structured_sequence, idr_sequence, 
                                                                  linear_window, spatial_window, 
                                                                  convert_to_custom=True, use_cython=True):
        
        """
        This function takes two sequences and returns a sequence1 x sequence2
        2D np.array.
        Note - the matrix here is the raw unsliding-windowed matrix so will be
        len(seq) by len(seq) long. For visualizing local regions of inter-residue
        interaction use the calculate_sliding_epsilon() function.

        In reality, you should just do all pairwise-residues ONCE 
        at the start and then look 'em up, but this code below does the
        dynamic non-redundant on-the-fly calculation of unique pairwise
        residues.

        Note we default to using a Cython implementation which is 3.5x faster
        than the pure Python implementation. For now we're leaving in the 
        option to fall back to a Python implementation but that may be 
        removed at some point...

        Parameters
        ---------------
        sequence : str
            Amino acid sequence of interest

        sequence2: str
            Second amino acid sequence of interest

        use_cython : bool
            Flag to select whether to use the cythonized version of the code
            or the Python version. The cythonized version reduces the time
            to about 5-10% of the Python version. 

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
                                           use_charge_weighting=True,
                                           use_aliphatic_weighting=True,
                                           use_cython=True):
        
        """
        Calculate heterotypic matrix and then weight the matrix based on
        the various flags defined here.  

        Parameters
        ---------------
        sequence1 : str
            Amino acid sequence of interest

        sequence2 : str
            Amino acid sequence of interest
                                                                                     
        
        prefactor : float 
            Model specific value to plug into the local charge weighting of 
            the matrix 

        use_charge_weighting : bool
            Flag to select whether weight the matrix by local sequence charge 

        use_aliphatic_weighting : bool
            Flag to select whether weight the matrix by local patches of aliphatic 
            residues

        use_cython : bool
            Flag to select whether to use the cythonized version of the code
            or the python version. The cythonized version is reduces the time
            to about 5-10% of the python version. 

        Returns
        ------------------
        np.array
            Returns an (n x n) matrix with pairwise interactions; recall
            that negative values are attractive and positive are repulsive, 
            this matrix is weighted by local sequence context of the two 
            inputed sequences.

        """
        
        # compute matrix - note if s1 and s2 are the same that's fine
        matrix = self.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=convert_to_custom, use_cython=use_cython)

        # weight the matrix by local sequence charge
        if use_charge_weighting:

            if charge_prefactor == None:
                charge_prefactor = self.charge_prefactor 

            # calculate attractive and repulsive masks. Note as of 2024-04-07 we ONLY populate the repulsive mask, so the
            # attractive matrix (first element here) is returned to _ so we just ignore i.
            (_, repulsive_mask) = parsing_aminoacid_sequences.get_charge_weighted_mask(sequence1, sequence2)
            
            np.set_printoptions(threshold=np.inf)
            
                  
            try:

                # for the matrix*w_mask calculation, MOST elements in that w_mask array
                # are zero, so most values get zeroed out other than charge residues. Then
                # multiplying by the charge_prefactor which is a scalar gives you a forcefield
                # specific weighting factor for charge residues which gets added/subtracted off
                # the actual matrix to give you a weighted matrix

                # we ADD the attractive matrix, so negative values become more negative
                
                # NOTE that one implementation split the charge weighting into attractive
                # and repulsive components but this turned out to be suboptimal for charge,
                # however we're leaving the code here for other types of weighting in the
                # future...
                # w_matrix = matrix   + (matrix*attractive_mask*charge_prefactor)

                # we SUBSTRACT the repulsive matrix, so positive numbers become smaller
                # (but still positive if charge_prefactor is < 1)
                w_matrix = matrix - (matrix*repulsive_mask*charge_prefactor)

                """
                print('Repulsive Mask')
                print(repulsive_mask)
                print('Matrix')
                print(matrix)
                print('Weighted Matrix')
                print(w_matrix)
                print('delta')
                print(matrix - w_matrix)
                """
                
            except Exception as e:
                print(e)
                raise Exception('Possible issue: INVALID charge_prefactor, check to ensure self.charge_prefactor is defined.')
                print(e)
        else:
            w_matrix = matrix

        # weight the matrix by local patches of aliphatic residues
        if use_aliphatic_weighting:
            w_ali_mask = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(sequence1, sequence2)
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
                            use_charge_weighting=True,
                            use_aliphatic_weighting=True):
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

        use_charge_weighting : bool
            Flag to select whether weight the matrix by local sequence charge 

        use_aliphatic_weighting : bool
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
        return epsilon_stateless.get_sequence_epsilon_vectors(sequence1,
                                                              sequence2,
                                                              self,
                                                              use_charge_weighting=use_charge_weighting,
                                                              use_aliphatic_weighting=use_aliphatic_weighting)

    ## ------------------------------------------------------------------------------
    ## 
    def calculate_epsilon_value(self,
                          sequence1,
                          sequence2,
                          use_charge_weighting=True,
                          use_aliphatic_weighting=True):

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

        use_charge_weighting : bool
            Flag to select whether weight the matrix by local sequence charge 

        use_aliphatic_weighting : bool
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
        return epsilon_stateless.get_sequence_epsilon_value(sequence1,
                                          sequence2,
                                          self,
                                          use_charge_weighting=use_charge_weighting,
                                          use_aliphatic_weighting=use_aliphatic_weighting)

    ## ------------------------------------------------------------------------------
    ## 
    def calculate_sliding_epsilon(self,
                                  sequence1,
                                  sequence2,
                                  window_size=31,
                                  use_charge_weighting=True,
                                  use_aliphatic_weighting=True,
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

        use_charge_weighting : bool
            Flag to select whether weight the matrix by local sequence charge

        use_aliphatic_weighting : bool
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
            sequence position from sequence1 and sequence2 to the matrix. Note
            that matrix positions index in protein space, so the first residue
            is 1

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
            attractive_matrix, repulsive_matrix = epsilon_stateless.get_attractive_repulsive_matrices(in_matrix, self.null_interaction_baseline)

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
                                                           use_charge_weighting=use_charge_weighting,
                                                           use_aliphatic_weighting=use_aliphatic_weighting)

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
            
        # finally, determine indices for sequence1 - add 1 to return indices
        # in protein numbering instead of python numbering. 
        start = int((window_size-1)/2) + 1
        end   = (l1 - start) + 1
        seq1_indices = np.arange(start,end+1)

        # and sequence2
        start = int((window_size-1)/2) + 1
        end   = (l2 - start) + 1
        seq2_indices = np.arange(start,end+1)

        # make sure everything is hunky dory...
        assert len(seq1_indices) == everything.shape[0]
        assert len(seq2_indices) == everything.shape[1]
        
                
        return (everything, seq2_indices, seq1_indices)




class ArbitraryFilterInteractionMatrixContructor(InteractionMatrixConstructor):
    def __init__(self,
                 parameters,
                 weight_function=None,
                 sequence_converter=False,
                 charge_prefactor=None,
                 null_interaction_baseline=None, 
                 compute_forcefield_dependencies=False):
        '''Initializes the arbitrary filtering needed for IDR structure interations
        Mostly the same as the InterationMatrixConstructor but with a needed weighting function for the filter.
        
        The InteractionMatrixConstructor is a class which houses user-facing
        functions for calculating inter-residue interactions. This is appropriate
        if you want more fine-grained control over epsilon-associated calculations.
        Alternatively, using one of the finches.frontend modules may be an easier 
        route if you're just doing "standard" types of analysis.

        Philosophically speaking, the goal here is to separate the biophysical
        model used to determine interaction parameters into a set of classes 
        housed in finches.forcefields, and then the class InteractionMatrixConstructor       
        takes one of those models in as initializing input and provides a 
        standardized way to calculate the same types of interaction properties 
        derived from many different biophysical models. In software engineering, this
        is known as an "interface" design pattern.

        For example, using Mpipi_GGv1, one might do:

            # import key modules
            from finches.forcefields.mpipi import Mpipi_model
            from finches.epsilon_calculation import InteractionMatrixConstructor
            
            # Initialize a finches.forcefields.Mpipi.Mpipi_model object
            Mpipi_GGv1_params = Mpipi_model(version = 'Mpipi_GGv1')

            # initialize an InteractionMatrixConstructor
            IMC = InteractionMatrixConstructor(parameters = Mpipi_GGv1_params)

        The InteractionMatrixConstructor object (aka IMC) then provides functionality 
        for calculating inter-residue interactions using the energetics in the 
        underlying model. Moreover, the IMC object can also then be updated in a
        variety of ways.
        
        Parameters
        -----------
        parameters : finches.forcefield.<model>.<model_object>
            Instance of one of the forcefield objects found in finches.forcefields 
            module. This object contains all of the parameters for the model, such
            that the InteractionMatrixConstructor calls on a series of functions
            that a model presents to calculate the associated interactions in
            a consistent way.

            In this way, different interaction models can be distributed but the 
            same analysis code (using an InteractionMatrixConstructor) can always
            be used.

            This parameters object has two key functions that are required to be
            implemented.

            * parameters.ALL_RESIDUES_TYPES : list of lists which define residues
              which are allowed to occur in the same type of sequence; e.g., we expect
              a protein list, an RNA list, a DNA list etc. Note that EVERY residue in
              all lists must have a pair-wise interaction parameter calculatable via
              the compute_interaction_parameter() function (described below)
                        
            * parameters.compute_interaction_parameter(r1,r2) : function which takes
              two valid residues (i.e. any pair from the residues defined in 
              ALL_RESIDUES_TYPES) and returns a value that reports on the relative
              preferential interaction between those residues.

            * parameters.CONFIGS: dictionary with some general default 
              precomputed values for the forcefield, including: 'charge_prefactor' 
              and, 'null_interaction_baseline', which will be used if these are
              not explicitly passed into this constructor.
                      
        sequence_converter : function 
            A function that takes in a sequence and converts the sequence to 
            an alternative sequnence that matches the acceptable residue types of the 
            parameters object. If no function is provided, a default function that 
            returns the input sequences is provided. This can be useful if 
            we need to mask sequences in a certain way.

        charge_prefactor : float 
            Model-specific value that defines how the charge weighting is scaled. 
            Charge weighting is implemented in a way that runs oppositely 
            charged residues are less repulsive for one another than they might 
            otherwise be, mimicking the fact that charged sidechains can point away 
            from one another and/or be influenced by pKa shifts. The 
            charge_prefactor is a scalar which in effect, scales the strength of this 
            scaling has and typically needs to be tuned on a per forcefield basis. 
            Note that this value must be between 0 and 1.

        null_interaction_baseline : float  
            Model-specific threshold to differentiate between attractive and repulsive
            interactions. By default, this baseline is parameterized based on the attractive/
            repulsive interaction associated with a poly(GS) sequence, which we expect
            to behave as a Gaussian chain, however, we can over-ride the default value. 
            

            NOTE -  null_interaction_baseline is specific to the parameter version. 
                    The null_interaction_baseline is the value used to split matrix. 
                    This has been built such that this value recapitulates PolyGS for                    
                    the specific input model being used. Precomputed null_interaction 
                    baseline can be found in the CONFIGS dictionary of the parameters
                    object.

                    To compute a new null_interaction_baseline for a new forcefield
                    see:

                        data.forcefield_dependencies.get_null_interaction_baseline

        compute_forcefield_dependencies : bool 
            Flag to specify whether to recompute the model-specific 
            charge_prefactor and null_interaction_baseline used when computing epsilon if
            these are not available at runtime. 
        '''
        #initialize the parent class
        super().__init__(parameters,
                 sequence_converter,
                 charge_prefactor,
                 null_interaction_baseline, 
                 compute_forcefield_dependencies)
        #initialize the weighting function to use for computation later
        self.weight_function = weight_function
        
    
    '''
    properties
    '''
    @property
    def threshold_offset(self) ->  float:
        '''This computes the value that corresponds to the original epsilons 0 value.'''
        #check if _threshold_offset has been previously calculated
        if hasattr(self, '_threshold_offset'):
            return self._threshold_offset
        else: #if it hasnt check that the weight_function is not 
            if self.weight_function is None:
                return None
            else: #need to calculate the _threshold_offset
                seq_gs = 'GS'*20
                #seq_gs = 'GS'*2 + 'G'
                # r_gs = np.array([-2,-1,0,1,2])
                # self._threshold_offset = self.calc_filtered_region(seq_gs, r_gs*3.5, seq_gs, r_gs, offset=False)
                mat,_ = self.compute_offset_term(seq_gs,seq_gs,2,2*3.5)
                self._threshold_offset= np.mean(mat)
                return self._threshold_offset
                
    
    '''
    methods
    '''
    
    def compute_offset_term(self, seq1 : str,seq2 : str, window_extent : int,
                                                       window_equivilent_spatial_distance : float,
                                                       full_extent : bool = False):
        '''basically a copy of the front end function that does the same thing'''
        #check that there is a wieghting function that this object was initialized with
        filter_func = self.weight_function
        if filter_func is None:
            raise Exception(f"There is no weighting function in this object. You must have initialized with a weighting function for this to work")
        
        #find the indexes to be used to loop based on the filter window
        if full_extent == True:
            ids1 = np.arange(len(seq1))
            ids2 = np.arange(len(seq2))
        else:
            ids1 = np.arange(window_extent, len(seq1)-window_extent,dtype=int)
            ids2 = np.arange(window_extent, len(seq1)-window_extent,dtype=int)
            
        #get all the pairs of ids to loop over
        pairs = list(product(ids1,ids2))
        
        #compute the conversion factor
        seq_dist_factor = window_equivilent_spatial_distance/window_extent
        
        #loop over the ids
        ret_mat = np.zeros((len(seq1),len(seq2)),dtype=float)
        for (i,j) in pairs:
            #do the same for the idr
            m = i- window_extent
            if m < 0:
                m = 0
            n = i + window_extent+1
            if n > len(seq1):
                n = len(seq1)
            idr_negihbor_resid1 = np.arange(m,n)
            #grab the sequence bit
            idr_str1 = [seq1[k] for k in idr_negihbor_resid1]
            #grabe the distances
            idr_local_distance1 = np.array([seq_dist_factor*np.abs(k-i) for k in idr_negihbor_resid1])
            
            #do the same for the idr
            #do the same for the idr
            m = j- window_extent
            if m < 0:
                m = 0
            n = j + window_extent+1
            if n > len(seq2):
                n = len(seq2)
            idr_negihbor_resid2 = np.arange(m,n)
            #grab the sequence bit
            idr_str2 = [seq2[k] for k in idr_negihbor_resid2]
            #grabe the distances
            idr_local_distance2 = np.array([seq_dist_factor*np.abs(k-j) for k in idr_negihbor_resid2])
            
            #compute the 
            ret_mat[i,j] = self.calc_filtered_region(idr_str1, idr_local_distance1, idr_str2, idr_local_distance2, offset=False)
                
        return ret_mat, (ids1, ids2)
        
        
    def apply_weighted_averaging(self, interaction_vec : np.ndarray, distance_from_center_vec : np.ndarray) -> float:
        '''This function applies the weighted average to the 
        
        '''
        #determine the wieghts based on the weighting function
        weights = self.weight_function(interaction_vec)
        #compute the weighted interaction
        weighted_interaction =  weights* interaction_vec
        #return the weighted average
        nsum = np.sum(weighted_interaction)
        dsum = np.sum(weights)
        res = nsum/dsum
        return res
    
    def determine_referenced_interaction_strength_vec(self, seq : str, ref_let : str):
        return np.array([self.lookup[ref_let][vvv] for vvv in seq])
    
    def calc_filtered_region(self, seq1 : np.ndarray, r1 : np.ndarray, 
                             seq2 : np.ndarray, r2 : np.ndarray,
                             offset=True):
        '''This function calculates the filtered value of the region based on the 
        
        
        Parameters
        ----------
        seq1 : numpy.ndarray
            This is the first sequence that is in the window associated with the first protein
        r1 : numpy.ndarray
            Distance values in the seq1 dimension
        seq2 : numpy.ndarray
            This is the second sequence that is in the window associated with the second protein
        r2 : numpy.ndarray
            Distance values in the seq2 dimension
        split : bool
            This is the flag that determines if the data is going to get split. 
            If it is going to get split then this will split the interaction values
            prior to filtering between the values above/inclusive and strictly below the 
            split_thresh.
        split_thresh : float
            This is the threshold that splits the interaction strength between "attractive" and 
            "repulsive". 
            
        Returns
        -------
        ret_val : numpy.ndarray or tuple
            If the split bool is False then this function returns a numpy.ndarray
            that is the value for the filtered matrix at the interaction locations
            of interest. If the split boolean is True, then the ouput is a tuple
            where the first value is the mean for the repulsive interactions and
            the second is the mean for the attactive interactions.      
        '''
        #check that seq1 has the same length of r1 and same for seq2 and r2
        if len(seq1) != len(r1) or len(seq2) != len(r2):
            raise Exception(f"Error in Calc_filtered_region. Length of sequence did not match length of radius")
        
        #generate the tuple of all posibble combos from seq1 and seq2
        #(radius1, radius2, interaction value)
        linear_index_pairings = list(product(range(len(r1)),range(len(r2)))) #fairly fast way to get this without a nested for loop
        data_concat = np.array([[r1[i], r2[j], self.lookup[seq1[i]][seq2[j]]]for (i,j) in linear_index_pairings])
        
        #pull the component distances and turn them into a real radius based on an orthogonality argument
        disp_mat = data_concat[:,0:2]
        distance_vec = np.linalg.norm(disp_mat, axis=1)
        
        #determine functional behavior based on if a split in the data is to be considered on interaction strenght
        ret_val = None
        interaction_vec = data_concat[:,2]
    # if split:
    #     #split the matrix into positive and negative values
    #     pos_binary = interaction_vec>=split_thresh
    #     neg_binary = interaction_vec<split_thresh
    #     pos_vec = (pos_binary)*interaction_vec - split_thresh
    #     neg_vec = (neg_binary)*interaction_vec - split_thresh
    #     #apply the weighting function to the local neighborhood
    #     filt_pos = self.apply_weighted_averaging(pos_vec, distance_vec)
    #     filt_neg = self.apply_weighted_averaging(neg_vec, distance_vec)
    #     #create a tuple to pass back
    #     ret_val = (filt_pos*np.sum(pos_binary) +  filt_neg*np.sum(neg_binary))/len(interaction_vec) #remember that positive is repulsive
    # else:
    #     #compute the weighted average
        ret_val = self.apply_weighted_averaging(interaction_vec, distance_vec)
        if offset:
            ret_val = ret_val - self.threshold_offset
        
        #return
        return ret_val
    
    def remove_empty_rows_or_columns(result_matrix : np.ndarray) -> tuple:
        '''Removing a row or column if it is all zeros.
        
        This function removes the rows or columns that are all zeroes and then
        catalogs the remaining rows and columns to ensure the user still knows which are there.
        
        Parameters
        ----------
        result_matrix : np.ndarray
            This is the matrix that is the result of the 
        
        Returns
        -------
        tuple 
            This is tuple with the reduced matrix first, the first axis indexes, and then the second axis
            index that remained post the reduction.
        '''
        #check that there are ony 2 dimensions to the matrix passed
        if len(result_matrix.shape) != 2:
            raise Exception(f"Removing empty rows assumes that are only two dimensions in the matrix. The passed matrix has {len(result_matrix.shape)} dimensions")
        
        #find the number of columns and rows
        num_rows, num_cols = result_matrix.shape
        
        #loop over the rwos and check if all the values are 0s
        rm_row_indexes = []
        for i in range(num_rows):
            #pull the values
            row_vec = result_matrix[i]
            #check if the row vec is all 0s
            if np.all(row_vec == 0):
                rm_row_indexes.append(i)
        #do the same for ht ecolumns
        rm_col_indexes = []
        for j in range(num_cols):
            #pull the column of interest
            col_vec = result_matrix[:,j]
            #check if the column is all 0s
            if np.all(col_vec == 0):
                rm_col_indexes.append(j)
                
        #remove the columns and rows that have all zeros
        cleaned_rows = np.delete(rm_row_indexes, result_matrix, axis = 0)
        cleaned_mat = np.delete(rm_col_indexes, cleaned_rows, axis = 1)
        
        #grabe the indexes that are still left in each axis
        p1_ind = [k for k in range(num_rows) if k not in rm_row_indexes]
        p2_ind = [k for k in range(num_cols) if k not in rm_col_indexes]
        
        #return the output
        return cleaned_mat, p1_ind, p2_ind