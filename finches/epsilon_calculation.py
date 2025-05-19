"""
Class to build Interation Matrix from Mpipi forcefield By


values : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06

Updated by: Nicholas Razo
2025-05-16
"""
import numpy as np
from scipy.ndimage import convolve1d
from scipy import signal
import math

from .data import forcefield_dependencies
from . import parsing_aminoacid_sequences

from .PDB_structure_tools import build_column_mask_based_on_xyz

from .utils import matrix_manipulation
from . import epsilon_stateless

# -------------------------------------------------------------------------------------------------
CANONICAL_AMINOACID_ORDERED = ['M', 'G', 'K', 'T', 'Y', 'A', 'D', 'E', 'V', 'L', 'Q', 'W', 'R', 'F', 'S', 'H', 'N', 'P', 'C', 'I']

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
    def _update_lookup_vec_dif(self) -> None:
        """
        Function which wipes any previous inter-residue interaction difference vector representation
        computes it assuming that the lookup_vec is always updated. Please note that these vectors
        are signed. It assumes that the first index is the 'initial' vector and the second
        vector is the final vector.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # find the length of the lookup vector
        lookvec_len = len(self.lookup_vec)

        # loop over each amino acid in the list that defines the encoding order
        ending_result = []
        for i in range(lookvec_len):
            temp = []
            for j in range(lookvec_len):
                #take the difference between the initial and final vectors interaction
                vi = self.lookup_vec[i]
                vf = self.lookup_vec[j]
                temp.append(vf - vi)

            #add the temp to the ending result
            ending_result.append(temp)

        # set the result to difference lookup vec
        self.lookup_vec_dif =  ending_result

    def _update_lookup_vec(self) -> None:
        """
        Function which wipes any previous inter-residue interaction vectorization scheme
        and recomputes the parameters (the 'lookup_vec'). This DOES NOT update the 
        lookup dictionary and the vectorized form is computed based on lookup. You must
        call _update_lookup_dict prior to callig this function.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # loop over each amino acid in the list that defines the encoding order
        ending_result = []
        for a1 in CANONICAL_AMINOACID_ORDERED:
            #this is the interaction list for a amino acid
            temp = []
            for a2 in CANONICAL_AMINOACID_ORDERED:
                temp.append(self.lookup[a1][a2])
            # append the amino acid to the end result
            ending_result.append(temp)

        #save the result as a numpy array
        self.lookup_vec = np.array(ending_result)

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

        # update the lookup_vec (vectorial form of the lookup dict)
        self._update_lookup_vec()

        # lastly update the difference vectors
        self._update_lookup_vec_dif()


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
    def vector_encode_seq(self, sequence : str) -> list[np.ndarray]:
        """
        This function is here to produce a vector encoding of a sequence.
        The vector encoding is the interaction term for that residue to all
        other amino acids.
        The result is a list of numpy arrays. The arrays are 'row' vectors.

        Parameters
        ----------
        sequence : str
            This is the string you wish to convert to its amino acid interaction representation

        Returns
        -------
        list[numpy.ndarray]
            This is the list of numpy array encodings for interaction strength. The length of this will
            be the same as the input string.
        """
        # loop over each character in the string
        ret_val = []
        for aa in sequence:
            # find the index in the list for the current amino acid
            aa_idx = CANONICAL_AMINOACID_ORDERED.index(aa)
            ret_val.append(self.lookup_vec[aa_idx])

        # return the converted sequence
        return ret_val
    
    def vector_decode_seq(self, sequence : list[np.ndarray]) -> list[dict[str,np.ndarray]]:
        """
        Decodes a sequence of interaction vectors into a more interpretable format.
        
        This function takes a list of interaction vectors and converts each vector into a 
        dictionary that maps amino acid one-letter codes to their corresponding interaction 
        values. This makes it easier to interpret the interaction profile for each position
        in the sequence.

        Parameters
        ----------
        sequence : list[np.ndarray]
            List of interaction vectors, where each vector contains interaction values 
            with all canonical amino acids

        Returns
        -------
        list[dict[str,np.ndarray]]
            List of dictionaries, one per position in the sequence. Each dictionary maps 
            amino acid one-letter codes to their interaction values
            
        Example
        -------
        >>> imc = InteractionMatrixConstructor(parameters)
        >>> vec_seq = imc.vector_encode_seq('MG')
        >>> decoded = imc.vector_decode_seq(vec_seq)
        >>> decoded[0]['A']  # gets interaction value between M and A at position 0
        array([0.5])
        """
        #loop over every position in the sequence
        ret_val = []
        for i,vec in enumerate(sequence):
            # loop over each amino acid at each position to learn about the interaciton
            temp = {}
            for j, aa in enumerate(CANONICAL_AMINOACID_ORDERED):
                temp[aa] = vec[j]
            # add the value tp the return list
            ret_val.append(temp)
    
    def position_encode_seq(self, sequence : str) -> list[int]:
        """
        This function is here to produce a postiion encoding of a sequence (CANONICAL_AMINOACID_ORDERED).
        
        The result is a list of integers that correspond to the index in CANONICAL_AMINOACID_ORDERED.

        Parameters
        ----------
        sequence : str
            This is the string you wish to convert to its amino acid interaction representation

        Returns
        -------
        list[int]
            A list of indexes that correspond to CANONICAL_AMINOACID_ORDERED
        """
        # loop over each character in the string
        ret_val = []
        for aa in sequence:
            # find the index in the list for the current amino acid
            aa_idx = CANONICAL_AMINOACID_ORDERED.index(aa)
            ret_val.append(aa_idx)

        # return the converted sequence
        return ret_val
    
    def position_decode_seq(self, sequence : list[int]) -> str:
        """
        Decodes a positionally encoded sequence back into an amino acid sequence string.
        
        This function takes a list of integer indices and converts each index back to its 
        corresponding amino acid using the CANONICAL_AMINOACID_ORDERED list. The amino acids
        are then joined together into a single string.

        Parameters
        ----------
        sequence : list[int]
            List of integer indices corresponding to positions in CANONICAL_AMINOACID_ORDERED

        Returns
        -------
        str
            The decoded amino acid sequence as a string
            
        Example
        -------
        >>> imc = InteractionMatrixConstructor(parameters)
        >>> encoded = [0, 1, 2]  # corresponds to ['M', 'G', 'K']
        >>> imc.position_decode_seq(encoded)
        'MGK'
        """
        # loop over every character in the seqence
        ret_val = []
        for aa in sequence:
            #convert the integer into a character
            ret_val.append(CANONICAL_AMINOACID_ORDERED[aa])

        # convert the list to a string and return it
        return ''.join(ret_val)

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




    ######################################################################
    ##                                                                  ##
    ##                                                                  ##
    ##              FUNCTIONS FOR SEQUENCE CHEMISTRY                    ##
    ##                                                                  ##
    ##                                                                  ##
    ######################################################################


    def moving_weighted_average_of_vectors_valid(self, vector_list : list[np.ndarray], kernel_weights : np.ndarray, 
                                                 mode : str ='nearest', cval : float =0.0):
        """
        Computes a moving weighted average over a sequence of vectors, returning only
        the "valid" part where the kernel fully overlaps the input data.

        Parameters
        ----------
        vector_list : list[np.ndarray]
            A list of 1D NumPy arrays (vectors). Assumed all vectors have same length.

        kernel_weights : np.ndarray
            A 1D NumPy array representing the filter kernel.

        mode : str, optional
            The mode parameter for handling boundaries during convolution.
            Default is 'nearest'.

        cval : float, optional
            Value to fill past edges of input if mode is 'constant'.
            Default is 0.0.

        Returns
        -------
        np.ndarray
            A 2D NumPy array where each row is a resulting averaged vector.
            The sequence will be shorter than the input by (len(kernel_weights) - 1).
            If len(vector_list) < len(kernel_weights), returns empty array.

        Raises
        ------
        ValueError
            If vector_list is empty, kernel_weights is empty, or kernel_weights sum to zero.
        """
        if not vector_list:
            return np.array([]) # Or np.empty((0,0)) if features known, but best to be consistent
        if not isinstance(kernel_weights, np.ndarray) or kernel_weights.ndim != 1 or kernel_weights.size == 0:
            raise ValueError("kernel_weights must be a non-empty 1D NumPy array.")

        try:
            data_matrix = np.array(vector_list)
        except Exception as e:
            raise ValueError(f"Could not convert vector_list to a NumPy array. Ensure all vectors have the same dimension. Error: {e}")

        if data_matrix.ndim == 1: # List of scalars was passed
            if len(vector_list) > 0 and isinstance(vector_list[0], (int, float, np.number)):
                data_matrix = data_matrix.reshape(-1, 1)
            else: # Single vector passed, make it a row in a 2D array
                data_matrix = data_matrix.reshape(1, -1)


        if data_matrix.ndim != 2:
            raise ValueError("Input vector_list must produce a 2D array when stacked (a sequence of vectors).")
        
        num_input_vectors, num_features = data_matrix.shape
        if num_features == 0 and num_input_vectors > 0: # list of empty arrays
            # Kernel length for slicing calculation
            K = len(kernel_weights)
            if num_input_vectors < K:
                return np.empty((0, 0))
            valid_len = num_input_vectors - K + 1
            return np.empty((valid_len, 0))
        elif num_features == 0 and num_input_vectors == 0: # empty list
            return np.empty((0,0))


        kernel_sum = np.sum(kernel_weights)
        if np.isclose(kernel_sum, 0):
            raise ValueError("Sum of kernel weights is close to zero, cannot compute a weighted average.")
        
        normalized_kernel = kernel_weights / kernel_sum
        K = len(normalized_kernel)

        if num_input_vectors < K:
            # Not enough data points for even one full kernel application
            return np.empty((0, num_features))

        # Perform convolution, this will produce an output of the same shape as data_matrix along axis 0
        averaged_vectors_matrix_full = convolve1d(
            data_matrix,
            weights=normalized_kernel,
            axis=0,
            mode=mode,
            cval=cval
        )

        # Calculate slice indices for the "valid" part
        # The 'valid' output has length N - K + 1
        # For a centered kernel (default origin in convolve1d):
        # Trim (K-1)//2 from the start
        # Trim K//2 from the end (which means the end slice index is N - K//2)
        
        start_slice_index = (K - 1) // 2
        # The end_slice_index for Python slicing (exclusive) needs to result in N - K + 1 elements.
        # end_slice_index = start_slice_index + (num_input_vectors - K + 1)
        # Or, calculated from the end:
        end_slice_index = num_input_vectors - (K // 2) # num_input_vectors is N

        # Ensure start_slice_index is not greater than end_slice_index (handles N<K implicitly too)
        if start_slice_index >= end_slice_index :
            return np.empty((0, num_features))


        valid_averaged_vectors = averaged_vectors_matrix_full[start_slice_index:end_slice_index, :]

        return valid_averaged_vectors
    
    def compute_region_interaction_vectors(self, sequence : str, region_size : int, 
                                           kernal_type : str = 'flat',
                                           **kwargs) -> np.ndarray:
        """
        Computes interaction vectors for regions of a sequence using a sliding window approach.
        Each vector represents the average interaction profile for a region of specified size.

        Parameters
        ----------
        sequence : str
            The amino acid sequence to analyze.

        region_size : int
            The size of the sliding window/region to compute interactions for.

        kernal_type : str, optional
            The type of kernel to use for weighted averaging.
            Options are 'flat' (uniform) or 'gaussian'.
            Default is 'flat'.

        **kwargs : dict
            Additional keyword arguments:
            - std : float
                Standard deviation for Gaussian kernel (only used if kernal_type='gaussian').
                Default is 1.0.

        Returns
        -------
        np.ndarray
            A 2D array where each row represents the averaged interaction vector 
            for a region. Shape is (n_regions, n_interaction_types) where n_regions
            depends on sequence and region size, and n_interaction_types is the 
            length of the interaction vector for each residue.

        Notes
        -----
        The output array will be shorter than the input sequence due to edge effects
        from the sliding window. Only positions where the window fully overlaps 
        the sequence are included.
        """
        # encode the sequence
        enc_seq = self.vector_encode_seq(sequence=sequence)

        # determine the window that the user requested
        if kernal_type == 'flat':
            kern = np.ones(region_size, dtype=float)
        if kernal_type == 'gaussian':
            #get the standard deviation if it was passed otherwise set it equal to 1
            std = kwargs.get('std', 1.0)
            kern = signal.windows.gaussian(region_size, std=std)

        # call the moving average filter 
        return self.moving_weighted_average_of_vectors_valid(enc_seq, kernel_weights=kern)
    
    def compute_region_chemical_heterogeneity_vectors(self, sequence : str, region_size : int) -> np.ndarray:
        """
        Computes chemical heterogeneity vectors for sliding windows along a sequence.
        
        For each window of size region_size, this function evaluates the chemical 
        heterogeneity by considering all possible pairs of positions within the window
        and calculating their interaction vector differences. The absolute values of 
        these differences are averaged to produce a measure of chemical heterogeneity 
        for each window position.

        Parameters
        ----------
        sequence : str
            The amino acid sequence to analyze.
        region_size : int
            The size of the sliding window to use for calculating heterogeneity.

        Returns
        -------
        np.ndarray
            A 2D numpy array where each row represents the averaged chemical heterogeneity
            vector for a window position. The number of rows will be len(sequence) - region_size,
            and the number of columns matches the dimensionality of the interaction vectors.
        """
        # find the length of the sequence
        seq_len = len(sequence)

        #position encode the sequence
        enc_seq = self.position_encode_seq(sequence)

        # get all unique combinations of local indexes within a window
        all_local_combos = []
        for i in range(region_size-1):
            for j in np.arange(i+1,region_size):
                all_local_combos.append((i,j))

        # loop over each window
        ret_val = []
        for k in range(seq_len - region_size + 1):
            #loop over each unique local index combination
            temp = []
            for (first_idx, second_idx) in all_local_combos:
                #pull the encoded positions for each index
                p1 = enc_seq[first_idx + k]
                p2 = enc_seq[second_idx + k]

                #get the difference in chemical specificity for that pair of residues being swapped
                vec_dif = self.lookup_vec_dif[p1][p2]

                # append the absolute value of the vector difference to the temporary list
                temp.append(np.abs(vec_dif))

            #compute the mean value across the columns (average value of each vec_dif)
            mean_vec = np.mean(np.array(temp), axis=0)

            # append this to the return value
            ret_val.append(mean_vec)

        #return the vectorial heterogeneity values
        return np.array(ret_val)
                
    
    def determine_region_interaction_norm(self, avg_vec_seq : np.ndarray) -> np.ndarray:
        """
        Calculates the Euclidean norm (magnitude) of interaction vectors for each position.
        
        This function computes the L2 norm of each row vector in the averaged vector sequence.
        It can be interepreted as providing two measures: interaction strength or heterogeneity.
        When passing a sequence of intera ction vectors it provides a measure of the overall
        interaction strength at each position regardless of whether the interactions are favorable 
        or unfavorable. when passing a sequence of interaction differences it provides a measure
        of chemicical heterogeneity in that region.

        Parameters
        ----------
        avg_vec_seq : np.ndarray
            A 2D numpy array where each row represents the interaction vector for a position
            in the sequence. The columns correspond to interactions with each canonical amino acid.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the L2 norm of each row vector, representing the
            overall interaction strength at each position in the sequence.
        """
        seq_list = []
        for i in range(len(avg_vec_seq)):
            # pull the vector for the interactions of that residue
            v1 = avg_vec_seq[i]

            # take the norm of the vector
            v1_norm = np.linalg.norm(v1)

            # add the value to the list
            seq_list.append(v1_norm)

        #return 
        return np.array(seq_list)
    
    def determine_region_logo_manipulation(self, avg_vec_seq: np.ndarray, threshold: float = 0.0, 
                                                operation: str = 'lt') -> list[dict[str,float]]:
        """
        Analyzes an averaged vector sequence to determine amino acid interactions based on a threshold.
        
        For each position in the averaged vector sequence, this function identifies which amino 
        acids have interaction values that satisfy the specified threshold condition and returns 
        their absolute magnitudes. Values that don't meet the condition are set to 0.0.

        Parameters
        ----------
        avg_vec_seq : np.ndarray
            A 2D numpy array where each row represents the interaction vector for a position
            in the sequence. The columns correspond to interactions with each canonical amino acid.

        threshold : float, optional
            The threshold value to compare interaction values against. Default is 0.

        operation : str, optional
            The comparison operation to use. Options are:
            - 'lt': less than (value < threshold)
            - 'gt': greater than (value > threshold)
            Default is 'lt'.

        Returns
        -------
        list[dict[str,float]]
            A list of dictionaries, one per position in avg_vec_seq. Each dictionary maps
            amino acid one-letter codes to their interaction strength (absolute value of 
            the interaction). Amino acids not meeting the threshold condition are assigned 0.0.

        Raises
        ------
        ValueError
            If operation is not 'lt' or 'gt'.
        """
        if operation not in ['lt', 'gt']:
            raise ValueError("Operation must be either 'lt' or 'gt'")

        seq_list = []
        for i in range(len(avg_vec_seq)):
            temp = {}
            for j in range(len(CANONICAL_AMINOACID_ORDERED)):
                val = avg_vec_seq[i,j]
                if (operation == 'lt' and val < threshold) or (operation == 'gt' and val > threshold):
                    temp[CANONICAL_AMINOACID_ORDERED[j]] = np.abs(val)
                else:
                    temp[CANONICAL_AMINOACID_ORDERED[j]] = 0.0
            seq_list.append(temp)

        return seq_list
    
    def boltzmann_normalize(self, avg_vec_seq: np.ndarray, kt: float = 1.0, mode: str = 'region') -> np.ndarray:
        """
        Performs Boltzmann normalization on a sequence of interaction vectors.

        Parameters
        ----------
        avg_vec_seq : np.ndarray
            A 2D numpy array where each row represents a position in the sequence and
            each column represents interaction values with canonical amino acids.
            Shape is (n_positions, n_amino_acids)

        kt : float, optional
            Temperature factor (kT) used in Boltzmann normalization. Higher values lead
            to more uniform distributions. Default is 1.0.

        mode : str, optional
            Normalization mode to use:
            - 'region': Normalize each column independently (per amino acid interaction)
            - 'residue': Normalize each row independently (per sequence position)
            - 'whole': Normalize entire matrix using all values
            Default is 'region'.

        Returns
        -------
        np.ndarray
            Boltzmann normalized array with same shape as input. Values are normalized
            such that they sum to 1.0 within each normalization group as defined by mode.

        Raises
        ------
        ValueError
            If mode is not one of 'region', 'residue', or 'whole'
            If kt <= 0
        """
        if kt <= 0:
            raise ValueError("kt must be positive")

        if mode not in ['region', 'residue', 'whole']:
            raise ValueError("mode must be one of: 'region', 'residue', 'whole'")

        # Create copy to avoid modifying original
        vec_seq = avg_vec_seq.copy()
        
        # Convert to Boltzmann factors: exp(-E/kT)
        boltz_factors = np.exp(-vec_seq / kt)

        if mode == 'region':
            # Normalize each column (amino acid interaction) independently
            # Sum along rows (axis=0) to get normalization factors for each column
            norm_factors = np.sum(boltz_factors, axis=0)
            normalized = boltz_factors / norm_factors[np.newaxis, :]

        elif mode == 'residue':
            # Normalize each row (sequence position) independently
            # Sum along columns (axis=1) to get normalization factors for each row
            normalized = boltz_factors / np.sum(boltz_factors, axis=1)[:, np.newaxis]

        else:  # mode == 'whole'
            # Normalize entire matrix using single normalization factor
            normalized = boltz_factors / np.sum(boltz_factors)

        return normalized










