"""
This Script serves as custom model template for a user to imput and build in a new custom 
forcefield class. The returned object functions as a class object in the same way 
that finches.mpipi or finches.calvados does.

By: Garrett M. Ginell
2023.10.01
"""

import itertools
import numpy as np
import finches
from os.path import exists



# The block below defines key metadata associated with a specific for
CUSTOM_MODEL_CONFIGS = {}


CUSTOM_MODEL_CONFIGS['VERSION_1'] = {}

#
CUSTOM_MODEL_CONFIGS['VERSION_1']['charge_prefactor'] = 0.2
CUSTOM_MODEL_CONFIGS['VERSION_2']['null_interaction_baseline'] = -0.1



class custom_model:

    def __init__(self, version, input_dictionary, valid_residue_groups=[],
                     condition_dictionary={}, condition_dependence_function=None):
        """
        The custom_model class defines an custom_model Object which lets you calculate 
        and return the pairwise interaction potential for the beads and values defined in 
        the inputed input_dictionary. The condition_dictionary allows the user to define 
        conditions of the model that can tune the values in the inputed input_dictionary 
        the keys of the condition_dictionary are set as properies of the returned class 
        object. 

        *NOTE* - The following condition variables are NOT allowed: 
                 ['ALL_RESIDUES_TYPES', 'conditions', 'compute_interaction_parameter', 'version',
                 'condition_dependence_function', 'valid_residue_groups']


 
        Parameters
        ---------------
        version : str
            Defines the name of the version of the parameters for the passed model

        input_dictionary : dictionary
            The directory where the pairwise interactions are defined. This should 
            be formated as a dictionary where the keys are string seperated by '_' 
            defining the pair of interacting residues and value is the interaction 
            values for that pair. Every pairwise interaction must be defined in
            input dictionary. Example:

                my_dict = {'A_A': -1, 'A_B': 1, 'B_B': -2, 'B_C': 1, 'A_C':-2.4,
                           'C_C':2}

            This inputed dict is then converted to a dictionary of dictionaries where 
            all keys are residues and the subdictionary contain k:v pairing of the 
            interaction strengths. This is then set the class property, below is an 
            example base off the inputed dict above:

                self.parameters = {'A': {'A':-1 ,  'B': 1 , 'C':-2.4},
                                   'B': {'A': 1 ,  'B':-2 , 'C': 1},
                                   'C': {'A':-2.4, 'B': 1 , 'C': 2}}

        valid_residue_groups : list
            This is a list of lists which defines with sets of input residues are allowed 
            to be in the same chain. Defult = None which automatically allows all residues 
            to be in the same chain. This flag controls the self.ALL_RESIDUES_TYPES property. 
            As an example, following the example above:

                if valid_residue_groups = None
                    self.ALL_RESIDUES_TYPES = [['A','B','C']]

                if valid_residue_groups = [['A','B'], ['C']]
                    self.ALL_RESIDUES_TYPES = [['A','B'], ['C']]

                    This indicates the following valitity of chains:
                        'AAAAB' = True 
                        'AABCA' = False
                        'CCCCC' = True 

                    IE bead 'C' can not be in a chain with bead 'A' or 'B'.

            This is particularly useful when in an example where beads 'C' represent 
            RNA while the rest of the beads represent amino acids.


        condition_dictionary : dictionary
            Defines a dictionary of condition keys and condition values which are 
            converted and set as instances of the returned model object. For example: 

            This allows the user to define a base 'SALT' condition for their inputed 
            parameters. The condition_dependence_function then takes in all the inputed 
            conditions and defines how the parameters will be tuned as a funtion of the
            passed specific conditions. As an example in the Mpipi forcefield, the condition
            dictionay would include the following keys ['dielectric', 'salt']. In the example
            above, an example condition_dictionary is below, the key is set as a propery of 
            the class with tha value as the value. 

                condition_dict = {'pH': 7.5, 'salt': 0.2}

            this dictionary results in the class having the following properties:

                self.pH = 7.5 
                self.salt = 0.2

            if a condition_dictionary is set a condition_dependence_function MUST be defined 
            which takes in all of the defined conditions and returns a value corrisponding to 
            a updated value from self.parameters[i][j].

        condition_dependence_function : funtion 
            Defines a function that takes in i, j, self.parameters[i][j], and every condition 
            and returns a updated value of self.parameters[i][j] at the specified conditions. 

            *NOTE* - condition_dependence_function is called to compute self.parameters[i][j] 
                     whenever any conditions are passed 
            
            Example function with require inputs base on the example above: 
                condition_dependence_function(i, j, self.parameters[i][j], salt=None, pH=None)

                condition_dependence_function() must return an acceptable value of self.parameters[i][j]
                
        Returns
        -------------
        
        model_object : obj 
            Returned is a finches.forcefields.calvados object that can then be passed to 
            the Interaction_Matrix_Constructor class. 
        """

        ##
        ## check and parse the input dictionary
        ##
        l_parameters = {}
        residue_list = []
        for r in input_dictionary:
            if len(r) != 3:
                raise Exception(f'INVALID residue pair of {r}, ensure each residue is 1 character in length')
            try:
                r0, r1 = r.split('_')
                v = input_dictionary[r]
            except Exception as e:
                print(f'INVALID residue pair of {r}, ensure pair sting contains a "_" and can be split by "_"')

            residue_list.append(r0)
            residue_list.append(r1)

            for l_r, l_r1 in {r0:r1, r1:r0}.items():
                if l_r not in l_parameters:
                    l_parameters[l_r] = {l_r1:v}
                else:
                    if l_r1 not in l_parameters[l_r]:
                        l_parameters[l_r][l_r1] = v 
                    else:
                        raise Exception(f'ERROR - duplicate pair defintion of {r} found in input_dictionary')


        self.parameters = l_parameters
        self.version = version

        ##
        ## check condition dictionary for invalid condition names
        ##
        if len(condition_dictionary.keys()) > 0:

            if 'ALL_RESIDUES_TYPES' in condition_dictionary:
                raise Exception(f'ERROR defining conditions - ALL_RESIDUES_TYPES - is an invalid conditions name.')
            if 'conditions' in condition_dictionary:
                raise Exception(f'ERROR defining conditions - conditions - is an invalid conditions name.')
            if 'compute_interaction_parameter' in condition_dictionary:
                raise Exception(f'ERROR defining conditions - compute_interaction_parameter - is an invalid conditions name.')
            if 'condition_dependence_function' in condition_dictionary:
                raise Exception(f'ERROR defining conditions - condition_dependence_function - is an invalid conditions name.')
            if 'valid_residue_groups' in condition_dictionary:
                raise Exception(f'ERROR defining conditions - valid_residue_groups - is an invalid conditions name.')
            if 'version' in condition_dictionary:
                raise Exception(f'ERROR defining conditions - version - is an invalid conditions name.')
                    
            # define possible conditions
            self.conditions = list(condition_dictionary.keys())

            # initiate conditions 
            for key, value in condition_dictionary.items():

                # set condition as property and assign the defult value (IE 1st position in the tuple)
                setattr(self, key, value[0])


            # check and set condition_dependence_function as a method of the class 
            if callable(condition_dependence_function):

                # check that the function can get take in the defined conditions
                for c in self.conditions:
                    if not hasattr(condition_dependence_function, c):
                        raise Exception(f'''ERROR initializing condition_dependence_function - condition_dependence_function must take in
                            all conditions defined in the condition_dictionary''')

                # NOTE ADD MORE CHECKS FOR USER DEFINED CONDITION FUNCTION

                # if it's a function, set it as a method of the class
                self.condition_dependence_function = condition_dependence_function

            else: 
                raise Exception(f'ERROR defing condition_function, ensure condition_function is callable')
       
        else:
            # set to null if no conditions are defined 
            self.conditions = None
            self.condition_dependence_function = None 


        # NOTE THIS IS A NESTED LIST FOR WHICH:
        #   every residue in each sublist can occur in the same sequeuce, for sequences with residues 
        #   found in multible sublist and error will be thrown
        if len(valid_residue_groups) > 0:

            # check to make sure all beads in input_dictionary are included in 
            #   the valid allowed bead type pairs 
            all_l_beads = [r for sublist in valid_residue_groups for r in sublist]
            if set(all_l_beads) != set(self.parameters.keys()):
                raise Exception('ERROR - not all residues in input input_dictionary defined in valid_residue_groups')

            # specific allocation of valid beads found in the same chain
            self.ALL_RESIDUES_TYPES = valid_residue_groups
        else: 
            # all beads found are allowed in the same chain
            self.ALL_RESIDUES_TYPES = [list(self.parameters.keys())]


        try:
            self.CONFIGS = CUSTOM_MODEL_CONFIGS[self.version]
        except KeyError:
            raise Exception('Version was not recognized - this probably should have been checked before now...')


        

    # ----------------------------------------------------------
    #
    def compute_interaction_parameter(self, residue_1, residue_2, **conditions):
        """
        NOTE - the name of this function must match name in other forcefield modules

        Standalone function that computes pariwise interaction parameter
        between two residue types based on self.parameters and the conditions 
        functions. 
        
        Parameters
        --------------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes
        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        Returns
        -----------
        tuple
            Returns a tuple with several values:
            [0] - float -  interaction parameter based on conditions
            [1] - dict  -  dictionary of conditions and there values
        
        """

        if self.conditions:
            l_cond = {}
            for c in self.conditions:
                if c in conditions.keys():
                    l_cond[c] = conditions[c]
                else:
                    l_cond[c] = getattr(self, c)

            interaction_param = self.condition_dependence_function(residue_1, residue_2, 
                                                                  self.parameters[residue_1][residue_2], **l_cond)
        else:
            # get the interacion parameter
            interaction_param = self.parameters[residue_1][residue_2]
            l_cond = {'ALL':None}

        return (interaction_param, l_cond)

