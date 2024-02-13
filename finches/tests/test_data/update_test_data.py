"""
This MODULE contains the functions to write the files 
used in the test. 

NOTE - running any of the below functions could overwrite 
       the accepted TRUE values for the values used in the 
       test. be sure to pass the correct file handle if you 
       intend to write new values.
"""
import numpy as np
import os

import finches

from test_sequences import test_sequences, test_condition_dict

# ..........................................................................................
#
def write_test_pairwise_homotypic_matrix(filepath, model, model_name):

    data_file = f'{filepath}/test_{model_name}_homotypic_matrix'
    X_local = model 

    with open(data_file+'.npz','w') as fh:
        pass

    otl = []
    for t in test_sequences:
        ot = X_local.calculate_pairwise_homotypic_matrix(t)
        otl.append(ot)
        print(ot.shape)

    np.savez(data_file, otl)

# ..........................................................................................
#
def write_test_pairwise_heterotypic_matrix(filepath, model, model_name):

    data_file = f'{filepath}/test_{model_name}_heterotypic_matrix'
    X_local = model 

    with open(data_file+'.npz','w') as fh:
        pass

    otl = []
    for t in test_sequences:
        ot = X_local.calculate_pairwise_heterotypic_matrix(t,t0)
        otl.append(ot)
        print(ot.shape)

    np.savez(data_file, otl)

# ..........................................................................................
#
def write_test_weighted_matrix(filepath, model, model_name):

    data_file = f'{filepath}/test_{model_name}_weighted_matrix'
    X_local = model 
    
    with open(data_file+'.npz','w') as fh:
        pass

    otl = []
    for t in test_sequences:
        ot = X_local.calculate_weighted_pairwise_matrix(t, t0)
        otl.append(ot)
        print(ot.shape)
        
    otl1 = []
    for t in test_sequences:
        ot = X_local.calculate_weighted_pairwise_matrix(t, t0, use_charge_weighting=False)
        otl1.append(ot)
        print(ot.shape)

    otl2 = []
    for t in test_sequences:
        ot = X_local.calculate_weighted_pairwise_matrix(t, t0, use_aliphatic_weighting=False)
        otl2.append(ot)
        print(ot.shape)
        
    np.savez(data_file, DEFAULT=otl, NOCHARGE=otl1,  NOuse_aliphatic_weighting=otl2)

#..........................................................................................
#
def write_test_matrix_manipulation(filepath, model):

    from finches.epsilon_calculation import get_attractive_repulsive_matrixes
    from finches.epsilon_calculation import mask_matrix

    data_file = f'{filepath}/test_matrix_manipulation'
    X_local = model 

    with open(data_file+'.npz','w') as fh:
        pass

    all_manipulated = {}

    test_matrix = X_local.calculate_weighted_pairwise_matrix(test_sequences[1],t0)
    all_manipulated['test_matrix'] = test_matrix

    otla, otlr = get_attractive_repulsive_matrixes(all_manipulated['test_matrix'],-0.15)
    all_manipulated['attractive_repulsive_matrixes'] = otla, otlr

    mask = np.random.choice([0, 1], size=test_matrix.shape)
    all_manipulated['bionary_mask'] = mask

    out_masked = epsilon_calculation.mask_matrix(test_matrix, mask)
    all_manipulated['masked_matrix'] = out_masked


    vector_o0 = epsilon_calculation.flatten_matrix_to_vector(test_matrix, orientation=0)
    vector_o1 = epsilon_calculation.flatten_matrix_to_vector(test_matrix, orientation=1)
    all_manipulated['flattened_vector'] = [vector_o0, vector_o1]

    # write file 
    np.savez(data_file, **all_manipulated)

#..........................................................................................
#
def write_test_seq_epsilon_and_vectors(filepath, model, model_name):

    data_file = f'{filepath}/{model_name}_seq_epsilon_and_vectors'
    X_local = model 

    with open(data_file+'.npz','w') as fh:
        pass

    all_manipulated = {}
    names = ['t', 't0']
    for i, t in enumerate([test_sequences[1], t0]):
        
        n = names[i]
        
        attractive_vector, repulsive_vector = epsilon_calculation.get_sequence_epsilon_vectors(t,
                                                                         t0,
                                                                         X_local,
                                                                         prefactor=None,
                                                                         null_interaction_baseline=None,
                                                                         use_charge_weighting=True,
                                                                         use_aliphatic_weighting=True)

        all_manipulated[f'{n}_t0_DEFAULT'] = [attractive_vector, repulsive_vector]

        # test passed prefactor and null_interaction_baseline
        attractive_vector, repulsive_vector = epsilon_calculation.get_sequence_epsilon_vectors(t,
                                                                         t0,
                                                                         X_local,
                                                                         prefactor=0.25,
                                                                         null_interaction_baseline=-0.15,
                                                                         use_charge_weighting=True,
                                                                         use_aliphatic_weighting=True)

        all_manipulated[f'{n}_t0_prefactor_baseline'] = [attractive_vector, repulsive_vector]


        # test no weighting
        attractive_vector, repulsive_vector = epsilon_calculation.get_sequence_epsilon_vectors(t,
                                                                         t0,
                                                                         X_local,
                                                                         use_charge_weighting=False,
                                                                         use_aliphatic_weighting=False)

        all_manipulated[f'{n}_t0_NOWEIGHTING'] = [attractive_vector, repulsive_vector]



    # write file 
    np.savez(data_file, **all_manipulated)


#..........................................................................................
#
def write_FH_out_data(filepath, model, model_name):

    data_file = f'{filepath}/{model_name}_FH_outdata'
    X_local = model 

    with open(data_file+'.npz','w') as fh:
        pass

    all_manipulated = {}
    names = ['t0']
 
    # test salt JUST outdiagrams
    out_diagrams = finches.epsilon_to_FHtheory.build_SALT_dependent_phase_diagrams(t0, model, test_condition_dict['test_SALT'])[1]
    all_manipulated[f'SALT_t0_defult'] = out_diagrams

    # test pH JUST outdiagrams
    out_diagrams = finches.epsilon_to_FHtheory.build_PH_dependent_phase_diagrams(t0, model, test_condition_dict['test_PH'])[1]
    all_manipulated[f'PH_t0_defult'] = out_diagrams
    
    # test DIELETRIC JUST outdiagrams
    out_diagrams = finches.epsilon_to_FHtheory.build_DIELECTRIC_dependent_phase_diagrams(t0, model, test_condition_dict['test_DIELECTRIC'])[1]
    all_manipulated[f'DIELECTRIC_t0_defult'] = out_diagrams


    # write file 
    np.savez(data_file, **all_manipulated)





