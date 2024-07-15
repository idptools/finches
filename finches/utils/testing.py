import numpy as np
from matplotlib import pyplot as plt
from finches.utils import folded_domain_utils as fdu
from finches import Mpipi_frontend

from goose import create

standard_devv = 3
#guassian = lambda x : np.exp(-0.5* (x/standard_devv)**2)/np.sqrt(2*np.pi)
lin_drop = lambda x : 1 - (x - 5.5*3)
mf = Mpipi_frontend(arb_window_func=lin_drop)

#import finches

#clear the output
import os
os.system('cls' if os.name == 'nt' else 'clear')


#location of a pdb
pdb_path = "/Users/nrazo/Documents/IDR_Struct_Interaction_PIMMS/Testing/Code and Settup/AF-P38398-F1-model_v4.pdb"

#create a structure
fdu.extract_and_write_domains(pdb_path, 'Modified_PDB.pdb', 1,95)
folded_dom = fdu.FoldeDomain("Modified_PDB.pdb")

#seq 1
seq1 = create.sequence(50)
a = folded_dom.calculate_surface_matrix_epsilon(seq1, mf.IMC_object, 3,5.5*3)
plt.imshow(a)
plt.colorbar()
plt.show()
print(np.mean(a))

#seq 2
seq2 = create.sequence(50)
b = folded_dom.calculate_surface_matrix_epsilon(seq2, mf.IMC_object, 3,5.5*3)
plt.imshow(b)
plt.colorbar()
plt.show()
print(np.mean(b))


#try the two sequences this way
c = mf.calc_idr_idr_psuedo_spatial_interaction_matrix(seq1,seq2,3,5.5*3)
plt.imshow(c)
plt.colorbar()
plt.show()
print(np.mean(c))



