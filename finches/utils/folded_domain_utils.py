import mdtraj as md
import numpy as np

from scipy.spatial import distance
import networkx as nx


# sidechain/backbone max SASA data

# NB: note on approach for identifying surface residues
#
#

# MAX_SASA_DATA calculated by running all-atom simulations of GXG tripeptides
# with an excluded volume simulation (i.e. all attractive non-bonded interactions
# turned off 
MAX_SASA_DATA = {'A': [7.581871795654296875e+01, 7.607605743408203125e+01],
                 'C': [1.154064483642578125e+02, 6.787722015380859375e+01],
                 'D': [1.302558288574218750e+02, 7.182710266113281250e+01],
                 'E': [1.617985687255859375e+02, 6.805746459960937500e+01],
                 'F': [2.093871002197265625e+02, 6.598278808593750000e+01],
                 'G': [0.000000000000000000e+00, 1.149752731323242188e+02],
                 'H': [1.808149414062500000e+02, 6.750666809082031250e+01],
                 'I': [1.727196502685546875e+02, 6.034464645385742188e+01],
                 'K': [2.058575897216796875e+02, 6.871156311035156250e+01],
                 'L': [1.720360412597656250e+02, 6.451246643066406250e+01],
                 'M': [1.847660064697265625e+02, 6.778076934814453125e+01],
                 'N': [1.427441253662109375e+02, 6.680493164062500000e+01],
                 'P': [1.342914733886718750e+02, 5.583909606933593750e+01],
                 'Q': [1.733262939453125000e+02, 6.660184478759765625e+01],
                 'R': [2.364875640869140625e+02, 6.673487854003906250e+01],
                 'S': [9.587133026123046875e+01, 7.287202453613281250e+01],
                 'T': [1.309214324951171875e+02, 6.421310424804687500e+01],
                 'V': [1.431178131103515625e+02, 6.172962188720703125e+01],
                 'W': [2.545694122314453125e+02, 6.430991363525390625e+01],
                 'Y': [2.225183105468750000e+02, 7.186695098876953125e+01]}

# amino acid conversion 
THREE_TO_ONE = {'ALA': 'A',
                'CYS': 'C',
                'ASP': 'D',
                'GLU': 'E',
                'PHE': 'F',
                'GLY': 'G',
                'HIS': 'H',
                'ILE': 'I',
                'LYS': 'K',
                'LEU': 'L',
                'MET': 'M',
                'ASN': 'N',
                'PRO': 'P',
                'GLN': 'Q',
                'ARG': 'R',
                'SER': 'S',
                'THR': 'T',
                'VAL': 'V',
                'TRP': 'W',
                'TYR': 'Y'}



# internal notes 2024-10-20
# NB: In earlier versions of finches, FoldedDomain was defined as FoldeDomain (missing the
# ending 'd' of folded. To maintain backwards compatibility, we alias the old name to the
# new correct one at the end of this class. 
class FoldedDomain:

    # ................................................................................
    #
    #
    def __init__(self,
                 pdbfilename,
                 start=None,
                 end=None,
                 probe_radius=1.4,
                 residue_overide_mapping={},
                 surface_thresh=0.10,
                 sasa_mode='v1',
                 ignore_warnings=False,
                 SASA_ONLY=False,
                 ):
        """
        Class to handle folded domains and perform folded domain structure 
        analysis related to epsilon-associated interacions.

        Object variables available after initialization:

        self.sequence: string of the full amino acid sequence of the folded domain
        self.traj: mdtraj trajectory object of the folded domain
        self.sasa: numpy array of the SASA of each residue in the folded domain
        self.surface_residues: list of surface residues
        self.surface_vector: numpy array of where each element is 0 (interior) or 1 (surface)
        self.surface_indices: numpy array of the indices of the surface residues
        self.surface_positions: x,y,z coordinate of the surface residues
        

        # further, a few things do not automatically initialize but will be
        # calculated as needed:
        self.all_shortes_paths:
        self.surface_neighbours:
        self.surface_epsilon:

        self.surface_graph: networkX graph of surface residues
        self.surface_neighbours: 
        self.surface_distance_surface:
        self.surface_distance_straight_line:


        Parameters
        ----------

        pdbfilename: str
            path to pdb file

        start: int
            start residue index

        end: int
            end residue index

        probe_radius: float
            probe radius for SASA calculation (default 1.4)

        residue_overide_mapping: dict
            dictionary to map non-standard residue names to standard residue names
        
        surface_thresh: float
            threshold for surface residues (default 0.10)

        sasa_mode: str
            mode for SASA calculation (default 'v1'). Could be v1 or v2. Default is v1.

            v1 means we calculate the SASA of the residue, but then compare ONLY to the
            the sidechain SASA and define a residue as solvent exposed if the RESIDUE SASA
            is above surface_thresh * max sidechain SASA. This means glycines are always solvent
            exposed, unless they are fully hidden, in which case max sidechain SASA = 0 and 
            residue SASA = 0 and 0 is not > 0. Not we focus on 

            v2 means we calculate the SASA of the residue, but then compare to the summed max
            SASA of the sidechain with the backbone. This is more stringent 

        ignore_warnings: bool
            ignore warnings (default False)

        SASA_ONLY: bool
            Only calculate SASA and return (default False). If this is set, all other
            functionality will fail.

        """

        if sasa_mode not in ['v1', 'v2']:
            raise ValueError('sasa_mode must be v1 or v2')
        
        # safety feature for now...
        if start is not None and end is not None:
            print("WARNING: Using non standard start and end residues has not been throughly tested and if this is super important to you please contact Alex and he'll make sure it's really working correctly. We recommended excising out a domain using the extract_and_write_domain() function and then using the full PDB file.")
            if ignore_warnings is False:
                raise Exception('This warning tiggers an exception; to ignore set ignore_warnings=True')

            
        # parse pdbfile
        p = md.load_pdb(pdbfilename)

        # handle start/end residues if passed
        if start is not None and end is not None:
            p = p.atom_slice(p.topology.select(f'resid {start} to {end}'))

        elif start is not None or end is not None:
            raise ValueError('Either start and end passed or neither passed; one of start or end was passed')

        # assign trajectory info
        self.traj = p

        # calculate SASA of residues;
        # 100* to convert from nm^2 to A^2, and *0.1 for probe radius to convert from A to nm
        self.sasa = 100*md.shrake_rupley(p, mode='residue', probe_radius=probe_radius*0.1)[0]

        # create full amino acid sequence
        s = ''
        for r in p.topology.residues:
            three_letter = str(r)[0:3]

            try:
                s = s + THREE_TO_ONE[three_letter]
            except KeyError:

                if three_letter in residue_overide_mapping:
                    s = s + override_mapping
                else:                
                    print('Encountered residue name {three_letter} that is not a default AA nor in overide mapping')                    
        self.sequence = s

        # bail here
        if SASA_ONLY:
            return

        # find solvent surface res
        surface_vector = []
        surface_indices = []
        for i in range(len(self.sasa)):

            if sasa_mode == 'v1':
                solvent_accessible = self.sasa[i] > surface_thresh*MAX_SASA_DATA[self.sequence[i]][0]
                
            elif sasa_mode == 'v2':
                solvent_accessible = self.sasa[i] > surface_thresh*(MAX_SASA_DATA[self.sequence[i]][0] + MAX_SASA_DATA[self.sequence[i]][1])
                
            
            if solvent_accessible:
                surface_vector.append(1)
                surface_indices.append(i)
            else:
                surface_vector.append(0)
                
        self.surface_vector  = surface_vector
        self.surface_fraction = sum(surface_vector)/len(surface_vector)
        self.surface_indices = surface_indices
        self.surface_positions = {}

        # if we're here cycle through surface residues
        for idx in self.surface_indices:

            # get atom indices for this residue

            atom_indices = p.topology.select(f'sidechain and resid {idx}')
            if len(atom_indices) == 0:                
                atom_indices = p.topology.select(f'resid {idx} and name CA')

            #print(f"{idx} => {len(atom_indices)}")
            self.surface_positions[idx] = md.compute_center_of_mass(p.atom_slice(atom_indices))[0]

        # initialize this as empty strings, which are populated in the
        # calculate_surface_sequences() function when needed
        self._surface_seq = {}
        for idx in self.surface_indices:
            self._surface_seq[idx] = ""
            

        # initialize expensive things to None - these will be calculated
        # at execution time via the property functions below
        self._all_shortest_paths = None
        self._surface_neighbours = None
        self._surface_neighbour_sequences = None
        self._surface_graph = None
        self._surface_neighbours = None 
        self._surface_distance_surface = None
        self._surface_distance_straight_line = None

    # ....................................................................
    #
    def __len__(self):
        return len(self.sequence)
        
    # ....................................................................
    #
    @property 
    def all_shortest_paths(self):

        # if not yet initialized (this will trigger self.surface_graph to be
        # built as well
        if self._all_shortest_paths is None:
            self._all_shortest_paths = dict(nx.all_pairs_dijkstra_path(self.surface_graph), weight='weight')
        return self._all_shortest_paths

    # ....................................................................
    #
    @property
    def surface_graph(self):
        if self._surface_graph is None:
            self.get_nearest_neighbour_res()
        return self._surface_graph

    # ....................................................................
    #    
    @property
    def surface_neighbours(self):
        if self._surface_neighbours is None:
            self.get_nearest_neighbour_res()
        return self._surface_neighbours

    # ....................................................................
    #    
    @property
    def surface_neighbour_sequences(self):
        if self._surface_neighbour_sequences is None:
            self.get_nearest_neighbour_res()
        return self._surface_neighbour_sequences
    
    
    # ....................................................................
    #
    @property
    def surface_distance_surface(self):
        if self._surface_distance_surface is None:
            self.get_nearest_neighbour_res()
            
        return self._surface_distance_surface

    # ....................................................................
    #    
    @property
    def surface_distance_straight_line(self):
        if self._surface_distance_straight_line is None:
            self.get_nearest_neighbour_res()
        return self._surface_distance_straight_line

    
    # ....................................................................
    #    
    def __str__(self):
        return f'FoldeDomain object with {len(self.sequence)} residues'

    
    # ....................................................................
    #        
    def __repr__(self):
        return str(self)


        
    # ................................................................................
    #
    #
    def get_nearest_neighbour_res(self, distance_thresh=9.0):
        """

        This function will find the nearest neighbours of each residue in 
        the protein. It will evaluate values for the following object variables
        that can be accessed after the function is called:

        * self.surface_neighbours: dictionary that reports on the indices of residues 
        that are within a certain distance of each residue (defined by the distance 
        threshold parameter)

        * self.surface_distance_surface: dictionary that maps between each pair 
        of surface residues and the distance between them (in over-surface distance)

        * self.surface_distances_straight_line: dictionary that maps between each pair 
        of surface residues and the distance between them (in straight-line euclidean 
        distance)
        
        * self.surface_graph: a networkx graph connecting surface residues with edges. 

        Parameters
        ----------

        distance_thresh: float
            distance threshold for nearest neighbour search (default 9)

        Returns
        -------
        None
        
        """


        # rescale distance threshold into nanometers
        distance_thresh_A = distance_thresh/10

        # build positions array
        position_array = np.array(list(self.surface_positions.values()))

        # map position in the positions array back to residue index
        dist_idx_to_res_idx = {}
        for i in range(len(self.surface_indices)):
            dist_idx_to_res_idx[i] = self.surface_indices[i]

        
        # calculate distance matrix
        dist_matrix = distance.cdist(position_array, position_array, 'euclidean')

        # build dictionary that maps in residue ID space euclidean distances
        surface_distance = {}

        # d1 = 1,2,3,... indice into the distance matrix
        for d1 in range(len(dist_matrix)):

            # residue index that maps to this distance index
            r1 = dist_idx_to_res_idx[d1]

            # if this is a new residue index...
            if r1 not in surface_distance:
                surface_distance[r1] = {}

            # for each distance index (d2 = 1,2,3,  )
            for d2 in range(len(dist_matrix)):

                # residue index that maps to this distance index
                r2 = dist_idx_to_res_idx[d2]

                surface_distance[r1][r2] = dist_matrix[d1][d2]*10
                
        

        # build a dictionary of neighbours
        neighbours = {}

        # for each residue, find all residues within 0.9 nm
        for dist_index in range(len(dist_matrix)):
            
            all_idx = np.where(dist_matrix[dist_index] < distance_thresh_A)[0]
            

            # get residue index for this distance index
            res_idx = dist_idx_to_res_idx[dist_index]
            neighbours[res_idx] = []

            for dist_index2 in all_idx:
                r = dist_idx_to_res_idx[dist_index2]
                
                neighbours[res_idx].append([r, dist_matrix[dist_index][dist_index2]*10])


        # build a graph
        G = nx.Graph()

        # Add edges to the graph, weighted by distances
        for node, connections_dist in neighbours.items():
            for connected_node in connections_dist:

                connect_id = connected_node[0]
                connect_dist = connected_node[1]

                G.add_edge(node, connect_id, weight=connect_dist)

        all_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

        self._surface_graph = G
        self._surface_neighbours = neighbours
        self._surface_distance_surface = all_shortest_paths
        self._surface_distance_straight_line = surface_distance
        self._surface_neighbour_sequences = {}
        
        for i in self.surface_neighbours:
            local_string = ""
            for j in self.surface_neighbours[i]:
                local_string = local_string + self.sequence[j[0]]
            self._surface_neighbour_sequences[i] = local_string
            


        
            
    # ................................................................................
    #
    #
    def __extract_residues(self, inseq, targets):
        """
        Internal function that removes all residues in a list that 
        are in a target list, returning the modified list and the
        removed elements.

        Parameters
        ----------
        inseq: str
            Input amino acid sequence

        targets: list
            List of residues to remove

        Returns
        -------
        tuple

            [0] Modified list with residues removed

            [1] List of removed residues in reverse order they 
                appeared in the original sequence.
        
        """

        inseq = list(inseq)

        removed_elements = []
        for i in range(len(inseq) - 1, -1, -1):
            if inseq[i] in targets:
                removed_elements.append(inseq.pop(i))

        return inseq, removed_elements
                

    
    # ................................................................................
    #
    #            
    def calculate_surface_epsilon(self, input_sequence, IMCObject):
        """
        This function calculates the surface epsilon values for each residue in the
        protein. The function will calculate the surface epsilon for each residue in 
        the protein and return this in a surface_epsilon dictionary.

        Briefly, this function works by doing the following:

        (1) Takeing each solvent accessible residue

        (2) Finding the neighbour residues near that residue. Note we can redefine
            the distance threshold used to define neighbours by running 
            get_nearest_neighbour_res() function.

        (3) Re-organizing the sequence order of the neighbor string depending on what
            the center residue is. If the center residue is charged, re-order so all
            charged residues of that type are next to it. If the center residue is a 
            hydrophobe, reorder so the hydrophobes are next to it. We do this so we
            take the local chemical environment into account for the epsilon 
            calculation using the center residue to define what types(s) of local 
            chemistry we care about.   

        (4) Calculating the mean epsilon score between the "neighbor string" and 
            the passed input string.

        (5) Divide that epsilon score by the number of residues in the neighbor string
            to get the average mean-field epsilon value for that residue.

        Note this approach is probably ok if the input sequence is either quite 
        short or a repetitive sequence, but effectively it calculates the mean-field
        attraction/repulsion beteween each residue on the surface (using its local context
        to define that interaction) and the ENTIRE input sequence. 

        Parameters
        ----------
        input_sequence: str
            Amino acid sequence of the input sequence

        IMCObject: IMCObject
            IMCObject that contains the epsilon calculate, this
            can obtained as a object variable from an Mpipi_frontend
            or CALAVDOS_frontend object.


        Returns
        -------
        Dict
            Dictionary that maps between residue index and surface epsilon value. The
            return dictionry has index position as key and then a list as values, where
            each list has three positions:
            [0] - the residue associated with that position
            [1] - the neibouring residues, where neighbours are those residues within
                  some distance threshold of the surface residue of interest
            [2] - the surface epsilon value for that residue.

        """

        hydrophobes = ['I' , 'V', 'L', 'A', 'M']
        negative = ['D', 'E']
        positive = ['K', 'R']

        surface_eps = {}
        
        for idx in self.surface_indices:

            if self._surface_seq[idx] == '':

                center_resid = self.sequence[idx]

                # get indices of all residues neighbouring this residue, excluding itself
                neighbor_resid = [i[0] for i in self.surface_neighbours[idx]]

                # build a local sequence list
                local_seq = [self.sequence[n] for n in neighbor_resid]
                og_seq = "".join(local_seq)
                reordered_seq = ''

                # if center res is hydrophobe 
                if center_resid in hydrophobes:
                    local_seq, removed = self.__extract_residues(local_seq, hydrophobes)
                    reordered_seq = ''.join(removed) + ''.join(local_seq)
                elif center_resid in negative:
                    local_seq, removed = self.__extract_residues(local_seq, negative)
                    reordered_seq = ''.join(removed) + ''.join(local_seq)
                elif center_resid in positive:
                    local_seq, removed = self.__extract_residues(local_seq, positive)
                    reordered_seq = ''.join(removed) + ''.join(local_seq)
                else:
                    reordered_seq = ''.join(local_seq)

            else:
                reordered_seq = self._surface_seq[idx]
                
            # initial value 
            tmp = IMCObject.calculate_epsilon_value(reordered_seq, input_sequence)/len(reordered_seq)
            
            surface_eps[idx] = [center_resid, reordered_seq, tmp]
            
        return surface_eps


    # ................................................................................
    #
    #            
    def calculate_mean_surface_epsilon(self, input_sequence, IMCObject):
        """
        This function calculates the mean surface epsilon for the protein. The mean
        surface epsilon is a measure of the average chemical specificity for a given
        guest sequence, averaged across the entire protein. 

        Parameters
        ----------
        input_sequence: str
            Amino acid sequence of the input sequence

        IMCObject: IMCObject
            IMCObject that contains the epsilon calculate. The IMCObject is
            typically obtained from an Mpipi_frontend or CALAVDOS_frontend object.

        Returns
        -------
        float
            Mean surface epsilon value for the protein.
        """
        

        surface_eps = self.calculate_surface_epsilon(input_sequence, IMCObject)

        return np.mean([i[2] for i in surface_eps.values()])

    # ................................................................................
    #
    #            
    def calculate_attractive_surface_epsilon(self, input_sequence, IMCObject, threshold=0):
        """
        This function calculates the attractive surface epsilon values for each surface residue,
        a list of values that are the residues which find themselves in an attractive environment
        given the input sequence. The function calculates the surface epsilon for each residue in 
        the protein, and then filters out those that are above the threshold value, leaving
        only the attractive residues. 

        NOTE if you want to relate these to residue position you should use the 
        calculate_surface_epsilon() function instead.
                
        Parameters
        ----------
        input_sequence: str
            Amino acid sequence of the input sequence

        IMCObject: IMCObject
            IMCObject that contains the epsilon calculate. The IMCObject is an initialized
            FINCHES-derived object that enables epsilon calculations. It can be found in 
            the Mpipi_frontend or CALAVDOS_frontend objects, as an IMC_object variable.

        threshold: float
            The threshold value for the attractive epsilon values. If the epsilon value is
            below this threshold, it will be excluded from the calculation. By default
            this is zero and in general can't imagine when you'd want to change this...

        Returns
        -------
        list
            List of attractive epsilon values for the protein, excluding those below 
            the threshold.

        """

        surface_eps = self.calculate_surface_epsilon(input_sequence, IMCObject)

        return [i[2] for i in surface_eps.values() if i[2] < threshold]
        
    # ................................................................................
    #
    #            
    def calculate_repulsive_surface_epsilon(self, input_sequence, IMCObject, threshold=0):
        """
        This function calculates the repulsive surface epsilon values for each surface residue,
        a list of values that are the residues which find themselves in an repulsive environment
        given the input sequence. The function calculates the surface epsilon for each residue in 
        the protein, and then filters out those that are below the threshold value, leaving
        only the repulsive residues.

        NOTE if you want to relate these to residue position you should use the 
        calculate_surface_epsilon() function instead.

        Parameters
        ----------
        input_sequence: str
            Amino acid sequence of the input sequence

        IMCObject: IMCObject
            IMCObject that contains the epsilon calculate. The IMCObject is an initialized
            FINCHES-derived object that enables epsilon calculations. It can be found in 
            the Mpipi_frontend or CALAVDOS_frontend objects, as an IMC_object variable.

        threshold: float
            The threshold value for the repulsive epsilon values. If the epsilon value is
            below this threshold, it will be excluded from the calculation. By default
            this is zero and in general can't imagine when you'd want to change this...

        Returns
        -------
        list
            List of repulsive epsilon values for the protein, excluding those below 
            the threshold.

        """

        surface_eps = self.calculate_surface_epsilon(input_sequence, IMCObject)

        return [i[2] for i in surface_eps.values() if i[2] >= threshold]

    
            
    # ................................................................................
    #
    #            
    def calculate_IWD(self, residues, positions = None, distance_mode='surface', calculate_null=False, number_null_iterations=100):
        """
        This function will calculate the IWD between a set 
        of residues, where we define the residue groups based 
        on the residues list.

        Distance mode can be set to surface, in which an approximation
        for the distance between residues is calculated based on the
        surface distance between residues. Alternatively, the distance
        can be calculated as the straight line distance between residues
        which ignores the protein surface.

        Finally, it can often be hard to know if the IWD value that is
        returned is significant. To help with this, we can calculate the
        null distribution of IWD values for the same number of residues
        over the same structure and compare the IWD value to this null.
        This happens if calculate_null is set to True.

        NB this has not be super well tested yet...

        Parameters
        ----------

        residues: list
            list of residues to calculate IWD between

        positions: list
            list of residue positions to calculate IWD between.
            Default is None, in which case the residues list is used.

        distance_mode: str
            distance mode to use for IWD calculation. Can be either 'surface' 
            or 'straight line' Default is 'surface'.

        calculate_null: bool
            if True, calculate the null distribution of IWD values. Default is False.

        number_null_iterations: int
            number of iterations to use for null distribution calculation. 
            Default is 100.

        Returns
        -------

        [0] --> IWD: float
                    The IWD value for the residues of interest

        [1] --> number res of interest : int
                    Number of residues

        [2] --> iwd_null: list
                Distribution of IWD values if the same number of 
                residues .

        """
            

        # define distance function based on distance mode
        if distance_mode == 'surface':
            def dist(i1,i2):
                return self.surface_distance_surface[i1][i2]
        elif distance_mode == 'straight line':
            def dist(i1,i2):
                return self.surface_distance_straight_line[i1][i2]
        else:
            raise ValueError('distance mode must be either "surface" or "straight_line"')

        
        relevant_indices = []

        # if we have a list of positions, use those instead of the residue selector list
        if positions is not None:
            for idx in positions:
                if idx in self.surface_indices:
                    relevant_indices.append(idx)

        # otherwise, use the residue selector list
        else:
            for idx, res in enumerate(self.sequence):
                if res in residues:
                    if idx in self.surface_indices:
                        relevant_indices.append(idx)

        if len(relevant_indices) < 2:
            return (None, None, None)

        # calculate the inverse weighted distance as defined by
        # equation 1 in Schueler-Furman & Baker, 2003
        summation = 0
        npairs = 0        
        for i1 in range(0, len(relevant_indices)-1):

            # index 1
            idx1 = relevant_indices[i1]
            
            for i2 in range(i1+1, len(relevant_indices)):

                # index 2
                idx2 = relevant_indices[i2]
                
                summation = summation + 1/dist(idx1,idx2)
                npairs = npairs + 1

        iwd = summation/npairs

        if calculate_null is False:
            return [iwd, len(relevant_indices), -1]

        else:
            
            iwd_null = []
            for i in range(number_null_iterations):

                # randomly sample positions from the surface
                random_positions = np.random.choice(self.surface_indices, len(relevant_indices), replace=False)
                iwd_null.append(self.calculate_IWD([], positions = random_positions, distance_mode=distance_mode)[0])

            return [iwd, len(relevant_indices), iwd_null]


        
    # ................................................................................
    #
    #                    
    def write_SASA_vis_file(self, filename='sasa_binary.txt'):
        """
        This function will write a file that can be used to visualize the surface 
        residues in a protein. Specifically this assigns a "1" to solvent exposed
        residues and a "0" to buried residues.

        Parameters
        ----------
        filename : str
            The name of the file to write.

        Returns
        -------
        None

        """


        with open(filename,'w') as fh:
            for i in range(len(self.sasa)):
                fh.write(f'{i+1} A {self.surface_vector[i]}\n')


    # ................................................................................
    #
    #                    
    def write_epsilon_vis_file(self, surface_epsilon, filename='surface_epsilon.txt'):
        """
        This function will write a file that can be used to visualize the surface 
        residues in a protein. Specifically this assigns a "1" to solvent exposed
        residues and a "0" to buried residues.

        Parameters
        ----------
        filename : str
            The name of the file to write.

        Returns
        -------
        None

        """

        if surface_epsilon is None:
            raise ValueError('Surface epsilon has not been calculated. Run calculate_surface_epsilon first.')

        max_idx = max(surface_epsilon)
        min_idx = min(surface_epsilon)
        
        with open(filename,'w') as fh:
            for i in range(min_idx, max_idx+1):
                if i not in surface_epsilon:
                    fh.write(f'{i+1} A 0\n')
                else:
                    fh.write(f'{i+1} A {round(surface_epsilon[i][2],3)}\n')


    def calculate_idr_surface_patch_interactions(self, 
                                   interacting_sequence, 
                                   IMCObject, 
                                   idr_tile_size = 31,
                                   patch_radius=12):
        """
        NOTE: You can either make an fdobj by feeding in a path to a PDB 
        or you can premake your fdobj and use that. Either fdobj or path_to_pdb
        must be set to None or this function will rais an exception.
        
        function to calculate the interaction between some chemistry
        and the surface patches of a sequence. Patches are defined
        by residues within distance thresh of the central residue. 

        TO DO ; UPDATE THESE DOCS

        Parameters
        -----------
        interacting_sequence : str
            the sequence that interacts with the protein in the PDB as a string

        IMCObject: IMCObject
            IMCObject that contains the epsilon calculate. The IMCObject is an initialized
            FINCHES-derived object that enables epsilon calculations. It can be found in 
            the Mpipi_frontend or CALAVDOS_frontend objects, as an IMC_object variable.
            
        patch_radius : float
            the radius from the center of the patch to allow residues to contribute
            to the patch. Default is 12 Angstroms.

        Returns
        --------
        tupe
            [0] dict of the patch mean interactions
            [1] matrix for visualizing IDR:FD surface
            [2] mean vector for visualizing IDR:FD surface


        """

        if (idr_tile_size % 2) == 0:
            raise Exception("IDR tile window size must be an odd integer.")

        if (idr_tile_size > len(interacting_sequence)):
            raise Exception("IDR tile window size must be less than the length of the provided sequence.")
        
        half_window = int((idr_tile_size - 1)/2)

        # get nearest neighbors and surface neighbors
        self.get_nearest_neighbour_res(distance_thresh=patch_radius)
        
        neighbors = self.surface_neighbours

        # get sequence
        seq = self.sequence

        # dict to hold patches
        interaction_dict = {}
        
        # now iterate over all possible patches
        for patch_ind_center in neighbors:
            resinds = neighbors[patch_ind_center]

            patch_mean_eps = sum([IMCObject.calculate_epsilon_value(interacting_sequence, seq[a]) for a, dist in resinds])/len(interacting_sequence)

            interacting_residues=''.join([seq[a] for a, dist in resinds])
            residue_indices=[a for a,n in resinds]
            interaction_dict[patch_ind_center]={'mean_epsilon':patch_mean_eps, 'interacting_residues':interacting_residues, 'residue_indices':residue_indices, 'resinds_with_dist':resinds}
        
        for i in interaction_dict.keys():
            res_data = interaction_dict[i]

            tmp = []
            for j in range(half_window, len(interacting_sequence) - half_window):
                tmp.append(IMCObject.calculate_epsilon_value(interacting_sequence[(j - half_window):(j + half_window + 1)], res_data['interacting_residues']))


            interaction_dict[i]['idr_epsilon_vector'] = tmp

        # pre-initailize the empty vector
        empty_vector = np.array([np.nan] * len(list(interaction_dict.values())[0]['idr_epsilon_vector']))
        
        collect_epsilon_vectors = []
        
        for i in range(len(self)):

            # if residue i is a surface residue
            if interaction_dict.get(i):
                collect_epsilon_vectors.append(np.array(interaction_dict[i]['idr_epsilon_vector']))
            else:
                collect_epsilon_vectors.append(empty_vector)

        vector_mean = np.array([x['idr_epsilon_vector'] for x in interaction_dict.values()]).mean(axis=0)
        
        return [interaction_dict, np.array(collect_epsilon_vectors), vector_mean]
    

    def _guassian_filter_vectors(self, vectors : np.ndarray, dist : np.ndarray, stdev : float = 2.2):
        """
        This function calculates a guassian filter for the values in the vector set.
        """
        # compute out the weights vector
        weights = np.exp(-0.5 * (dist / stdev) ** 2)

        # normalize the weights
        weights = weights / np.sum(weights)

        # apply the weights to the vectors
        filtered_vectors = np.matmul(vectors.T, weights).T

        # return the filtered vectors
        return filtered_vectors



    def compute_context_interaction_vectors(self, 
                                            IMCObject,
                                            patch_radius : float = 12.0,
                                            filter_std : float = 2.2) -> np.ndarray:
        """
        Computes interaction vectors for each surface residue taking local context into consideration.
    
        This function calculates how each surface patch interacts with amino acids by considering
        the local chemical environment around each residue within the specified patch radius.
        The interactions are weighted using a Gaussian filter based on distance from the patch center.

        Parameters
        ----------
        IMCObject : IMCObject
            Object containing methods for sequence encoding and interaction calculations
        patch_radius : float, optional
            Radius in Angstroms to define the local patch size, by default 12.0
        filter_std : float, optional
            Standard deviation for Gaussian distance weighting, by default 2.2

        Returns
        -------
        np.ndarray
            Array of interaction vectors for each surface patch
        """

        # check to ensure out inputs were valid

        # get nearest neighbors and surface neighbors
        self.get_nearest_neighbour_res(distance_thresh=patch_radius)

        # get the patch neighborhoods
        neighbors = self.surface_neighbours

        # get the sequence for the surface
        aa_seq = self.sequence

        # encode the sequence into a number based on its amino acid positions
        vector_encoded_seq = IMCObject.vector_encode_seq(aa_seq)

        # iterate over every patch
        patch_vectors = []
        for patch_ind_center in neighbors:
            # rescinds is for the residue index in the sequence and the distance from the center point
            resinds = neighbors[patch_ind_center]

            # pull the vector encodings and distances (residue index, distance)
            vector_patch = np.array([vector_encoded_seq[idx] for idx, r in resinds])

            # pull the distance away from the center residue
            distance_path = np.array([r for idx,r in resinds])

            # filter the data points 
            patch_spec_vec = self._guassian_filter_vectors(vector_patch, distance_path, filter_std)

            # add it to the patch list
            patch_vectors.append(patch_spec_vec)

        # return the patch vectors as a numpy array
        return np.array(patch_vectors)
    

    def compute_context_interaction_dict(self, 
                                   IMCObject,
                                   patch_radius: float = 12.0,
                                   filter_std: float = 2.2) -> list[dict[str,float]]:
        """
        Converts patch interaction vectors into dictionaries mapping amino acids to interaction values.

        Parameters
        ----------
        IMCObject : IMCObject
            Object containing methods for sequence encoding and interaction calculations
        patch_radius : float, optional
            Radius in Angstroms to define the local patch size, by default 12.0
        filter_std : float, optional
            Standard deviation for Gaussian distance weighting, by default 2.2

        Returns
        -------
        list[dict[str,float]]
            List of dictionaries where each dictionary maps amino acid letters to their 
            interaction values for a given patch
        """

        # compute the interaction vectors
        interaction_mat = self.compute_context_interaction_vectors(IMCObject=IMCObject,
                                                                   patch_radius=patch_radius,
                                                                   filter_std=filter_std)



        # transform the matrix into a list of dictionaries for the interaction
        interaction_dict = IMCObject.vector_decode_seq(interaction_mat)

        # return the interaction dictionary 
        return interaction_dict
    

    def compute_maximal_attractor(self, IMCObject,
                            patch_radius: float = 12.0,
                            filter_std: float = 2.2) -> tuple[list[str], np.ndarray]:
        """
        Identifies the amino acids that would interact most favorably with each surface patch.

        For each patch, determines which amino acid would have the strongest attractive
        interaction based on the local chemical environment.

        Parameters
        ----------
        IMCObject : IMCObject
            Object containing methods for sequence encoding and interaction calculations
        patch_radius : float, optional
            Radius in Angstroms to define the local patch size, by default 12.0
        filter_std : float, optional
            Standard deviation for Gaussian distance weighting, by default 2.2

        Returns
        -------
        tuple[list[str], np.ndarray]
            Two-element tuple containing:
            - List of amino acid letters that maximize attraction for each patch
            - Array of the corresponding interaction values
        """

        # compute the interaction vectors for each patch
        interact_mat = self.compute_context_interaction_vectors(IMCObject=IMCObject,
                                                                patch_radius=patch_radius,
                                                                filter_std=filter_std)
        
        # find the locations of the strongest interacting values and the values themselves
        max_vals = np.min(interact_mat, axis=1)
        max_idx = np.argmin(interact_mat, axis=1)

        # convert the indexes into amino acids
        strong_interact_seq = self.IMC_object.position_decode_seq(max_idx)

        #return the values
        return strong_interact_seq, max_vals 
    
    def compute_maximal_repulsor(self, IMCObject,
                                  patch_radius: float = 12.0,
                                  filter_std: float = 2.2) -> tuple[list[str], np.ndarray]:
        """
        Identifies the amino acids that would interact least favorably with each surface patch.

        For each patch, determines which amino acid would have the strongest repulsive
        interaction based on the local chemical environment.

        Parameters
        ----------
        IMCObject : IMCObject
            Object containing methods for sequence encoding and interaction calculations
        patch_radius : float, optional
            Radius in Angstroms to define the local patch size, by default 12.0
        filter_std : float, optional
            Standard deviation for Gaussian distance weighting, by default 2.2

        Returns
        -------
        tuple[list[str], np.ndarray]
            Two-element tuple containing:
            - List of amino acid letters that maximize repulsion for each patch
            - Array of the corresponding interaction values
        """

        # compute the interaction vectors for each patch
        interact_mat = self.compute_context_interaction_vectors(IMCObject=IMCObject,
                                                                patch_radius=patch_radius,
                                                                filter_std=filter_std)
        
        # find the locations of the strongest interacting values and the values themselves
        max_vals = np.max(interact_mat, axis=1)
        max_idx = np.argmax(interact_mat, axis=1)

        # convert the indexes into amino acids
        strong_interact_seq = self.IMC_object.position_decode_seq(max_idx)

        #return the values
        return strong_interact_seq, max_vals 
    
    def compute_interaction_strength(self, 
                               IMCObject,
                               patch_radius: float = 12.0,
                               filter_std: float = 2.2,
                               direction: str = 'lt',
                               threshold: float = 0.0) -> np.ndarray:
        """
        Calculates the strength of attractive or repulsive interactions for each surface patch.

        Parameters
        ----------
        IMCObject : IMCObject
            Object containing methods for sequence encoding and interaction calculations
        patch_radius : float, optional
            Radius in Angstroms to define the local patch size, by default 12.0
        filter_std : float, optional
            Standard deviation for Gaussian distance weighting, by default 2.2
        direction : str, optional
            Direction of interaction to measure - 'lt' for attraction, 'gt' for repulsion,
            by default 'lt'
        threshold : float, optional
            Threshold value for considering interactions, by default 0.0

        Returns
        -------
        np.ndarray
            Array of interaction strength values for each patch
        """

        # lets calculate the interaction vectors
        interaction_mat = self.compute_context_interaction_vectors(IMCObject=IMCObject,
                                                                   patch_radius=patch_radius,
                                                                   filter_std=filter_std)

        # determine if we are lookimg at attraction or repulsion
        # you will use the opposite size as you jsut have to replace the values that do not match
        if direction == 'lt': #attraction
            im1 = interaction_mat
            im1[im1 > threshold] = threshold
        if direction == 'gt': #repulsion
            im1 = interaction_mat
            im1[im1 < threshold] = threshold
        # compute the interaction strength
        interaction_strength = np.linalg.norm(im1,axis=1)

        # return the interaction strength
        return interaction_strength
    
    def interaction_permiscutivity(self, IMCObject,
                             patch_radius: float = 12.0,
                             filter_std: float = 2.2,
                             onlyfrac: bool = False) -> np.ndarray:
        """
        Calculates the interaction permiscutivity for each surface patch.

        Permiscutivity measures how promiscuous a patch is in its interactions by comparing
        the overall interaction strength to the maximal attractive interaction.

        Parameters
        ----------
        IMCObject : IMCObject
            Object containing methods for sequence encoding and interaction calculations
        patch_radius : float, optional
            Radius in Angstroms to define the local patch size, by default 12.0
        filter_std : float, optional
            Standard deviation for Gaussian distance weighting, by default 2.2
        onlyfrac : bool, optional
            If True, returns only the fractional term without affine projection, by default False

        Returns
        -------
        np.ndarray
            Array of permiscutivity values for each patch
        """

        # compute the most attractive values
        attrac = self.compute_maximal_attractor(IMCObject=IMCObject, patch_radius=patch_radius, filter_std=filter_std)[1]
        # take the absolute value of this number
        attrac = np.abs(attrac)

        # compute the interaction vector (this is the attractive interaction)
        # this will always be positive in value since we take a norm
        interact_vec = self.compute_interaction_strength(IMCObject=IMCObject,
                                                         patch_radius=patch_radius,
                                                         filter_std=filter_std,
                                                         direction='lt',
                                                         threshold=0.0)

        # take the difference and divide to get the fractional value
        frac = (interact_vec - attrac)/attrac

        # check if the user is only interested in the fractional term
        if onlyfrac:
            return frac
        
        # apply the affine projection
        inner = frac + np.ones(len(frac))
        affine = np.power(inner, 2)
        
        return affine
    
    def compute_patch_heterogeneity_vectors(self, IMCObject, patch_radius : float = 12.0) -> np.ndarray:
        """Computes the the heterogeneity of amino acids over different patches on the surface."""

        # get nearest neighbors and surface neighbors
        self.get_nearest_neighbour_res(distance_thresh=patch_radius)

        # get the patch neighborhoods
        neighbors = self.surface_neighbours

        # get the sequence for the surface
        aa_seq = self.sequence

        # iterate over every patch
        patch_heterogeneity_vecs = []
        for patch_ind_center in neighbors:
            # rescinds is for the residue index in the sequence and the distance from the center point
            resinds = neighbors[patch_ind_center]

            # get the string of amino acid characters (residue index, distance)
            aa_list = [aa_seq[idx] for idx, r in resinds]

            # compute out what the residues are in this region (no filtering is done so the order does not matter)
            subseq = "".join(aa_list)

            # compute the heterogeneity of the patch (set the window size to the length of the sub sequence to get back one vector)
            patch_het_vec = IMCObject.compute_region_chemical_heterogeneity_vectors(subseq, len(subseq))

            # add the patch vector to the list of the rest of them
            patch_heterogeneity_vecs.append(patch_het_vec)

        # return the patch heterogeneity vectors as a numpy array like we did before
        return np.array(patch_heterogeneity_vecs)
    
    def compute_patch_heterogeneity_dict(self, IMCObject, patch_radius : float = 12.0) -> list[dict[str,float]]:
        """Computes the context dependant heterogeniety for each amino acid on each patch"""
        # compute the heterogeneity vectors for each patch
        het_vecs = self.compute_patch_heterogeneity_vectors(IMCObject=IMCObject,
                                                            patch_radius=patch_radius)
        
        # transform the matrix into a list of dictionaries for the heterogeneity
        het_dict = IMCObject.vector_decode_seq(het_vecs)

        # return the heterogeneity dictionary 
        return het_dict
    
    def compute_interaction_heterogeneity(self, IMCObject, patch_radius : float = 12.0) -> np.ndarray:
        """Computes a single number for the interactioin heterogeneity by taking the L2 norm of the vectors
        at each location.
        """
        # compute the heterogeneity vectors for each patch
        het_vectors = self.compute_patch_heterogeneity_vectors(IMCObject=IMCObject,
                                                               patch_radius=patch_radius)
        
        # compute the L2 norm of the heterogeneity vectors
        interaction_heterogeneity = np.linalg.norm(het_vectors, axis=1)

        # return the interaction heterogeneity
        return interaction_heterogeneity
    
    def get_patch_sequence_logo(self, IMCObject,
                            patch_radius: float = 12.0,
                            filter_std: float = 2.2,
                            threshold: float = 0.0) -> tuple[dict[str,float], str, dict[str,float], np.ndarray]:
        """Computes the info for a sequence logo"""
        # find the psuedo sequence that corresponds to each patch
        # get the patch neighborhoods
        neighbors = self.surface_neighbours
        # get the sequence for the surface
        aa_seq = self.sequence
        # iterate over every patch
        patch_seq = []
        for patch_ind_center in neighbors:
            # rescinds is for the residue index in the sequence and the distance from the center point
            patch_aa = aa_seq[patch_ind_center]
            patch_seq.append(patch_aa)

        # combine to patch sequence together
        patch_seq = ''.join(patch_seq)

        # start by computing the interaction
        interaction_mat = self.compute_context_interaction_vectors(IMCObject=IMCObject,
                                                                   patch_radius=patch_radius,
                                                                   filter_std=filter_std)
        
        # now produce a split of attracting and repulsive values into their own unique interaction matrixes
        attract_mat = interaction_mat.copy()
        attract_mat[attract_mat > threshold] = 0
        attract_mat = np.abs(attract_mat)
        repulse_mat = interaction_mat.copy()
        repulse_mat[repulse_mat < threshold] = 0
        repulse_mat = np.abs(repulse_mat)
        

        # convert the matrixes to dictionaries
        attract_dict = IMCObject.vector_decode_seq(attract_mat)
        repulse_dict = IMCObject.vector_decode_seq(repulse_mat)

        # now get the residue numbers
        res_nums = np.arange(1,len(patch_seq)+1)

        # return the data
        return attract_dict, patch_seq, repulse_dict, res_nums
        
        
        



        
    
        

# < end of class> - note we define this here because we actually want to alias the old
# name to the new name....
FoldeDomain = FoldedDomain


# ................................................................................
#
#
def extract_and_write_domains(pdb_file: str, outfile: str, start: int, end: int, reset_indices:bool=True) -> None:
    """
    This function will extract a domain from a pdb file and write it 
    to a new file. Note this function will renumber the residues and atoms
    to start at 1 assuming reset_indices is True.

    Parameters
    ----------
    pdb_file : str
        The path to the pdb file.

    outfile : str
        The path to the output file.

    start : int
        The start residue of the domain.

    end : int
        The end residue of the domain.

    reset_indices : bool
        If True, the function will reset the 
        residue and atom indices to start at 1.

    Returns
    -------
    None

    """

    # read pdb file
    p = md.load(pdb_file)

    # extract domain
    p = p.atom_slice(p.topology.select(f'resid {start} to {end}'))

    # if we want to reset the indices to star at 1...
    if reset_indices:
        # recode residue indices
        for i, residue in enumerate(p.topology.residues):
            residue.index = i + 1  # Setting new residue index starting from 1
            residue.resSeq = i + 1  # Setting new residue index starting from 1

        # recode atom indices
        for i, atom in enumerate(p.topology.atoms):
            atom.index = i+1
            atom.serial = i+1

    # write pdb file
    p.save_pdb(outfile)


    

#def inter_domain_epsilon(fd1, fd2):
#    for 


            
            

        

        
        
