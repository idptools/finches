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





class FoldeDomain:

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
                 ignore_warnings=False):
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
        
        #set the local distance parameters
        self._surface_neighbor_distance = 9.0

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
    def get_nearest_neighbour_res(self, distance_thresh:float=None):
        """

        This function will find the nearest neighbours of each residue in 
        the protein. It will evaluate values for the following object variables
        that can be accessed after the function is called:

        * self.surface_neighbours: dictionary that reports on the indeices of induces 
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
        #update the last used distance
        if distance_thresh is None:
            distance_thresh = self._surface_neighbor_distance
        else:
            self._surface_neighbor_distance = distance_thresh


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
    def calculate_surface_matrix_epsilon(self, input_sequence, IMCObject,
                                         window_seq_distance_extent : int,
                                         window_struct_distance_extent : float,
                                         split = True,
                                         split_threshold : float = 0.0,
                                         idr_tail_exclusion : bool = False):
        """
        Nick's Version
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
        
        window_struct_distance_extent : float
            This is the radius (calculated as a graph distance) of the filtering window in
            the unit of the PDB file.
            
        window_seq_distance_extent : int
            This is the radius of the window in terms of number of residues. This should match with the 
            structural one in spacial units. If window_struct_distance_extent is 8
            and the average residue distance is 0.5 in the same spatial units then 
            the window_seq_distance_extent should be 16.

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
        #amino acid constants
        hydrophobes = ['I' , 'V', 'L', 'A', 'M']
        negative = ['D', 'E']
        positive = ['K', 'R']
        
        #recompute the neighbors based on distance specified
        self.get_nearest_neighbour_res(distance_thresh=window_struct_distance_extent)

        #creating a blank for saving the result
        surface_eps = {}
        
        #sequence distance factor
        seq_dist_factor = window_struct_distance_extent/window_seq_distance_extent
        
        #determine which method to use the idr in
        if idr_tail_exclusion: #chop off the ends of the idr
            adj_idr_idxs = np.arange(window_seq_distance_extent, len(input_sequence)-window_seq_distance_extent,dtype=int)
        else: #use the full idr
            adj_idr_idxs = np.arange(0,len(input_sequence))
            #settup the return matrix output (preallocate)
        ret_mat = np.zeros((len(self.surface_indices),len(input_sequence)), dtype=float)
        if split: #if we split we need to pass twice the information back.
            ret_mat2 = np.zeros((len(self.surface_indices),len(input_sequence)), dtype=float)
        #loop over the structure idxs and idr idxs
        for ll,struct_idx_center in enumerate(self.surface_indices):
            for idr_idx_center in adj_idr_idxs:
                #get the residues and respective distances for th estructured state
                # get indices of all residues neighbouring this residue, excluding itself
                struct_neighbor_resid = [i[0] for i in self.surface_neighbours[struct_idx_center]]
                struct_neighbor_resid.insert(0,struct_idx_center)       
                #grab the characters
                struct_str = [self.sequence[k] for k in struct_neighbor_resid]
                # build a local sequence list
                struct_local_distance = [self.surface_distance_surface[struct_idx_center][k] for k in struct_neighbor_resid]
                
                #do the same for the idr (add 1 due to the non inclusive nature of arange)
                idr_negihbor_resid = np.arange(idr_idx_center-window_seq_distance_extent,idr_idx_center+window_seq_distance_extent+1)
                #grab the sequence bit
                idr_str = [input_sequence[k] for k in idr_negihbor_resid]
                #grabe the distances
                idr_local_distance = [seq_dist_factor*np.abs(k-idr_idx_center) for k in idr_negihbor_resid]
                
                #pass the sequence and distances for each window to the filter obj
                if split: #splitting the attractive and repulsive components
                    pos_vv,neg_vv = IMCObject.calc_filtered_region(struct_str, struct_local_distance, idr_str, idr_local_distance,
                                                                                split = True, split_thresh=split_threshold)
                    ret_mat[ll,idr_idx_center] = pos_vv
                    ret_mat2[ll,idr_idx_center] = neg_vv
                else: #Dont split them and just take the average
                    ret_mat[ll,idr_idx_center] = IMCObject.calc_filtered_region(struct_str, struct_local_distance, idr_str, idr_local_distance,
                                                                                split = False)
                    
        # #determine if the extra whitespace on the exterior of the vectors needs to get removed
        # if remove_extra_space:
        #     if split:
        #         rmp, struct_idxsp, idr_idxsp = IMCObject.remove_empty_rows_or_columns(ret_mat)
        #         rmn, struct_idxsn, idr_idxsn = IMCObject.remove_empty_rows_or_columns(ret_mat2)
                
        #         #recombine them
        #         ret_mat = (rmp, struct_idxsp, idr_idxsp)
        #         ret_mat2 = (rmn, struct_idxsn, idr_idxsn)
        #     else:
        #         pass
        
        #combine the data if nessisary
        if split:
            ret_mat = (ret_mat, ret_mat2)
        return ret_mat
        


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
                


# ................................................................................
#
#
def extract_and_write_domains(pdb_file, outfile, start, end, reset_indices=True):
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


            
            

        

        
        
