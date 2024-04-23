import metapredict as meta
from finches.forcefields.mPiPi import mPiPi_model
from finches import epsilon_to_FHtheory
from finches import epsilon_calculation

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

import matplotlib

# ensure text is editable in illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# set to define axes linewidths
matplotlib.rcParams['axes.linewidth'] = 0.5


class FinchesFrontend:
    """
    Base class for the FinchesFrontend. This class should not be instantiated directly, but
    instead should be used as a base class for other classes that implement the FinchesFrontend
    interface.

    Note that depending on the derived model being used, different methods may be implemented
    in the derived class. It is recommended that before creating a new derived class, the
    methods in this class are reviewed to ensure that the derived class implements the same
    methods. Also, if any of this is confusing, please familiarize yourself with inheritance
    in Python.

    """

    def __init__(self):

        # ensure we don't accidentally instantiate this class!
        if type(self) == FinchesFrontend:
            raise TypeError("FinchesFrontened class should not be instantiated directly, but instead derived classes should be used.")
        # this must be defined in the subclass
        self.IMC_object = None


    # ....................................................................................
    #
    #        
    def intermolecular_idr_matrix(self,
                                  seq1,
                                  seq2,
                                  window_size=31,
                                  use_cython=True,
                                  use_aliphatic_weighting=True,
                                  use_charge_weighting=True,
                                  disorder_1=True,
                                  disorder_2=True):

        """
        Returns the interaction matrix for the two sequences. Specifically this involves
        decomposing the two sequences into window_size fragments and calculating the inter-fragment
        epsilon values using a sliding window approach.

        Note that we don't pad the sequence here, so the edges of the matrix start
        and end at indices that depend on the window size. To avoid confusion, the
        function also returns the indices for sequence1 and sequence2.

        Parameters
        --------------
        seq1 : str
            Input sequence 1

        seq2 : str
            Input sequence 2

        window_size : int
            The window size to use for the interaction matrix calculation. 
            Default is 31. 

        use_cython : bool
            Whether to use the cython implementation of the interaction matrix calculation.
            Default is True. 

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme for the interaction matrix
            calculation. This weights local aliphatic residues based on the number of
            aliphatic residues adjacent to them. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme for the interaction matrix. This
            weights local charged residues based on the number of charged residues adjacent
            to them. Default is True.

        disorder_1 : bool
            Whether to generate the disorder profile for sequence 1. Default is True. If False,
            a uniform disorder profile is used (all values=1).

        disorder_2 : bool
            Whether to generate the disorder profile for sequence 2. Default is True. If False,
            a uniform disorder profile is used (all values=1).

        Returns
        --------------
        tuple
            A tuple containing the interaction matrix, disorder profile for sequence 1, and disorder
            profile for sequence 2. 

            [0] : This is interaction matrix, and is itself a tuple of 3 elements. The first 
            is the matrix of sliding epsilon values, and the second and 3rd are the indices that map            
            sequence position from sequence1 and sequence2 to the matrix

            [1] disorder profile for sequence 1. Will be all 1s if disorder_1 is False

            [2] disorder profile for sequence 2. Will be all 1s if disorder_2 is False

        """                                  
        
        # compute the matrix
        B = self.IMC_object.calculate_sliding_epsilon(seq1,
                                                      seq2,
                                                      window_size=window_size,
                                                      use_cython=use_cython,
                                                      use_aliphatic_weighting=use_aliphatic_weighting,
                                                      use_charge_weighting=use_charge_weighting)


        # compute disorder profile for sequence 1 assuming disorder_1 is set to True
        if disorder_1:            
            disorder_1 = meta.predict_disorder(seq1)[B[1][0]:B[1][-1]+1]
        else:
            disorder_1 = np.array([1]*B[0].shape[0])

        # compute disorder profile for sequence 1 assuming disorder_2 is set to True
        if disorder_2:
            disorder_2 = meta.predict_disorder(seq2)[B[2][0]:B[2][-1]+1]
        else:
            disorder_2 = np.array([1]*B[0].shape[1])
            
        # this assertion clause just checks our x values match for matrix dimensions vs. disorder profile
        assert B[0].shape[0] == len(disorder_1)

        # and our y protein
        assert B[0].shape[1] == len(disorder_2)

        return (B, disorder_1, disorder_2)


    
    # ....................................................................................
    #
    #            
    def epsilon(self,
                seq1,
                seq2,
                use_aliphatic_weighting=True,
                use_charge_weighting=True):
        """
        Returns the epilson value associated with the two sequences. 

        Parameters
        --------------
        seq1 : str
            Input sequence 1

        seq2 : str
            Input sequence 2

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme for the interaction matrix
            calculation. This weights local aliphatic residues based on the number of
            aliphatic residues adjacent to them. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme for the interaction matrix. This
            weights local charged residues based on the number of charged residues adjacent
            to them. Default is True.

        Returns
        --------------
        float
            The epsilon value for the two sequences.

        """        
        return self.IMC_object.calculate_epsilon_value(seq1,
                                                       seq2,
                                                       use_aliphatic_weighting=use_aliphatic_weighting,
                                                       use_charge_weighting=use_charge_weighting)

    
    # ....................................................................................
    #
    #            
    def interaction_figure(self,
                           seq1,
                           seq2,
                           window_size=31,
                           use_cython=True,
                           use_aliphatic_weighting=True,
                           use_charge_weighting=True,
                           tic_frequency=100,
                           seq1_domains=[],
                           seq2_domains=[],
                           vmin=-3,
                           vmax=3,
                           cmap='PRGn',
                           fname=None,
                           zero_folded=True,
                           disorder_1=True,
                           disorder_2=True):
                           
    
        """
        Function to generate an interaction matrix figure between two sequences. This does
        all the calculation on the backend and formats a figure with parallel disorder tracks 
        alongside the interaction matrix.
        
        Parameters
        --------------
        seq1 : str
            Input sequence 1

        seq2 : str
            Input sequence 2

        window_size : int
            Size of the window to use for the interaction matrix calculation. Note
            this must be an odd number and will be converted to an odd number if it
            is not. Default is 31.

        use_cython : bool
            Whether to use the cython implementation of the interaction matrix (always
            use this if you can). Default is True.

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme for the interaction matrix
            calculation. This weights local aliphatic residues based on the number of
            aliphatic residues adjacent to them. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme for the interaction matrix. This
            weights local charged residues based on the number of charged residues adjacent
            to them. Default is True.

        tic_frequency : int
            Frequency of the TICs on the plot. Default is 100.

        seq1_domains : list
            List of tuples/lists containing the start and end positions of domains in 
            sequence 1. Means these can be easily higlighted in the plot.

        seq2_domains : list
            List of tuples/lists containing the start and end positions of domains in
            sequence 2. Means these can be easily higlighted in the plot.

        vmin : float
            Minimum value for the interaction matrix color scale. Default is -3.

        vmax : float
            Maximum value for the interaction matrix color scale. Default is 3.
        
        cmap : str
            Colormap to use for the interaction matrix. Default is 'PRGn'.

        fname : str
            Filename to save the figure to. If None, the figure will be displayed

        zero_folded : bool
            Whether to zero out the interaction matrix for folded residues. Default is 
            True.

        disorder_1 : bool
            Whether to include the disorder profile for sequence 1. Default is True.

        disorder_2 : bool
            Whether to include the disorder profile for sequence 2. Default is True.


        Returns
        --------------
        A tuple containing the figure and the axes objects for the main plot, the top
        disorder plot, the right disorder plot and the colorbar.

            fig : matplotlib.figure.Figure (from plt.figure()

            im : matplotlib.image.AxesImage (from plt.imshow())

            ax_main : matplotlib.axes.Axes (from plt.subplot2grid()

            ax_top : matplotlib.axes.Axes  (from plt.subplot2grid()

            ax_right : matplotlib.axes.Axes  (from plt.subplot2grid()

            ax_colorbar : matplotlib.axes.Axes  (from plt.subplot2grid()
    

        """

        # note - when this is called from a derived class it's the DERIVED CLASS
        # version of the intermolecular_idr_matrix function that's called, so RNA
        # is handelled in this correctly
        B, disorder_1, disorder_2 = self.intermolecular_idr_matrix(seq1,
                                                                   seq2,
                                                                   window_size=window_size,
                                                                   use_cython=use_cython,
                                                                   use_aliphatic_weighting=use_aliphatic_weighting,
                                                                   use_charge_weighting=use_charge_weighting)


        if zero_folded:        
            try:
                folded_1 = meta.predict_disorder_domains(seq1).folded_domain_boundaries
                folded_2 = meta.predict_disorder_domains(seq2).folded_domain_boundaries

                

            except Exception as e:
                folded_1 = []
                folded_2 = []


        else:
            folded_1 = []
            folded_2 = []

        B1_start = B[1][0]
        B2_start = B[2][0]

            
        for i in range(B[1][0]-1, B[1][-1]-1):
            for j in range(B[2][0]-1, B[2][-1]-1):

                
                for fd in folded_1:
                    if i >= fd[0] and i <= fd[1]:
                        
                        B[0][i - B1_start, j - B2_start ] = 0

                for fd in folded_2:
                    if j >= fd[0] and j <= fd[1]:
                        B[0][i - B1_start, j - B2_start ] = 0


        # extract out the interaction matrix
        matrix = B[0]
        
        # Create a figure and a grid of subplots; note we can tweak the figure size params to 
        # make the figure look good depending on how long the two sequences are, here 5.5 and 5
        # works but this is adjustable ofc.
        fig = plt.figure(figsize=(8.5, 8.5), dpi=350)
    
        # Main matrix plot; create axis and then plot using seismic so 0=white
        ax_main = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
        im = ax_main.imshow(matrix.T, extent=[B[1][0], B[1][-1], B[2][0], B[2][-1]], origin='lower', aspect='auto', vmax=vmax, vmin=vmin, cmap=cmap)
    
        # edt here to change tickmarks;  note again the tic_frequency, this again
        # probably can be edited manually depending on the system
        ax_main.set_xticks(np.arange(B[1][0],B[1][-1], tic_frequency))
        ax_main.set_yticks(np.arange(B[2][0],B[2][-1], tic_frequency))
        ax_main.tick_params(axis='x', rotation=45)  # Rotates the x-tick labels by 45 degrees
    
    
        ## .....................................................................
        # Bar plot for X protein 
        ax_top = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=ax_main)
    
        # plot disorder profile 
        ax_top.bar(B[1], disorder_1, width=1, color='k', alpha=0.3)
    
        # if we wanted to plot mean per-residue inteaction value, uncomment
        ax_top.set_xlim(B[1][0], B[1][-1])

        # disorder goes 0 to 1
        ax_top.set_ylim(0,1.05)
        
    
        # highlight some specific regions manually
        for r in seq1_domains:
            region_start = r[0]
            region_end   = r[1]
            ax_top.axvspan(region_start,region_end, color='k', linewidth=0, alpha=0.7)

    
        ## .....................................................................
        # Bar plot for Y protein
        ax_right = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
        ax_right.barh(B[2], disorder_2, align='center', height=1, color='k', alpha=0.3)
    
        ax_right.set_yticks(ax_main.get_yticks())
        ax_right.set_ylim(ax_main.get_ylim())

        # disorder goes 0 to 1
        ax_right.set_xlim(0,1.05)
    
        # highlight some specific regions manually
    
    
        for r in seq2_domains:
            region_start = r[0]
            region_end   = r[1]
            ax_right.axhspan(region_start,region_end, color='k', linewidth=0, alpha=0.7)
    
        ax_colorbar = plt.subplot2grid((4, 4), (0, 3))
        cbar = fig.colorbar(im, cax=ax_colorbar, orientation='vertical')
        ax_colorbar.yaxis.set_ticks_position('left')
        ax_colorbar.yaxis.set_label_position('left')

        plt.tight_layout()
        
        # finally save the figure
        if fname is not None:
            plt.savefig(fname, dpi=350)

        return fig, im, ax_main, ax_top, ax_right, ax_colorbar


    # ....................................................................................
    #
    #
    def per_residue_attractive_vector(self,
                                      seq1,
                                      seq2,
                                      window_size=31,
                                      use_cython=True,
                                      use_aliphatic_weighting=True,
                                      use_charge_weighting=True,                                      
                                      return_total=False,
                                      attractive_threshold=0,
                                      smoothing_window=20,
                                      poly_order=3):
        
        """
        Function to calculate the per-residue attractive vector for a given pair 
        of sequences. This is calculated as the sum of the attractive interactions 
        for each residue in the first sequence with all residues in the second 
        sequence. Specifically, this is an average over all attractive values (i.e.
        where value < 0) using the inter-sequence matrix.

        If return_total is True, the function will return the total sum of attractive
        interactions between the two sequences instead of the average.

        This is (potentially) interesting inasmuch as if we just tak the AVERAGE of
        a region it may be very attractive in some place but repulsive elsewhere, 
        however, repulsive regions in an IDR can avoid each other while attractive
        things attract, so this allows you to identify the putative 'sticker' regions
        without confounding by repulsive regions. 


        Parameters
        ----------
        seq1 : str
            The first sequence

        seq2 : str
            The second sequence

        window_size : int
            The window size for the intermolecular matrix. Default is 31.

        use_cython : bool
            Whether to use the cython implementation of the intermolecular matrix
            calculation. Default is True.

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme. Default is True.
   
        return_total : bool
            If True, return the total sum of attractive interactions between 
            the two sequences. Sometimes you may want this.
        
        attractive_threshold : float
            The threshold for what is considered attractive. Default is 0 (i.e. 
            only negative values are considered attractive). If changed anything
            less than this value will be considered attractive.

        smoothing_window : int
            The window size for the Savgol filter. This is used to
            smooth the per-residue attractive vector. If set to False no
            smoothing is applied. Default is 20.

        poly_order : int
            The polynomial order for the Savgol filter. This is used to
            smooth the per-residue attractive vector. If set to False no
            smoothing is applied. Default is 3.


        Returns
        -------
        tuple of np.arrays

            [0] : np.array
                Indices of the residues in the first sequence

            [1] : np.array
                The per-residue attractive vector

        """

        # do the thing; note this class from the derived class so model-specific
        # sanity checking is handled implictly here
        B = self.intermolecular_idr_matrix(seq1, seq2, window_size=window_size, use_cython=use_cython, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)[0]

        # extract the raw matrix
        raw_matrix = B[0]

        # get dem indices
        idx = np.arange(B[1][0], B[1][-1]+1)

        # create a mask for attractive values
        attractive_mask = raw_matrix < attractive_threshold

        # sum attractive values in each column
        attractive_sums = np.sum(raw_matrix * attractive_mask, axis=1)

        if len(attractive_sums) != len(idx):
            print(len(attractive_sums), len(idx))
            raise ValueError('Length of attractive sums does not match length of indices; this is a bug.')

        # count attractive values in each column
        attractive_counts = np.sum(attractive_mask, axis=1)

        # Avoid division by zero for columns with no attractive values
        attractive_counts[attractive_counts == 0] = 1

        # alculate the average of attractive values in each column
        if return_total:
            vals = attractive_sums
        else:
            vals = attractive_sums / attractive_counts

        # if no smoothing is requested
        if smoothing_window is False or poly_order is False:
            pass
        else:
            try:
                vals = savgol_filter(vals, smoothing_window, poly_order)
            except Exception as e:
                print('')
                print('Error when trying to apply savgol filter; error message below')
                raise(e)


        return idx, vals


    # ....................................................................................
    #
    #
    def protein_nucleic_vector(self, seq, fragsize=31, smoothing_window=30, poly_order=3):
        """
        Function to calculate the per-residue attractive vector for a given protein
        sequence. This is calculated as the sliding-window average of a fragsize
        region of the protein with a fragsize region of poly-U RNA. 

        The two-position return vector returns the indices of the residues in the
        protein sequence and the per-residue attractive vector. Note that indices
        START at fragsize-1/2 and END at len(seq)-fragsize+1/2. This is because the
        sliding window is centered on each residue.

        Parameters
        ----------
        seq : str
            The protein sequence

        fragsize : int
            The size of the sliding window. Must be an odd number. Default is 31.

        smoothing_window : int
            The window size for the Savgol filter. This is used to smooth the
            per-residue attractive vector. If set to False no smoothing is applied.

        poly_order : int
            The polynomial order for the Savgol filter. This is used to smooth the
            per-residue attractive vector. If set to False no smoothing is applied.


        Returns
        -------
        tuple of np.arrays

            [0] : np.array
                Indices of the residues in the protein sequence

            [1] : np.array
                The per-residue attractive vector

        """

        if fragsize % 2 == 0:
            raise Exception('fragsize must be odd')

        
        # define the function to calculate the epsilon value for a given sequence; the closure
        # used with the fragsize and IMC_object variables is to avoid having to pass these
        # as arguments to the function
        def RNA_bind(seq):
            return epsilon_calculation.get_sequence_epsilon_value(seq, len(seq)*'U', self.IMC_object)/fragsize

        # initialize the return vector
        return_vector = []
        idx = []

        # if the sequence is longer than the fragment size, calculate the per-residue
        # attractive vector for each fragment
        if len(seq) > fragsize:
        
            for i in range(0,1+(len(seq)-fragsize)):
                return_vector.append(RNA_bind(seq[i:i+fragsize]))
                idx.append(i+(fragsize-1)/2)
                
                
                
        # if the sequence is shorter than the fragment size, calculate the per-residue
        # attractive vector for the whole sequence, which is all we can really do here
        else:
            return_vector.append(RNA_bind(seq))
            idx = [int(len(seq)/2)]

        return_vector = np.array(return_vector)
        idx = np.array(idx)

        if smoothing_window is False or poly_order is False:
            return [idx, return_vector]
        else:
            return [idx, savgol_filter(return_vector, smoothing_window, poly_order)]


    # ....................................................................................
    #
    #
    def build_phase_diagram(self, seq, use_aliphatic_weighting=True, use_charge_weighting=True):
        """
        Function to build a homotypic phase diagram for a given sequence. This is done by
        calculating the overall epsilon for the sequence, and then combining this with 
        closed-form expressions for the binodal and spinodal lines.

        The return 

        Parameters
        ----------
        seq : str
            The protein sequence

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme. Default is True.

        Returns
        -------
        tuple of np.arrays

            [0] - Dilute phase concentrations (array of len=N) in Phi
            [1] - Dense phase concentrations (array of len=N) in Phi
            [2] - List with [0]: critical T and [1]: Critical phi
            [3] - List of temperatures that match with the dense and dilute phase concentrations
            [4] - Dilute phase concentrations (array of len=N) in Phi for spinodal
            [5] - Dense phase concentrations (array of len=N) in Phi for spinodal
            [6] - List with [0]: critical T and [1]: Critical phi  for spinodal
            [7] - List of temperatures that match with the dense and dilute phase concentrations for spinodal

        This seems somewhat overwhelming, but to plot the resulting binodal we just need to do:

        # assuming mf = is a frontend object
        B = mf.build_phase_diagram(seq)

        # binodal low arm
        plt.plot(B[0], B[3], 'blue', label='sequence name')

        # binodal high arm
        plt.plot(B[1], B[3], 'blue')

        plt.legend()

        """
        eps = self.epsilon(seq, seq, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)
        
        return epsilon_to_FHtheory.epsilon_to_phase_diagram(seq, eps)



    def plot_phase_diagram(self,
                           seq,
                           use_aliphatic_weighting=True,
                           use_charge_weighting=True,
                           line_color='k',
                           line_style='-',
                           line_width=0.5,
                           xlim=None,
                           ylim=None,
                           xlog=False,
                           width=1.2,
                           height=2.2,
                           filename=None):
        """
        Function to plot the phase diagram for a given sequence. This is done by
        calculating the overall epsilon for the sequence, and then combining this with
        closed-form expressions for the binodal.

        Parameters
        ----------
        seq : str
            The protein sequence

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme. Default is True.

        xlim : tuple
            The x-axis limits. Default is None meaning it is determined automatically.
            if provided must be a 2-position tuple e.g. [xmin, xmax].

        ylim : tuple
            The y-axis limits. Default is None meaning it is determined automatically.
            if provided must be a 2-position tuple e.g. [ymin, ymax].

        xlog : bool
            Whether to plot the x-axis in log scale. Default is False.

        width : float
            The width of the figure in inches. Default is 1.2.

        height : float
            The height of the figure in inches. Default is 2.2.

        filename : str
            The filename to save the figure. Default is None, meaning the figure 
            is not saved. If provided, the filename must include the extension.

        Returns
        -------
        Tuple

            [0] - The complex tuple associated with the phase diagram (see the function
                  signature of build_phase_diagram for more details)

            [1] - Figure object

            [2] - Axes object


        """                           
                           
        B = self.build_phase_diagram(seq, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)

        fig = plt.figure(figsize=(width, height), dpi=450)
        ax = plt.gca()
                           
        plt.plot(B[0], B[3], color=line_color, ls=line_style, lw=line_width)
        plt.plot(B[1], B[3], color=line_color, ls=line_style, lw=line_width)
        
        plt.ylabel(r'$T (AU)$', fontsize=7)
        plt.xlabel(r'$\rm\phi$', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlog is True:
            plt.xscale('log')

        plt.tight_layout()
            
        if filename is not None:
            plt.savefig(filename, dpi=450)
            

        return [B, fig, ax]


    def plot_multiple_phase_diagrams(self,
                                     seq_dict,
                                     use_aliphatic_weighting=True,
                                     use_charge_weighting=True,
                                     tc_ref=None,
                                     line_style='-',
                                     line_width=0.5,
                                     xlim=None,
                                     ylim=None,
                                     xlog=False,
                                     width=2.2,
                                     height=1.2,
                                     filename=None):
        """
        Function to plot the phase diagram for a given sequence. This is done by
        calculating the overall epsilon for the sequence, and then combining this with
        closed-form expressions for the binodal.

        Parameters
        ----------
        seq_dict : dict
            Dictionary where keys are sequence names and values are a 2-position list, 
            where the first element is the sequence and the second element the line
            to plot. 

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme. Default is True.

        xlim : tuple
            The x-axis limits. Default is None meaning it is determined automatically.
            if provided must be a 2-position tuple e.g. [xmin, xmax].

        ylim : tuple
            The y-axis limits. Default is None meaning it is determined automatically.
            if provided must be a 2-position tuple e.g. [ymin, ymax].

        xlog : bool
            Whether to plot the x-axis in log scale. Default is False.

        width : float
            The width of the figure in inches. Default is 1.2.

        height : float
            The height of the figure in inches. Default is 2.2.

        filename : str
            The filename to save the figure. Default is None, meaning the figure 
            is not saved. If provided, the filename must include the extension.

        Returns
        -------
        Tuple

            [0] - The complex tuple associated with the phase diagram (see the function
                  signature of build_phase_diagram for more details)

            [1] - Figure object

            [2] - Axes object


        """
        
        
        fig = plt.figure(figsize=(width, height), dpi=450)
        ax = plt.gca()

        if tc_ref is not None:
            if tc_ref not in seq_dict:
                raise ValueError('The reference sequence is not in the dictionary.')

            local_seq = seq_dict[tc_ref][0]

            
            B_ref = self.build_phase_diagram(local_seq, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)
            tc_ref = max(B_ref[3])
        else:
            tc_ref = 1

        # for each sequence in the dictionary
        all_phase_diagrams = []
        for k in seq_dict:
            
            B = self.build_phase_diagram(seq_dict[k][0], use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)

            plt.plot(B[0], B[3]/tc_ref, color=seq_dict[k][1], ls=line_style, lw=line_width, label=k)
            plt.plot(B[1], B[3]/tc_ref, color=seq_dict[k][1], ls=line_style, lw=line_width)

            all_phase_diagrams.append(B)

        if np.isclose(tc_ref,1):
            plt.ylabel(r'$T (AU)$', fontsize=7)            
        else:
            plt.ylabel(r'$T/T_c$', fontsize=7)
            
            
            
        plt.xlabel(r'$\rm\phi$', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlog is True:
            plt.xscale('log')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=4, frameon=False)            

        plt.tight_layout()
            
        if filename is not None:
            plt.savefig(filename, dpi=450)
            

        return [all_phase_diagrams, fig, ax]
            


    
        
        



        
        
    
    
                           

        


        

    

    
