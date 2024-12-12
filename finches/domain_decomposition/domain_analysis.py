import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import metapredict as meta
from scipy.signal import find_peaks

class PeakError(Exception):
    # custom exception
    pass


class InteractingRegions:
    """
    Data structure to represent interacting regions between two sequences. 
    This is just to make it more convenient to work with local
    interacting regions
    The Struct (class) has the following attributes:

    r1_start : int
        The start index of the first sequence.

    r1_end : int
        The end index of the first sequence.    

    r1_sequence : str   
        The sequence of the first region.

    r2_start : int
        The start index of the second sequence.

    r2_end : int
        The end index of the second sequence.

    r2_sequence : str
        The sequence of the second region.
    
    """
    def __init__(self, 
                 r1_start, 
                 r1_end, 
                 r1_sequence, 
                 r2_start, 
                 r2_end, 
                 r2_sequence,
                 window_size,
                 sequence_1,
                 sequence_2,
                 ) -> None:
        
        # initialize the six attributes
        for key, value in locals().items():
            if key != "self":  # Exclude 'self' from being set
                setattr(self, key, value)   

        window_half = int((window_size-1)/2)

        self.r1_start_full = r1_start - window_half
        self.r1_end_full   = r1_end   + window_half

        self.r2_start_full = r2_start - window_half
        self.r2_end_full   = r2_end   + window_half

        # we do a -1 at the start to convert from amino acid space to Python indexing 
        # space
        self.r1_sequence_full = sequence_1[self.r1_start_full-1:self.r1_end_full]
        self.r2_sequence_full = sequence_2[self.r2_start_full-1:self.r2_end_full]

        # set default values for the rectangle
        self.edgecolor = 'r'
        self.rectangle_kwargs = {}
        
    @property
    def rectangle_info(self):
        """
        Property that returns a list which can be added to a list of list to draw
        rectangles. Note to pass additional keyword arguments to the rectangle,
        you can set them in the rectangle_kwargs attribute. For example, if you want
        to set the facecolor of the rectangle to 'blue', you can do the following:
        ```
        ir = InteractingRegions(...)
        ir.rectangle_kwargs = {'facecolor': 'blue'}
        ```        
        """

        return [self.r1_start, self.r1_end, self.r2_start, self.r2_end, {'edgecolor': self.edgecolor, **self.rectangle_kwargs}]         
    
    def __str__(self):
        return f"InteractingRegions(seq_1: {self.r1_start}-{self.r1_end} ({self.r1_end-self.r1_start}), seq2:{self.r2_start}-{self.r2_end}) ({self.r2_end-self.r2_start})"
    
    def __repr__(self):
        return str(self)

        
#....................................................................................
#
#         
def extract_regions(frontend_obj, 
                    seq1, 
                    seq2,
                    window_size=31,
                    criteria="less", 
                    criteria_threshold=-0.2, 
                    baseline=0.3, 
                    min_region_area=500,
                    min_region_size=20,
                    penalize_opposite=False,
                    offset_threshold=0.2):                                          

                    
    '''
    Given a frontend object and two sequences, return the intermap and 
    regions of interaction.

    Parameters:
    ------------------
    frontend_obj : finches.frontend.frontend_base.FinchesFrontend
        The frontend object to use for the analysis. This object 
        should have a method called intermolecular_idr_matrix that 
        takes two sequences as input and returns the intermap.

    seq1 : str
        The first sequence to use for the analysis.

    seq2 : str
        The second sequence to use for the analysis.

    window_size : int
        The window size to use for the intermap calculation. Default 
        is 31.

    criteria : str
        The criteria to use for the analysis. Can be 'less' or 
        'greater'. Default is 'less'.

    criteria_threshold : float
        The threshold to use for the criteria, i.e. the value
        upon which masking is done to define regions.
        Default is -0.2.

    baseline : float
        The baseline to use for the analysis. Default is 0.3.

    min_region_area : int
        The minimum region area to use for the analysis.
        Default is 500.

    min_region_size : int
        The minimum region size (in amino acids) to use for the analysis. 
        Default is 20.

    penalize_opposite : bool
        Whether to penalize opposite interactions. Default is False.

    offset_threshold : float
        The threshold to use for penalizing opposite interactions. 
        Default is 0.2 to mirror default criteria_threshold.


    Returns:
    ------------------
    interactor_pairs : list
        A list of InteractingRegions objects that represent the interacting regions.

    '''

    # guarentees odd window size; this is the design pattern we use throughout
    if window_size % 2 == 0:
        print(f"Warning: window size is even, rounding up to next odd number {window_size+1}")
        window_size = window_size + 1

    # compute the intermap
    intermap = frontend_obj.intermolecular_idr_matrix(seq1, seq2, window_size=window_size)[0][0].transpose()

    # get the interaction regions
    regions = get_bidirectional_interaction_regions(intermap,
                                                    criteria=criteria,
                                                    criteria_threshold=criteria_threshold,
                                                    baseline=baseline,
                                                    min_region_area=min_region_area,
                                                    min_region_size=min_region_size,
                                                    penalize_opposite=penalize_opposite,
                                                    offset_threshold=offset_threshold)                                          
   
    # build out the actual interacting regions
    interactor_pairs = []
    for region in regions:

        # seq start and end are in sequence space (i.e. 1 thru len(seq))
        seq_1_start = convert_intermap_index_to_residue_number(region[0][0], window_size)
        seq_1_end   = convert_intermap_index_to_residue_number(region[0][1], window_size)
                                                            
        seq_2_start = convert_intermap_index_to_residue_number(region[1][0], window_size)
        seq_2_end   = convert_intermap_index_to_residue_number(region[1][1], window_size)

        # we do a -1 at the start to convert from amino acid space to Python indexing 
        # space
        seq_1_region = seq1[seq_1_start-1:seq_1_end]
        seq_2_region = seq2[seq_2_start-1:seq_2_end]
        interactor_pairs.append(InteractingRegions(seq_1_start, seq_1_end, seq_1_region, seq_2_start, seq_2_end, seq_2_region, window_size, seq1, seq2))            
            

    return (interactor_pairs, regions, intermap)



#....................................................................................
#
#         
def convert_intermap_index_to_residue_number(index, window_size=31):  
    """
    Given an index in an intermap, return the residue number it corresponds to.

    Parameters:
    ------------------
    index : int
        The index in the intermap.

    window_size : int
        The window size used to generate the intermap.

    Returns:
    ------------------
    int
        The residue number corresponding to the index.
    """

    if window_size % 2 == 0:
        raise ValueError(f"Window size ({window_size}) must be odd; this should have been caught earlier")
    
    window_half = int((window_size-1)/2)

    return index + window_half


#....................................................................................
#
#
def peak_width(data, peak_index, baseline):
    '''
    Function that returns the width of a peak in a given data set at a 
    given x value using a given baseline value. In short, "what x range
    does the peak span before dropping below the baseline value?". 

    Parameters:
    ------------------
    data : np.array
        The data to analyze. This should be a 1D numpy array.

    peak_index : int
        The index of the peak of interest along the data array.

    baseline : float
        The baseline value to use for the analysis. This is the value where 
        the peak 'ends', defined as a specific value.

    Parameters:
    ------------------
    tuple
        Returns a tuple of the form (peak_width, left_index, right_index),

    '''
    # Determine peak value and check for validity

    # The 'y value' of the data/peak at peak_index
    peak_value = data[peak_index]  

    # If the peak value falls below the given baseline. Should not occur in any case 
    # where this function is used, but designed to catch critical errors.

    # Raise exception if peak value is below baseline. This should not
    if peak_value < baseline:  
        raise PeakError("Peak value is below baseline value. This should not occur in this context.")
        
    ## Find the left crossing point

    # Set initial left_index to peak_index, we're now looking for the left edge of the peak, where
    # the 'edge' here is the index at which the x-value falls below the baseline value.
     
    # to do this, we step left-wards from the starting point until we drop below the baseline 
    # value OR we get to the start of the array
    left_index = peak_index  
    while left_index > -1 and data[left_index] >= baseline:          
        left_index -= 1  
    left_index = left_index + 1  

    ## Find the right crossing point
    # repea the same thing moving right
    right_index = peak_index  
    while right_index < len(data) and data[right_index] >= baseline:          
        right_index += 1      
    right_index -= 1  

    # +1 to include both endpoints
    width = right_index - left_index + 1  

    # Return width, left_index, and right_index
    return width, left_index, right_index  
             

#....................................................................................
#
#         
def remove_included_ranges(ranges):
    '''
    Given a list of index ranges (ie 1-10, 5-12, etc) in form [(1,10),
     (5,12)], remove any ranges fully included within another range in 
     the list.

     Parameters:
    ------------------
    ranges : list
        A list of index ranges in the form [(i1,i2),(i1,i2),...].

    Returns:
    ------------------
    list
        A list of index ranges with any fully included ranges removed.
    '''
 
    # Sort ranges by left_index, then by right_index
    sorted_ranges = sorted(ranges, key=lambda x: (x[1], x[2]))
    
    # Filter for included ranges
    filtered_ranges = []

    # For each range tuple, in order
    for current in sorted_ranges:  

        ## Check if current range is included in any of the ranges already in filtered_ranges
        included = False  # Default included (whether current range is included in any other range)

        # For each already-accepted range
        for existing in filtered_ranges:  

            # If current is included in existing
            if existing[1] <= current[1] and existing[2] >= current[2]:  
                included = True  # Say so
                break  # And stop looking

        if not included:  # If current range is not included in any existing range
            filtered_ranges.append(current)  # Add it to the filtered list

    return filtered_ranges  # Return filtered ranges



#....................................................................................
#
#         
def get_column_regions(array, 
                        criteria="less", 
                        criteria_threshold=-0.2, 
                        baseline=0.3,
                        min_region_size=20,
                        penalize_opposite=False,
                        offset_threshold=0.2):


    '''
    Given a 2d numpy array (an intermap), define regions based on peaks in 
    column counts of interaction values meeting a threshold criterion.

    Parameters: 
    ------------------
    array : np.array
        The array to analyze. This should be a 2D numpy array that represents
        an intermap.

    criteria : str
        The criteria to use for the analysis. Can be 'less' or 
        'greater'. Default is 'less'.

    threshold : float
        The threshold to use for the criteria, i.e. the value
        upon which masking is done to define regions. Default is -0.2.

    baseline : float
        The baseline to use for the analysis. Default is 0.3.

    penalize_opposite : bool
        Whether to penalize opposite interactions. Default is False.

    offset_threshold : float
        The threshold to use for penalizing opposite interactions. 
        Default is 0.2 to mirror default criteria_threshold.


    Returns:
    ------------------
    tuple 
        A tuple of the form (bounds_filtered, counts), where bounds_filtered
        is a list of accepted peaks with included ranges removed, and counts
        is the raw column count data of interaction values meeting the threshold
        criterion (i.e. counts = is the sum of the masked intermap along the 
        columns). The bounds_filtered list elements are themselelves tuples of
        the form (width, left_index, right_index).
    '''
    
    # Filter for values in array that meet criterion and threshold
    if criteria == "less":
        within_range = (array <= criteria_threshold)
    elif criteria == "greater":
        within_range = (array >= criteria_threshold)
    else:
        raise ValueError(f"criteria must be 'less' or 'greater', not {criteria}")

    # Count of interaction values that meet the criterion/threshold in each column
    # counts then is a 1D array     
    counts = np.sum(within_range, axis=0)  

    # not implemented, BUT we could also rather than masking, make the penality
    # absolute; i.e. this would involve multiplying offset matrix a large number
    # so no peaks will be found if the residues are overlapping with a 'bad' region
    if penalize_opposite:
        if criteria == "less":
            offset = (array >= offset_threshold)
        elif criteria == "greater":
            offset = (array <= offset_threshold)

        # apply offset based on penalize_opposite
        counts = counts - np.sum(offset, axis=0)

    # Smooth counts via moving average to denoise slightly
    counts_smooth = np.convolve(counts, [1,1,1], mode='same')  

     # Use scipy find_peaks to get indices of peaks in counts_smooth
    peaks_raw = find_peaks(counts_smooth)[0] 

    # Get width/start index/end index parameters from each of the raw peaks, skipping
    # any peaks that fail
    bounds = []
    for peak in peaks_raw:
        try:
            bounds.append(peak_width(counts_smooth, peak, baseline*counts_smooth[peak]))
        except PeakError:
            pass

    # filter out peaks that are too small
    bounds_filtered = []
    for i in bounds:

        # we use >= to to mirror logic of the get_interaction_regions() function
        if i[0] >= min_region_size:
            bounds_filtered.append(i)

    return bounds_filtered, counts  # Return bounds_filtered, counts


#....................................................................................
#
#         
def get_interaction_regions(array, 
                            criteria="less", 
                            criteria_threshold=-0.2, 
                            baseline=0.3, 
                            min_region_area=500, 
                            min_region_size=20,
                            penalize_opposite=False,
                            offset_threshold=0.2):

    '''
    Given a 2d numpy array (intermap), use get_column_regions iteratively to find 2d 
    regions of specified interaction character and size.

    array is the 2d numpy array to analyze.
    criteria is whether to look for values 'less' than or 'greater' than threshold.
    threshold is the thresholding value for interaction values.

    baseline is the coefficient multiplied by peak height that is passed onto peak_width. 
    Note that it is passed here as a multiple of peak height, but is absolute in peak_width.
    min_region_area is the minimal acceptable area of a region.

    Returns:
    regions defined as a list of regions of format [((lower_left_x, lower_right_x), (lower_left_y, upper_left_y))], with x being the axis of sequence 1 and y being the axis of sequence 2.
    counts, carried forwards from the first of 3 get_column_regions for visualization purposes.

    Parameters:
    ------------------
    array : np.array
        The array to analyze. This should be a 2D numpy array that represents
        an intermap.

    criteria : str
        The criteria to use for the analysis. Can be 'less' or 
        'greater'. Default is 'less'.

    criteria_threshold : float
        The threshold to use for the criteria, i.e. the value
        upon which masking is done to define regions. Default is -0.2.

    baseline : float
        The baseline to use for the analysis. Default is 0.3.

    min_region_area : int
        The minimum region area to use for the analysis. Area here is
        defined in terms of x * y in the array. Default is 500 (e.g. 20x25).

    penalize_opposite : bool
        Whether to penalize opposite interactions. Default is False.

    offset_threshold : float
        The threshold to use for penalizing opposite interactions. 
        Default is 0.2 to mirror default criteria_threshold.

    Returns:
    ------------------
    regions : list
        A list of regions that represent the interacting regions. Each region
        is a tuple of the form ((lower_left_x, lower_right_x), (lower_left_y, upper_left_y))
    '''

    # List containing regions fitting given parameters
    regions = []  

    bounds, counts = get_column_regions(array, 
                                        criteria=criteria, 
                                        criteria_threshold=criteria_threshold, 
                                        baseline=baseline, 
                                        min_region_size=min_region_size, 
                                        penalize_opposite=penalize_opposite,
                                        offset_threshold=offset_threshold)
                                          

    # For each columnar region (or 'peak')
    for peak in bounds:  

        # submap is that region, sliced out and turned on its side
        submap = array[:, peak[1]:peak[2]]  

        # Get columnar regions of the submap, which are actually row slices of the original intermap/array
        submap_bounds = get_column_regions(submap.T, 
                                            criteria=criteria,  
                                            criteria_threshold=criteria_threshold, 
                                            baseline=baseline, 
                                            min_region_size=min_region_size,
                                            penalize_opposite=penalize_opposite,
                                            offset_threshold=offset_threshold)[0]  

        # For each columnar region of submap
        for submap_peak in submap_bounds:  

            # Turn it on its side again (now back in our original orientation) and do the same
            subsubmap = array[submap_peak[1]:submap_peak[2], peak[1]:peak[2]]  

            # Find the peaks/regions in subsubmap
            subsubmap_bounds = get_column_regions(subsubmap, 
                                                  criteria=criteria,  
                                                  criteria_threshold=criteria_threshold, 
                                                  baseline=baseline, 
                                                  min_region_size=min_region_size,
                                                  penalize_opposite=penalize_opposite,
                                                  offset_threshold=offset_threshold)[0]  

            # For each final region in a region in a region
            for subsubmap_peak in subsubmap_bounds:  
                
                # Define the index boundaries within the original array/intermap
                low_left_x = peak[1] + subsubmap_peak[1]

                low_left_y = submap_peak[1]
                
                #peak[0]
                width = subsubmap_peak[0] 
                
                height = submap_peak[0]
                
                area = width * height

                # 'region' is the actual slice out of the array (and is not currently used, but could be returned if desired); 'region_bounds' defines it                
                region = array[submap_peak[1]:submap_peak[2], subsubmap_peak[1]:subsubmap_peak[2]]
                
                region_bounds = ((low_left_x, low_left_x+width), (low_left_y, low_left_y+height))

                # If region area is big enough, add to regions
                if area >= min_region_area and width >= min_region_size and height >= min_region_size:
                    regions.append(region_bounds)

    return regions, counts


#....................................................................................
#
#         
def rectangles_overlap(R1, R2):
    '''
    Given two regions of the form ((lower_left_x, lower_right_x), (lower_left_y, upper_left_y)), 
    do they overlap? Returns a boolean.

    Parameters:
    ------------------
    R1 : tuple
        The first region of interest; a tuple of the form 
        ((lower_left_x, lower_right_x), (lower_left_y, upper_left_y)).

    R2 : tuple
        The second region of interest; a tuple of the form 
        ((lower_left_x, lower_right_x), (lower_left_y, upper_left_y)).

    Returns:    
    ------------------
    bool
        True if the rectangles overlap, False otherwise
    '''
    # Unpack the rectangle coordinates
    # left = left hand side of rectangle (min x)
    # right = right hand side of rectangle  (max x)
    # bottom = bottom of rectangle (min y)
    # top = top of rectangle (max y)
    (left1, right1), (bottom1, top1) = R1
    (left2, right2), (bottom2, top2) = R2

    # Check if one rectangle is completely to the left or right of the other
    if right1 < left2 or right2 < left1:
        return False

    # Check if one rectangle is completely above or below the other
    if top1 < bottom2 or top2 < bottom1:
        return False

    # If neither of the above conditions is true, the rectangles overlap
    return True


#....................................................................................
#
#         
def rectangle_area(rect):
    '''
    Given a region of the form ((lower_left_x, lower_right_x), (lower_left_y, upper_left_y)), 
    calculate the area of the rectangle.

    Parameters:
    rect is the region of interest.
    
    '''
    # Unpack the rectangle coordinates
    (left, right), (bottom, top) = rect
    
    # Calculate the width and height
    width = right - left
    height = top - bottom
    
    # Return the area
    return width * height


#....................................................................................
#
#         
def get_bidirectional_interaction_regions(array, 
                                          criteria="less", 
                                          criteria_threshold=-0.2, 
                                          baseline=0.3, 
                                          min_region_area=500,
                                          min_region_size=20,
                                          penalize_opposite=False,
                                          offset_threshold=0.2):                                          

    '''
    Running get_interaction_regions will have a heavy bias against 
    'horizontal' rectangles. We can mitigate this by running it 'vertically' 
    AND 'horizontally' on the same array,and then filtering for overlap to capture everything.

    Parameters:
    ------------------
    array : np.array
        The array to analyze. This should be a 2D numpy array that represents
        an intermap

    criteria : str
        The criteria to use for the analysis. Can be 'less' or 
        'greater'. Default is 'less'.

    criteria_threshold : float
        The threshold to use for the criteria, i.e. the value
        upon which masking is done to define regions. Default is 0.0

    baseline : float
        The baseline to use for the analysis. Default is 0.3.

    min_region_area : int
        The minimum region area to use for the analysis. Area here is
        defined in terms of x * y in the array. Default is 500 (e.g. 20x25).

    min_region_size : int
        The minimum region size (in amino acids) to use for the analysis. 
        Default is 20.

    penalize_opposite : bool
        Whether to penalize opposite interactions. Default is False.

    offset_threshold : float
        The threshold to use for penalizing opposite interactions. 
        Default is 0.2 to mirror default criteria_threshold.

    Returns:
    ------------------
    regions : list
        A list of regions that represent the interacting regions. Each region
        is a tuple of the form ((lower_left_x, lower_right_x), (lower_left_y, upper_left_y))

    '''

    # sanity check criteria
    if criteria not in ["less", "greater"]:
        raise ValueError(f"criteria must be 'less' or 'greater', not {criteria}")
    
    array_T = array.T

    # get regions in both directions
    regions_1,  counts_1 = get_interaction_regions(array,   criteria=criteria, criteria_threshold=criteria_threshold, baseline=baseline, min_region_area=min_region_area, min_region_size=min_region_size, penalize_opposite=penalize_opposite, offset_threshold=offset_threshold)
    regions_2_, counts_2 = get_interaction_regions(array_T, criteria=criteria, criteria_threshold=criteria_threshold, baseline=baseline, min_region_area=min_region_area, min_region_size=min_region_size, penalize_opposite=penalize_opposite, offset_threshold=offset_threshold)
    regions_2 = [(j,i) for (i,j) in regions_2_]
        
    # combine get all the regions from both lists
    regions = []
    for r in regions_1:
        regions.append(r)
    for r in regions_2:
        regions.append(r)
    
    # algorithm to merge overlapping rectangles
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                if rectangles_overlap(regions[i], regions[j]):
                    merged = True
                    regions[i] =  (((min(regions[i][0][0], regions[j][0][0]), max(regions[i][0][1], regions[j][0][1])), 
                                    (min(regions[i][1][0], regions[j][1][0]), max(regions[i][1][1], regions[j][1][1]))))
                    regions.pop(j)
                    break
            if merged:
                break            

    return regions
    
#....................................................................................
#
#         
def plot_regions_on_intermap(array, regions_lists, colors, filename=None):
    '''
    Convenient function that will plot rectangles onto an intermap.

    Parameters:
    ------------------
    array : np.array
        The array to plot. This should be a 2D numpy array that represents
        an intermap.

    regions_lists : list   
        A list of lists of regions. Each list in the list should contain 
        a set of regions that are to be plotted with the same color.

    colors : list   
        A list of colors to use for each set of regions in regions_lists.

    filename : str
        The filename to save the plot to. If None, the plot will be shown
        instead of saved.

    Returns:
    ------------------
    None
        Nothing is returned, but a plot is shown.

    '''
    if len(regions_lists) != len(colors):
        return
    fig, ax1 = plt.subplots(figsize=(12, 6))
    cax = ax1.imshow(array, cmap="PRGn", vmin=-2.5, vmax=2.5)
    fig.colorbar(cax, ax=ax1, label='Repulsion/Interaction Strength')
    for i in range(len(regions_lists)):
        for region in regions_lists[i]:
            rect = patches.Rectangle((region[0][0], region[1][0]), 
                                    region[0][1]-region[0][0], 
                                    region[1][1]-region[1][0],
                                    linewidth=2, 
                                    edgecolor=colors[i], 
                                    facecolor='none',
                                    alpha=1)

            ax1.add_patch(rect)
    ax1.set_xlabel("Sequence 1")
    ax1.set_ylabel("Sequence 2")
    ax1.invert_yaxis()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
