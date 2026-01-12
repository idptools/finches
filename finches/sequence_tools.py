"""
Module to store functions needed to manipulate sequences


NOTE - Some of the functions below also exist in housetools but are 
       hardcoded in here to eliminate dependencies.

By : Garrett M. Ginell & Alex S. Holehouse 
2023-3-13
"""

def calculate_NCPR(s):
    """
    Simple function which calculates the net charge per residue of a
    protein sequence.

    Parameters
    --------------
    s : str
        Input amino acid sequence

    Returns
    ---------------
    float
        Returns the net charge per residue of the input sequence

    """

    # define charges
    charges = {'R':1, 'K':1, 'E':-1, 'D':-1}

    # set up counter
    total_charge = 0

    # iterate sequence and add up charges
    for r in s:
        if r not in charges:
            pass
        else:
            total_charge += charges[r]
            
    return total_charge / len(s)


def calculate_FCR(s):
    """
    Simple function which calculates the fraction of charged residues

    Parameters
    --------------
    s : str
        Input amino acid sequence

    Returns
    ---------------
    float
        Returns the fraction of charged residues in the input sequence

    """

        # define charges
    charges = {'R':1, 'K':1, 'E':1, 'D':1}

    # set up counter
    total_charge = 0

    # iterate sequence and add up charges
    for r in s:
        if r not in charges:
            pass
        else:
            total_charge += charges[r]
            
    return total_charge / len(s)


def calculate_FCR_and_NCPR(s):
    """
    Simple function which calculates the fraction of charged residues (FCR)
    and net charge per residue (NCPR) of a protein sequence. 

    Parameters
    --------------
    s : str
        Input amino acid sequence

    Returns
    ---------------
    list
        Returns a list of the FCR and NCPR of the input sequence


    """

    # define charges
    pos = set(['R','K'])
    neg = set(['E','D'])

    # set up counter
    total_pos = 0
    total_neg = 0


    # iterate sequence and add up charges
    for r in s:
        if r in pos:
            total_pos = total_pos + 1
        elif r in neg:
            total_neg = total_neg + 1
        else:
            pass
        
    return [(total_pos + total_neg)/len(s), (total_pos - total_neg)/len(s)]


## ------------------------------------------------------------------ 
##
def mask_sequence(sequence, target_residues):
    """
    Convert an amino acid sequence to a binary mask based on target residues.

    Each position in the output is 1 if that residue is in `target_residues`,
    otherwise 0.

    Parameters
    ----------
    sequence : str
        Input amino acid sequence (e.g., "ACDEFGK")

    target_residues : list or set
        Residues to mark as 1 in the mask (e.g., ['K', 'R'] for basic residues)
    
    Returns
    -------
    list of int
        Binary mask where 1 = residue in target_residues, 0 = not in target

    Examples
    --------
    >>> mask_sequence("ACDEFGK", ['A', 'G'])
    [1, 0, 0, 0, 0, 1, 0]
    
    >>> mask_sequence("KKEKK", ['K', 'R'])  # Mark basic residues
    [1, 1, 0, 1, 1]

    """
    # Convert to set for O(1) lookup
    target_set = set(target_residues)
    
    # Build mask: 1 if residue is a target, 0 otherwise
    mask = [1 if residue in target_set else 0 for residue in sequence]
    
    return mask

## ---------------------------------------------------------------------------
##
def get_neighbors_window_of3(position, sequence):
    """
    Extract a 3-residue window centered on the given position.

    Returns the residue at `position` along with its immediate N-terminal 
    and C-terminal neighbors (i.e., positions -1 and +1).

    At sequence boundaries, the window is truncated:
    - At position 0: returns only positions 0 and 1 (2 residues)
    - At last position: returns only the last 2 residues

    Parameters
    ----------
    position : int
        Index of the central residue (0-based)

    sequence : str
        The amino acid sequence

    Returns
    -------
    str
        Substring of 1-3 residues centered on the given position

    Examples
    --------
    >>> get_neighbors_window_of3(5, "ACDEFGHIK")
    'FGH'  # positions 4, 5, 6
    
    >>> get_neighbors_window_of3(0, "ACDEFGHIK")
    'AC'  # positions 0, 1 (no N-terminal neighbor)
    
    >>> get_neighbors_window_of3(8, "ACDEFGHIK")
    'IK'  # positions 7, 8 (no C-terminal neighbor)

    """
    # Calculate window boundaries
    # Start: one residue before, but not before position 0
    window_start = max(0, position - 1)
    
    # End: one residue after (exclusive), but not past sequence end
    window_end = min(len(sequence), position + 2)
    
    return sequence[window_start:window_end]



##------------------------------------------------------------------ 
#
def extract_fragments(mask, max_gap=1):
    """
    Extract contiguous fragments from a binary mask, grouping 1s that are 
    separated by at most `max_gap` zeros.

    A "fragment" is a run of 1s (possibly with small gaps of 0s within).
    Gaps larger than `max_gap` zeros break the sequence into separate fragments.

    Parameters
    ----------
    mask : list of int
        Binary mask where 1 = hit, 0 = non-hit
        Example: [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]

    max_gap : int, optional
        Maximum consecutive 0s allowed within a fragment. Default is 1.
        - max_gap=1: A single 0 between 1s keeps them in the same fragment
        - max_gap=0: Any 0 breaks the fragment (only consecutive 1s stay together)

    Returns
    -------
    list of str
        Each string is a fragment showing the pattern of 1s and 0s.
        Leading/trailing 0s are stripped from each fragment.

    Examples
    --------
    >>> extract_fragments([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], max_gap=1)
    ['111011', '1', '11', '101101']
    
    Breaking down the example above (max_gap=1, so "00" splits):
      Mask string: "001110110001001100101101"
                    ^^------^^-^--^^--------
      Splits at:    00      00 0  00  (but single 0s are allowed within fragments)
    
    - '111011': "11101" with the single 0 kept inside
    - '1': the isolated 1 at position 11
    - '11': positions 14-15
    - '101101': positions 18-23, single 0s allowed within

    >>> extract_fragments([1, 1, 0, 0, 1, 1], max_gap=1)
    ['11', '11']  # Gap of 2 zeros splits into two fragments

    >>> extract_fragments([1, 1, 0, 0, 1, 1], max_gap=2)
    ['110011']  # Gap of 2 zeros is now allowed, stays as one fragment

    """
    # Convert mask to string for easy splitting
    # e.g., [1, 0, 0, 0, 1] -> "10001"
    mask_string = ''.join(str(bit) for bit in mask)
    
    # Build the delimiter: (max_gap + 1) consecutive zeros
    # This is the pattern that breaks fragments apart
    # e.g., max_gap=1 means "00" splits, so delimiter = "00"
    delimiter = '0' * (max_gap + 1)
    
    # Split on the delimiter to get candidate fragments
    # e.g., "1110110001" with delimiter "00" -> ["111011", "01"]
    raw_fragments = mask_string.split(delimiter)
    
    # Clean up each fragment:
    # - Strip leading/trailing zeros (we only care about the 1s pattern)
    # - Filter out empty strings
    fragments = []
    for fragment in raw_fragments:
        cleaned = fragment.strip('0')
        if cleaned:  # Skip empty fragments
            fragments.append(cleaned)
    
    return fragments

##------------------------------------------------------------------ 
#
def count_nearby_hits(mask, max_gap=1, window_size=4):
    """
    Count nearby "hits" (1s) for each hit position in a binary mask.

    For each position with a 1, counts how many 1s are within a local window,
    including itself. Positions with 0 remain 0 in the output.

    The counting respects "clusters" - groups of 1s separated by at most 
    `max_gap` zeros are considered together, while larger gaps break the
    counting window.

    Parameters
    ----------
    mask : list of int
        Binary mask where 1 = hit position, 0 = non-hit position

    max_gap : int, optional
        Maximum number of consecutive 0s allowed within a cluster.
        Larger gaps split the sequence into separate clusters. Default is 1.        

    window_size : int, optional  
        Maximum distance (in either direction) to look for neighbors.
        Default is 4.
        (Old parameter name: max_distance)

    Returns
    -------
    list of int
        Same length as input mask. Each 0 stays 0, each 1 is replaced
        with the count of 1s in its local window (including itself).

    Examples
    --------
    >>> count_nearby_hits([0, 0, 1, 1, 1, 0, 0])
    [0, 0, 3, 3, 3, 0, 0]  # Each 1 sees all three 1s in cluster
    
    >>> count_nearby_hits([1, 0, 0, 0, 1])  # gap > max_gap=1, so separate clusters
    [1, 0, 0, 0, 1]  # Each 1 only sees itself

    """
    # Handle deprecated parameter names for backward compatibility
    if not mask or sum(mask) == 0:
        return mask.copy() if isinstance(mask, list) else list(mask)

    # Step 1: Find clusters of hits (groups of 1s with at most max_gap 0s between)
    clusters = extract_fragments(mask, max_gap=max_gap)
    
    # Step 2: For each hit position, determine which cluster it belongs to
    #         and count hits in its local window within that cluster
    neighbor_counts = []
    cluster_idx = 0
    position_in_cluster = 0
    
    for value in mask:
        if value == 0:
            # Non-hit positions stay 0
            neighbor_counts.append(0)
        else:
            # This is a hit - count neighbors in its cluster window
            cluster = clusters[cluster_idx]
            cluster_len = len(cluster)
            
            # Define window bounds within this cluster
            # (centered on current position, extending up to window_size in each direction)
            window_start = max(0, position_in_cluster - window_size)
            window_end = min(cluster_len, position_in_cluster + window_size + 1)
            
            # Count 1s in this window
            window = cluster[window_start:window_end]
            hit_count = sum(1 for char in window if char == '1')
            neighbor_counts.append(hit_count)
            
            # Move to next position within cluster
            position_in_cluster += 1
            
            # Check if we need to advance to next cluster
            # (we've consumed all 1s in current cluster)
            ones_in_cluster = cluster.count('1')
            if position_in_cluster >= ones_in_cluster:
                cluster_idx += 1
                position_in_cluster = 0
    
    return neighbor_counts


##------------------------------------------------------------------ 
#
def show_sequence_HTML(seq, blocksize=10, newline=50, fontsize=14, 
                        font_family='Courier', colors={},header=None,
                        bold_positions=[],bold_residues=[],opaque_positions=[],
                        return_raw_string=True):
    """
    FUNCTION FROM - sparrow.Protein.show_sequence 


    Function that generates an HTML colored string that either renders in the browser or returns the 
    html string. Contains various customizable components.

    Parameters
    -------------

    blocksize : int
        Defines how big blocks of residues are. Blocks are equal to blocksize or the newline parameter, whicever is smaller. 
        Default=10. If set to -1 uses length of the sequence.

    newline : int
        Defines how many residues are shown before a newline is printed. Default is 50. If set to -1 uses the length of
        the sequence.

    fontsize : int
        Fontsize used. Default is 14

    font_family : str
        Which font family (from HTML fonts) is used. Using a non-monospace font makes no sense as columns will be 
        unaligned. Default is Courier. 

    colors : dict
        Dictionary that allows overiding of default color scheme. Should be of format key-value as 'residue'-'color' where 
        residue is a residue in the string and color is a valid HTML color (which can be a Hexcode, standard HTML color name). 
        Note that this also lets you define colors for non-standard amino acids should these be useful. Default is an empty 
        dictionary. Note also that the standard amino acid colorings are defined at sparrow.data.amino_acids.AA_COLOR
        

    header : str
        If provided, this is a string that provides a FASTA-style header (with a leading carrett included). Default None.

    bold_positions : list
        List of positions (indexing from 1 onwards) which will be bolded. Useful for highlighting specific regions. Note that this
        defines individual residues so (for example) to bold residues 10 to 15 would require bold_positions=[10,11,12,13,14,15]. 
        Default is an empty list.

    bold_residues : list
        List of residue types that can be bolded. Useful for highlighting specific residue groups.  Default is an empty list.
    
    opaque_positions : list
        List of positions (indexing from 1 onwards) which will be grey and slighlty opaque. Useful for highlighting specific regions. 
        Note that this defines individual residues so (for example) to bold residues 10 to 15 would require 
        bold_positions=[10,11,12,13,14,15]. Default is an empty list.

    return_raw_string : bool
        If set to true, the function returns the actual raw HTML string, as opposed to an in-notebook rendering. 
        Default is TRUE as function is named

    Returns
    ----------
    None or str
        If return_raw_string is set to true then an HTML-compatible string is returned.
    """


    from IPython.display import display # dependency for showing sequence
    from IPython.display import HTML  # dependency for showing sequence


    if blocksize > newline:
        newline = blocksize

    if blocksize == -1:
        blocksize = len(seq)
        newline = len(seq)


    if blocksize < 1:
        raise 


    colorString = '<p style="font-family:%s;font-size: %ipx">'%(font_family, fontsize)

    if header:
        colorString = colorString + "><b>%s</b><br>"%(str(header))
        

    count = -1
    for residue in seq:

        count = count + 1

        if count > 0:
            if count % newline == 0:
                colorString = colorString + "<br>"
            
            elif count % blocksize == 0:
                colorString = colorString + " "


        if residue not in AA_COLOR and residue not in colors:
            print('Warning: found invalid amino acid (%s and position %i'%(residue, count+1))
            colorString = colorString + '<span style="color:%s"><b>%s</b></span>' % ('black', residue)
        else:

            # override with user-suppplied pallete if present
            if residue in colors:
                c = colors[residue]

            # else fall back on the standard pallete 
            else:
                c = AA_COLOR[residue]

             # check if residue should be light grey and opaque
            # This overrides other coloring 
            if count+1 in opaque_positions:
                 c = '#a9a9a9'

            # if the residue type OR residue position is to be bolded...
            if residue in bold_residues or (count+1) in bold_positions:
                colorString = colorString + '<span style="color:%s"><b>%s</b></span>' % (c, residue)
            else:
                colorString = colorString + '<span style="color:%s">%s</span>' % (c, residue)

    colorString = colorString +"</p>"
            
    if return_raw_string:
        return colorString
    else:
        display(HTML(colorString))


# annotation of amino acid coloring.
AA_COLOR = {'Y':'#ff9d00','W':'#ff9d00','F':'#ff9d00','A':'#171616','L':'#171616','M':'#171616',
            'I':'#171616','V':'#171616','Q':'#04700d','N':'#04700d','S':'#04700d','T':'#04700d',
            'H':'#04700d','G':'#04700d','E':'#ff0d0d','D':'#ff0d0d','R':'#2900f5','K':'#2900f5',
            'C':'#ffe70d','P':'#cf30b7'}
