"""
Module to store functions needed to manipulate sequences


NOTE - Some of the functions below also exist in housetools but are 
       hardcoded in here to eliminate dependencies.

By : Garrett M. Ginell & Alex S. Holehouse 
2023-3-13
"""

## ------------------------------------------------------------------ 
##
def mask_sequence(sequence, target_residues):
    """
    FUNCTION FROM - housetools.sequence_tools.sequence_masking

    Function converts sequence to a binary mask (list where each
    position is either 1 or 0) based on the residues passed in
    target_residues.

    Parameters
    --------------
    sequence : str
        Input amino acid sequence

    target_residues : list 
        A list of residues which will be used to to mask the sequence
        into 1s and zeros
    
    Returns
    ---------------
    Mask : list
        List where every position is either a 0 or a 1, depending on if 
        the original residue was in the target_residue list or not.

    """

    value_track = []

    #iterate sequence and build track 
    for i in sequence:
        if str(i) in target_residues:
            value_track.append(1)
        else:
            value_track.append(0)      
            
    #check if track matches len of sequence, if not throw error 
    if len(value_track) != len(sequence):
        raise Exception('Masking ERROR - mask legnth does not match sequence length')
    
    return value_track

## ---------------------------------------------------------------------------
##
def _get_neighboors_3(i, sequence):
    """
    """
    if i == 0:
        s2= sequence[:i+2]
    elif i == len(sequence):
        s2=sequence[i-1:]
    else:
        s2= sequence[i-1:i+2]
    return s2

##------------------------------------------------------------------ 
#
def extract_fragments(mask, max_separation=1):
    """
    Converts a binary mask of 1s and 0s to fragments based on the max_separation
    (largest gap between two fragments).

    For example, if max_separation = 1 then:
    
       In:  [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
        
       Out: ['111011', '1', '11', '11', '101101']
    
    Parameters
    --------------
    mask : list 
       Binary mask of 0s and 1s

    max_separation : int 
        Define maximum number of 0s betweens 1s before a new fragment is identified 

    Returns
    ---------------
    list
        Returns list of strings where each element is a fragment extracted from the
        input mask

    """

    mask = [str(s) for s in mask]
    spliter= ''.join(['0'] * (max_separation+1))
    frag_list = ''.join(mask).split(spliter)
    
    return [f.strip('0') for f in frag_list if f ]

##------------------------------------------------------------------ 
#
def MASK_n_closest_nearest_neighbors(mask, max_separation=1, max_distance=4):
    """
    FUNCTION FROM - housetools.sequence_tools.sequence_masking

    Takes in mask and converts this residues to into none binary mask 
    based on the relitive posision for the hit residues to each other 

    Parameters
    --------------
    mask : list 
        Mask of 1s and 0s 

    max_separation : int 
        Define maximum number of 0s betweens 1s that is allowed when counting the
        nearest neighbors to the target residue (default = 1) 

    max_distance : int 
        define maximum linear distance that is summed when counting the
        nearest neighbors to the target residue ie the window size (default = 4) 

    Returns
    ---------------
    mask : list
        returns new vector mask where target residues are assigned the sum of there
        nearest neibors
    """
    w_half = max_distance    
    
    # extract fragments per aliphatic residue (IE split mask by mask seperator, iterate by aliphatic and get sum within fragment 
    # for centered window around aliphatic residue
    frags = extract_fragments(mask,max_separation=max_separation)
    per_ali_frags=[] 
    
    for frag in frags:
        l = len(frag)
        # handle residues where nearest neighbor fragment is less than the window size
        if l <= w_half:
            for r in frag:
                if r == '1':
                    per_ali_frags.append(frag)
        else: 
            # parse frags larger than window size
            l_mask = frag
            out_mask=[]
            for i,r in enumerate(l_mask):
                if r == '1':
                    if w_half >= i and i+w_half <= l:
                        per_ali_frags.append(l_mask[:i+w_half+1])
                    elif  i+w_half > l:
                        per_ali_frags.append(l_mask[i-w_half:])
                    elif w_half <= i <= l-w_half: 
                        per_ali_frags.append(l_mask[i-w_half:i+w_half+1])
                    else:
                        raise Exception('Parsing ERROR')      
    
    # get sum of neighbors in fragments
    nearest_neighbor_sums_mask = []
    ali_count=0
    for i in mask:
        if i == 1:
            nearest_neighbor_sums_mask.append(sum([int(r) for r in per_ali_frags[ali_count]]))
            ali_count+=1
        else:
            nearest_neighbor_sums_mask.append(i)
    
    return nearest_neighbor_sums_mask
