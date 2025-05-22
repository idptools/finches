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
def get_neighbors_window_of3(i, sequence):
    """
    Function that takes in an index position and sequence and returns 
    the portion of the sequence based off the index (i) and the 
    1 neighboring residues before and after that index for a window size 
    of 3 residues.

    NOTE - if the index is at the begining or end of the sequence, the 
            returned string may not be a window size of 3.

    Parameters
    ----------
    i : int
        Set which position in the sequence to reference

    sequence : string 
        the sequence to reference

    Returns
    -------
    s2 : str
        Portion of sequence which includes and the residues N- 
        and C-terminal of the index position

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
    based on the relative posision for the hit residues to each other 

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
    
    # extract fragments per aliphatic residue (IE split mask by mask separator, iterate by aliphatic and get sum within fragment 
    # for centered window around aliphatic residue

    # frags is a series of sublists where each sublist is a fragment of the mask and contains aliphatics with a max of 
    # one 'gap' between congigous aliphatics
    frags = extract_fragments(mask, max_separation=max_separation)

    # this list has an aliphatic fragment for each aliphatic residue in the mask,
    # ie len(per_ali_frags) == sum(mask)
    per_ali_frags=[] 
    
    for frag in frags:

        # get fragment size
        l = len(frag)

        # if the fragment is equal to or smaller than the window size
        if l <= w_half:

            # for each residue in the fragment
            for r in frag:

                # if the residue is an aliphatic, add it to the list
                if r == '1':
                    per_ali_frags.append(frag)
        else: 
            # parse frags larger than window size
            l_mask = frag
            for i,r in enumerate(l_mask):

                # if the residue is an aliphatic...
                if r == '1':

                    # 
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
