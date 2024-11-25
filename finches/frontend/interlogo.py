from IPython.display import HTML
from IPython.display import display
import numpy as np

#####

AA_COLOR = {'Y':'#ff9d00',
            'W':'#ff9d00',
            'F':'#ff9d00',
            'A':'#171616',
            'L':'#171616',
            'M':'#171616',
            'I':'#171616',
            'V':'#171616',
            'Q':'#04700d',
            'N':'#04700d',
            'S':'#04700d',
            'T':'#04700d',
            'H':'#04700d',
            'G':'#04700d',
            'E':'#ff0d0d',
            'D':'#ff0d0d',
            'R':'#2900f5',
            'K':'#2900f5',
            'C':'#ffe70d',
            'P':'#cf30b7'}

AA_CHARGE_COLOR = {'Y':'#171616',
            'W':'#171616',
            'F':'#171616',
            'A':'#171616',
            'L':'#171616',
            'M':'#171616',
            'I':'#171616',
            'V':'#171616',
            'Q':'#171616',
            'N':'#171616',
            'S':'#171616',
            'T':'#171616',
            'H':'#171616',
            'G':'#171616',
            'E':'#ff0d0d',
            'D':'#ff0d0d',
            'R':'#2900f5',
            'K':'#2900f5',
            'C':'#171616',
            'P':'#171616'}


#####


def styled_text(letters : str, 
                colors : list[str], 
                sizes : list,
                return_text : bool = False,
                quantile_threshold : float = 0.70):
                
    """
    Generate styled text in HTML format.

    Parameters
    ----------
    letters : str
        The text to be styled.

    colors : list[str]
        A list of colors for each letter in the text.

    sizes : list
        A list of sizes for each letter in the text.

    return_text : bool
        If True, returns the HTML as a string. Otherwise, returns an HTML object.

    quantile_threshold : float  
        The quantile threshold for the size of the letters. Must be between 0 and 1.

    Returns
    -------
    HTML
        An HTML object that can be displayed in Jupyter notebooks.


    """
    if quantile_threshold <= 0 or quantile_threshold >= 1:
        raise ValueError("The quantile threshold must be between 0 and 1.")

    # Check if inputs are consistent
    if len(letters) != len(colors) or len(letters) != len(sizes):
        raise ValueError("The length of letters, colors, and sizes must be the same.")
    
    # Generate HTML string
    threshold = np.quantile(sizes, quantile_threshold)

    styled_html = ""
    idx=0
    for letter, color, size in zip(letters, colors, sizes):
        if size < threshold:
            color = '#cccccc'
        styled_html = styled_html + f"<span style='color:{color}; font-size:{int(size)}px'>{letter}</span>"
        idx = idx + 1
        if idx % 50 == 0:
            styled_html = styled_html + "<br>"
    
    if return_text:
        return styled_html
    else:
        return HTML(styled_html)


def check_sequence_validity(seq: str):
    '''Checks that the sequence is a valid amino acid sequence'''
    #check that the sequence is all valid amino acids
    valid_letters = list(AA_COLOR.keys())
    for aa in seq:
        if aa not in valid_letters:
            raise Exception(f"Invalid amino acid {aa} was passed.")


def chemical_context_seq_plot(seq : str, 
                              scale_vec : np.ndarray,
                              min_font_sz :int = 8,
                              max_font_sz : int = 30,
                              quantile_threshold : float = 0.70): 
    """
    Produces a string of the amino acid sequence with colors for their 
    chemical context with the size being based on the scale vector.
    the size is specified in em units.

    Parameters
    ----------

    seq : str
        The amino acid sequence to be displayed.

    scale_vec : np.ndarray  
        A vector of values that will be used to scale the font size of the amino acids.

    min_font_sz : int
        The minimum font size for the amino acids.  

    max_font_sz : int   
        The maximum font size for the amino acids.  

    quantile_threshold : float  
        The quantile threshold for the size of the letters. Must be between 0 and 1.


    Returns 
    ----------

    None        
    ---
    """

    # check that the sequence is all valid amino acids
    check_sequence_validity(seq)

    if quantile_threshold <= 0 or quantile_threshold >= 1:
        raise ValueError("The quantile threshold must be between 0 and 1.")

    min_val = np.min(scale_vec)
    max_val = np.max(scale_vec)

    # scale scale_vec to be between min_font_sz and max_font_sz
    sizes_vec = min_font_sz + (scale_vec - min_val) * (max_font_sz - min_font_sz) / (max_val - min_val)

    # create the matched hex values for each amino acid
    color_list = [AA_COLOR[aa] for aa in seq]
 
    html_obj = styled_text(seq, color_list, sizes_vec.tolist(), quantile_threshold=quantile_threshold)
    display(html_obj)


def color_aminoacids(seq : str, 
                    aminos_of_interest : str,
                    font_sz = 12):
    '''Produces the charge state of a sequence'''
    #check that the sequence is all valid amino acids
    check_sequence_validity(seq)

    #match the values and turn them into a list
    sizes_vec = [font_sz for k in range(len(seq))]
    #size_txt_list = [f"{sz}px" for sz in sizes_vec]

    #create the matched hex values for each amino acid
    color_list = [AA_COLOR[aa] if aa in aminos_of_interest else '#3f3f3f' for aa in seq]

    html_obj = styled_text(seq, color_list, sizes_vec)
    display(html_obj)


def plot_attractive_logo(s1,
                         s2, 
                         xf,
                         window_size=31,
                         quantile_threshold=0.7):
    """
    Plots the attractive logo for two sequences. 

    Parameters
    ----------
    s1 : str
        The first sequence.

    s2 : str
        The second sequence.

    xf : finches.frontend.frontend_base.FinchesFrontend object        
        The frontend object that will be used to calculate the 
        attractive vector.

    window_size : int
        The window size for the attractive vector. Default is 31.

    quantile_threshold : float
        The quantile threshold above which we color the residues
        in the logo.

    """            
    
    xa = xf.per_residue_attractive_vector(s1,s2, window_size=window_size)
    norm_attractive = abs(xa[1] - np.max(xa[1]))

    seq_start = xa[0][0]-1
    seq_end   = xa[0][-1] 

    chemical_context_seq_plot(s1[seq_start:seq_end], norm_attractive, max_font_sz=40, min_font_sz=3, quantile_threshold=quantile_threshold)


def plot_repulsive_logo(s1,
                         s2, 
                         xf,
                         window_size=31,
                         quantile_threshold=0.7):
    
    """
    Plots the repulsive logo for two sequences. 

    Parameters
    ----------
    s1 : str
        The first sequence.

    s2 : str
        The second sequence.

    xf : finches.frontend.frontend_base.FinchesFrontend object        
        The frontend object that will be used to calculate the 
        attractive vector.

    window_size : int
        The window size for the attractive vector. Default is 31.

    quantile_threshold : float
        The quantile threshold above which we color the residues
        in the logo.

    """
    
    xa = xf.per_residue_repulsive_vector(s1,s2, window_size=window_size)
    norm_repulsive = xa[1] + abs(np.min(xa[1]))

    seq_start = xa[0][0]-1
    seq_end   = xa[0][-1] 

    chemical_context_seq_plot(s1[seq_start:seq_end], norm_repulsive, max_font_sz=40, min_font_sz=3, quantile_threshold=quantile_threshold)

        