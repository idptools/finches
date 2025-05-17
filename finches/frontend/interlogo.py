from IPython.display import HTML
from IPython.display import display
import numpy as np

from typing import List, Dict, Union, Optional # For type hinting

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












# Helper function (remains the same as your previous version)
def _generate_char_stack_html(
    position_data: Dict[str, float],
    color_map: Dict[str, str],
    height_per_unit_size: float,
    char_width_px: float,
    sort_largest_top: bool,
    min_char_height_px: float,
    min_font_size_px: float,
    font_family: str
) -> str:
    """
    Generate HTML for a vertical stack of characters at a single position in the logo.

    Parameters
    ----------
    position_data : Dict[str, float]
        Dictionary mapping characters to their sizes/weights
    color_map : Dict[str, str]
        Dictionary mapping characters to their display colors
    height_per_unit_size : float
        Scaling factor for converting size values to pixel heights
    char_width_px : float
        Width of the character column in pixels
    sort_largest_top : bool
        If True, largest characters appear at top; if False, at bottom
    min_char_height_px : float
        Minimum height for any character div
    min_font_size_px : float
        Minimum font size for displayed characters
    font_family : str
        CSS font-family specification

    Returns
    -------
    str
        HTML string for the character stack
    """
    # Return empty div if no position data
    if not position_data:
        return f'<div style="width: {char_width_px:.2f}px; height: {min_char_height_px:.2f}px; box-sizing: border-box;"></div>'

    # Small value to filter out near-zero sizes
    epsilon = 1e-6
    
    # Sort characters by size, filtering out invalid or near-zero values
    sorted_chars_with_size = sorted(
        [(char, size) for char, size in position_data.items() if isinstance(size, (int, float)) and size > epsilon],
        key=lambda item: item[1],
        reverse=sort_largest_top
    )

    # Return empty div if no valid characters after filtering
    if not sorted_chars_with_size:
        return f'<div style="width: {char_width_px:.2f}px; height: {min_char_height_px:.2f}px; box-sizing: border-box;"></div>'

    # Generate HTML for each character in the stack
    stack_html_parts = []
    for char, size in sorted_chars_with_size:
        # Calculate character height and enforce minimum
        char_height_px = max(min_char_height_px, float(size) * float(height_per_unit_size))
        char_color = color_map.get(char, 'black')

        # Calculate appropriate font size based on character height
        current_font_size = max(float(min_font_size_px), char_height_px * 0.8)
        if char_height_px < current_font_size * 1.1 and char_height_px > min_char_height_px:
            current_font_size = char_height_px * 0.9
        current_font_size = max(float(min_font_size_px), current_font_size)
        
        # Replace character with space if font size is too small
        char_display = char
        if current_font_size < 1:
            char_display = "&nbsp;"

        # Generate HTML for individual character
        char_div_html = f"""
        <div style="
            height: {char_height_px:.2f}px;
            width: 100%;
            color: {char_color};
            font-family: {font_family};
            font-weight: bold;
            font-size: {current_font_size:.2f}px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        ">{char_display}</div>"""
        stack_html_parts.append(char_div_html)
    
    # Wrap all character divs in container
    stack_wrapper_div_html = f'<div style="width: {char_width_px:.2f}px; margin: 0 auto; line-height: 0; font-size: 0;">' + "".join(stack_html_parts) + '</div>'
    return stack_wrapper_div_html

def display_protein_qlogo_visualization(
    top_logo_data: List[Dict[str, float]],
    middle_sequence: List[str],
    bottom_logo_data: List[Dict[str, float]],
    color_map: Dict[str, str] = AA_COLOR,
    integer_sequence: Optional[List[Union[int, str]]] = None,
    normalize_logo_heights_globally: bool = True, # New: Normalize heights globally for top/bottom logos
    height_per_unit_size: float = 50.0, # If normalizing, this is max height for the globally largest value
    position_width_px: float = 25.0,
    middle_row_char_height_px: float = 30.0,
    middle_row_font_size_px: float = 20.0,
    middle_row_background_color: str = "#f9f9f9",
    logo_min_char_height_px: float = 2.0,
    logo_min_font_size_px: float = 8.0,
    logo_font_family: str = "'Consolas', 'Menlo', 'Courier New', monospace'",
    integer_row_height_px: float = 25.0,
    integer_row_font_size_px: float = 12.0,
    integer_row_font_color: str = "#333333",
    integer_row_background_color: str = "#ffffff"
):
    """
    Display a multi-row protein sequence logo visualization.

    Parameters
    ----------
    top_logo_data : List[Dict[str, float]]
        List of dictionaries containing character frequencies for top logo
    middle_sequence : List[str]
        The sequence to display in the middle row
    bottom_logo_data : List[Dict[str, float]]
        List of dictionaries containing character frequencies for bottom logo
    color_map : Dict[str, str]
        Mapping of characters to their display colors
    integer_sequence : Optional[List[Union[int, str]]]
        Optional sequence of integers/strings to display in bottom row
    normalize_logo_heights_globally : bool
        If True, normalizes character heights across entire logo
    height_per_unit_size : float
        Base height scaling factor for logo characters
    position_width_px : float
        Width of each position column in pixels
    middle_row_char_height_px : float
        Height of middle row characters in pixels
    middle_row_font_size_px : float
        Font size for middle row characters
    middle_row_background_color : str
        Background color for middle row
    logo_min_char_height_px : float
        Minimum height for logo characters
    logo_min_font_size_px : float
        Minimum font size for logo characters
    logo_font_family : str
        Font family for all text
    integer_row_height_px : float
        Height of optional integer row
    integer_row_font_size_px : float
        Font size for integer row
    integer_row_font_color : str
        Text color for integer row
    integer_row_background_color : str
        Background color for integer row
    """
    if not middle_sequence:
        print("Middle sequence cannot be empty.")
        return

    num_positions = len(middle_sequence)
    
    # --- Prepare Top Logo Data (Normalization if requested) ---
    processed_top_logo_data = top_logo_data
    if normalize_logo_heights_globally and top_logo_data:
        global_max_val = 0.0
        for pos_dict in top_logo_data:
            if isinstance(pos_dict, dict) and pos_dict:
                for size_val in pos_dict.values():
                    if isinstance(size_val, (int, float)) and size_val > 0 and size_val > global_max_val:
                        global_max_val = size_val
        
        if global_max_val > 1e-6:
            normalized_list = []
            for pos_dict in top_logo_data:
                if not isinstance(pos_dict, dict) or not pos_dict:
                    normalized_list.append({})
                    continue
                current_normalized_dict = {}
                for char, size_val in pos_dict.items():
                    if isinstance(size_val, (int, float)) and size_val > 0:
                        current_normalized_dict[char] = size_val / global_max_val
                    else: 
                        current_normalized_dict[char] = size_val # carry over non-positive or non-numeric
                normalized_list.append(current_normalized_dict)
            processed_top_logo_data = normalized_list

    # --- Prepare Bottom Logo Data (Normalization if requested) ---
    processed_bottom_logo_data = bottom_logo_data
    if normalize_logo_heights_globally and bottom_logo_data:
        global_max_val = 0.0
        for pos_dict in bottom_logo_data:
            if isinstance(pos_dict, dict) and pos_dict:
                for size_val in pos_dict.values():
                    if isinstance(size_val, (int, float)) and size_val > 0 and size_val > global_max_val:
                        global_max_val = size_val
        
        if global_max_val > 1e-6:
            normalized_list = []
            for pos_dict in bottom_logo_data:
                if not isinstance(pos_dict, dict) or not pos_dict:
                    normalized_list.append({})
                    continue
                current_normalized_dict = {}
                for char, size_val in pos_dict.items():
                    if isinstance(size_val, (int, float)) and size_val > 0:
                        current_normalized_dict[char] = size_val / global_max_val
                    else:
                        current_normalized_dict[char] = size_val
                normalized_list.append(current_normalized_dict)
            processed_bottom_logo_data = normalized_list

    html_rows_content = ["", "", ""]
    if integer_sequence is not None:
        html_rows_content.append("")

    td_logo_style_common = f"padding: 0; border-left: 1px solid #f0f0f0; border-right: 1px solid #f0f0f0; background-color: transparent;"
    td_top_logo_style = f"{td_logo_style_common} vertical-align: bottom;"
    td_bottom_logo_style = f"{td_logo_style_common} vertical-align: top;"
    td_data_row_style = f"padding: 0; vertical-align: middle; border-left: 1px solid #ddd; border-right: 1px solid #ddd; background-color: transparent;"

    # --- Generate Top Logo Row ---
    for i in range(num_positions):
        pos_data = processed_top_logo_data[i] if i < len(processed_top_logo_data) else {}
        stack_html = _generate_char_stack_html(
            pos_data, color_map, height_per_unit_size, position_width_px, True,
            logo_min_char_height_px, logo_min_font_size_px, logo_font_family
        )
        html_rows_content[0] += f"<td style='{td_top_logo_style}'>{stack_html}</td>"

    # --- Generate Middle Sequence Row ---
    for i in range(num_positions):
        char = middle_sequence[i] if i < len(middle_sequence) else "&nbsp;"
        char_color = color_map.get(char, 'black')
        middle_char_div_html = f"""
        <div style="
            width: {position_width_px:.2f}px; height: {middle_row_char_height_px:.2f}px; 
            line-height: {middle_row_char_height_px:.2f}px; color: {char_color}; 
            background-color: {middle_row_background_color}; font-family: {logo_font_family};
            font-weight: bold; font-size: {middle_row_font_size_px:.2f}px;
            text-align: center; margin: auto; box-sizing: border-box;
            border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;
        ">{char}</div>"""
        html_rows_content[1] += f"<td style='{td_data_row_style}'>{middle_char_div_html}</td>"

    # --- Generate Bottom Logo Row ---
    for i in range(num_positions):
        pos_data = processed_bottom_logo_data[i] if i < len(processed_bottom_logo_data) else {}
        stack_html = _generate_char_stack_html(
            pos_data, color_map, height_per_unit_size, position_width_px, False,
            logo_min_char_height_px, logo_min_font_size_px, logo_font_family
        )
        html_rows_content[2] += f"<td style='{td_bottom_logo_style}'>{stack_html}</td>"

    # --- Generate Integer Sequence Row ---
    if integer_sequence is not None:
        for i in range(num_positions):
            value = integer_sequence[i] if i < len(integer_sequence) else "&nbsp;"
            integer_div_html = f"""
            <div style="
                width: {position_width_px:.2f}px; height: {integer_row_height_px:.2f}px; 
                line-height: {integer_row_height_px:.2f}px; color: {integer_row_font_color}; 
                background-color: {integer_row_background_color}; font-family: {logo_font_family};
                font-size: {integer_row_font_size_px:.2f}px; text-align: center; margin: auto;
                box-sizing: border-box; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;
            ">{value}</div>"""
            html_rows_content[3] += f"<td style='{td_data_row_style}'>{integer_div_html}</td>"

    html_output_str = "<table style='border-collapse: collapse; margin: auto; border: 1px solid #ccc;'>"
    html_output_str += f"<tr style='border-bottom: 1px dashed #ccc;'>{html_rows_content[0]}</tr>"
    html_output_str += f"<tr style='border-bottom: 1px dashed #ccc;'>{html_rows_content[1]}</tr>"
    
    if integer_sequence is not None:
        html_output_str += f"<tr style='border-bottom: 1px dashed #ccc;'>{html_rows_content[2]}</tr>"
        html_output_str += f"<tr>{html_rows_content[3]}</tr>"
    else:
        html_output_str += f"<tr>{html_rows_content[2]}</tr>"
    html_output_str += "</table>"
    display(HTML(html_output_str))




