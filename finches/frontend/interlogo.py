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
# Helper function (remains the same)
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
    if not position_data:
        return f'<div style="width: {char_width_px:.2f}px; height: {min_char_height_px:.2f}px; box-sizing: border-box;"></div>'

    epsilon = 1e-6
    # Ensure size is float for comparison and calculation
    sorted_chars_with_size = sorted(
        [(char, float(size)) for char, size in position_data.items() if isinstance(size, (int, float)) and float(size) > epsilon],
        key=lambda item: item[1],
        reverse=sort_largest_top
    )

    if not sorted_chars_with_size:
        return f'<div style="width: {char_width_px:.2f}px; height: {min_char_height_px:.2f}px; box-sizing: border-box;"></div>'

    stack_html_parts = []
    for char, size in sorted_chars_with_size:
        char_height_px = max(min_char_height_px, size * float(height_per_unit_size))
        char_color = color_map.get(char, 'black')

        current_font_size = max(float(min_font_size_px), char_height_px * 0.8)
        if char_height_px < current_font_size * 1.1 and char_height_px > min_char_height_px:
            current_font_size = char_height_px * 0.9
        current_font_size = max(float(min_font_size_px), current_font_size)
        
        char_display = char
        if current_font_size < 1:
            char_display = "&nbsp;"

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
    
    stack_wrapper_div_html = f'<div style="width: {char_width_px:.2f}px; margin: 0 auto; line-height: 0; font-size: 0;">' + "".join(stack_html_parts) + '</div>'
    return stack_wrapper_div_html

# Main function to display the multi-row protein logo visualization
def display_protein_interaction_logo_visualization(
    top_logo_data: List[Dict[str, float]],
    middle_sequence: List[str],
    bottom_logo_data: List[Dict[str, float]],
    integer_sequence: Optional[List[Union[int, str]]] = None,
    color_map: Dict[str, str] = AA_COLOR,
    normalize_logo_heights_globally: bool = True,
    height_per_unit_size: float = 50.0,
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
    Displays a 3-row or 4-row protein logo visualization.
    - If `middle_sequence` is longer than `top_logo_data` or `bottom_logo_data` by an 
      even number of positions, the respective logo will be centered.
    - `normalize_logo_heights_globally`: If True, normalizes character sizes in logo data.
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
                        current_normalized_dict[char] = float(size_val) / global_max_val
                    else: 
                        current_normalized_dict[char] = size_val
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
                        current_normalized_dict[char] = float(size_val) / global_max_val
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

    # --- Generate Top Logo Row (with centering) ---
    len_current_top_logo_list = len(processed_top_logo_data)
    padding_top = 0
    if num_positions > len_current_top_logo_list:
        diff_top = num_positions - len_current_top_logo_list
        if diff_top % 2 == 0:
            padding_top = diff_top // 2
        # else: logo will be left-aligned if difference is not even, or handle as error/warning

    for i in range(num_positions):
        pos_data_for_column = {} 
        logo_data_index = i - padding_top
        if 0 <= logo_data_index < len_current_top_logo_list:
             pos_data_for_column = processed_top_logo_data[logo_data_index]
        
        stack_html = _generate_char_stack_html(
            pos_data_for_column, color_map, height_per_unit_size, position_width_px, True,
            logo_min_char_height_px, logo_min_font_size_px, logo_font_family
        )
        html_rows_content[0] += f"<td style='{td_top_logo_style}'>{stack_html}</td>"

    # --- Generate Middle Sequence Row --- (No change in logic here)
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

    # --- Generate Bottom Logo Row (with centering) ---
    len_current_bottom_logo_list = len(processed_bottom_logo_data)
    padding_bottom = 0
    if num_positions > len_current_bottom_logo_list:
        diff_bottom = num_positions - len_current_bottom_logo_list
        if diff_bottom % 2 == 0:
            padding_bottom = diff_bottom // 2
        # else: logo will be left-aligned

    for i in range(num_positions):
        pos_data_for_column = {}
        logo_data_index = i - padding_bottom
        if 0 <= logo_data_index < len_current_bottom_logo_list:
             pos_data_for_column = processed_bottom_logo_data[logo_data_index]
        
        stack_html = _generate_char_stack_html(
            pos_data_for_column, color_map, height_per_unit_size, position_width_px, False,
            logo_min_char_height_px, logo_min_font_size_px, logo_font_family
        )
        html_rows_content[2] += f"<td style='{td_bottom_logo_style}'>{stack_html}</td>"

    # --- Generate Integer Sequence Row --- (No change in logic here for data access)
    if integer_sequence is not None:
        for i in range(num_positions): # Assumes integer_sequence is same length as middle_sequence
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

    # --- Combine into an HTML Table ---
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