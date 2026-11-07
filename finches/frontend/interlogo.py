"""
InterLogos: sequence logos for FINCHES interaction profiles.

An InterLogo renders a sequence as coloured letters whose size follows a
per-residue score, in the style of a sequence logo, so that the residues driving
attraction (or repulsion) with a partner can be read directly off the sequence.
Letters are coloured by chemistry (aromatic orange, aliphatic black, polar green,
acidic red, basic blue, cysteine yellow, proline magenta), and letters whose score
falls below a quantile threshold are greyed out. The output is HTML, intended for
display in a Jupyter notebook.
"""

import numpy as np
from IPython.display import HTML
from IPython.display import display

# colour of each amino acid, by chemistry
AA_COLOR = {
    "Y": "#ff9d00",
    "W": "#ff9d00",
    "F": "#ff9d00",
    "A": "#171616",
    "L": "#171616",
    "M": "#171616",
    "I": "#171616",
    "V": "#171616",
    "Q": "#04700d",
    "N": "#04700d",
    "S": "#04700d",
    "T": "#04700d",
    "H": "#04700d",
    "G": "#04700d",
    "E": "#ff0d0d",
    "D": "#ff0d0d",
    "R": "#2900f5",
    "K": "#2900f5",
    "C": "#ffe70d",
    "P": "#cf30b7",
}


def styled_text(
    letters: str,
    colors: list[str],
    sizes: list,
    return_text: bool = False,
    quantile_threshold: float = 0.70,
):
    """
    Render letters as HTML spans with a colour and font size each.

    Letters whose size falls below the quantile_threshold quantile of all the
    sizes are drawn in light grey instead of their own colour, and a line break
    is inserted every 50 letters.

    Parameters
    ----------
    letters : str
        The text to be styled.

    colors : list[str]
        A colour (any CSS colour, e.g. a hex string) for each letter.

    sizes : list
        A font size in pixels for each letter.

    return_text : bool, optional
        If True, return the HTML as a string; otherwise return an IPython HTML
        object. Default is False.

    quantile_threshold : float, optional
        Quantile of the sizes below which letters are greyed out. Must be
        strictly between 0 and 1. Default is 0.70.

    Returns
    -------
    IPython.display.HTML or str
        The styled text, as an HTML object that displays in a Jupyter notebook,
        or as a string if return_text is True.

    Raises
    ------
    ValueError
        If quantile_threshold is not strictly between 0 and 1, or letters,
        colors and sizes differ in length.
    """
    if quantile_threshold <= 0 or quantile_threshold >= 1:
        raise ValueError("The quantile threshold must be between 0 and 1.")

    # Check if inputs are consistent
    if len(letters) != len(colors) or len(letters) != len(sizes):
        raise ValueError("The length of letters, colors, and sizes must be the same.")

    # Generate HTML string
    threshold = np.quantile(sizes, quantile_threshold)

    styled_html = ""
    for idx, (letter, color, size) in enumerate(zip(letters, colors, sizes), start=1):
        if size < threshold:
            color = "#cccccc"
        styled_html = (
            styled_html
            + f"<span style='color:{color}; font-size:{int(size)}px'>{letter}</span>"
        )
        if idx % 50 == 0:
            styled_html = styled_html + "<br>"

    if return_text:
        return styled_html
    else:
        return HTML(styled_html)


def check_sequence_validity(seq: str):
    """
    Check that a sequence contains only the twenty standard amino acids.

    Parameters
    ----------
    seq : str
        The amino acid sequence.

    Raises
    ------
    ValueError
        If the sequence contains any other character.
    """
    for aa in seq:
        if aa not in AA_COLOR:
            raise ValueError(f"Invalid amino acid {aa} was passed.")


def chemical_context_seq_plot(
    seq: str,
    scale_vec: np.ndarray,
    min_font_sz: int = 8,
    max_font_sz: int = 30,
    quantile_threshold: float = 0.70,
):
    """
    Display a sequence as an InterLogo, with letter size set by a per-residue score.

    Each letter is coloured by its chemistry (see AA_COLOR) and sized linearly
    between min_font_sz (the lowest score) and max_font_sz (the highest), in
    pixels. Letters whose size falls below the quantile_threshold quantile are
    greyed out. A flat score vector has nothing to rank, so every letter gets
    min_font_sz.

    Parameters
    ----------
    seq : str
        The amino acid sequence to be displayed.

    scale_vec : np.ndarray
        One score per residue of seq; larger scores give larger letters.

    min_font_sz : int, optional
        Font size in pixels for the lowest score. Default is 8.

    max_font_sz : int, optional
        Font size in pixels for the highest score. Default is 30.

    quantile_threshold : float, optional
        Quantile of the letter sizes below which letters are greyed out. Must be
        strictly between 0 and 1. Default is 0.70.

    Returns
    -------
    None
        The logo is displayed with IPython.display.display().

    Raises
    ------
    ValueError
        If seq contains a non-standard amino acid, or quantile_threshold is not
        strictly between 0 and 1.
    """

    # check that the sequence is all valid amino acids
    check_sequence_validity(seq)

    if quantile_threshold <= 0 or quantile_threshold >= 1:
        raise ValueError("The quantile threshold must be between 0 and 1.")

    # scale scale_vec to be between min_font_sz and max_font_sz; a flat vector
    # (e.g. no attractive contacts anywhere) has nothing to rank, so every letter
    # gets the minimum size rather than a divide-by-zero NaN
    scale_vec = np.asarray(scale_vec, dtype=float)
    min_val = np.min(scale_vec)
    max_val = np.max(scale_vec)
    if max_val == min_val:
        sizes_vec = np.full(len(scale_vec), float(min_font_sz))
    else:
        sizes_vec = min_font_sz + (scale_vec - min_val) * (
            max_font_sz - min_font_sz
        ) / (max_val - min_val)

    # create the matched hex values for each amino acid
    color_list = [AA_COLOR[aa] for aa in seq]

    html_obj = styled_text(
        seq, color_list, sizes_vec.tolist(), quantile_threshold=quantile_threshold
    )
    display(html_obj)


def color_aminoacids(seq: str, aminos_of_interest: str, font_sz=12):
    """
    Display a sequence with chosen amino acids highlighted in their chemistry colour.

    Every letter has the same size; residues in aminos_of_interest are drawn in
    their AA_COLOR colour and all others in dark grey.

    Parameters
    ----------
    seq : str
        The amino acid sequence to be displayed.

    aminos_of_interest : str
        The amino acids to highlight, e.g. "DERK" for the charged residues.

    font_sz : int, optional
        Font size in pixels. Default is 12.

    Returns
    -------
    None
        The sequence is displayed with IPython.display.display().

    Raises
    ------
    ValueError
        If seq contains a non-standard amino acid.
    """
    # check that the sequence is all valid amino acids
    check_sequence_validity(seq)

    sizes_vec = [font_sz] * len(seq)

    # create the matched hex values for each amino acid
    color_list = [AA_COLOR[aa] if aa in aminos_of_interest else "#3f3f3f" for aa in seq]

    html_obj = styled_text(seq, color_list, sizes_vec)
    display(html_obj)


def _plot_profile_logo(s1, s2, xf, window_size, quantile_threshold, attractive):
    """
    Shared implementation of plot_attractive_logo() and plot_repulsive_logo().

    Parameters
    ----------
    attractive : bool
        If True, letters grow with attraction (a more negative attractive
        profile); if False, with repulsion (a more positive repulsive profile).

    All other parameters are as documented in the public functions.
    """
    if attractive:
        positions, values = xf.per_residue_attractive_vector(
            s1, s2, window_size=window_size
        )
        # more negative = more attractive = bigger letter. chemical_context_seq_plot
        # rescales linearly between the smallest and largest score, so only the
        # ordering matters and the sign flip is all that is needed
        scores = -values
    else:
        positions, values = xf.per_residue_repulsive_vector(
            s1, s2, window_size=window_size
        )
        scores = values

    # the profile covers the window centres only, so show that part of s1
    seq_start = positions[0] - 1
    seq_end = positions[-1]

    chemical_context_seq_plot(
        s1[seq_start:seq_end],
        scores,
        max_font_sz=40,
        min_font_sz=3,
        quantile_threshold=quantile_threshold,
    )


def plot_attractive_logo(s1, s2, xf, window_size=31, quantile_threshold=0.7):
    """
    Display an InterLogo of the residues in s1 that attract s2.

    The per-residue attractive profile of s1 against s2
    (FinchesFrontend.per_residue_attractive_vector()) sets the letter sizes: the
    more attractive a position, the larger its letter. The profile is defined for
    window centres only, so the first and last window_size // 2 residues of s1 are
    not shown.

    Parameters
    ----------
    s1 : str
        The sequence to draw.

    s2 : str
        The partner sequence.

    xf : FinchesFrontend
        The frontend object (e.g. Mpipi_frontend()) used to compute the profile.

    window_size : int, optional
        The window size for the profile. Default is 31.

    quantile_threshold : float, optional
        Letters whose size falls below this quantile are greyed out. Must be
        strictly between 0 and 1. Default is 0.7.

    Returns
    -------
    None
        The logo is displayed with IPython.display.display().
    """
    _plot_profile_logo(s1, s2, xf, window_size, quantile_threshold, True)


def plot_repulsive_logo(s1, s2, xf, window_size=31, quantile_threshold=0.7):
    """
    Display an InterLogo of the residues in s1 that repel s2.

    The per-residue repulsive profile of s1 against s2
    (FinchesFrontend.per_residue_repulsive_vector()) sets the letter sizes: the
    more repulsive a position, the larger its letter. The profile is defined for
    window centres only, so the first and last window_size // 2 residues of s1 are
    not shown.

    Parameters
    ----------
    s1 : str
        The sequence to draw.

    s2 : str
        The partner sequence.

    xf : FinchesFrontend
        The frontend object (e.g. Mpipi_frontend()) used to compute the profile.

    window_size : int, optional
        The window size for the profile. Default is 31.

    quantile_threshold : float, optional
        Letters whose size falls below this quantile are greyed out. Must be
        strictly between 0 and 1. Default is 0.7.

    Returns
    -------
    None
        The logo is displayed with IPython.display.display().
    """
    _plot_profile_logo(s1, s2, xf, window_size, quantile_threshold, False)
