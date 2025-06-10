from IPython.display import display, HTML

def display_char_dtw_alignment(s1: str, s2: str, path: list):
    """
    Generates and displays an HTML/SVG visual for a DTW alignment of two character sequences.

    Args:
        s1 (str): The first sequence.
        s2 (str): The second sequence.
        path (list): A list of (i, j) tuples representing the DTW path.
    """
    # --- SVG layout constants ---
    CHAR_WIDTH = 30
    FONT_SIZE = 20
    V_SPACING = 80
    TOP_PADDING = 30
    SIDE_PADDING = 20
    LINE_STROKE_WIDTH = 1.5

    # Calculate SVG dimensions
    svg_width = max(len(s1), len(s2)) * CHAR_WIDTH + 2 * SIDE_PADDING
    svg_height = 2 * FONT_SIZE + V_SPACING + 2 * TOP_PADDING

    # --- Build the SVG string ---
    svg_elements = []

    # Add Sequence 1 characters
    y1 = TOP_PADDING + FONT_SIZE
    for i, char in enumerate(s1):
        x = SIDE_PADDING + i * CHAR_WIDTH + (CHAR_WIDTH / 2)
        svg_elements.append(
            f'<text x="{x}" y="{y1}" font-size="{FONT_SIZE}px" font-family="monospace" text-anchor="middle">{char}</text>'
        )

    # Add Sequence 2 characters
    y2 = y1 + V_SPACING
    for i, char in enumerate(s2):
        x = SIDE_PADDING + i * CHAR_WIDTH + (CHAR_WIDTH / 2)
        svg_elements.append(
            f'<text x="{x}" y="{y2}" font-size="{FONT_SIZE}px" font-family="monospace" text-anchor="middle">{char}</text>'
        )

    # Add lines for the DTW path pairings
    for i, j in path:
        x1 = SIDE_PADDING + i * CHAR_WIDTH + (CHAR_WIDTH / 2)
        line_y1 = y1 + 5 # Start line just below the character
        
        x2 = SIDE_PADDING + j * CHAR_WIDTH + (CHAR_WIDTH / 2)
        line_y2 = y2 - FONT_SIZE # End line just above the character
        
        # Color line green if characters match, red otherwise
        color = "green" if s1[i] == s2[j] else "crimson"
        
        svg_elements.append(
            f'<line x1="{x1}" y1="{line_y1}" x2="{x2}" y2="{line_y2}" stroke="{color}" stroke-width="{LINE_STROKE_WIDTH}" opacity="0.8"/>'
        )
    
    # --- Assemble the final HTML and display it ---
    svg_content = "".join(svg_elements)
    html_string = f"""
    <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px;">
        <svg width="{svg_width}" height="{svg_height}">
            {svg_content}
        </svg>
    </div>
    """
    
    display(HTML(html_string))