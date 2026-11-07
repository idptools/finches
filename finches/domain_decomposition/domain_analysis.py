"""
Automatic extraction of interacting regions from FINCHES intermaps.

An intermap often contains a handful of rectangular blocks where one stretch
of sequence 1 interacts consistently with one stretch of sequence 2. This
module finds those blocks automatically and reports them in sequence space.

The approach works on a thresholded intermap. Every cell that meets a criterion
(for example, an epsilon value below -0.2) counts as a hit. Summing hits down
each column gives a 1D profile whose peaks mark columns that take part in many
interactions; each peak is then sliced out, turned on its side and searched
again along the rows, and once more along the columns, so that what survives is
a rectangle that is dense in hits in both directions. Because this favours
rectangles in one orientation, the search is run on both the intermap and its
transpose and overlapping rectangles are merged
(see get_bidirectional_interaction_regions()).

Region bounds are reported in intermap index space as
((x_start, x_end), (y_start, y_end)), half-open (the end index is excluded),
where x indexes the columns of the array and y the rows. extract_regions()
converts these into 1-based, inclusive residue numbers.

Developed by Alex Keeley and Alex Holehouse.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks


class PeakError(Exception):
    """
    Raised by peak_width() when the value at the requested peak index lies
    below the baseline, which means it is not a peak at that baseline.
    """

    pass


class InteractingRegions:
    """
    A pair of interacting regions, one in each sequence.

    This is a lightweight container that makes it easy to work with the regions
    returned by extract_regions(). All positions are 1-based residue numbers,
    and start and end are both inclusive.

    Attributes
    ----------
    r1_start, r1_end : int
        First and last residue of the region in sequence 1. These are the
        centres of the first and last intermap windows in the region.

    r1_sequence : str
        The sequence of the region in sequence 1 (residues r1_start to r1_end).

    r2_start, r2_end : int
        First and last residue of the region in sequence 2.

    r2_sequence : str
        The sequence of the region in sequence 2.

    window_size : int
        The window size of the intermap the region was extracted from.

    sequence_1, sequence_2 : str
        The two full sequences.

    r1_start_full, r1_end_full, r2_start_full, r2_end_full : int
        The region extended by half a window at each end, i.e. every residue
        that contributes to at least one window in the region.

    r1_sequence_full, r2_sequence_full : str
        The sequences of the extended regions.

    edgecolor : str
        Edge colour used by rectangle_info. Default is 'r'.

    rectangle_kwargs : dict
        Additional matplotlib Rectangle keyword arguments used by
        rectangle_info. Default is an empty dictionary.
    """

    def __init__(
        self,
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
        """
        Create an InteractingRegions object.

        Parameters
        ----------
        r1_start, r1_end : int
            First and last residue (1-based, inclusive) of the region in
            sequence 1.

        r1_sequence : str
            The sequence of the region in sequence 1.

        r2_start, r2_end : int
            First and last residue (1-based, inclusive) of the region in
            sequence 2.

        r2_sequence : str
            The sequence of the region in sequence 2.

        window_size : int
            The (odd) window size of the intermap the region came from.

        sequence_1, sequence_2 : str
            The two full sequences.
        """
        self.r1_start = r1_start
        self.r1_end = r1_end
        self.r1_sequence = r1_sequence
        self.r2_start = r2_start
        self.r2_end = r2_end
        self.r2_sequence = r2_sequence
        self.window_size = window_size
        self.sequence_1 = sequence_1
        self.sequence_2 = sequence_2

        window_half = int((window_size - 1) / 2)

        self.r1_start_full = r1_start - window_half
        self.r1_end_full = r1_end + window_half

        self.r2_start_full = r2_start - window_half
        self.r2_end_full = r2_end + window_half

        # we do a -1 at the start to convert from amino acid space to Python indexing
        # space
        self.r1_sequence_full = sequence_1[self.r1_start_full - 1 : self.r1_end_full]
        self.r2_sequence_full = sequence_2[self.r2_start_full - 1 : self.r2_end_full]

        # set default values for the rectangle
        self.edgecolor = "r"
        self.rectangle_kwargs = {}

    @property
    def rectangle_info(self):
        """
        The region in the format taken by the plot_rectangles argument of the
        frontend interaction_figure() method.

        In that figure the intermap cell for residue p spans p to p + 1, so the
        rectangle runs from each start residue to one past each end residue and
        covers every residue in the region. Extra matplotlib Rectangle keyword
        arguments can be set via the rectangle_kwargs attribute, for example::

            ir = interactor_pairs[0]
            ir.rectangle_kwargs = {'linestyle': '--'}
            mf.interaction_figure(seq1, seq2, plot_rectangles=[ir.rectangle_info])

        Returns
        -------
        list
            [r1_start, r1_end + 1, r2_start, r2_end + 1, kwargs], where kwargs
            holds the edge colour plus anything in rectangle_kwargs.
        """

        return [
            self.r1_start,
            self.r1_end + 1,
            self.r2_start,
            self.r2_end + 1,
            {"edgecolor": self.edgecolor, **self.rectangle_kwargs},
        ]

    def __str__(self):
        """
        Return a one-line summary of the two regions.

        Returns
        -------
        str
            The start, end and length of each region.
        """
        return (
            f"InteractingRegions(seq1: {self.r1_start}-{self.r1_end} "
            f"({self.r1_end - self.r1_start + 1}), seq2: {self.r2_start}-{self.r2_end} "
            f"({self.r2_end - self.r2_start + 1}))"
        )

    def __repr__(self):
        """
        Return the same summary as __str__().

        Returns
        -------
        str
            The start, end and length of each region.
        """
        return str(self)


# ....................................................................................
#
#
def extract_regions(
    frontend_obj,
    seq1,
    seq2,
    window_size=31,
    criteria="less",
    criteria_threshold=-0.2,
    baseline=0.3,
    min_region_area=500,
    min_region_size=20,
    penalize_opposite=False,
    offset_threshold=0.2,
):
    """
    Compute the intermap between two sequences and extract its interacting
    regions.

    This is the main entry point to the module. It builds the intermap with
    the frontend's intermolecular_idr_matrix(), finds rectangular blocks that
    meet the criterion with get_bidirectional_interaction_regions(), and
    converts each block into a pair of regions in sequence space.

    Parameters
    ----------
    frontend_obj : finches.frontend.frontend_base.FinchesFrontend
        The frontend object used to compute the intermap (for example an
        Mpipi_frontend or CALVADOS_frontend).

    seq1 : str
        The first sequence (the columns of the returned intermap).

    seq2 : str
        The second sequence (the rows of the returned intermap).

    window_size : int, optional
        The window size for the intermap. Must be odd; an even value is rounded
        up to the next odd number with a warning. Default is 31.

    criteria : str, optional
        'less' to look for attractive regions (values at or below
        criteria_threshold) or 'greater' to look for repulsive regions (values
        at or above it). Default is 'less'.

    criteria_threshold : float, optional
        The intermap value a cell has to reach to count as a hit. Default is
        -0.2.

    baseline : float, optional
        Fraction of a peak's height at which the peak is taken to end when
        measuring its width (see get_column_regions()). Default is 0.3.

    min_region_area : int, optional
        Smallest region, in intermap cells (width x height), that is kept.
        Default is 500 (e.g. 20 x 25).

    min_region_size : int, optional
        Smallest width and smallest height, in intermap cells (equivalently
        residues), that a region can have. Default is 20.

    penalize_opposite : bool, optional
        If True, cells of the opposite character (repulsive when looking for
        attractive regions and vice versa) are subtracted from the hit counts,
        so regions do not extend across strongly opposing interactions.
        Default is False.

    offset_threshold : float, optional
        Magnitude of the value a cell has to reach to count as opposite when
        penalize_opposite is True: at or above +offset_threshold for
        criteria='less', at or below -offset_threshold for criteria='greater'.
        Default is 0.2, mirroring the default criteria_threshold.

    Returns
    -------
    tuple
        (interactor_pairs, regions, intermap), where:

        - interactor_pairs is a list of InteractingRegions objects, one per
          region, with start and end in 1-based residue numbering, inclusive at
          both ends;
        - regions is the list of region bounds in intermap index space, as
          returned by get_bidirectional_interaction_regions();
        - intermap is the intermap the regions were extracted from. It is the
          transpose of the matrix returned by intermolecular_idr_matrix(), so
          rows are windows of seq2 and columns are windows of seq1.
    """

    # guarantees odd window size; this is the design pattern we use throughout
    if window_size % 2 == 0:
        warnings.warn(
            f"window size is even, rounding up to next odd number {window_size + 1}"
        )
        window_size = window_size + 1

    # compute the intermap
    intermap = frontend_obj.intermolecular_idr_matrix(
        seq1, seq2, window_size=window_size
    )[0][0].transpose()

    # get the interaction regions
    regions = get_bidirectional_interaction_regions(
        intermap,
        criteria=criteria,
        criteria_threshold=criteria_threshold,
        baseline=baseline,
        min_region_area=min_region_area,
        min_region_size=min_region_size,
        penalize_opposite=penalize_opposite,
        offset_threshold=offset_threshold,
    )

    # build out the actual interacting regions
    interactor_pairs = []
    for region in regions:
        # seq start and end are in sequence space (i.e. 1 thru len(seq)) and are
        # inclusive at both ends. Region bounds are [start, end) in intermap index
        # space, so the exclusive end index is converted to the last included column
        seq_1_start = convert_intermap_index_to_residue_number(
            region[0][0], window_size
        )
        seq_1_end = convert_intermap_index_to_residue_number(
            region[0][1] - 1, window_size
        )

        seq_2_start = convert_intermap_index_to_residue_number(
            region[1][0], window_size
        )
        seq_2_end = convert_intermap_index_to_residue_number(
            region[1][1] - 1, window_size
        )

        # we do a -1 at the start to convert from amino acid space to Python indexing
        # space
        seq_1_region = seq1[seq_1_start - 1 : seq_1_end]
        seq_2_region = seq2[seq_2_start - 1 : seq_2_end]
        interactor_pairs.append(
            InteractingRegions(
                seq_1_start,
                seq_1_end,
                seq_1_region,
                seq_2_start,
                seq_2_end,
                seq_2_region,
                window_size,
                seq1,
                seq2,
            )
        )

    return (interactor_pairs, regions, intermap)


# ....................................................................................
#
#
def convert_intermap_index_to_residue_number(index, window_size=31):
    """
    Convert a 0-based intermap index into the residue number at the centre of
    that window.

    Parameters
    ----------
    index : int
        The 0-based row or column index in the intermap.

    window_size : int, optional
        The (odd) window size used to generate the intermap. Default is 31.

    Returns
    -------
    int
        The residue number corresponding to the index, in 1-based sequence
        numbering (i.e. 1 through len(seq)). Column ``index`` of the intermap is
        the window centred on this residue, which matches the index vectors
        returned by the frontend's intermolecular_idr_matrix().

    Raises
    ------
    ValueError
        If window_size is even.
    """

    if window_size % 2 == 0:
        raise ValueError(
            f"Window size ({window_size}) must be odd; this should have been caught earlier"
        )

    window_half = int((window_size - 1) / 2)

    return index + window_half + 1


# ....................................................................................
#
#
def peak_width(data, peak_index, baseline):
    """
    Measure how far a peak extends either side of its maximum before dropping
    below a baseline value.

    In short, "what index range does the peak span before the data fall below
    the baseline?". The walk stops at the ends of the array.

    Parameters
    ----------
    data : np.ndarray
        The 1D data containing the peak.

    peak_index : int
        The index of the peak of interest.

    baseline : float
        The absolute value below which the peak is considered to have ended.

    Returns
    -------
    tuple
        (width, left_index, right_index), where left_index and right_index are
        the first and last indices at or above the baseline (both inclusive) and
        width = right_index - left_index + 1.

    Raises
    ------
    PeakError
        If the value at peak_index is itself below the baseline.
    """

    # The 'y value' of the data/peak at peak_index
    peak_value = data[peak_index]

    # A peak cannot sit below its own baseline. This should not occur in any case
    # where this function is used, but is designed to catch critical errors.
    if peak_value < baseline:
        raise PeakError(
            "Peak value is below baseline value. This should not occur in this context."
        )

    ## Find the left crossing point

    # step left-wards from the peak until we drop below the baseline value OR we
    # get to the start of the array
    left_index = peak_index
    while left_index > -1 and data[left_index] >= baseline:
        left_index -= 1
    left_index = left_index + 1

    ## Find the right crossing point
    # repeat the same thing moving right
    right_index = peak_index
    while right_index < len(data) and data[right_index] >= baseline:
        right_index += 1
    right_index -= 1

    # +1 to include both endpoints
    width = right_index - left_index + 1

    return width, left_index, right_index


# ....................................................................................
#
#
def remove_included_ranges(ranges):
    """
    Remove any range that lies entirely inside another range in the list.

    Parameters
    ----------
    ranges : list of tuple
        Ranges in the (width, left_index, right_index) form returned by
        peak_width(), with right_index inclusive.

    Returns
    -------
    list of tuple
        The ranges, sorted by left index, with every range that is fully
        contained in another one removed.
    """

    # Sort ranges by left_index, then by right_index
    sorted_ranges = sorted(ranges, key=lambda x: (x[1], x[2]))

    filtered_ranges = []
    for current in sorted_ranges:
        # keep current only if no already-accepted range contains it
        included = False
        for existing in filtered_ranges:
            if existing[1] <= current[1] and existing[2] >= current[2]:
                included = True
                break

        if not included:
            filtered_ranges.append(current)

    return filtered_ranges


# ....................................................................................
#
#
def get_column_regions(
    array,
    criteria="less",
    criteria_threshold=-0.2,
    baseline=0.3,
    min_region_size=20,
    penalize_opposite=False,
    offset_threshold=0.2,
):
    """
    Find runs of columns in a 2D array that contain many cells meeting a
    threshold criterion.

    Cells meeting the criterion are counted down each column, the counts are
    smoothed with a width-3 moving sum, and every peak in the smoothed profile
    is measured with peak_width() at baseline times its own height. Peaks at
    least min_region_size columns wide are kept.

    Parameters
    ----------
    array : np.ndarray
        The 2D array to analyse (an intermap, or a slice of one).

    criteria : str, optional
        'less' to count cells at or below criteria_threshold, or 'greater' to
        count cells at or above it. Default is 'less'.

    criteria_threshold : float, optional
        The value a cell has to reach to count. Default is -0.2.

    baseline : float, optional
        Fraction of each peak's height at which the peak is taken to end.
        Default is 0.3.

    min_region_size : int, optional
        Smallest width, in columns, of a peak that is kept. Default is 20.

    penalize_opposite : bool, optional
        If True, cells of the opposite character are subtracted from each
        column's count. Default is False.

    offset_threshold : float, optional
        Magnitude of the value a cell has to reach to count as opposite: at or
        above +offset_threshold for criteria='less', at or below
        -offset_threshold for criteria='greater'. Default is 0.2.

    Returns
    -------
    tuple
        (bounds_filtered, counts), where bounds_filtered is a list of the kept
        peaks as (width, left_index, right_index) tuples (right_index
        inclusive), and counts is the unsmoothed per-column count of cells
        meeting the criterion (minus opposite cells if penalize_opposite is
        True).

    Raises
    ------
    ValueError
        If criteria is not 'less' or 'greater'.
    """

    # Filter for values in array that meet criterion and threshold
    if criteria == "less":
        within_range = array <= criteria_threshold
    elif criteria == "greater":
        within_range = array >= criteria_threshold
    else:
        raise ValueError(f"criteria must be 'less' or 'greater', not {criteria}")

    # Count of interaction values that meet the criterion/threshold in each column
    counts = np.sum(within_range, axis=0)

    # optionally subtract cells of the opposite character: repulsive cells when
    # looking for attractive regions, attractive cells when looking for repulsive
    # ones. offset_threshold is a magnitude, and its sign follows the criterion.
    if penalize_opposite:
        if criteria == "less":
            offset = array >= abs(offset_threshold)
        else:
            offset = array <= -abs(offset_threshold)

        counts = counts - np.sum(offset, axis=0)

    # Smooth counts with a width-3 moving sum to denoise slightly. We take the
    # centre of the full convolution rather than mode="same", which returns three
    # values for an array with fewer than three columns.
    n = len(counts)
    counts_smooth = np.convolve(counts, np.ones(3, dtype=int), mode="full")[1 : n + 1]

    # Use scipy find_peaks to get indices of peaks in counts_smooth
    peaks_raw = find_peaks(counts_smooth)[0]

    # Get width/start index/end index parameters from each of the raw peaks, skipping
    # any peaks that fail
    bounds = []
    for peak in peaks_raw:
        try:
            bounds.append(
                peak_width(counts_smooth, peak, baseline * counts_smooth[peak])
            )
        except PeakError:
            pass

    # filter out peaks that are too small; >= mirrors get_interaction_regions()
    bounds_filtered = [i for i in bounds if i[0] >= min_region_size]

    return bounds_filtered, counts


# ....................................................................................
#
#
def get_interaction_regions(
    array,
    criteria="less",
    criteria_threshold=-0.2,
    baseline=0.3,
    min_region_area=500,
    min_region_size=20,
    penalize_opposite=False,
    offset_threshold=0.2,
):
    """
    Find rectangular regions of a 2D array that are dense in cells meeting a
    threshold criterion.

    get_column_regions() is applied three times: to the array, to find runs of
    columns; to each run turned on its side, to find runs of rows within it;
    and to each resulting block, to trim its columns again. Blocks that are
    large enough in both directions are returned. This favours rectangles in
    one orientation, which is why get_bidirectional_interaction_regions() runs
    it on both the array and its transpose.

    Parameters
    ----------
    array : np.ndarray
        The 2D array to analyse (an intermap).

    criteria : str, optional
        'less' or 'greater'; see get_column_regions(). Default is 'less'.

    criteria_threshold : float, optional
        The value a cell has to reach to count. Default is -0.2.

    baseline : float, optional
        Fraction of each peak's height at which the peak is taken to end.
        Default is 0.3.

    min_region_area : int, optional
        Smallest region area, in cells (width x height), that is kept. Default
        is 500 (e.g. 20 x 25).

    min_region_size : int, optional
        Smallest width and smallest height, in cells, that a region can have.
        Default is 20.

    penalize_opposite : bool, optional
        Whether to subtract cells of the opposite character; see
        get_column_regions(). Default is False.

    offset_threshold : float, optional
        Magnitude of the value a cell has to reach to count as opposite; see
        get_column_regions(). Default is 0.2.

    Returns
    -------
    tuple
        (regions, counts), where regions is a list of region bounds
        ((x_start, x_end), (y_start, y_end)), half-open, with x the column index
        and y the row index, and counts is the per-column count from the first
        call to get_column_regions(), returned for visualisation.
    """

    # List containing regions fitting given parameters
    regions = []

    bounds, counts = get_column_regions(
        array,
        criteria=criteria,
        criteria_threshold=criteria_threshold,
        baseline=baseline,
        min_region_size=min_region_size,
        penalize_opposite=penalize_opposite,
        offset_threshold=offset_threshold,
    )

    # For each columnar region (or 'peak')
    for peak in bounds:
        # submap is that region, sliced out and turned on its side. Note that
        # peak_width() returns an inclusive right index, hence the +1
        submap = array[:, peak[1] : peak[2] + 1]

        # Get columnar regions of the submap, which are actually row slices of the original intermap/array
        submap_bounds = get_column_regions(
            submap.T,
            criteria=criteria,
            criteria_threshold=criteria_threshold,
            baseline=baseline,
            min_region_size=min_region_size,
            penalize_opposite=penalize_opposite,
            offset_threshold=offset_threshold,
        )[0]

        # For each columnar region of submap
        for submap_peak in submap_bounds:
            # Turn it on its side again (now back in our original orientation) and do the same
            subsubmap = array[
                submap_peak[1] : submap_peak[2] + 1, peak[1] : peak[2] + 1
            ]

            # Find the peaks/regions in subsubmap
            subsubmap_bounds = get_column_regions(
                subsubmap,
                criteria=criteria,
                criteria_threshold=criteria_threshold,
                baseline=baseline,
                min_region_size=min_region_size,
                penalize_opposite=penalize_opposite,
                offset_threshold=offset_threshold,
            )[0]

            # For each final region in a region in a region
            for subsubmap_peak in subsubmap_bounds:
                # Define the index boundaries within the original array/intermap
                low_left_x = peak[1] + subsubmap_peak[1]
                low_left_y = submap_peak[1]

                width = subsubmap_peak[0]
                height = submap_peak[0]
                area = width * height

                # half-open bounds in index space
                region_bounds = (
                    (low_left_x, low_left_x + width),
                    (low_left_y, low_left_y + height),
                )

                # If region area is big enough, add to regions
                if (
                    area >= min_region_area
                    and width >= min_region_size
                    and height >= min_region_size
                ):
                    regions.append(region_bounds)

    return regions, counts


# ....................................................................................
#
#
def rectangles_overlap(R1, R2):
    """
    Test whether two regions overlap or touch.

    Parameters
    ----------
    R1, R2 : tuple
        Regions of the form ((x_start, x_end), (y_start, y_end)).

    Returns
    -------
    bool
        True if the rectangles overlap. Rectangles that only share an edge
        (one's end equals the other's start) also count as overlapping, so that
        get_bidirectional_interaction_regions() merges adjacent blocks.
    """
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


# ....................................................................................
#
#
def rectangle_area(rect):
    """
    Area of a region.

    Parameters
    ----------
    rect : tuple
        A region of the form ((x_start, x_end), (y_start, y_end)), half-open.

    Returns
    -------
    int or float
        (x_end - x_start) * (y_end - y_start), the number of cells covered.
    """
    (left, right), (bottom, top) = rect
    return (right - left) * (top - bottom)


# ....................................................................................
#
#
def get_bidirectional_interaction_regions(
    array,
    criteria="less",
    criteria_threshold=-0.2,
    baseline=0.3,
    min_region_area=500,
    min_region_size=20,
    penalize_opposite=False,
    offset_threshold=0.2,
):
    """
    Find interacting regions in both orientations of an array and merge them.

    get_interaction_regions() is biased against rectangles in one orientation,
    so it is run on the array and on its transpose, the second set of regions is
    transposed back, and any regions that overlap or touch are repeatedly merged
    into their bounding rectangle until none do.

    Parameters
    ----------
    array : np.ndarray
        The 2D array to analyse (an intermap).

    criteria : str, optional
        'less' or 'greater'; see get_column_regions(). Default is 'less'.

    criteria_threshold : float, optional
        The value a cell has to reach to count. Default is -0.2.

    baseline : float, optional
        Fraction of each peak's height at which the peak is taken to end.
        Default is 0.3.

    min_region_area : int, optional
        Smallest region area, in cells, that is kept. Default is 500.

    min_region_size : int, optional
        Smallest width and smallest height, in cells, that a region can have.
        Default is 20.

    penalize_opposite : bool, optional
        Whether to subtract cells of the opposite character; see
        get_column_regions(). Default is False.

    offset_threshold : float, optional
        Magnitude of the value a cell has to reach to count as opposite. Default
        is 0.2.

    Returns
    -------
    list of tuple
        The merged regions, each ((x_start, x_end), (y_start, y_end)),
        half-open, with x the column index and y the row index of array.

    Raises
    ------
    ValueError
        If criteria is not 'less' or 'greater'.
    """

    # sanity check criteria
    if criteria not in ["less", "greater"]:
        raise ValueError(f"criteria must be 'less' or 'greater', not {criteria}")

    shared = dict(
        criteria=criteria,
        criteria_threshold=criteria_threshold,
        baseline=baseline,
        min_region_area=min_region_area,
        min_region_size=min_region_size,
        penalize_opposite=penalize_opposite,
        offset_threshold=offset_threshold,
    )

    # get regions in both directions, transposing the second set back
    regions_1, _ = get_interaction_regions(array, **shared)
    regions_2, _ = get_interaction_regions(array.T, **shared)
    regions = list(regions_1) + [(j, i) for (i, j) in regions_2]

    # merge overlapping rectangles until no pair overlaps
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if rectangles_overlap(regions[i], regions[j]):
                    merged = True
                    regions[i] = (
                        (
                            min(regions[i][0][0], regions[j][0][0]),
                            max(regions[i][0][1], regions[j][0][1]),
                        ),
                        (
                            min(regions[i][1][0], regions[j][1][0]),
                            max(regions[i][1][1], regions[j][1][1]),
                        ),
                    )
                    regions.pop(j)
                    break
            if merged:
                break

    return regions


# ....................................................................................
#
#
def plot_regions_on_intermap(array, regions_lists, colors, filename=None):
    """
    Draw an intermap with sets of regions outlined on top of it.

    The array is drawn with imshow on a fixed -2.5 to 2.5 PRGn colour scale,
    with row 0 at the bottom, and each region is outlined so that it covers
    exactly the cells in its half-open bounds.

    Parameters
    ----------
    array : np.ndarray
        The 2D intermap to draw, for example the intermap returned by
        extract_regions() (rows are seq2, columns are seq1).

    regions_lists : list of list
        One list of regions per colour. Each region is
        ((x_start, x_end), (y_start, y_end)) in index space.

    colors : list
        One matplotlib colour per list in regions_lists.

    filename : str, optional
        If given, the figure is saved to this path; otherwise it is shown with
        plt.show(). Default is None.

    Returns
    -------
    tuple
        (fig, ax), the matplotlib figure and axes.

    Raises
    ------
    ValueError
        If regions_lists and colors do not have the same length.
    """
    if len(regions_lists) != len(colors):
        raise ValueError(
            f"regions_lists ({len(regions_lists)}) and colors ({len(colors)}) must "
            "have the same length; pass one colour per list of regions"
        )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    cax = ax1.imshow(array, cmap="PRGn", vmin=-2.5, vmax=2.5)
    fig.colorbar(cax, ax=ax1, label="Repulsion/Interaction Strength")
    for region_list, color in zip(regions_lists, colors):
        for region in region_list:
            # imshow centres cell i on i, so cell edges sit at i - 0.5
            rect = patches.Rectangle(
                (region[0][0] - 0.5, region[1][0] - 0.5),
                region[0][1] - region[0][0],
                region[1][1] - region[1][0],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                alpha=1,
            )

            ax1.add_patch(rect)
    ax1.set_xlabel("Sequence 1")
    ax1.set_ylabel("Sequence 2")
    ax1.invert_yaxis()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    return fig, ax1
