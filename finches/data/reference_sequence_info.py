from afrc import AnalyticalFRC


def build_ref_GS_AFRC_distances():
    ## --- init GS dict --- ##
    GS_seq_max='GS'*1000
    GS_seqs_dic={}
    for i in range(len(GS_seq_max)):
        GS_seqs_dic[i] = AnalyticalFRC(GS_seq_max[0:i]).get_mean_end_to_end_distance()
    return GS_seqs_dic



GS_seqs_dic = build_ref_GS_AFRC_distances()