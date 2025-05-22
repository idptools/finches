Predicting mean-field epsilon values
=======================================

What is epsilon?
------------------
Epsilon is a mean-field parameter that describes the average interaction between two IDRs. It is a potentially useful value to ask how sequence changes are expected to alter the overall attraction or repulsion between two sequences. 

How to calculate epsilon with FINCHES:
---------------------------------------

We recommend calculating epislon using the frontend objects.

.. code-block:: python

    # import the Mpipi_frontend object
    from finches import Mpipi_frontend, CALVADOS_frontend

    # initialize the frontend objects (note can change salt)
    mf = Mpipi_frontend()
    cf = CALVADOS_frontend()

    # laf-1 RGG domain (1-168)
    s1 = 'MESNQSNNGGSGNAALNRGGRYVPPHLRGGDGGAAAAASAGGDDRRGGAGGGGYRRGGGNSGGGGGGGYDRGYNDNRDDRDNRGGSGGYGRDRNYEDRGYNGGGGGGGNRGYNNNRGGGGGGYNRQDRGDGGSSNFSRGGYNNRDEGSDNRGSGRSYNNDRRDNGGD'

    # laf-1 CTD (622-708)
    s2 = 'LEGMSGDMRSGGGYRGRGGRGNGQRFGGRDHRYQGGSGNGGGGNGGGGGFGGGGQRSGGGGGFQSGGGGGRQQQQQQRAQPQQDWWS'

    # calculate epsilon
    eps = mf.epsilon(s1, s2)
    print(eps)    
    >> -11.832997767002091

    # calculate epsilon
    eps = cf.epsilon(s1, s2)
    print(eps)    
    >> -28.8043554142898

As a reminder, the order in which sequences are passed to calculate epsilon matters; the intuition here is that sequence 1 is being "dipped" into a bath of sequence 2, so if we double the length of sequence 1 we double the interaction strength (whereas if we double the length of sequence 2 we do not double the interaction strength).

The epislon value is also used for calculating phase diagrams. 

Caveats and considerations
------------------------------

1. For sequences that are chemically homogenous (e.g. low-complexity IDRs) epislon can be quite useful to capture the overall interaction strength. However, for sequences that are chemically heterogeneous (e.g. IDRs with several chemically-distinct subdomains) the epsilon value is likely less useful, because it is an average across the two sequences, yet two sequences only need a subregion that is highly attractive to interact with one another. As such, for interpreting whethere two IDRs are likely to interact, epislon is potentially sensitive to false negatives. 


