
S1-12: Sample #1, 12th slide (corresponding to AP -1.94 mm).
# representative brain slice images, correspond to the Fig. 2e. 

.csv data was extracted from slide image by using IMARIS cell detection function. 
DAPI.csv: DAPI signals were used to delineate boundaries by colocalization with the reference atlas.
AC.csv: Neurons projecting to the auditory cortex (AC), identified by GFP signals.
IC.csv: Neurons projecting to the inferior colliculus (IC), identified by tdTomato signals.
Colocal.csv: Neurons projecting to both AC and IC, identified by the presence of both GFP and tdTomato signals.

1st column: X coordinate (ML).
2nd column: Y coordinate (DV; inverted, corrected by "camroll").
3rd column: Z coordinate (AP; values are same for this dataset).

If you run the code, figures will be poped up with interaction 
Figure 1:
Draw the boundary line to delineate the PPC-M region.

Figures 2 & 4:
Draw a line along the ventricle first (illustrated in Red), followed by the pia line (in blue) 
The code will fit these lines using a regression algorithm with DAPI signals

Figure 3:
Designate the PPC-L region by drawing a polygon with sequential clicks.
Double-click to complete the polygon.

Figure 1000:
Left panel: Plot the number of cells as a function of normalized cortical depth.
Right panel: Plot the fraction of cells as a function of normalized cortical depth.
Depth normalization: 1 represents the pia, 0 represents the ventricle.
