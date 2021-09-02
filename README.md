# EIAA
Edge-Induced-Atoms-Attention

# Atom attention
Each Atom is represented by the weigthed sum of all linked atoms and itself
The attention weight between two atoms is induced by the linked bond  

# Edge attention
Each Bond is represented by the weigthed sum of two linked atoms and itself  
The attention weight of the three part (atom_0, atom_1 and bond) is determined by the dot-product of bond and each part      
The attention of the bond consisting of multiple atoms, like π, is under development

# Requirement
At least 24G GPU memory is required for a 6-layer model  
If Edge-Attention is turned off, the demand for video memory is halved
