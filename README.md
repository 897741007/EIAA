# EIAA
Edge-Induced-Atoms-Attention

# Atom attention
Each Atom is represented by the weigthed sum of all linked atoms and itself  
The attention weight between two atoms is induced by the Bond between them  

# Edge attention
Each Bond is represented by the weigthed sum of two linked atoms and itself  
The attention weight of the three part (atom_0, atom_1 and bond) is determined by the dot-product of bond and each part      
The attention for the bond consisting of multiple atoms, like π, is under development

# Prediction on the whole graph
A super Node is introduced to capture the information of the whole graph  
The super Node is linked to all the atoms by a special bond, "self-link", while all the atoms are not linked to the super Node  
Which means an unidirectional link is set from super Node to all atoms  
"self-link" is different from any other chemical bonds, and also, each Atom is linked to itself by "self-link"  

# Requirement
At least 24G GPU memory is required for a 6-layer model  
If Edge-Attention is turned off, the demand for GPU memory is halved
