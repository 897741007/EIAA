from rdkit import Chem
import numpy as np
from time import time
from random import shuffle
import torch
from tqdm import tqdm

class SMI_grapher():
    def __init__(self, for_predictor=False, device='cuda', lower_aromatic=False, specify_bond=True):
        # param : lower_aromatic --> if treat aromatic atom as the different, 
        #                            e.g. if True, the aromatic 'C' is represented in 'c'
        # param : specify_bond --> if keep the feature of different type of bonds
        #                            e.g. if False, all the bonds are regarded as a link
        self.CLS = for_predictor
        self.device = device
        self.dict = {}
        self.lower_aromatic = lower_aromatic
        self.specify_bond = specify_bond
        if self.CLS:
            self.dict_size = len(self.dict) + 2
        else:
            self.dict_size = len(self.dict) + 1
        self.pad_atom_id = 0
        self.get_coding_dict()
    
    def draw_pi_bond(self, adjm, benzene_ring_info, identical_bond=False, pi=4):
        # draw the π-bond on adjacency matrix
        # param identical_bond : if treat the link between the atoms in a benzene ring as identical link
        #                        if False, these links are divided into three types : adjacent_link, meta_link and para_link
        adj_m = np.copy(adjm)
        if identical_bond:
            for r in benzene_ring_info:
                for a_idx in r:
                    for c_idx in r:
                        if a_idx != c_idx:
                            adj_m[a_idx][c_idx] = pi
        else:
            link_pi = [pi, pi+1, pi+2]
            for br in benzene_ring_info:
                br_length = len(br)
                br_exp = br + br + br
                for a_idx in range(br_length):
                    if br_length == 6:
                        link_lib = [[br_exp[a_idx+br_length-1], br_exp[a_idx+br_length+1]],
                        [br_exp[a_idx+br_length-2], br_exp[a_idx+br_length+2]], 
                        [br_exp[a_idx+br_length+3]]]
                    else:
                        link_lib = [[br_exp[a_idx+br_length-1], br_exp[a_idx+br_length+1]],
                        [br_exp[a_idx+br_length-2], br_exp[a_idx+br_length+2]], 
                        []]
                    for link in range(3):
                        for c_idx in link_lib[link]:
                            adj_m[br[a_idx]][c_idx] = link_pi[link]
        return adj_m

    def get_benzene_ring(self, mol):
        rings = mol.GetRingInfo()
        a_ring = rings.AtomRings()
        benzene_rings = []
        for r in a_ring:
            bond = mol.GetBondBetweenAtoms(r[0],r[1])
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                benzene_rings.append(r)
        return benzene_rings

    def Smi2Graph_pri(self, smi):
        # SMILES --> PDBBlock --> Graph
        # The π-bond of benzene ring is transformed into the structure of single double alternating
        # return adj_m : adjacency matrix, the information of bond type is remain
        # return atom_info : atom information
        # return ion_info : ion (charge) information
        # is faster than Smi2Graph_1
        def complete_upper_trimatrix(upper_matrix):
            # complete the upper triangular matrix to a symmetric matrix
            complete_matrix = np.copy(upper_matrix)
            a, _ = upper_matrix.shape
            for i in range(a):
                for j in range(0, i):
                    complete_matrix[i][j] = complete_matrix[j][i]
            return complete_matrix

        mol = Chem.MolFromSmiles(smi)
        # extracting atomic information
        atom_info = []
        ion_info = []
        for a in mol.GetAtoms():
            a_symbol = a.GetSymbol()
            if self.lower_aromatic:
                if a.GetIsAromatic():
                    a_symbol = a_symbol.lower()
            atom_info.append(a_symbol)
            ion_info.append(a.GetFormalCharge())
        atom_n = len(atom_info)
        # generating adjacency matrix
        if self.specify_bond:
            pb = Chem.MolToPDBBlock(mol)
            mol_info = pb.split('\n')[:-2]
            link_info = []
            for line in mol_info:
                info = [i for i in line.split(' ') if i]
                if info[0] == 'CONECT':
                    link_info.append([int(i) for i in info[1:]])
            adj_m = np.zeros((atom_n, atom_n), dtype=int)
            for rec in link_info:
                atom_0 = rec[0]-1
                for a1 in rec[1:]:
                    atom_1 = a1 - 1
                    adj_m[atom_0][atom_1] += 1
            adj_m = complete_upper_trimatrix(adj_m)
        else:
            adj_m = Chem.GetAdjacencyMatrix(mol)

        return adj_m, atom_info, ion_info, mol

    def Smi2Graph(self, smi, pi=4):
        # SMILES --> Graph
        # The π-bond of benzene ring is regard as a new type of bond which is different from single and double bond
        # param pi : the representation of π-bond, valid 4
        # return adj_m : adjacency matrix, the information of bond type is remain
        # return atom_info : atom information
        # return ion_info : ion (charge) information
        adj_m, atom_info_s, ion_info, mol = self.Smi2Graph_pri(smi)
        # advanced benzene ring mark
        if self.specify_bond:
            BRs = self.get_benzene_ring(mol)
            adj_m = self.draw_pi_bond(adj_m, BRs, pi=pi)
    
        return adj_m, atom_info_s, ion_info
    
    def fit_new(self, smis):
        self.dict = {}
        with tqdm(total=len(smis)) as pbar:
            for smi in smis:
                mol = Chem.MolFromSmiles(smi)
                for atom in mol.GetAtoms():
                    a_idx = atom.GetSymbol()
                    if self.lower_aromatic:
                        if atom.GetIsAromatic():
                            a_idx = a_idx.lower()
                    if a_idx in self.dict:
                        self.dict[a_idx] += 1
                    else:
                        self.dict[a_idx] = 1
                pbar.update()
        self.get_coding_dict()
        if self.CLS:
            self.dict_size = len(self.dict) + 2
        else:
            self.dict_size = len(self.dict) + 1
    
    def fit_add(self, smis):
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            for atom in mol.GetAtoms():
                a_idx = atom.GetSymbol()
                if a_idx in self.dict:
                    self.dict[a_idx] += 1
                else:
                    self.dict[a_idx] = 1
        self.get_coding_dict()
        if self.CLS:
            self.dict_size = len(self.dict) + 2
        else:
            self.dict_size = len(self.dict) + 1
    
    def get_coding_dict(self):
        if self.CLS:
            # [PAD] : 0
            # [CLS] : 1
            self.coding_dict = {a:idx+2 for idx, a in enumerate(self.dict.keys())}
        else:
            # [PAD] : 0
            self.coding_dict = {a:idx+1 for idx, a in enumerate(self.dict.keys())}


    def adjm_padding(self, adj_m, atoms, ions, padding_length):
        # Padding the three part of the molecular garph-representation
        n_atoms = len(atoms)
        adding_length = padding_length - n_atoms
        p_atoms = atoms + [self.pad_atom_id,] * adding_length
        p_ions = ions + [0,] * adding_length
        p_adj_m = np.zeros((padding_length, padding_length), dtype=int)
        p_adj_m[:n_atoms, :n_atoms] = adj_m
        return p_adj_m, p_atoms, p_ions
    
    def draw_selflink(self, adj_m):
        # draw the self-link on adjm
        # the self-link is different with chemical bonds
        a, _ = adj_m.shape
        adj_mx = np.copy(adj_m)
        adj_mx[adj_mx>0] += 1
        for idx in range(a):
            adj_mx[idx][idx] = 1
        return adj_mx

    def add_cls(self, adj_m, atoms, ions, cls_id, org_len):
        a, _ = adj_m.shape
        c_adj_m = np.zeros((a+1, a+1), dtype=int)
        c_adj_m[1:,1:] = adj_m
        c_adj_m[0,1:org_len] = 1
        c_atoms = [cls_id] + atoms
        c_ions = [0] + ions
        return c_adj_m, c_atoms, c_ions
                    
    def provide_batch_info(self, batch_smis):
        batch_adj_m = []
        batch_atoms = []
        batch_ions = []
        padding_length = 0
        for smi in batch_smis:
            adj_m, atoms, ions = self.Smi2Graph(smi)
            batch_adj_m.append(adj_m)
            batch_atoms.append([self.coding_dict[a] for a in atoms])
            batch_ions.append(ions)
            k = len(atoms)
            if k > padding_length:
                padding_length = k
        pbatch_adj_m = []
        pbatch_atoms = []
        pbatch_ions = []
        #self_link = np.identity(padding_length, dtype=int)
        for adj_m, atoms, ions in zip(batch_adj_m, batch_atoms, batch_ions):
            
            p_adj_m, p_atoms, p_ions = self.adjm_padding(adj_m, atoms, ions, padding_length)
            p_adj_m = self.draw_selflink(p_adj_m)
            if self.CLS:
                p_adj_m, p_atoms, p_ions = self.add_cls(p_adj_m, p_atoms, p_ions, 1, len(atoms)+1)
            
            pbatch_adj_m.append(p_adj_m)
            pbatch_atoms.append(p_atoms)
            pbatch_ions.append(p_ions)
        pbatch_adj_m = np.array(pbatch_adj_m)
        pbatch_atoms = np.array(pbatch_atoms)
        pbatch_ions = np.array(pbatch_ions)
        pbatch_adj_m = torch.Tensor(pbatch_adj_m).long().to(self.device)
        pbatch_atoms = torch.Tensor(pbatch_atoms).long().to(self.device)
        pbatch_ions = torch.Tensor(pbatch_ions).long().to(self.device)
        
        return (pbatch_adj_m, pbatch_atoms, pbatch_ions)
    
    def data_provider(self, data_set, batch_size, mode='cls', do_random=False):
        # data provider
        # the size of last batch maybe lower than batch_size to ensure all the data are sampled once in one epoch 
        # e.g. data_set : [1,2,3,4,5,6], batch_size : 4
        #      sampling result: [1, 2, 3, 4] [5, 6] [1, 2, 3, 4] [5, 6] [1, 2, 3, 4] ...
        total_amount = len(data_set)
        if do_random:
            shuffle(data_set)
        start_idx = 0
        end_idx = batch_size
        while 1:
            if end_idx >= total_amount:
                batch_data = data_set[start_idx:]
                start_idx = 0
                end_idx = start_idx + batch_size
                if do_random:
                    shuffle(data_set)
            else:
                batch_data = data_set[start_idx:end_idx]
                start_idx += batch_size
                end_idx += batch_size
            batch_smis = [r[0] for r in batch_data]
            batch_labels = [r[1] for r in batch_data]
            if mode == 'cls':
                batch_labels = torch.Tensor(batch_labels).long().to(self.device)
            else:
                batch_labels = torch.Tensor(batch_labels).to(self.device).reshape(-1,1)
            t_batch_smis = self.provide_batch_info(batch_smis)
            yield t_batch_smis, batch_labels

if __name__ == '__main__':
    #smi = 'CC(=O)Oc1ccccc1C(=O)O'
    #smi = 'C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1'
    #smi = 'Nc1nc(O)c2nn[nH]c2n1'
    #smi = 'N[C@@](F)(C)C(=O)O'
    #smi = 'O/N=C1C(=C\\c2ccccc2)/N2CCC/1CC2'
    #smi = 'C[n+]1ccoc1[NH-]'
    smi = 'Sc1cccc2c(S)cccc12'

    n_class = 4
    embedding_dim = 16
    adj_m, atoms, ions = Smi2Graph_0(smi)
    #badj_emb = adjacency_embedding(adj_m, n_class, embedding_dim)
