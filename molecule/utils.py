# Placeholder for utils like SMILES conversion, visualization
def nx_to_smiles(graph):
    try:
        atoms = nx.get_node_attributes(graph, 'element')
        bonds = list(graph.edges(data=True))

        mol = Chem.RWMol()
        atom_index = {}

        for node, element in atoms.items():
            if element == 'benzene':
                element = 'C'  # Treat 'benzene' as carbon for atom creation
            atom = Chem.Atom(element)
            idx = mol.AddAtom(atom)
            atom_index[node] = idx

        for start_node, end_node, bond in bonds:
            bond_type = bond.get('bond_type', 1)
            if bond_type == 1:
                chem_bond_type = Chem.BondType.SINGLE
            elif bond_type == 2:
                chem_bond_type = Chem.BondType.DOUBLE
            else:
                chem_bond_type = Chem.BondType.SINGLE  # Default to single bond
            mol.AddBond(atom_index[start_node], atom_index[end_node], chem_bond_type)

        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        #print(f"Error in converting graph to SMILES: {e}")
        return None

def evaluate_molecule(graph, episode, num_episodes):
    smiles = nx_to_smiles(graph)
    if smiles is None:
        return -30, None  # Invalid molecule penalty

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -30, smiles  # Invalid molecule penalty, but return SMILES for debugging

    try:
        rdmolops.SanitizeMol(mol)
    except Exception as e:
        return -30, smiles  # Invalid molecule penalty, but return SMILES for debugging

    molecular_weight = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    num_atoms = mol.GetNumAtoms()
    num_rings = mol.GetRingInfo().NumRings()

    weight_reward = -min(10, molecular_weight /50)  
    logP_reward = -min(20, abs(logP))  
    atom_reward = min(50, 1.21**num_atoms) 
    ring_reward = min(50, num_rings * 25)

    total_reward = weight_reward + logP_reward + atom_reward + ring_reward
    total_reward = max(-100, min(150, total_reward))

    return total_reward, smiles

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    elements = nx.get_node_attributes(graph, 'element')
    labels = {node: f"{node}: {elements[node]}" for node in graph.nodes}
    colors = [graph.nodes[n]['element'] for n in graph.nodes]
    color_map = {'C': 'green', 'H': 'white', 'O': 'red', 'N': 'blue'}
    node_colors = [color_map.get(element, 'gray') for element in colors]
    nx.draw(graph, pos, labels=labels, with_labels=True, node_color=node_colors, edge_color='black', node_size=700, font_size=12)
    plt.show()
