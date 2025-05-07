# Placeholder for molecule environment.
class ChemicalMolecule:
    def __init__(self, custom_gcn):
        self.graph = nx.Graph()
        self.custom_gcn = custom_gcn
        self.queue = deque()
        self.max_nodes = 50 
        self.reset()

        self.rules = {
            'C': ['C', 'H', 'O', 'N', 'benzene'],
            'H': ['C', 'H', 'O', 'N', 'benzene'],
            'O': ['C', 'H', 'O', 'N', 'benzene'],
            'N': ['C', 'H', 'O', 'N', 'benzene'],
            'benzene': ['C', 'H', 'O', 'N', 'benzene'],
        }


    def add_node(self, element, parent_node):
        self.node_counter += 1
        new_node = self.node_counter
        self.graph.add_node(new_node, element=element, degree=0, numb=new_node)
        self.graph.add_edge(parent_node, new_node, bond_type=1)
        self.queue.append(new_node)
        return new_node

    def add_benzene_ring(self, parent_node):
        ring_size = 6  # Total size of the ring
        elements = ['C'] * (ring_size - 1)
        new_nodes = [parent_node]

        # Add 5 new carbon atoms
        for i in range(ring_size - 1):
            self.node_counter += 1
            new_node = self.node_counter
            element = elements[i]
            self.graph.add_node(new_node, element=element, degree=0, numb=new_node)
            new_nodes.append(new_node)
            self.queue.append(new_node)

        # Add edges with alternating double bonds
        bond_orders = [1, 2] * 3  # Alternating single (1) and double (2) bonds
        for i in range(ring_size - 1):
            bond_type = bond_orders[i % len(bond_orders)]
            self.graph.add_edge(new_nodes[i], new_nodes[i + 1], bond_type=bond_type)

        # Close the ring
        bond_type = bond_orders[(ring_size - 1) % len(bond_orders)]
        self.graph.add_edge(new_nodes[-1], new_nodes[0], bond_type=bond_type)


    def step(self, action, continue_generation, parent_node):
        element = action

        if element == 'benzene':
            self.add_benzene_ring(parent_node)
        elif element == 'none':
            pass  
        else:
            self.add_node(element, parent_node)

        if not continue_generation:
            self.queue.popleft() 

    def reset(self):
        self.graph.clear()
        self.node_counter = 1
        elements = ['C', 'H', 'O', 'N']
        initial_element = random.choice(elements) 
        self.graph.add_node(1, element=initial_element, degree=0, numb=1)
        self.queue.clear()
        self.queue.append(1)

    def is_queue_empty(self):
        if len(self.queue) == 0:
            return True
        if self.node_counter > max_steps_per_episode:
            return True
        return False

    def get_current_node(self):
        if self.queue:
            return self.queue[0]
        else:
            return None

    def networkx_to_pyg(self):
        node_features = []

        for node in self.graph.nodes(data=True):
            one_hot_element = self.one_hot_encoding(node[1]["element"])
            node_features.append(one_hot_element)

        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_index = []
        edge_attr = []

        for edge in self.graph.edges(data=True):
            edge_index.append((edge[0]-1, edge[1]-1))
            edge_attr.append(edge[2]["bond_type"])

        if len(edge_index) == 0:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0,), dtype=torch.float)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)

        return data
    
    def one_hot_encoding(self, element):
        elements = ['C', 'H', 'O', 'N', 'benzene', 'none']
        one_hot = [0] * len(elements)
        index = elements.index(element)
        one_hot[index] = 1
        return one_hot
    
    def get_node_embeddings(self):
        data = self.networkx_to_pyg()
        node_embeddings = self.custom_gcn(data)
        return node_embeddings

    def get_state_representation(self, parent_node):
        node_embeddings = self.get_node_embeddings()
        current_node_index = parent_node - 1 
        parent_node_embedding = node_embeddings[current_node_index]

        parent_node_embedding = parent_node_embedding.unsqueeze(0)
        return parent_node_embedding


    def get_intermediate_reward(self, parent_node, child_node_type):
        if parent_node not in self.graph.nodes:
            return -10  # Penalty for non-existent node

        if child_node_type == 'none':
            # Neutral reward for 'none' action
            return 0

        parent_node_type = self.graph.nodes[parent_node]['element']
        if child_node_type in self.rules.get(parent_node_type, []):
            reward = 1
            if self.graph.number_of_nodes() <= 10:
                reward += 2
            if nx.cycle_basis(self.graph):
                reward += 5
            return reward
        else:
            return -5


    def calculate_final_reward(self, episode, num_episodes):
        return evaluate_molecule(self.graph, episode, num_episodes)


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

    weight_reward = -min(10, molecular_weight /50)  # 분자량이 클수록 패널티 감소
    logP_reward = -min(20, abs(logP))  # logP 값이 낮을수록 좋음
    atom_reward = min(50, 1.21**num_atoms)  # 원자 수가 많을수록 보상 증가, 최대값 제한
    ring_reward = min(50, num_rings * 25)  # 고리가 많을수록 보상 증가, 최대값 제한

    total_reward = weight_reward + logP_reward + atom_reward + ring_reward

    total_reward = max(-100, min(150, total_reward))

    return total_reward, smiles


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
                chem_bond_type = Chem.BondType.SINGLE 
            mol.AddBond(atom_index[start_node], atom_index[end_node], chem_bond_type)

        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        print(f"Error in converting graph to SMILES: {e}")
        return None
