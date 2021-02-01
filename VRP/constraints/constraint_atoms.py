from flloat.parser.ltlf import LTLfParser
import numpy as np


def node_visited(node, f_state_info):
    info = f_state_info['demand']
    return (info[node] <= 0)

def step_distance(f_state_info):
    cost = f_state_info['cost']
    max_cost = f_state_info['max_cost']
    return (cost <= max_cost // 3)

def node_unreachable(node, f_state_info):
    info = f_state_info['demand']
    return (info[node] > 0)


atoms = {
    'node_27_visited': (lambda x: node_visited(27, x)),
    'node_24_visited': (lambda x: node_visited(24, x)),
    'step_cost': step_distance,
    'node_24_unreachable': (lambda x: node_unreachable(24, x)),
    'node_27_unreachable': (lambda x: node_unreachable(27, x))
}

cvrp_constraints = ['G(node_24_unreachable & node_27_unreachable)',
                    'G(node_24_visited -> node_27_visited)']



# constraints = ['G(is_replenish_constraint -> ((X!is_replenish_constraint)&(XX!is_replenish_constraint)))']

def construct_formula(constraint):
    parser = LTLfParser()
    formula = parser(constraint)
    return formula


class CVPRConstraints(object):

    def __init__(self, constraint):
        self.constraint = constraint
        self.constraint_formula =construct_formula(constraint)
        self.constraint_automata = self.constraint_formula.to_automaton().determinize()
        self.num_constraint_states = len(self.constraint_automata.states)
        self.constraint_atoms = self.constraint_formula.find_labels()
        self.automata_state = self.constraint_automata.initial_state
        self.is_accepted = True

    def reset(self):
        self.is_accepted = True
        self.automata_state = self.constraint_automata.initial_state

    
    def step(self, f_state):
        def _is_automaton_state_rejected(dfa, state):
            if dfa.is_accepting(state):
                return False
            transitions = dfa.get_transitions_from(state)
            for t in transitions:
                if t[-1] != state:
                    return False
            return True
        
        if not self.is_accepted:
            self.reset()
        else:
            atom_labels = {}
            constraint = self.constraint
            _atom_label = dict()
            for atom in self.constraint_atoms:
                _atom_label[atom] = atoms[atom](f_state)
            atom_labels[constraint] = _atom_label
            self.automata_state = self.constraint_automata.get_successor(self.automata_state, _atom_label)
            if _is_automaton_state_rejected(self.constraint_automata, self.automata_state):
                self.is_accepted = False