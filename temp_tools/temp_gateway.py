"""
Original code to extract model from .mat file. 
This code has been refactored and added to matlab_model_gateway.py
Use it a a reference to test the refactored code.
"""

from __future__ import absolute_import

import re
from collections import OrderedDict
from uuid import uuid4
from warnings import warn

import numpy as np
from numpy import array, inf, isinf

from cobra.core import Metabolite, Model, Reaction
from cobra.util import create_stoichiometric_matrix
from cobra.util.solver import set_objective


try:
    from scipy import io as scipy_io
    from scipy import sparse as scipy_sparse
except ImportError:
    scipy_sparse = None
    scipy_io = None


def load_matlab_model(infile_path, variable_name=None, inf=inf):
    """Load a cobra model stored as a .mat file.

    Parameters
    ----------
    infile_path: str
        path to the file to to read
    variable_name: str, optional
        The variable name of the model in the .mat file. If this is not
        specified, then the first MATLAB variable which looks like a COBRA
        model will be used
    inf: value
        The value to use for infinite bounds. Some solvers do not handle
        infinite values so for using those, set this to a high numeric value.

    Returns
    -------
    cobra.core.Model.Model:
        The resulting cobra model

    """
    if not scipy_io:
        raise ImportError("load_matlab_model requires scipy")

    data = scipy_io.loadmat(infile_path)
    possible_names = []
    if variable_name is None:
        # skip meta variables
        meta_vars = {"__globals__", "__header__", "__version__"}
        possible_names = sorted(i for i in data if i not in meta_vars)
        if len(possible_names) == 1:
            variable_name = possible_names[0]
    if variable_name is not None:
        return from_mat_struct(data[variable_name], model_id=variable_name, inf=inf)
    for possible_name in possible_names:
        try:
            return from_mat_struct(data[possible_name], model_id=possible_name, inf=inf)
        except ValueError:
            pass
    # If code here is executed, then no model was found.
    raise IOError("no COBRA model found")


def from_mat_struct(mat_struct, model_id=None, inf=inf):
    """Create a model from the COBRA toolbox struct.

    The struct will be a dict read in by scipy.io.loadmat

    """
    m = mat_struct
    if m.dtype.names is None:
        raise ValueError("not a valid mat struct")
    if not {"rxns", "mets", "S", "lb", "ub"} <= set(m.dtype.names):
        raise ValueError("not a valid mat struct")
    if "c" in m.dtype.names:
        c_vec = m["c"][0, 0]
    else:
        c_vec = None
        warn("objective vector 'c' not found")
    model = Model()
    if model_id is not None:
        model.id = model_id
    elif "description" in m.dtype.names:
        description = m["description"][0, 0][0]
        if not isinstance(description, str) and len(description) > 1:
            model.id = description[0]
            warn("Several IDs detected, only using the first.")
        else:
            model.id = description
    else:
        model.id = "imported_model"
    for i, name in enumerate(m["mets"][0, 0]):
        new_metabolite = Metabolite()
        new_metabolite.id = str(name[0][0])
        if all(var in m.dtype.names for var in ["metComps", "comps", "compNames"]):
            comp_index = m["metComps"][0, 0][i][0] - 1
            new_metabolite.compartment = m["comps"][0, 0][comp_index][0][0]
            if new_metabolite.compartment not in model.compartments:
                comp_name = m["compNames"][0, 0][comp_index][0][0]
                model.compartments[new_metabolite.compartment] = comp_name
        else:
            new_metabolite.compartment = _get_id_compartment(new_metabolite.id)
            if new_metabolite.compartment not in model.compartments:
                model.compartments[new_metabolite.compartment] = (
                    new_metabolite.compartment
                )
        try:
            new_metabolite.name = str(m["metNames"][0, 0][i][0][0])
        except (IndexError, ValueError):
            pass
        try:
            new_metabolite.formula = str(m["metFormulas"][0][0][i][0][0])
        except (IndexError, ValueError):
            pass
        try:
            new_metabolite.charge = float(m["metCharge"][0, 0][i][0])
            int_charge = int(new_metabolite.charge)
            if new_metabolite.charge == int_charge:
                new_metabolite.charge = int_charge
        except (IndexError, ValueError):
            pass
        model.add_metabolites([new_metabolite])
    new_reactions = []
    coefficients = {}
    for i, name in enumerate(m["rxns"][0, 0]):
        new_reaction = Reaction()
        new_reaction.id = str(name[0][0])
        new_reaction.lower_bound = float(m["lb"][0, 0][i][0])
        new_reaction.upper_bound = float(m["ub"][0, 0][i][0])
        if isinf(new_reaction.lower_bound) and new_reaction.lower_bound < 0:
            new_reaction.lower_bound = -inf
        if isinf(new_reaction.upper_bound) and new_reaction.upper_bound > 0:
            new_reaction.upper_bound = inf
        if c_vec is not None:
            coefficients[new_reaction] = float(c_vec[i][0])
        try:
            new_reaction.gene_reaction_rule = str(m["grRules"][0, 0][i][0][0])
        except (IndexError, ValueError):
            pass
        try:
            new_reaction.name = str(m["rxnNames"][0, 0][i][0][0])
        except (IndexError, ValueError):
            pass
        try:
            new_reaction.subsystem = str(m["subSystems"][0, 0][i][0][0])
        except (IndexError, ValueError):
            pass
        new_reactions.append(new_reaction)
    model.add_reactions(new_reactions)
    set_objective(model, coefficients)
    coo = scipy_sparse.coo_matrix(m["S"][0, 0])
    for i, j, v in zip(coo.row, coo.col, coo.data):
        model.reactions[j].add_metabolites({model.metabolites[i]: v})
    return model


# precompiled regular expressions
_bracket_re = re.compile(r"\[(?P<compartment>[a-z]+)\]$")
_underscore_re = re.compile(r"_(?P<compartment>[a-z]+)$")


def _get_id_compartment(id):
    """Extract the compartment from the id string."""
    bracket_search = _bracket_re.search(id)
    if bracket_search:
        return bracket_search.group("compartment")
    underscore_search = _underscore_re.search(id)
    if underscore_search:
        return underscore_search.group("compartment")
    return None
