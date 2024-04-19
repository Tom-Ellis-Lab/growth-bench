import re
import warnings

from typing import Union

import cobra
from cobra.util import solver as cobra_solver
import scipy
import numpy as np


# precompiled regular expressions
_bracket_re = re.compile(r"\[(?P<compartment>[a-z]+)\]$")
_underscore_re = re.compile(r"_(?P<compartment>[a-z]+)$")


def load_model_from_mat(
    file_path, variable_name=None, inf=float("inf")
) -> Union[cobra.Model, None]:
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

    # Load the MATLAB file
    try:
        mat_data: np.ndarray = scipy.io.loadmat(file_path)
    except ImportError as e:
        raise ImportError(f"Failed to load the .mat file: {e}")

    meta_vars = {"__globals__", "__header__", "__version__"}
    model_keys = [name for name in mat_data if name not in meta_vars]

    if variable_name:
        if variable_name in mat_data:
            result = create_cobra_model_from_dict(
                mat_data[variable_name], model_id=variable_name, inf=inf
            )
            return result
        else:
            raise IOError(f"Variable '{variable_name}' not found in the file.")

    for key in model_keys:
        try:
            return create_cobra_model_from_dict(mat_data[key], model_id=key, inf=inf)
        except ValueError:
            continue  # Try the next possible model variable if current one fails.
    raise IOError("No COBRA model found in the file.")


def create_cobra_model_from_dict(
    dict_model: np.ndarray, model_id=None, inf=float("inf")
) -> cobra.Model:
    """Create a model from the COBRA toolbox struct.

    Parameters
    ----------
    dict_model: np.ndarray
        The MATLAB struct containing the model
    model_id: str, optional
        The ID to use for the model. If not provided, the ID will be taken
    inf: value
        The value to use for infinite bounds. Some solvers do not handle
        infinite values so for using those, set this to a high numeric value.

    Returns
    -------
    cobra.core.Model.Model:
        The resulting cobra model

    """

    _is_dict_model_with_fields(dict_model=dict_model)
    c_vec = _get_objective_vector_c(dict_model=dict_model)

    cobra_model = cobra.Model()
    _set_model_id(model=cobra_model, mat_struct=dict_model, model_id=model_id)

    # Adding metabolites
    for index, name in enumerate(dict_model["mets"][0, 0]):
        new_metabolite = cobra.Metabolite()
        new_metabolite.id = str(name[0][0])

        _set_metabolite_compartment(
            cobra_model=cobra_model,
            metabolite=new_metabolite,
            mat_struct=dict_model,
            index=index,
        )
        _add_metabolite_properties(
            dict_model=dict_model, metabolite=new_metabolite, index=index
        )
        cobra_model.add_metabolites([new_metabolite])

    # Adding reactions
    new_reactions = []
    coefficients = {}
    for index, name in enumerate(dict_model["rxns"][0, 0]):
        new_reaction = cobra.Reaction()
        new_reaction.id = str(name[0][0])
        # Set reaction bounds
        _set_reaction_bounds(
            reaction=new_reaction,
            lower_bound=dict_model["lb"][0, 0][index][0],
            upper_bound=dict_model["ub"][0, 0][index][0],
            inf=inf,
        )
        # Set additional reaction properties
        _set_reaction_properties(
            reaction=new_reaction, dict_model=dict_model, index=index
        )
        # Set coefficients if c_vec is provided
        if c_vec is not None:
            coefficients[new_reaction] = float(c_vec[index][0])

        # Add the new reaction to the list
        new_reactions.append(new_reaction)
    cobra_model.add_reactions(new_reactions)

    # Set the objective coefficients
    cobra_solver.set_objective(cobra_model, coefficients)

    # Adding stoichiometry
    coo = scipy.sparse.coo_matrix(dict_model["S"][0, 0])
    for i, j, v in zip(coo.row, coo.col, coo.data):
        cobra_model.reactions[j].add_metabolites({cobra_model.metabolites[i]: v})

    return cobra_model


def _is_dict_model_with_fields(dict_model: np.ndarray) -> bool:
    """Check if dict_model is a valid structured array with the required fields.

    Parameters
    ----------
    dict_model: np.ndarray
        The MATLAB struct containing the model

    Returns
    -------
    bool:
        True if the struct is a valid COBRA model, False otherwise
    """
    required_fields = {"rxns", "mets", "S", "lb", "ub"}
    if dict_model.dtype.names is None or not required_fields <= set(
        dict_model.dtype.names
    ):
        raise ValueError("not a valid mat struct")
    return True


def _get_objective_vector_c(dict_model: np.ndarray) -> Union[np.ndarray, None]:
    """Attempt to extract the objective vector 'c' from dict_model

    Parameters
    ----------
    dict_model: np.ndarray
        The MATLAB struct containing the model

    Returns
    -------
    np.ndarray:
        The objective vector 'c' if it exists, None otherwise
    """
    if "c" in dict_model.dtype.names:
        result = dict_model["c"][0, 0]  # Extract 'c' if it exists
    else:
        result = None  # Set c_vec to None if 'c' is not found in the structure
        warnings.warn("objective vector 'c' not found")
    return result


def _set_model_id(
    model: cobra.Model, mat_struct: np.ndarray, model_id: Union[str, None]
) -> None:
    """Set the model ID based on the provided model_id or description from mat_struct.

    Parameters
    ----------
    model: cobra.Model
        The model to set the ID for
    mat_struct: np.ndarray
        The MATLAB struct containing the model
    model_id: str
        The ID to use for the model. If not provided, the ID will be taken

    Returns
    -------
    None
    """
    if model_id is not None:
        model.id = model_id
    elif "description" in mat_struct.dtype.names:
        description = mat_struct["description"][0, 0][0]
        if not isinstance(description, str) and len(description) > 1:
            model.id = description[0]
            warnings.warn("Several IDs detected, only using the first.")
        else:
            model.id = description
    else:
        model.id = "imported_model"


def _set_metabolite_compartment(
    cobra_model: cobra.Model,
    metabolite: cobra.Metabolite,
    mat_struct: np.ndarray,
    index: int,
) -> None:
    """Set the compartment for a metabolite, fetching names if available

    Parameters
    ----------
    metabolite : cobra.Metabolite
        The metabolite to set the compartment for
    mat_struct : np.ndarray
        The MATLAB struct containing the model
    index : int
        The index of the metabolite in the struct

    Returns
    -------
    None
    """
    required_fields = ["metComps", "comps", "compNames"]
    if all(field in mat_struct.dtype.names for field in required_fields):
        comp_index = mat_struct["metComps"][0, 0][index][0] - 1
        compartment = mat_struct["comps"][0, 0][comp_index][0][0]
        metabolite.compartment = compartment
        if compartment not in cobra_model.compartments:
            comp_name = mat_struct["compNames"][0, 0][comp_index][0][0]
            cobra_model.compartments[compartment] = comp_name
    else:
        metabolite.compartment = _get_id_compartment(metabolite.id)
        if metabolite.compartment not in cobra_model.compartments:
            cobra_model.compartments[metabolite.compartment] = metabolite.compartment


def _get_id_compartment(id) -> Union[str, None]:
    """Extract the compartment from the id string.

    Parameters
    ----------
    id : str
        The id string to extract the compartment from.

    Returns
    -------
    str
        The compartment string extracted from the id.
    """
    bracket_search = _bracket_re.search(id)
    if bracket_search:
        return bracket_search.group("compartment")
    underscore_search = _underscore_re.search(id)
    if underscore_search:
        return underscore_search.group("compartment")
    return None


def _add_metabolite_properties(
    dict_model: np.ndarray,
    metabolite: cobra.Metabolite,
    index: int,
):
    """Set properties to the metabolite if available.

    Parameters
    ----------
    dict_model : np.ndarray
        The MATLAB struct containing the model
    metabolite : cobra.Metabolite
        The metabolite to set the properties for
    index : int
        The index of the metabolite in the struct

    Returns
    -------
    None
    """
    try:
        metabolite.name = str(dict_model["metNames"][0, 0][index][0][0])
    except (IndexError, ValueError):
        pass
    try:
        metabolite.formula = str(dict_model["metFormulas"][0][0][index][0][0])
    except (IndexError, ValueError):
        pass
    try:
        metabolite.charge = float(dict_model["metCharge"][0, 0][index][0])
        int_charge = int(metabolite.charge)
        if metabolite.charge == int_charge:
            setattr(metabolite, "charge", int_charge)
    except (IndexError, ValueError):
        pass


def _set_reaction_bounds(
    reaction: cobra.Reaction, lower_bound: float, upper_bound: float, inf: float
) -> None:
    """Set the bounds for the reaction, adjusting for infinity.

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to set the bounds for
    lower_bound : float
        The lower bound for the reaction
    upper_bound : float
        The upper bound for the reaction
    inf : value
        The value to use for infinite bounds. Some solvers do not handle
        infinite values so for using those, set this to a high numeric value.

    Returns
    -------
    None

    """
    reaction.lower_bound = float(lower_bound)
    reaction.upper_bound = float(upper_bound)
    if np.isinf(reaction.lower_bound) and reaction.lower_bound < 0:
        reaction.lower_bound = -inf
    if np.isinf(reaction.upper_bound) and reaction.upper_bound > 0:
        reaction.upper_bound = inf


def _set_reaction_properties(
    reaction: cobra.Reaction, dict_model: np.ndarray, index: int
) -> None:
    """Safely set additional properties of a reaction from the struct.

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to set the properties for
    dict_model : np.ndarray
        The MATLAB struct containing the model
    index : int
        The index of the reaction in the struct

    Returns
    -------
    None

    """
    # Properties to set if available
    properties = {
        "gene_reaction_rule": "grRules",
        "name": "rxnNames",
        "subsystem": "subSystems",
    }
    for property_name, field in properties.items():
        try:
            property_value = str(dict_model[field][0, 0][index][0][0])
            setattr(reaction, property_name, property_value)
        except (IndexError, ValueError):
            pass
