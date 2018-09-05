# -*- coding: utf-8 -*-
'''
    This maps the names of IEEE and FCL options to their implementation.
    For the IEEE names I'm working from the XML standard (IEEE 1855-2016).
    For the FCL names I'm following fuzzylite/src/imex/FclImporter.cpp
    I only record FCL names if they're different from the IEEE ones.

    @author: james.power@mu.ie Created on Wed Aug 22 11:59:59 2018
'''

from collections import OrderedDict
import numpy as np

import skfuzzy.membership as fuzzmf
import skfuzzy.control.fuzzyvariable as fuzzvar
import skfuzzy.control as ctrl
import skfuzzy.control.term as fuzzterm

import extramf
import tnorms

# ############################
# ### Membership functions ###
# ############################

# Return skfuzzy version, or one of the extras:

_IEEE_MF = {  # IEEE name: (fuzz-mf,  split-parameters?)
    'triangular':    (fuzzmf.trimf, False),
    'rightlinear':   (extramf.rightlinearmf, True),
    'leftlinear':    (extramf.leftlinearmf, True),
    'pi':            (fuzzmf.pimf, True),
    'gaussian':      (fuzzmf.gaussmf, True),
    'rightgaussian': (extramf.rightgaussmf, True),
    'leftgaussian':  (extramf.leftgaussmf, True),
    'trapezoid':     (fuzzmf.trapmf, False),
    's':             (fuzzmf.smf, True),
    'z':             (fuzzmf.zmf, True),
    'rectangular':   (extramf.rectanglemf, True),
    'singleton':     (extramf.singletonmf, True),
    'pointset':      (extramf.pointsetmf, False),
}

# jFuzzyLogic likes these names:
_JFUZZYLOGIC_MF = {
    'trian': (fuzzmf.trimf, False),
    'trape': (fuzzmf.trapmf, False),
    'gauss': (fuzzmf.gaussmf, True),
    'gauss2': (fuzzmf.gauss2mf, True),
    'gbell': (fuzzmf.gbellmf, True),
    'sigm':  (extramf.jfl_sigmf, True),
}

# These are some other MFs I found, mostly from fuzzylite
_FCL_MF = {  # FCL name: (fuzz-mf,  split-parameters?)
    'bell':              (extramf.fl_bellmf, True),
    'concave':           (extramf.concavemf, True),
    'cosine':            (extramf.cosinemf, True),
    'gaussianproduct':   (extramf.gaussprod, True),
    'pishape':           (fuzzmf.pimf, True),
    'pointlist':         (extramf.pointsetmf, False),
    'ramp':              (extramf.rampmf, True),
    'rectangle':         (extramf.rectanglemf, True),
    'sigmoid':           (fuzzmf.sigmf, True),
    'sigmoiddifference': (fuzzmf.dsigmf, True),
    'sigmoidproduct':    (fuzzmf.psigmf, True),
    'spike':             (extramf.spikemf, True),
    'sshape':            (fuzzmf.smf, True),
    'triangle':          (fuzzmf.trimf, False),
    'zshape':            (fuzzmf.zmf, True),
}

# ################################
# ### Defuzzification methods: ###
# ################################

# return a string that skfuzzy.defuzzify.defuzz() can be called with.

_IEEE_DEFUZZ = {
    'cog': 'centroid',
    'coa': 'bisector',
    'lm':  'som',
    'rm':  'lom',
    'mom': 'mom'
}

_FCL_DEFUZZ = {
    'mm':  'mom',
    'cogs': 'centroid',
    # 'cogs':  WeightedAverage, not implemented
    # 'cogss': WeightedSum, not implemented
}


# #####################################
# ### Aggregation (AND/OR) methods: ###
# #####################################

# Note that these all return a FuzzyAggregationMethods object
# that is, you get both and/or when you lookup either one of them.

_IEEE_AND = {
    'min':    tnorms.min_max(),
    'prod':   tnorms.product_sum(),
    'bdif':   tnorms.bounded(),
    'drp':    tnorms.drastic(),
    'eprod':  tnorms.einstein(),
    'hprod':  tnorms.hamacher(),
    'nilmin': tnorms.nilpotent(),
}

_IEEE_OR = {
    'max':    tnorms.min_max(),
    'probor': tnorms.product_sum(),
    'bsum':   tnorms.bounded(),
    'drs':    tnorms.drastic(),
    'esum':   tnorms.einstein(),
    'hsum':   tnorms.hamacher(),
    'nilmax': tnorms.nilpotent(),
}

_FCL_AND = {
    'dprod':  tnorms.drastic(),
    'nmin':   tnorms.nilpotent(),
}

_FCL_OR = {
    'asum':   tnorms.product_sum(),  # 'algebraic sum'
    'dsum':   tnorms.drastic(),
    # 'nsum' is not implemented
    'nmax':   tnorms.nilpotent(),
}

_JFUZZYLOGIC_AND = {
    'dmin':    tnorms.drastic(),
    'hamacher':  tnorms.hamacher(),
    'nipmin': tnorms.nilpotent(),

}

_JFUZZYLOGIC_OR = {
    'asum':   tnorms.product_sum(),  # 'algebraic sum'
    'dmax':    tnorms.drastic(),
    'einstein':  tnorms.einstein(),
    'nipmax': tnorms.nilpotent(),
}


# ######################################
# ### Class to map names to objects: ###
# ######################################

class NameMapper(object):
    '''
        Just three dicts, mapping names to: mfs, defuzz methods and norms.
        These are loaded up with the IEEE and FCL names
    '''
    def __init__(self):
        '''
            Initialise lists of known mfs, defuzz methods and and/or methods.
            Can load in names from IEEE XML standard as well as FCL.
        '''
        self.known_mfs = {}       # Membership functions
        self.defuzz_methods = {}  # Defuzzification methods
        self.and_names = {}       # And function to be used in rules
        self.or_names = {}        # Or function to be used in rules

    def load_ieee_names(self):
        '''Load in the names used by the IEEE (XML) standard'''
        self.known_mfs.update(_IEEE_MF)
        self.defuzz_methods.update(_IEEE_DEFUZZ)
        self.and_names.update(_IEEE_AND)
        self.or_names.update(_IEEE_OR)

    def load_fcl_names_too(self):
        '''
            Load in the names used by the IEC 1131-7 (FCL) draft standard
            Note: we assume you've already loaded in the IEEE names.
        '''
        self.known_mfs.update(_FCL_MF)
        self.defuzz_methods.update(_FCL_DEFUZZ)
        self.and_names.update(_FCL_AND)
        self.or_names.update(_FCL_OR)

    def load_jfl_names(self):
        self.known_mfs.update(_JFUZZYLOGIC_MF)
        self.and_names.update(_JFUZZYLOGIC_AND)
        self.or_names.update(_JFUZZYLOGIC_OR)

    def _report_error(self, msg, kind, pos=None):
        '''Simple error reporter (so override me)'''
        assert False, '{}: {}'.format(kind, msg)

    def _unsupported(self, msg):
        '''Raise an 'unsupported feature' error at the current position'''
        self._report_error(msg, 'unsupported feature')

    def translate_mf(self, mf_name):
        '''Translate a member-function name to an actual function'''
        if mf_name.lower() in self.known_mfs:
            return self.known_mfs[mf_name.lower()]
        else:
            self._unsupported('membership function "{}"'.format(mf_name))

    def translate_defuzz(self, df_name):
        '''Translate a given defuzz method to its skfuzzy name'''
        if df_name.lower() in self.defuzz_methods:
            return self.defuzz_methods[df_name.lower()]
        else:
            self._unsupported('defuzzify method "{}"'.format(df_name))

    def translate_and_or(self, and_name, or_name):
        '''
            Get the and/or function corresponding to the given names.
            If only one specified, the other will be its dual method.
            If both are specified, then take both, even if not dual.
            Return a FuzzyAggregationMethods object with both functions.
        '''
        # First check that both names, if specified, are valid:
        if and_name and and_name.lower() not in self.and_names:
            self._unsupported('and method "{}"'.format(and_name))
        if or_name and or_name.lower() not in self.or_names:
            self._unsupported('and method "{}"'.format(or_name))
        # Set up the default (is actually min/max):
        fam = fuzzterm.FuzzyAggregationMethods()
        # Now see if one/both have been specified
        if and_name and or_name:  # Set both separately:
            fam.and_func = self.and_names[and_name.lower()].and_func
            fam.or_func = self.or_names[or_name.lower()].or_func
        elif and_name:
            fam = self.and_names[and_name.lower()]
        elif or_name:
            fam = self.or_names[or_name.lower()]
        return fam


# #######################################
# ### Symbol Table for use by parser: ###
# #######################################

class SymbolTable(object):
    '''
        A very simple symbol table with a list of variables and rules.
        The interface mirros some methods of skfuzzy.control.ControlSystem
    '''
    def __init__(self, varlist=None):
        '''Set up an empty symbol table; optionally supply list of variables'''
        self.fb_name = None   # Name of function block (if any in file)
        self.variables = OrderedDict()   # Map variable label to FuzzyVariable
        self.all_rules = OrderedDict()   # Map rule label to Rule object
        if varlist:
            self.add_vars(varlist)

    def clear(self):
        ''' Empty all items in the symbol table'''
        self.fb_name = None
        self.variables.clear()
        self.all_rules.clear()

    def _report_error(self, msg, kind, pos=None):
        '''Simple error reporter (so override me)'''
        assert False, '{}: {}'.format(kind, msg)

    def add_vars(self, varlist):
        '''
            Add these variables to the set of those known to us.
            Will overwrite any previous variables with these names.
        '''
        for fvar in varlist:
            assert isinstance(fvar, fuzzvar.FuzzyVariable),\
                '{} should be a variable'.format(fvar)
            self.variables[fvar.label] = fvar

    def get_var_defn(self, varname):
        '''
            Gives the variable definition for this name; error if not there.
        '''
        if varname not in self.variables:
            self._report_error('Variable "{}" not found'.format(varname),
                               'scope error')
        return self.variables[varname]

    @property
    def antecedents(self):
        """Generator which yields Antecedents in the system."""
        for node in self.variables.values():
            if isinstance(node, ctrl.Antecedent):
                yield node

    @property
    def consequents(self):
        """Generator which yields Consequents in the system."""
        for node in self.variables.values():
            if isinstance(node, ctrl.Consequent):
                yield node

    @property
    def fuzzy_variables(self):
        '''Return an iterator over all the variable objects'''
        return self.variables.values()

    @property
    def rules(self):
        '''Return an iterator over all the rule objects'''
        return self.all_rules.values()

    def add_rule(self, rule):
        '''
            Add this rule to the list of those known to us.
            Will overwrite any previous rule with the same label.
        '''
        assert isinstance(rule, ctrl.Rule),\
            '{} should be a rule object'.format(rule)
        self.all_rules[rule.label] = rule
        return rule

    def set_rule_label(self, rule, new_label):
        '''
            Changing the rule label has consequences for our dict,
            so use this method rather than setting it directly.
            Will overwrite any previous rule with the same label.
        '''
        # Remove the old-labelled version, if there is one:
        self.all_rules.pop(rule.label, None)
        rule.label = new_label
        self.add_rule(rule)

    def __getitem__(self, key):
        '''
            Allows the parser to be accessed as a dict;
            the key should be a variable or rule name,
            returns the definition corresponding to that name (or error).
        '''
        if key in self.variables:
            return self.variables[key]
        elif key in self.all_rules:
            return self.all_rules[key]
        else:
            self._report_error('"{}" is not a known variable or rule name'
                               .format(key), 'scope error')

    def __str__(self):
        pstr = ''
        if self.fb_name:
            pstr += 'Function-Block "{}"\n'.format(self.fb_name)
        for var in self.fuzzy_variables:
            lo, hi = np.min(var.universe), np.max(var.universe)
            pstr += '{}, range := ({} .. {})\n'.format(var, lo, hi)
            pstr += '{}terms: {}\n'.format(' '*12, [t for t in var.terms])
        for rule in self.rules:
            pstr += 'Rule {}: {}\n'.format(rule.label, rule)
        return pstr
