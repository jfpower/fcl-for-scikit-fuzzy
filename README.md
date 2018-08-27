FCL parser with a scikit-fuzzy back-end
=======================================

This is a parser for the Fuzzy Control Language
[FCL](https://en.wikipedia.org/wiki/Fuzzy_Control_Language)
which has a back-end for `scikit-fuzzy`, a fuzzy logic toolkit for SciPy.

The basic use-case is to parse a FCL file and then use the fuzzy rules
in your `scikit-fuzzy` code.  For example:

```python
from fcl_parser import FCLParser

p = FCLParser()    # Create the parser
p.read_fcl_file('tipper.fcl')  # Parse a file

# ... and so on, as usual for skfuzzy:
cs = ctrl.ControlSystem(p.rules)

```

After reading a file the parser object has attributes to supply the
`rules` (as above) or the `antecedents`, the `consequents`, or all the
`fuzzy_variables` All these are represented via lists of their
corresponding `scikit-fuzzy` objects.


Other Entry Points
------------------

The parser can be used to accept program fragments, so you can
interleave its use with regular `scikit-fuzzy` code.

For example, in the following `scikit-fuzzy` code we set up the
tipping example in the usual way by specifying the variables and
defining some membership functions for the inputs:

```python
# First we set up the variables in the usual way:
food = ctrl.Antecedent(np.linspace(0, 10, 11), 'quality')
service = ctrl.Antecedent(np.linspace(0, 10, 11), 'service')
tip = ctrl.Consequent(np.linspace(0, 25, 26), 'tip')

# Auto-generate the membership functions for the inputs:
food.automf(3)
service.automf(3)
```

We can define the output variable using FCL code, in this case getting
the parser to parse a membership function `mf` definition:

```python
# Define a FCL parser-object:
p = FCLParser()
# Use FCL to define membership functions for the output:
tip['bad'] = p.mf('Triangle 0 0 13', tip.universe)
tip['middling'] = p.mf('Triangle 0 13 25', tip.universe)
tip['lots'] = p.mf('Triangle 13 25 25', tip.universe)

```
                        
Last, we can define the rules in FCL, and get a `scikit-fuzzy` rule
object for each of them if we like:

```python
# We need to tell the parser about the variables before we parse any rules:
p.add_vars([food, service, tip])

# Now use FCL to define three rules:
rule1 = p.rule('IF quality is poor OR service is poor THEN tip is bad')
rule2 = p.rule('IF service is average THEN tip is middling')
rule3 = p.rule('IF service is good OR quality is good tHEN tip is lots')

# To get the control system, just add the rules (from the parser):
tipping = ctrl.ControlSystem(p.rules)
```

There are some more examples of mixed FCL/skfuzzy use in the file
[tests/test_fcl_parser.py](./tests/test_fcl_parser.py)


Dependencies
------------

The scanner is written using
[PLY](http://www.dabeaz.com/ply/ply.html),
so you need to install PLY before the code here will work.

     $ pip install ply

You don't need to import this anywhere, my scanner code just needs it.
The parser is hand-written so we don't actually use the
parser-generation features of PLY.


What's implemented
------------------

Basically the subset of FCL that can be translated easily into
`scikit-fuzzy`.  That includes most parts of a standard
(Mamdani-style) fuzzy system.

At the moment the main options are for:
  * defuzzification methods: cog, coa, lm, rm, mom
  * membership functions: quite a collection; have a look in
  [fcl_symbols.py](./fcl_symbols.py) for a list.
  * and/or methods (norms and co-norms): again, quite a few,
  including (norms) min, prod, bdif, drp, eprod, hprod, nilmin
  and their co-norm duals.

I was doing this with an eye on the XML standard, hence the rather
large selection of membership functions and norms.

Most notably _not_ implemented (yet) are:
  * activation methods (hard-wired to `MIN`)
  * accumulation methods (hard-wired to `MAX`)
  * default values for variables

The parser accepts these, I just haven't figured out how to get them
into the `scikit-fuzzy` code, so they will be ignored for the moment.

Compliance
----------

First of all, I'm working from the draft of the FCL standard (IEC
TC65/WG 7/TF8), plus any examples I could find, so I may have missed a
few things.

Second, the parser does not enforce strict conformance to the FCL standard,
and is somewhat liberal in the kind of FCL code it will accept.

In particular:
  * Case is not relevant for keywords
  (so `Rule`, `rule`, `RULE` are all the same)
  but note that it is relevant for identifiers (e.g. variable names).
  * The semi-colon at the end of lines can be left out in most cases
  * The parser doesn't impose a strict ordering on the contents of (say)
  variable definitions, so you can mix `TERM`, `RANGE`, `METHOD`
  etc. in your preferred order.




