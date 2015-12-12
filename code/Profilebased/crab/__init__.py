"""
Crab
Recommendation System module in python
=================================

scikits.recommender is a Python module integrating classique recommendation
algorithms in the tightly-nit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to recommendation problems
that are accessible to everybody and reusable in various contexts:
being a flexible tool for scientific and industry purposes.

See http://github.com/muricoca/crab for complete documentation.

"""

import logging


try:
    from numpy.testing import nosetester

    class NoseTester(nosetester.NoseTester):
        """
        Subclass numpy's NoseTester to add doctests by default
        """
        def test(self, label='fast', verbose=1, extra_argv=['--exe'],
                        doctests=True, coverage=False):
            return super(NoseTester, self).test(label=label, verbose=verbose,
                                    extra_argv=extra_argv,
                                    doctests=doctests, coverage=coverage)
    test = NoseTester().test
    del nosetester
except:
    pass

__all__ = ['datasets', 'metrics', 'similarities', 'models', 'recommenders']


__version__ = '0.1.git'


class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""
    def emit(self, record):
        pass

logger = logging.getLogger('crab')
if len(logger.handlers) == 0:   # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
