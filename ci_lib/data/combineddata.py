'''Contains the `CombinedData` class'''

import logging
from .decompdata import DecompData
from .trialdata import TrialData


LOGGER = logging.getLogger(__name__)

class CombinedData(DecompData, TrialData):
    '''The standard class for handeling data decomposed data with trial data'''
