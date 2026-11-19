import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.decision_pipeline import load_logged_trial,DataContractError

@pytest.mark.parametrize('threshold',[0,-.01, .34,float('nan'),float('inf')])
def test_invalid_positivity_configuration_fails_closed(threshold):
    with pytest.raises(DataContractError,match='min_propensity'):
        load_logged_trial('nonexistent.csv',min_propensity=threshold)
