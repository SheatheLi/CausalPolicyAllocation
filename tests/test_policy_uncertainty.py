from src.policy_uncertainty import paired_gate
import numpy as np

def test_positive_strong_paired_lift():
    x=np.linspace(0,1,180)
    g=paired_gate(x+.2,x,seed=5)
    assert g.accepted and abs(g.lift-.2)<1e-12

def test_negative_effect_rejected():
    x=np.linspace(0,1,180)
    assert not paired_gate(x-.01,x).accepted

def test_low_support_refuses_positive():
    a=np.ones(180);b=np.zeros(180);w=np.array([1.]+[.001]*179)
    assert not paired_gate(a,b,w).accepted
