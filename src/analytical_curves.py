"""Exact Poisson support probabilities and sparse-limit information curves."""
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_design import DESIGNS, leading_information_constant, poisson_informative_probability

def exact_curves(out):
    rows=[]
    for name,spec in DESIGNS.items():
        for mean in [.25,1,3,10]:
            coeff=leading_information_constant(np.full(len(spec['nodes']),mean),spec['nodes'])
            for q in np.geomspace(.0001,1,81):
                exact=float(poisson_informative_probability(np.full(len(spec['nodes']),mean*q),spec['nodes']))
                rows.append(dict(design=name,reference_mean_per_window=mean,retention=q,
                    informative_probability=exact,leading_probability=coeff['probability_coefficient']*q**coeff['minimum_events'],
                    leading_information=coeff['information_coefficient']*q**coeff['minimum_events'],minimum_events=coeff['minimum_events']))
    pd.DataFrame(rows).to_csv(out/'exact_retention_curves.csv',index=False)

if __name__ == '__main__':
    output = Path('results/analytical_curves')
    output.mkdir(parents=True, exist_ok=True)
    exact_curves(output)
