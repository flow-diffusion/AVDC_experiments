import json
import argparse
import os
import  numpy as np

def main(args):
    results_root = args.results_root
    out_file = os.path.join(results_root, 'overall_results.json')

    # {results_root}/result_dict.json
    with open(os.path.join(results_root, 'result_dict.json'), 'r') as f:
        raw_result = json.load(f)

    results = {}
    overall = []

    for task, task_result in raw_result.items():
        task_dict = {}
        srs = task_result['success_rates']
        overall += srs
        task_dict['mean'] = np.mean(srs)
        task_dict['raw_success_rates'] = srs
        results[task] = task_dict
    
    results['overall'] = np.mean(overall)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='../results/results_AVDC_full', type=str)
    args = parser.parse_args()
    main(args)