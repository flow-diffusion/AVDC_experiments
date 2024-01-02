import json
import argparse
import os

def main(args):
    result_root = args.result_root
    out_file = os.path.join(result_root, 'overall_results.json')

    # {result_root}/{task}/result.json

    rooms = os.listdir(result_root)

    results = {}
    overall = []
    for room in rooms:
        room_result = {}
        srs = []
        print(result_root)
        tasks = os.listdir(os.path.join(result_root, room))
        for task in tasks:
            result_file = os.path.join(result_root, room, task, 'result.json')
            if not os.path.exists(result_file):
                continue
            with open(result_file, 'r') as f:
                result = json.load(f)
            sr = result['success_rate']
            srs.append(sr)
            room_result[task] = sr

        room_result['mean'] = sum(srs) / len(srs)
        results[room] = room_result
        overall += srs

    results['overall'] = sum(overall) / len(overall)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', default='../results/results_AVDC_thor', type=str)
    args = parser.parse_args()
    main(args)