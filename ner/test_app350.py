import re
import sys
import yaml
import spacy
from collections import defaultdict
from pathlib import Path


def get_app350_label(practice):
    practice_tuple = practice.rsplit('_', 1)
    if practice_tuple[-1] in ["1stParty", "3rdParty"]:
        label, party = practice_tuple

        if label.endswith('IP_Address'):
            return (party, "IP_Address")
        elif label.startswith('Location'):
            return (party, "Location")
        elif label in ['Contact', 'Demographic', 'Identifier']:
            return None, None
        else:
            label = label.replace('Contact_', '')
            label = label.replace('Demographic_', '')
            label = label.replace('Identifier_', '')
            return (party, label)

    return None, None


def main():
    nlp_dir = sys.argv[1]
    yml_dir = Path(sys.argv[2])

    nlp = spacy.load(nlp_dir)

    stat_table = dict()
    cat_table = defaultdict(list)
    with open("category.yml") as f:
        for cat, pattern_list in yaml.safe_load(f).items():
            stat_table[cat] = [0, 0, 0, 0]

            for pattern in pattern_list:
                regex = re.compile(r'\b' + pattern + r'\b', flags=re.I)
                cat_table[regex].append(cat)

    for yml_path in yml_dir.glob('*.yml'):
        data = yaml.safe_load(yml_path.open())

        for segment in data['segments']:
            text = segment["segment_text"]
            doc = nlp(text)

            ground_truth = set()
            for annot in segment["annotations"]:
                _, label = get_app350_label(annot["practice"])
                if label is not None:
                    ground_truth.add(label)

            #annot = {d["practice"] for d in segment["annotations"]}
            #if len(annot) == 0:
            #    continue

            data_types = set()
            for ent in doc.ents:
                if ent.label_ == 'DATA' and ent.root.pos_ in ['NOUN', 'PROPN']:
                    data_types.add(ent.text)

            data_cats = set()
            for dt in data_types:
                for pattern, cat_list in cat_table.items():
                    if pattern.search(dt):
                        data_cats.update(cat_list)

            for cat in stat_table:
                t1 = 2 if cat in data_cats else 0
                t2 = 1 if cat in ground_truth else 0
                stat_table[cat][t1 + t2] += 1

            # print(text)
            # print(ground_truth)
            # print(data_types)
            # print(data_cats)
            # print()

            #if len(data_types) > 0:
            #    print(*data_types, sep='\n')
        for cat, (TN, FN, FP, TP) in stat_table.items():
            try:
                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
            except ZeroDivisionError:
                recall = precision = 0.0

            print(cat, recall, precision)
        print()


if __name__ == "__main__":
    main()
