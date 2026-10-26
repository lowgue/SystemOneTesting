"""Build a visually reviewed multi-attribute evaluation; never use model predictions as labels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'eval/datasets/test_7_complete_tags_manifest.json'

def main():
    cars = json.loads((ROOT / 'eval/datasets/real_cars_metadata.json').read_text())
    colors = ['white','red','gray_silver','blue','gray_silver','gray_silver','gray_silver','white','white','red','black','gray_silver','blue','gray_silver','black','gray_silver','black','red','white','red']
    items = []
    for car, color in zip(cars, colors, strict=True):
        make = 'land_rover' if car['filename'] == 'real_ford_2.jpg' else car['make']
        items.append(dict(id=car['filename'], domain='vehicle', rel_path='eval/images/cars/'+car['filename'],
                          expected=dict(presence='yes', make=make, color=color, body=car['body_type']),
                          annotation='Visual color review; existing make/body metadata, corrected Land Rover make. Gray includes silver.'))
    for name in ['no_car_landscape.jpg','no_car_field.jpg']:
        items.append(dict(id='vehicle_'+name, domain='vehicle', rel_path='eval/images/cars/'+name,
                          expected=dict(presence='no',make='unknown',color='unknown',body='unknown'), annotation='Reviewed negative control.'))
    # Labels reviewed from actual images before inference, rather than Commons category defaults.
    people = {
        9: ('yes','not_visible','not_visible'),
        10: ('yes','none','shirt_tshirt'),
        13: ('yes','not_visible','shirt_tshirt'),
        17: ('no','backpack','jacket_coat'),
        18: ('no','none','jacket_coat'),
        21: ('no','backpack','jacket_coat'),
        23: ('no','bag_or_box','shirt_tshirt'),
        26: ('no','none','safety_vest'),
        28: ('yes','none','safety_vest'),
        29: ('not_visible','none','safety_vest'),
        48: ('no','bag_or_box','jacket_coat'),
        50: ('not_visible','not_visible','not_visible'),
    }
    original = json.loads((ROOT / 'eval/datasets/test_3_person_attributes_manifest.json').read_text())['items']
    for num in [*people,51,52,54,55,56,59,60]:
        old = original[num-1]
        vals = people.get(num, ('not_visible',)*3)
        expected = dict(zip(('helmet','carrying','clothing'),vals))
        expected['presence'] = 'yes' if num in people else 'no'
        items.append(dict(id=old['id'],domain='person',rel_path=old['rel_path'],expected=expected,
                          annotation='Independent visual review before inference; dominant person; distant attributes explicitly not_visible.'))
    for item in items:
        item['sha256'] = hashlib.sha256((ROOT/item['rel_path']).read_bytes()).hexdigest()
    doc = dict(test='Test 7: Complete attribute bundles', annotation_method='Agent visual review, not human adjudication; no predictions used. Existing vehicle body labels retained.',
               limitations=['Convenience sample reused from previous experiments; not an unseen test set.',
                            'Whole images with dominant target; detector/crop pipeline is not evaluated.',
                            'No CaptionerManager implementation available: contract is four fields per domain.',
                            'No confidence thresholds tuned; probabilities are not assumed calibrated.'],items=items)
    OUT.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')
    print(f'{OUT}: {len(items)} images, {4*len(items)} labels')

if __name__ == '__main__': main()
