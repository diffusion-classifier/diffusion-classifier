import pandas as pd
from diffusion.datasets import get_target_dataset

classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

templates = [
    'a blurry photo of a {}.',
]
if __name__ == '__main__':
    dataset = 'cifar10'
    target_dataset = get_target_dataset(dataset)

    prompt = [templates[0].format(cls) for cls in classes]
    classname = list(target_dataset.class_to_idx.keys())
    classidx = list(target_dataset.class_to_idx.values())

    # sanity checks
    assert len(prompt) == len(classname) == len(classidx)
    for i in range(len(prompt)):
        assert classname[i].lower().replace('_', '/') in prompt[
            i].lower(), f"{classname[i]} not found in {prompt[i].lower()}"

    # make pandas dataframe
    df = pd.DataFrame(data=dict(prompt=prompt,
                                classname=classname,
                                classidx=classidx))
    # save to csv
    df.to_csv(f'prompts/{dataset}_prompts.csv', index=False)
