from typing import List
import shutil
from natsort import natsorted
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def _copy_file_to_dataset_dir(files: List[Path], dataset_dir: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files, desc=f'Copying Files to {dataset_dir.name}'):
        shutil.copyfile(str(file), str(dataset_dir / file.name))


def _generate_blurry(x: np.ndarray, w: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    x = (x / 256).astype('float')
    channels = x.shape[-1]
    padding = np.int((x.shape[0] - w.shape[0]) / 2)
    w = np.lib.pad(w, padding, mode='constant', constant_values=0)

    # applying blur kernel on each channel in freq. domain
    y_f = np.zeros(x.shape, dtype=complex)
    for i in range(channels):
        y_f[:, :, i] = np.fft.fft2(x[:, :, i]) * np.fft.fft2(w)

    # converting to spatial domain
    y = np.zeros(x.shape)
    for i in range(channels):
        y[:, :, i] = np.fft.fftshift(np.fft.ifft2(y_f[:, :, i]).real)

    # adding noise
    noise = np.random.normal(0, noise_std, size=y.shape)
    y = y + noise
    y = np.clip(y, 0, 1) * 256
    y = y.astype(np.uint8)
    return y


def _get_np_array_from_image_path(path: Path) -> np.ndarray:
    image = Image.open(str(path))
    return np.array(image)


def _save_image_from_np_array(path: Path, image_np: np.ndarray):
    assert image_np.dtype == np.uint8, f'Image should be of type np.uint8.'
    Image.fromarray(image_np).save(str(path))


def _preprocess_blur(blur: np.ndarray) -> np.ndarray:
    blur = blur / blur.sum()
    return blur


def create_blurry_images_in_dataset(image_files: List[Path], blur_files: List[Path], blurry_dataset_dir: Path):
    blurry_dataset_dir.mkdir(parents=True, exist_ok=True)
    for image_file in tqdm(image_files, desc=f'Creating Blurry images for {blurry_dataset_dir.name}'):
        image_name = image_file.name
        blur_idx = np.random.randint(0, len(blur_files))
        image_np = _get_np_array_from_image_path(image_file)
        blur_np = _get_np_array_from_image_path(blur_files[blur_idx])
        blur_np = _preprocess_blur(blur_np)
        blurry_np = _generate_blurry(image_np, blur_np, noise_std=0.01)
        _save_image_from_np_array(blurry_dataset_dir / image_name, blurry_np, )


if __name__ == '__main__':
    # unzip celeba_data.zip file in training_set folder and run this script followed by run_model.py with training configs.
    dataset_dir = Path('./training_set/data/celebA')
    celeba_files = natsorted(list(dataset_dir.glob('*.jpg')))
    assert len(celeba_files) == 179999, f'There should be 179,999 files in the CelebA folder'
    train_image_files = celeba_files[:-22000]
    test_image_files = celeba_files[-22000:]

    train_blur_dir = Path('./training_set/data/blurs/train')
    train_blur_files = natsorted(list(train_blur_dir.glob('*.jpg')))
    test_blur_dir = Path('./training_set/data/blurs/test')
    test_blur_files = natsorted(list(test_blur_dir.glob('*.jpg')))

    print('Processing Training Data')
    train_dataset_dir = Path('./training_set/sharp')
    _copy_file_to_dataset_dir(train_image_files, train_dataset_dir)
    train_blurry_dataset_dir = Path('./training_set/blurry')
    create_blurry_images_in_dataset(image_files=train_image_files, blur_files=train_blur_files,
                                    blurry_dataset_dir=train_blurry_dataset_dir)

    print('Processing Test Data')
    test_dataset_dir = Path('./testing_set/sharp')
    _copy_file_to_dataset_dir(test_image_files, test_dataset_dir)
    test_blurry_dataset_dir = Path('./testing_set/blurry')
    create_blurry_images_in_dataset(image_files=test_image_files, blur_files=test_blur_files,
                                    blurry_dataset_dir=test_blurry_dataset_dir)


    print('Creating celebea_datalist.txt file...')
    sharp_dataset_dir = Path('./training_set/sharp')
    blurry_dataset_dir = Path('./training_set/blurry')
    sharp_images = natsorted(list(sharp_dataset_dir.glob('*.jpg')))
    blurry_images = natsorted(list(blurry_dataset_dir.glob('*.jpg')))
    with open('celeba_datalist.txt', 'w') as f:
        for sharp_image, blurry_image in zip(sharp_images, blurry_images):
            assert sharp_image.name == blurry_image.name, f'{sharp_image.name} vs {blurry_image.name}'
            line = f'sharp/{str(sharp_image.name)} blurry/{str(blurry_image.name)}\n'
            f.write(line)
