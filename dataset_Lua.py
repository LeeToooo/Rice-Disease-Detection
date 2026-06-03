from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import Resize, Compose, ToTensor
from PIL import Image


def resolve_dataset_root(root, max_depth=4):
    """Find the dataset root directory containing both train/ and test/."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return None

    if os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "test")):
        return root

    queue = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        try:
            entries = os.listdir(current)
        except PermissionError:
            continue

        for entry in entries:
            path = os.path.join(current, entry)
            if not os.path.isdir(path):
                continue
            if os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "test")):
                return path
            queue.append((path, depth + 1))

    return None


def normalize_category(name: str) -> str:
    return name.strip().lower().replace(' ', '_')


class Lua_dataset(Dataset):
    def __init__(self, root, is_train, transforms=None, categories=None):
        resolved_root = resolve_dataset_root(root)
        if resolved_root is None:
            raise FileNotFoundError(
                f"Could not resolve dataset root from {root}. "
                "Expected a directory containing 'train/' and 'test/'."
            )
        root = resolved_root

        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        raw_dirs = [folder for folder in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, folder))]

        raw_map = {}
        for folder in raw_dirs:
            normalized = normalize_category(folder)
            raw_map.setdefault(normalized, []).append(folder)

        if categories is None:
            self.categories = sorted(raw_map.keys())
        else:
            self.categories = [normalize_category(cat) for cat in categories]

        self.image_paths = []
        self.labels = []

        for index, category in enumerate(self.categories):
            raw_folders = raw_map.get(category, [])
            for raw_folder in raw_folders:
                subdir_path = os.path.join(data_path, raw_folder)
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)
                    if os.path.isfile(file_path):
                        self.image_paths.append(file_path)
                        self.labels.append(index)

        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # image = cv2.imread(self.image_paths[index])
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)             
        return image, label                             

if __name__ == "__main__":
    transforms = Compose([
        ToTensor(),
        Resize((240,240))
    ])
    Train_Data = Lua_dataset(root="E:\Học Sâu\Rice_Leaf_Diease-master",is_train=True,transforms=transforms)
    train_Dataloader = DataLoader(
        dataset = Train_Data,
        batch_size=8,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )
    for images, labels in train_Dataloader:
        print(images.shape , labels.shape)