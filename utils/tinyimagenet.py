import os, sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class TinyImageNet(Dataset):
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    FILENAME = "tiny-imagenet-200.zip"
    ZIP_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(self, root, train=True, transform=None, download=True):
        self.Train = bool(train)
        self.transform = transform

        root = os.path.expanduser(root)

        # 兼容两种传法：
        # 1) root="./data"            -> root_dir="./data/tiny-imagenet-200"
        # 2) root="./data/tiny-imagenet-200" -> root_dir=同上
        if os.path.basename(os.path.normpath(root)) == "tiny-imagenet-200":
            self.root_dir = root
            self.data_root = os.path.dirname(root)
        else:
            self.data_root = root
            self.root_dir = os.path.join(root, "tiny-imagenet-200")

        if download:
            self._download_if_needed()

        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        # 基本存在性检查
        if not (os.path.isdir(self.train_dir) and os.path.isdir(self.val_dir)):
            raise RuntimeError(
                f"TinyImageNet not found under: {self.root_dir}\n"
                f"Expected folders: train/, val/\n"
                f"Try download=True or put tiny-imagenet-200 there."
            )

        if self.Train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)
        self.targets = [t for _, t in self.images]
        # label name mapping
        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()
        with open(wnids_file, "r") as fo:
            for entry in fo.readlines():
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, "r") as fo:
            for entry in fo.readlines():
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _download_if_needed(self):
        # 已有数据就不下
        if os.path.isdir(self.root_dir) and os.path.isdir(os.path.join(self.root_dir, "train")):
            return
        os.makedirs(self.data_root, exist_ok=True)
        download_and_extract_archive(
            url=self.URL,
            download_root=self.data_root,
            filename=self.FILENAME,
            md5=self.ZIP_MD5,
            remove_finished=True,
        )

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)

        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images += 1

        self.len_dataset = num_images
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")

        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, "r") as fo:
            for data in fo.readlines():
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(self.val_img_to_class)
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, "rb") as f:
            sample = Image.open(f).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, tgt