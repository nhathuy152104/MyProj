from PIL import Image
import os.path
import torch
import warnings
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import glob

def load_image_path(key, out_field, d):
    out_field = Image.open(d).convert('L')
    return out_field

def convert_tensor(key, d):
    # d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    c=torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    d=(255.0-c)/255.0
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def convert_dict(k, v):
    return { k: v }

class COVID19(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = ['COVID', 'LUNG_OPACITY', 'NORMAL', 'VIRAL PNEUMONIA']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data




    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # True = training set, False = val/test set

        # Gọi hàm generate_ds và truyền cờ self.train vào để biết cần load tập nào
        self.data, self.targets = self.generate_ds(args, self.root, is_train=self.train)


    def generate_ds(self, args, root, is_train):
        num_class = args.num_classes

        data = []
        targets = []
        
        base_dir = root 
        
        # 1. Cố định danh sách class ĐÚNG TÊN THƯ MỤC và ĐÚNG THỨ TỰ (Index 0->3)
        target_classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
        
        num_class = min(num_class, len(target_classes))

        for i in range(num_class):
            class_name = target_classes[i]
            
            # 2. Trỏ thẳng vào thư mục 'images' bên trong từng class cụ thể
            class_images_dir = os.path.join(base_dir, class_name, 'images')
            
            # Kiểm tra an toàn: Nếu nhập sai đường dẫn hoặc thiếu thư mục
            if not os.path.exists(class_images_dir):
                print(f"[LỖI] Không tìm thấy thư mục: {class_images_dir}")
                continue
            
            # Lấy tất cả ảnh png, jpg, jpeg và sắp xếp cố định
            image_paths = sorted(glob.glob(os.path.join(class_images_dir, '*.*')))
            valid_exts = ('.png', '.jpg', '.jpeg')
            image_paths = [p for p in image_paths if p.lower().endswith(valid_exts)]
            
            # --- LOGIC CHIA TRAIN / VAL ---
            # Tính số lượng 1/6 dữ liệu
            val_size = len(image_paths) // 6
            
            if is_train:
                # Bỏ qua 1/6 đầu tiên, lấy 5/6 phần còn lại làm tập Train
                selected_images = image_paths[val_size:]
            else:
                # Lấy đúng 1/6 đầu tiên làm tập Val
                selected_images = image_paths[:val_size]
            
            # Gán nhãn i (0, 1, 2, 3) cho các ảnh thuộc class tương ứng
            for img_path in selected_images:
                data.append(img_path)
                targets.append(i)

        targets = torch.tensor(targets, dtype=torch.long)
        
        mode = "Train" if is_train else "Val"
        print(f"Đã load tập {mode}: {len(data)} ảnh")
        
        return data, targets
    # def generate_ds_test(self, args, root):
    #     # read 100 images per classes per style

    #     num_class = args.num_classes
    #     # num_style = args.num_styles
    #     num_img = args.test_shots * args.num_users

    #     data = []
    #     # targets = torch.zeros([num_class * num_style * num_img])
    #     targets = torch.zeros([num_class * num_img])
    #     files = os.listdir(os.path.join(root, 'data', 'raw_data', 'by_class'))

    #     for i in range(num_class):
    #         for k in range(num_img):
    #             img = os.path.join(root, 'data', 'raw_data', 'by_class', files[i], 'hsf_0', 'hsf_0'+'_00'+str("%03d"%(k))+'.png')
    #             data.append(img)
    #             targets[i * num_img + k] = i

    #     targets = targets.reshape([num_class * num_img])

    #     return data, targets