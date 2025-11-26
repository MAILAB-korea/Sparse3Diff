import os
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import tifffile  # TIFF 파일 처리를 위한 모듈

def load_datasets_from_directory(directory, transform=None, sample_gap=6, overlap=False, overlapping=None):
    dataset_list = []
    # 지원되는 확장자: npy, npz, tif, tiff
    for file in os.listdir(directory):
        if file.endswith(('.npy', '.npz', '.tif', '.tiff')):
            file_path = os.path.join(directory, file)
            dataset = MicroDiffDataset13Channel(
                data_path=file_path,
                transform=transform,
                sample_gap=sample_gap,
                overlap=overlap,
                overlapping=overlapping
            )
            dataset_list.append(dataset)
    if not dataset_list:
        raise ValueError("해당 디렉토리에 지원되는 파일이 없습니다. (npy, npz, tif, tiff)")
    combined_dataset = ConcatDataset(dataset_list)
    return combined_dataset

def visualize_13_channels(data_13ch, start_z=None, save_path=None):
    """
    13채널 데이터를 시각화하여 각 채널을 개별 서브플롯에 그립니다.

    Args:
        data_13ch (numpy.ndarray or torch.Tensor): [13, H, W] 형태의 데이터
        start_z (int, optional): 시작 슬라이스 번호.
        save_path (str, optional): 플롯을 저장할 경로. None이면 화면에 출력.
    """
    if isinstance(data_13ch, torch.Tensor):
        data_13ch = data_13ch.numpy()

    fig, axes = plt.subplots(1, 13, figsize=(26, 3))
    for i in range(13):
        axes[i].imshow(data_13ch[i], cmap='gray')
        axes[i].set_title(f"Ch {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

class MicroDiffDataset13Channel(Dataset):
    def __init__(self, data_path, transform=None, sample_gap=6, overlap=False, overlapping=None):
        """
        data_path: .npz, .npy, .tif 또는 .tiff 파일 경로 (데이터셋 파일)
        transform: 선택적 이미지 전처리 transform
        sample_gap: 두 reference slice(ids) 사이의 간격 (여기서는 6으로 고정)
        overlap: True이면 overlapping slice를 허용
        overlapping: overlap일 때 겹치는 slice 수 (제공되지 않으면 sample_gap // 2가 기본)
        
        stacking 구성:
            - ids1: 시작 인덱스 s
            - ids2: s + sample_gap
            - ids3: s + 2 * sample_gap
            그리고 전체 stacking은 연속된 13개 slice (s부터 s+12까지)를 사용함.
            이때 ids1, ids2, ids3는 각각 stacking된 결과의 1번째, 7번째, 13번째 채널에 해당.
            즉, s : 채널1, s+1~s+5 : 채널2~6, s+6 : 채널7, s+7~s+11 : 채널8~12, s+12 : 채널13.
        """
        self.transform = transform
        self.sample_gap = sample_gap  
        self.total_channels = self.sample_gap * 2 + 1  
        self.overlap = overlap
        self.overlapping = overlapping

        # 파일 확장자에 따른 데이터 로딩
        if data_path.endswith(('.npz', '.npy')):
            loaded_data = np.load(data_path, allow_pickle=True)
            # npz 파일인 경우, dict 형태로 저장되어 있을 가능성이 있음
            try:
                data_dict = loaded_data.item()
                self.slices = data_dict["slices"]  # Expected shape: (N, H, W)
                self.positions = data_dict["positions"]
            except AttributeError:
                # 단순 배열인 경우
                self.slices = loaded_data
                self.positions = np.arange(self.slices.shape[0])
        elif data_path.endswith(('.tif', '.tiff')):
            # TIFF 파일은 다중 페이지 이미지로 로드
            self.slices = tifffile.imread(data_path)  # Expected shape: (N, H, W)
            # 위치 정보가 파일에 없는 경우, 슬라이스 수에 따른 인덱스를 생성
            self.positions = np.arange(self.slices.shape[0])
        else:
            raise ValueError(f"지원되지 않는 파일 확장자: {data_path}")

        # 사용할 시작 인덱스들(ids) 생성
        self.ids = self._get_all_gt_img()
        #print("All ids:", self.ids)  # 전체 ids 출력

    def _get_all_gt_img(self):
        """
        시작 인덱스(s)를 생성합니다.
        각 s에 대해, s부터 s + total_channels - 1 까지의 slice를 stacking할 수 있어야 합니다.
        기본(non-overlap)일 경우, step은 sample_gap.
        만약 overlap 옵션을 사용하면 step을 조정할 수 있습니다.
        """
        z_total = len(self.positions)
        print("Total slices:", z_total)
        valid_range = z_total - self.total_channels  

        if self.overlap:
            overlapping = self.overlapping if self.overlapping is not None else self.sample_gap // 2
            step = self.sample_gap - overlapping
        else:
            step = self.sample_gap

        ids = np.arange(self.sample_gap // 2, valid_range, step)

        if ids.size > 0 and (ids[-1] + self.total_channels > z_total):
            ids = ids[:-1]

        return ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        주어진 시작 인덱스 s를 기준으로, s부터 s + 12까지 총 13개의 연속 slice를 stacking합니다.
        stacking된 결과에서 채널1은 ids1, 채널7은 ids2, 채널13은 ids3에 해당합니다.
        """
        start_idx = self.ids[idx]
        end_idx = start_idx + self.total_channels

        stack = self.slices[start_idx:end_idx]
        z_position = np.arange(start_idx, end_idx)

        if stack.shape[0] != self.total_channels:
            return None

        # 각 slice에 대해 min-max normalization 적용
        stack_tensor = torch.tensor(stack, dtype=torch.float32)
        min_vals = stack_tensor.amin(dim=(1, 2), keepdim=True)
        max_vals = stack_tensor.amax(dim=(1, 2), keepdim=True)
        normalized_stack = (stack_tensor - min_vals) / (max_vals - min_vals)
        normalized_stack = normalized_stack.numpy()

        if self.transform:
            normalized_stack = self.transform(normalized_stack)

        return {"image": normalized_stack, "z_position": z_position}

    @staticmethod
    def train_val_split(dataset, val_ratio=0.2, seed=None):
        dataset_size = len(dataset)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size

        if seed is not None:
            np.random.seed(seed)

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset

if __name__ == "__main__":
    # 단일 파일로 테스트할 경우 (npz 또는 tif)

    #root_file = '../../../data/microdiffusion_dataset/128_gt/Vasculature.npy'
    root_file = '../../../data/allen_additional_test/dna.tif'
    root_file = os.path.expanduser(root_file)

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    '''
    dataset = MicroDiffDataset13Channel(root_file, transform=train_transforms, sample_gap=6)
    train_dataset, val_dataset = MicroDiffDataset13Channel.train_val_split(dataset, val_ratio=0, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    for i, batch in enumerate(train_loader):
        img_batch = batch['image'].permute(0, 2, 1, 3)
        z_position_batch = batch['z_position']
        save_path = f"vis/data_vis/visualization_indi_{i}.png"
        visualize_13_channels(img_batch[0], start_z=batch["z_position"][0], save_path=save_path)
        print(f"Batch {i} image shape:", img_batch.shape)
        print(f"Batch {i} z_position shape:", z_position_batch.shape)
        break
    '''

    
    # 디렉토리 내의 모든 파일을 사용하여 데이터셋 생성 (npz, tif 등 지원)
    #root_dir = '../../../data/microdiffusion_dataset/128_gt/'
    root_dir = '../../../data/allen_additional_test/'

    combined_dataset = load_datasets_from_directory(root_dir, transform=train_transforms, sample_gap=6)

    train_dataset, val_dataset = MicroDiffDataset13Channel.train_val_split(combined_dataset, val_ratio=0.0, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    for i, batch in enumerate(train_loader):
        img_batch = batch['image'].permute(0, 2, 1, 3)  # (B, H, 13, W) -> (B, 13, H, W)
        z_position_batch = batch['z_position']
        save_path = f"vis/data_vis/visualization_combined_{i}.png"
        visualize_13_channels(img_batch[0], start_z=batch["z_position"][0], save_path=save_path)
        print(f"Batch {i} image shape:", img_batch.shape)
        print(f"Batch {i} z_position shape:", z_position_batch.shape)
        import pdb
        pdb.set_trace()
        break
    