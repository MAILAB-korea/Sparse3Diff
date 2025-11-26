import os
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


def visualize_7_channels(data_7ch, file_name=None, z_position=None, save_path=None):
    """
    7채널 데이터를 시각화하여 각 채널을 개별 서브플롯으로 그립니다.

    Args:
        data_7ch (numpy.ndarray or torch.Tensor): [7, 64, 64] 형태의 데이터
        file_name (str, optional): 현재 시각화 중인 파일 이름.
        z_position (array-like, optional): 해당 샘플의 7채널 위치 정보 (예: [start, start+1, ..., start+6])
        save_path (str, optional): 플롯을 저장할 경로. None일 경우 화면에만 표시.
    """
    # Tensor를 numpy로 변환 (필요한 경우)
    if isinstance(data_7ch, torch.Tensor):
        data_7ch = data_7ch.numpy()

    # 7개의 채널 각각을 서브플롯으로 시각화
    fig, axes = plt.subplots(1, 7, figsize=(20, 5))  # 1행 7열의 서브플롯
    for i in range(7):
        axes[i].imshow(data_7ch[i], cmap="gray")
        axes[i].set_title(f"Channel {i+1}")
        axes[i].axis("off")

    # z_position은 배열이므로, 첫 번째 값 또는 전체를 표시할 수 있음
    title_info = (
        f"File: {file_name}, z_position: {z_position.tolist() if hasattr(z_position, 'tolist') else z_position}"
    )
    plt.suptitle(title_info, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


class MicroDataset7Channel(Dataset):
    def __init__(self, root_dir, transform=None, z_sample=7):
        self.root_dir = root_dir
        self.transform = transform

        # 디렉토리 내 모든 TIFF 파일 로드
        self.files = sorted(
            [f for f in os.listdir(root_dir) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
        )

        # 사용할 슬라이스 정보(파일 경로, 시작 z 인덱스)를 저장
        self.z_sample = z_sample
        self.slice_info = self._create_slice_info()

    def _create_slice_info(self):
        slice_info = []
        for file_name in self.files:
            file_path = os.path.join(self.root_dir, file_name)
            with tiff.TiffFile(file_path) as tif_file:
                num_slices = len(tif_file.pages)
                # 시작 인덱스를 2부터 num_slices - 6까지 사용 (7장의 슬라이스를 확보하기 위함)
                for start_z in range(2, num_slices - self.z_sample):
                    # for start_z in range(2, num_slices - 6):
                    slice_info.append((file_path, start_z))
        return slice_info

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        file_path, start_z = self.slice_info[idx]

        with tiff.TiffFile(file_path) as tif_file:
            slices = []
            for z in range(start_z, start_z + self.z_sample + 1):
                slice_img = tif_file.pages[z].asarray().astype(np.float32)
                slice_img = torch.tensor(slice_img)

                # Min-max 정규화
                smin, smax = slice_img.min(), slice_img.max()
                if smax > smin:
                    slice_img = (slice_img - smin) / (smax - smin)
                else:
                    slice_img = torch.zeros_like(slice_img)

                slices.append(slice_img)

        slices_7ch = torch.stack(slices, dim=0)
        slices_7ch = slices_7ch.numpy()
        # print(slices_7ch.shape)

        # Transform 적용 (필요시)
        if self.transform:
            slices_7ch = self.transform(slices_7ch)

        # start_z 대신 각 채널에 해당하는 z position을 생성 (shape: (7,))
        z_position = np.arange(start_z, start_z + self.z_sample + 1)

        return {
            "image": slices_7ch,
            "file_name": os.path.basename(file_path),
            "z_position": z_position,  # shape: (7,)
        }

    @staticmethod
    def train_val_split(dataset, val_ratio=0.2, seed=None):
        dataset_size = len(dataset)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size

        if seed is not None:
            torch.manual_seed(seed)

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset


if __name__ == "__main__":
    root_dir = "../../data/crop_128_resize_crop_add_more/"
    root_dir = os.path.expanduser(root_dir)

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = MicroDataset7Channel(root_dir, transform=train_transforms)

    train_dataset, val_dataset = MicroDataset7Channel.train_val_split(dataset, val_ratio=0.2, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    for i, batch in enumerate(train_loader):
        # 이미지 텐서는 (B, C, H, W) 형태로 변환 (이미 transform 후라면 확인)
        img_batch = batch["image"].permute(0, 2, 1, 3)  # 예: (B, 7, H, W)
        # z_position은 이미 (batch_size, 7) 형태임
        z_position_batch = batch["z_position"]

        save_path = f"vis/data_vis/test_again_{i}.png"
        # 시각화 함수에 z_position 전달 (배치 내 첫 번째 sample)
        visualize_7_channels(
            img_batch[0],
            file_name=batch["file_name"][0],
            z_position=z_position_batch[0],
            save_path=save_path,
        )
        print(f"Visualization saved at: {save_path}")
        pdb.set_trace()
