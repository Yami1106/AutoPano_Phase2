#!/usr/bin/env python3

'''
Unsupervised Learning - Step 1

Loading the datset 

input :

Data/patch_tain:
i_PA.png
i_PB.png

Data/patch_val:
i_PA.png
i_PB.png

Output:

stacked : (2,H,W)-> input for Homographynet 
PA : (1,H,W)
PB : (1,H,W)
CA:  (4,2) corners in PATCH coordinates (x,y): [[0,0],[W-1,0],[W-1,H-1],[0,H-1]]
'''


from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

class UnsupervisedDataset(Dataset):
    def __init__(self, patch_dir:str, patch_hw=(128,128), return_h4pt_gt:bool=False,):
        # get the absolute path
        self.patch_dir = Path(patch_dir).expanduser().resolve()
        #define patch height and width
        self.patch_h,self.patch_w = patch_hw
        self.return_h4pt_gt = return_h4pt_gt

        #check if path exists
        if not self.patch_dir.exists():
            raise FileNotFoundError(f"patch_dir not found: {self.patch_dir}")

        self.pb_files: List[Path] = sorted(self.patch_dir.glob("*_PB.png"))
        if len(self.pb_files) == 0:
            raise FileNotFoundError(f"No *_PB.png found in: {self.patch_dir}")
        
        #patch corners in patch coordinates
        self.CA = torch.tensor([
            [0,0],
            [self.patch_w-1,0],
            [self.patch_w-1,self.patch_h-1],
            [0,self.patch_h-1]
        ], dtype=torch.float32,
        )

    def __len__(self):
        return len(self.pb_files)
    
    @staticmethod
    def read_gray_img(path:Path) ->np.ndarray:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found or unable to read: {path}")
        
        return img.astype(np.float32)/255.0 #normalize to [0,1]
    
    def __getitem__(self, index:int) ->Dict[str, torch.Tensor]:
        pb_path = self.pb_files[index]
        stem = pb_path.name.replace("_PB.png","")
        pa_path = self.patch_dir / f"{stem}_PA.png"

        PA = self.read_gray_img(pa_path)
        PB = self.read_gray_img(pb_path)
    
        # ceck if there is a size mismatch so that training does not break
        if PA.shape != (self.patch_h, self.patch_w) or PB.shape != (self.patch_h, self.patch_w):
            raise ValueError(
                f"Patch size mismatch for {stem}: "
                f"PA={PA.shape}, PB={PB.shape}, expected={(self.patch_h, self.patch_w)}"
            )
        
        # tensors

        PA_tensor = torch.from_numpy(PA).unsqueeze(0).float()  #(1,H,W)
        PB_tensor = torch.from_numpy(PB).unsqueeze(0).float()  #(1,H,W)
        stacked = torch.cat([PA_tensor, PB_tensor], dim=0).float()  #(2,H,W)

        sample = {
            "stacked": stacked,
            "PA": PA_tensor,
            "PB": PB_tensor,
            "CA": self.CA.clone(),  #(4,2)
            "stem": stem,   
        }

        # load H4pt which is ground truth homography 
        if self.return_h4pt_gt:
            h4pt_path = self.patch_dir / f"{stem}_H4Pt.npy"
            if h4pt_path.exists():
                h4 = np.load(str(h4pt_path)).astype(np.float32)  # (8,)
                sample["H4Pt_gt"] = torch.from_numpy(h4).float()

        return sample

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    this = Path(__file__).resolve()
    print("This path:", this)
    phase2 = this.parents[2]
    print("Phase2 path:", phase2)
    data = phase2 / "Data"

    train_dir = data / "patch_train"
    val_dir = data / "patch_val"
    if not val_dir.exists():
        val_dir = data / "patch_test"

    train_ds = UnsupervisedDataset(str(train_dir), patch_hw=(128,128), return_h4pt_gt=True)
    val_ds   = UnsupervisedDataset(str(val_dir), patch_hw=(128,128), return_h4pt_gt=True)

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))

    # sample check
    xtr = train_ds[0]
    xva = val_ds[0]
    print("Train sample stacked:", xtr["stacked"].shape, "stem:", xtr["stem"])
    print("Val sample stacked:", xva["stacked"].shape, "stem:", xva["stem"])

    # loader check
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    batch = next(iter(train_loader))
    print("Batch stacked:", batch["stacked"].shape)  # (B,2,H,W)

    # visualize one
    # PA = xtr["PA"].squeeze(0).numpy()
    # PB = xtr["PB"].squeeze(0).numpy()
    # plt.figure(figsize=(8,4))
    # plt.subplot(1,2,1); plt.imshow(PA, cmap="gray"); plt.title("PA"); plt.axis("off")
    # plt.subplot(1,2,2); plt.imshow(PB, cmap="gray"); plt.title("PB"); plt.axis("off")
    # plt.tight_layout()
    # plt.show()