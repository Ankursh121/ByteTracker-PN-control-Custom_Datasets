import os
import shutil
import logging
import yaml

logger = logging.getLogger("AntiDroneSystem.DatasetManager")

class DatasetManager:
    def __init__(self, target_dir=None):
        self.workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.target_dir = target_dir or os.path.join(self.workspace_dir, "dataset")
        
        # Candidates for source dataset detection
        self.candidates = [
            os.path.join(os.path.dirname(self.workspace_dir), "coco json drone detection", "dataset_merged"),
            os.path.join(os.path.dirname(self.workspace_dir), "coco json drone detection", "dataset"),
            os.path.join(self.workspace_dir, "drone_dataset"),
            os.path.join(self.workspace_dir, "dataset")
        ]

    def detect_source_dataset(self):
        """
        Scans candidate directories to find a valid YOLO dataset.
        Returns the path to the detected source dataset.
        """
        for path in self.candidates:
            if os.path.exists(path):
                # Verify it has some train or valid images
                train_img_path = os.path.join(path, "train", "images")
                valid_img_path = os.path.join(path, "valid", "images")
                val_img_path = os.path.join(path, "val", "images")
                
                if (os.path.exists(train_img_path) and 
                    (os.path.exists(valid_img_path) or os.path.exists(val_img_path))):
                    logger.info(f"Detected source dataset at: {path}")
                    return path
        
        # Fallback recursive search in parent workspace
        parent_dir = os.path.dirname(self.workspace_dir)
        for root, dirs, files in os.walk(parent_dir):
            if "train" in dirs and ("valid" in dirs or "val" in dirs):
                images_dir = os.path.join(root, "train", "images")
                if os.path.exists(images_dir):
                    logger.info(f"Recursively detected source dataset at: {root}")
                    return root
                    
        raise FileNotFoundError("Could not find any valid source drone dataset with train/valid structure.")

    def validate_and_reconstruct(self):
        """
        Copies, restructures, and validates the dataset into the standard layout:
          - target/images/train
          - target/images/val
          - target/labels/train
          - target/labels/val
        
        Performs label formatting validation, fixes class mapping to 0, 
        and detects corrupted/empty labels or mismatched pairs.
        """
        source_dir = self.detect_source_dataset()
        logger.info(f"Reconstruction target directory: {self.target_dir}")
        
        # Define paths
        subsets = [
            ("train", "train"),
            ("valid", "val"),
            ("val", "val")
        ]
        
        os.makedirs(self.target_dir, exist_ok=True)
        
        summary = {
            "train": {"images": 0, "labels": 0, "empty_labels": 0, "corrupted_labels": 0, "missing_labels": 0},
            "val": {"images": 0, "labels": 0, "empty_labels": 0, "corrupted_labels": 0, "missing_labels": 0},
            "class_distribution": {0: 0}
        }
        
        # Map source subsets to target
        processed_subsets = set()
        for src_sub, tgt_sub in subsets:
            if tgt_sub in processed_subsets:
                continue
                
            src_images_dir = os.path.join(source_dir, src_sub, "images")
            src_labels_dir = os.path.join(source_dir, src_sub, "labels")
            
            if not os.path.exists(src_images_dir):
                continue
                
            processed_subsets.add(tgt_sub)
            
            tgt_images_dir = os.path.join(self.target_dir, "images", tgt_sub)
            tgt_labels_dir = os.path.join(self.target_dir, "labels", tgt_sub)
            
            os.makedirs(tgt_images_dir, exist_ok=True)
            os.makedirs(tgt_labels_dir, exist_ok=True)
            
            # Process all images
            image_files = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for img_file in image_files:
                src_img_path = os.path.join(src_images_dir, img_file)
                tgt_img_path = os.path.join(tgt_images_dir, img_file)
                
                # Copy image if it doesn't exist or is different size
                if not os.path.exists(tgt_img_path) or os.path.getsize(src_img_path) != os.path.getsize(tgt_img_path):
                    shutil.copy2(src_img_path, tgt_img_path)
                
                summary[tgt_sub]["images"] += 1
                
                # Check label file
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}.txt"
                src_label_path = os.path.join(src_labels_dir, label_file)
                tgt_label_path = os.path.join(tgt_labels_dir, label_file)
                
                if not os.path.exists(src_label_path):
                    # Missing label - YOLO allows empty files for background, so create empty file
                    with open(tgt_label_path, 'w') as f:
                        pass
                    summary[tgt_sub]["missing_labels"] += 1
                    summary[tgt_sub]["empty_labels"] += 1
                    continue
                
                # Read and validate/fix label
                summary[tgt_sub]["labels"] += 1
                valid_lines = []
                corrupted = False
                
                try:
                    with open(src_label_path, 'r') as lf:
                        lines = lf.readlines()
                    
                    if not lines or len(lines) == 0 or all(not line.strip() for line in lines):
                        summary[tgt_sub]["empty_labels"] += 1
                    
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            corrupted = True
                            continue
                            
                        # Validate float casting and normalization bounds
                        try:
                            cls_id = int(parts[0])
                            coords = [float(p) for p in parts[1:]]
                            
                            # Ensure coordinates are within range [0, 1]
                            if any(c < 0.0 or c > 1.0 for c in coords):
                                corrupted = True
                                continue
                                
                            # Force class_id to 0 (drone) as per requirements
                            if cls_id != 0:
                                cls_id = 0
                            
                            valid_lines.append(f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords))
                            summary["class_distribution"][0] += 1
                        except ValueError:
                            corrupted = True
                            
                except Exception as e:
                    logger.warning(f"Error reading label file {src_label_path}: {e}")
                    corrupted = True
                
                if corrupted:
                    summary[tgt_sub]["corrupted_labels"] += 1
                    # Save empty file to prevent training failure or copy original safely
                    with open(tgt_label_path, 'w') as f:
                        pass
                else:
                    # Write corrected/validated lines
                    with open(tgt_label_path, 'w') as f:
                        f.write("\n".join(valid_lines) + ("\n" if valid_lines else ""))
        
        # Automatically generate / fix data.yaml
        self.generate_data_yaml()
        
        # Print summary
        self.print_summary(summary)
        return summary

    def generate_data_yaml(self):
        """
        Creates or updates data.yaml under target dataset directory.
        """
        yaml_content = {
            'path': self.target_dir.replace('\\', '/'), # Forward slashes for YOLO
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': {
                0: 'drone'
            }
        }
        
        yaml_path = os.path.join(self.target_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_content, f, default_flow_style=False)
        logger.info(f"Generated data.yaml at {yaml_path}")
        
        # Copy data.yaml to anti_drone_system root configs if needed for easy access
        configs_dir = os.path.join(self.workspace_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        shutil.copy2(yaml_path, os.path.join(configs_dir, "data.yaml"))

    def print_summary(self, summary):
        """Prints a clean text summary of the dataset validation results."""
        print("\n" + "="*50)
        print("           DATASET VALIDATION SUMMARY           ")
        print("="*50)
        print(f"Target Directory: {self.target_dir}")
        print("-"*50)
        for subset in ["train", "val"]:
            sub_sum = summary[subset]
            print(f"Subset: {subset.upper()}")
            print(f"  - Total Images:       {sub_sum['images']}")
            print(f"  - Total Label Files:  {sub_sum['labels']}")
            print(f"  - Empty Labels:       {sub_sum['empty_labels']}")
            print(f"  - Corrupted Labels:   {sub_sum['corrupted_labels']}")
            print(f"  - Missing Labels:     {sub_sum['missing_labels']}")
            print("-"*50)
        print("Class Distribution:")
        print("  - Class 0 (drone):   " + str(summary['class_distribution'][0]))
        print("="*50 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = DatasetManager()
    manager.validate_and_reconstruct()
