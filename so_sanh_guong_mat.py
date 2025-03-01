import cv2
import numpy as np
import math
from pathlib import Path
import os
import tensorflow as tf
import time

face_cascade = cv2.CascadeClassifier('/home/quoctin/CodePython/FaceRecognition test/haarcascade_frontalface_default.xml')
threshold = 60

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = "/GPU:0" 
        print("Using GPU")
    except RuntimeError as e:
        print(f"Error enabling GPU: {e}")
        device = "/CPU:0"
else:
    device = "/CPU:0"
    print("Using CPU")

def get_eigenface_features(face1, face2):
    with tf.device(device):
        try:
            faces = np.vstack([face1.flatten(), face2.flatten()])
            mean_face = np.mean(faces, axis=0)
            phi = faces - mean_face
            cov = np.dot(phi, phi.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(-eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            eigenfaces = np.dot(eigenvectors.T, phi)
            weights = np.dot(phi, eigenfaces.T)
            return weights
        except Exception as e:
            print(f"Error in get_eigenface_features: {e}")
            return None

def get_lbp_features(img):
    with tf.device(device):
        try:
            lbp = np.zeros_like(img)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    center = img[i, j]
                    code = 0
                    code |= (img[i-1, j-1] >= center) << 7
                    code |= (img[i-1, j] >= center) << 6
                    code |= (img[i-1, j+1] >= center) << 5
                    code |= (img[i, j+1] >= center) << 4
                    code |= (img[i+1, j+1] >= center) << 3
                    code |= (img[i+1, j] >= center) << 2
                    code |= (img[i+1, j-1] >= center) << 1
                    code |= (img[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            return hist
        except Exception as e:
            print(f"Error in get_lbp_features: {e}")
            return None

def get_hog_features(img):
    with tf.device(device):
        try:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
            
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            cell_size = 32
            n_cells_x = int(img.shape[1] / cell_size)
            n_cells_y = int(img.shape[0] / cell_size)
            n_bins = 9
            
            hog_features = []
            for y in range(n_cells_y):
                for x in range(n_cells_x):
                    cell_mag = mag[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
                    cell_ang = angle[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
                    
                    hist = np.zeros(n_bins)
                    for i in range(cell_size):
                        for j in range(cell_size):
                            bin_idx = int(cell_ang[i, j] / 40) % n_bins  
                            hist[bin_idx] += cell_mag[i, j]
                    
                    if np.sum(hist) > 0:
                        hist = hist / np.sqrt(np.sum(hist ** 2) + 1e-7)
                    
                    hog_features.extend(hist)
            
            return np.array(hog_features)
        except Exception as e:
            print(f"Error in get_hog_features: {e}")
            return None
    
def face_comparison(image_path1, image_path2):
    with tf.device(device):
        if not Path(image_path1).exists() or not Path(image_path2).exists():
            raise FileNotFoundError("Một hoặc cả hai file ảnh không tồn tại")
        
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            raise ValueError("Không thể đọc một hoặc cả hai ảnh")
        
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        
        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)
        
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    
        faces1 = face_cascade.detectMultiScale(img1, 1.1, 4)
        faces2 = face_cascade.detectMultiScale(img2, 1.1, 4)
    
        if len(faces1) == 0 or len(faces2) == 0:
            return False, 0.0
    
        face1 = max(faces1, key=lambda x: x[2] * x[3])
        face2 = max(faces2, key=lambda x: x[2] * x[3])
        
        x, y, w, h = face1
        face_img1 = img1[y:y+h, x:x+w]
        x, y, w, h = face2
        face_img2 = img2[y:y+h, x:x+w]
        
        face_img1 = cv2.resize(face_img1, (128, 128))
        face_img2 = cv2.resize(face_img2, (128, 128))
    
        lbp1 = get_lbp_features(face_img1)
        lbp2 = get_lbp_features(face_img2)
        
        hog1 = get_hog_features(face_img1)
        hog2 = get_hog_features(face_img2)
    
        eigen_weights = get_eigenface_features(face_img1, face_img2)
    
        # Tính độ tương đồng bằng nhiều phương pháp
        # 1. Khoảng cách Chi-square cho LBP
        chi_square_dist = np.sum((lbp1 - lbp2) ** 2 / (lbp1 + lbp2 + 1e-10))
        lbp_similarity = 100 * (1 - np.sqrt(chi_square_dist) / 10)  
        
        # 2. Khoảng cách cosine cho HOG
        hog_similarity = 100 * (np.dot(hog1, hog2) / (np.linalg.norm(hog1) * np.linalg.norm(hog2) + 1e-7))
        
        # 3. Khoảng cách Euclidean cho eigenface weights
        eigen_dist = np.linalg.norm(eigen_weights[0] - eigen_weights[1])
        eigen_similarity = 100 * math.exp(-eigen_dist / 1000)
        
        # 4. Tương quan bình phương giữa hai ảnh khuôn mặt
        face_corr = cv2.matchTemplate(face_img1, face_img2, cv2.TM_CCOEFF_NORMED)[0][0]
        template_similarity = 100 * (face_corr + 1) / 2  
        
        weights = [0.3, 0.3, 0.2, 0.2] 
        similarity_score = (
            weights[0] * lbp_similarity + 
            weights[1] * hog_similarity + 
            weights[2] * eigen_similarity + 
            weights[3] * template_similarity
        )
        
        similarity_percentage = round(similarity_score, 2)
        is_same_face = similarity_percentage >= threshold
        
        if is_same_face:
            return is_same_face, similarity_percentage
        else:
            return is_same_face, None

def process_folder(target_image_path, folder_path):
    folder_name = os.path.basename(folder_path)
    print(f"Checking folder: {folder_name}")
    
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    best_match_in_folder = None
    best_similarity_in_folder = 0
    
    for image_path in image_files:
        try:
            print(f"  Processing: {os.path.basename(image_path)}")
            is_same, similarity = face_comparison(target_image_path, image_path)
            image_name = os.path.basename(image_path)
            
            if is_same and similarity is not None:
                print(f"  Match found: {image_name} - Similarity: {similarity}%")
                
                if similarity > best_similarity_in_folder:
                    best_similarity_in_folder = similarity
                    best_match_in_folder = image_path
            else:
                target_name = os.path.basename(target_image_path)
                print(f"  Image {target_name} không trùng khớp với ảnh {image_name}")
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
    
    if best_match_in_folder is not None:
        print(f"  Best match in folder {folder_name}: {os.path.basename(best_match_in_folder)} with {best_similarity_in_folder}% similarity")
        return folder_name, (best_match_in_folder, best_similarity_in_folder)
    else:
        print(f"  No matching faces found in folder {folder_name}")
        return folder_name, None

def main():
    start_time = time.time()
    try:
        image_path1 = "/home/quoctin/CodePython/FaceRecognition test/Face_Recognition/Test_Image/Ly Tran Quoc Tin 2_0.png"
        base_folder_path = "/home/quoctin/CodePython/FaceRecognition test/Face_Recognition/Raw/"
        
        folder_paths = [os.path.join(base_folder_path, folder) for folder in os.listdir(base_folder_path) 
                       if os.path.isdir(os.path.join(base_folder_path, folder))]
        
        folder_results = {}
        
        for folder_path in folder_paths:
            try:
                print(f"\nProcessing folder: {os.path.basename(folder_path)}")
                folder_name, result = process_folder(image_path1, folder_path)
                if result is not None:
                    folder_results[folder_name] = result
            except Exception as e:
                print(f"Error processing folder {os.path.basename(folder_path)}: {e}")
        
        if folder_results:
            sorted_folders = sorted(folder_results.items(), key=lambda x: x[1][1], reverse=True)
            
            best_folder, (best_image, best_score) = sorted_folders[0]
            
            print("\n" + "="*50)
            print(f"Kết quả tốt nhất: Folder '{best_folder}' chứa ảnh trùng khớp cao nhất")
            print(f"Tên file: {os.path.basename(best_image)}")
            print(f"Độ tương đồng: {best_score}%")
            print("="*50)
            
            print("\nTổng hợp các kết quả theo folder:")
            for folder_name, (image_path, similarity) in sorted_folders:
                print(f"Folder: '{folder_name}' - Image {os.path.basename(image_path)} - Similarity percentage: {similarity}%")
        else:
            print("\nKhông tìm thấy khuôn mặt trùng khớp trong bất kỳ folder nào.")
        
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.20f} seconds")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
